use std::any::Any;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::RecordBatch;
use crossbeam_deque::Worker;

use crate::api::Chain;
use std::sync::atomic::AtomicBool;

use crate::operations::channels::{ChannelFactory, MpscSender, Sender, StealableChannelFactory, VoidSender};
use crate::operations::{BinaryFactory, BinaryOperator, JoinBuildFactory, JoinProbeFactory, UnaryFactory, UnaryOperator, UnaryOperatorFactory};
use crate::{OperatorFactory, OperatorSpec, RecordBatchOperatorSpec};
use super::RecordBatchOperatorFactory;

/// Type-erased unary operator factory for `RecordBatch -> RecordBatch` stages.
///
/// Wraps a type-erased head (`Box<dyn RecordBatchOperatorFactory>`) with a concrete
/// unary operation (e.g. filter, project, count). The `UF` type parameter is the
/// concrete `UnaryFactory` — it gets erased when this struct is boxed as
/// `Box<dyn RecordBatchOperatorFactory>`.
///
/// Created internally by `RecordBatchOperatorSpec::unary` — not constructed directly.
pub struct RecordBatchUnaryOperatorFactory<UF: UnaryFactory<RecordBatch, RecordBatch>> {
    pub(super) head: Box<dyn RecordBatchOperatorFactory>,
    pub(super) unary_factory: UF,
    pub(super) channel_factory: StealableChannelFactory<RecordBatch>,
    pub(super) siblings_left: Arc<AtomicUsize>,
}

impl<UF: UnaryFactory<RecordBatch, RecordBatch>> RecordBatchOperatorFactory
    for RecordBatchUnaryOperatorFactory<UF>
{
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> Chain {
        let (tx, rx) = self.channel_factory.build();
        let chain = self.head.build_stealable(tx);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }

    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> Chain {
        let (tx, rx) = self.channel_factory.build();
        let chain = self.head.build_stealable(tx);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }
}

/// Type-erased binary operator factory for `(RecordBatch, RecordBatch) -> RecordBatch` stages.
///
/// Wraps two type-erased heads (`Box<dyn RecordBatchOperatorFactory>`) with a concrete
/// binary operation (e.g. concat). The `BF` type parameter is the concrete
/// [`BinaryFactory`] — it gets erased when this struct is boxed as
/// `Box<dyn RecordBatchOperatorFactory>`.
pub struct RecordBatchBinaryOperatorFactory<
    BF: BinaryFactory<RecordBatch, RecordBatch, RecordBatch>,
> {
    pub(super) left_head: Box<dyn RecordBatchOperatorFactory>,
    pub(super) right_head: Box<dyn RecordBatchOperatorFactory>,
    pub(super) binary_factory: BF,
    pub(super) left_channel_factory: StealableChannelFactory<RecordBatch>,
    pub(super) right_channel_factory: StealableChannelFactory<RecordBatch>,
    pub(super) siblings_left: Arc<AtomicUsize>,
}

impl<BF: BinaryFactory<RecordBatch, RecordBatch, RecordBatch>> RecordBatchOperatorFactory
    for RecordBatchBinaryOperatorFactory<BF>
{
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> Chain {
        let (left_tx, left_rx) = self.left_channel_factory.build();
        let (right_tx, right_rx) = self.right_channel_factory.build();
        let left_chain = self.left_head.build_stealable(left_tx);
        let right_chain = self.right_head.build_stealable(right_tx);
        Chain::merge(
            left_chain,
            right_chain,
            Box::new(BinaryOperator::new(
                self.binary_factory.build_binary(),
                left_rx,
                right_rx,
                sender,
                self.siblings_left,
            )),
        )
    }

    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> Chain {
        let (left_tx, left_rx) = self.left_channel_factory.build();
        let (right_tx, right_rx) = self.right_channel_factory.build();
        let left_chain = self.left_head.build_stealable(left_tx);
        let right_chain = self.right_head.build_stealable(right_tx);
        Chain::merge(
            left_chain,
            right_chain,
            Box::new(BinaryOperator::new(
                self.binary_factory.build_binary(),
                left_rx,
                right_rx,
                sender,
                self.siblings_left,
            )),
        )
    }
}

/// Builds both sides of a hash join into a single [`Chain`] via [`Chain::parallel`].
///
/// The probe side's output (`RecordBatch`) becomes the factory's output. The build
/// side's output (`()`) is discarded via [`VoidSender`].
pub(super) struct JoinOperatorFactory {
    pub(super) build_head: Box<dyn RecordBatchOperatorFactory>,
    pub(super) probe_head: Box<dyn RecordBatchOperatorFactory>,
    pub(super) build_factory: JoinBuildFactory,
    pub(super) probe_factory: JoinProbeFactory,
    pub(super) build_channel: StealableChannelFactory<RecordBatch>,
    pub(super) probe_channel: StealableChannelFactory<RecordBatch>,
    pub(super) build_siblings_left: Arc<AtomicUsize>,
    pub(super) probe_siblings_left: Arc<AtomicUsize>,
    pub(super) gate: Arc<AtomicBool>,
}

impl JoinOperatorFactory {
    fn build_inner<S: Sender<RecordBatch> + 'static>(self, sender: S) -> Chain {
        // Build side: scan → channel → JoinBuild → VoidSender
        let (build_tx, build_rx) = self.build_channel.build();
        let build_chain = self.build_head.build_stealable(build_tx).with(Box::new(
            UnaryOperator::new(
                self.build_factory.build_unary(),
                build_rx,
                VoidSender,
                self.build_siblings_left,
            ),
        ));

        // Probe side: scan (gated) → channel → Probe → sender
        let (probe_tx, probe_rx) = self.probe_channel.build();
        let probe_chain = self.probe_head.build_stealable(probe_tx).gate_roots(self.gate).with(Box::new(
            UnaryOperator::new(
                self.probe_factory.build_unary(),
                probe_rx,
                sender,
                self.probe_siblings_left,
            ),
        ));

        Chain::parallel(build_chain, probe_chain)
    }
}

impl RecordBatchOperatorFactory for JoinOperatorFactory {
    fn build_stealable(self: Box<Self>, sender: Rc<Worker<RecordBatch>>) -> Chain {
        (*self).build_inner(sender)
    }

    fn build_collect(self: Box<Self>, sender: MpscSender<RecordBatch>) -> Chain {
        (*self).build_inner(sender)
    }
}

/// Bridges a `Box<dyn RecordBatchOperatorFactory>` back into `OperatorFactory<RecordBatch>`.
///
/// Used by [`RecordBatchOperatorSpec::materialize`] to feed type-erased factories back
/// into the generic [`UnaryOperatorFactory`] / [`OperatorSpec::read_parquet`] pipeline.
/// Uses [`Any`] downcasting to dispatch the sender to the correct build method at runtime.
pub struct RecordBatchFactoryBridge(pub Box<dyn RecordBatchOperatorFactory>);

impl OperatorFactory<RecordBatch> for RecordBatchFactoryBridge {
    fn build<S: Sender<RecordBatch> + 'static>(self: Box<Self>, sender: S) -> Chain {
        let sender_any: Box<dyn Any> = Box::new(sender);
        match sender_any.downcast::<Rc<Worker<RecordBatch>>>() {
            Ok(s) => self.0.build_stealable(*s),
            Err(sender_any) => match sender_any.downcast::<MpscSender<RecordBatch>>() {
                Ok(s) => self.0.build_collect(*s),
                Err(_) => panic!("RecordBatchFactoryBridge: unsupported sender type"),
            },
        }
    }
}
