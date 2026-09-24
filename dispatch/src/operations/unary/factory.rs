//! Factories that produce [`UnaryOperator`]s during the build step.
//!
//! Two factory types:
//!
//! - [`UnaryOperatorFactory`] — Chains a head factory (`OP`) with a [`UnaryFactory`] and
//!   a [`ChannelFactory`]. When built, it creates the channel, builds the head (passing
//!   it the channel's sender), and creates a [`UnaryOperator`] reading from the channel's
//!   receiver and writing to the downstream sender.
//!
//! - [`RootUnaryOperatorFactory`] — Same but for the root of a dataflow (no head).
//!   Uses a [`RootChannelFactory`] that only produces a receiver (the sender lives
//!   externally, e.g. a shared injector queue).

use crate::api::OperatorGraphBuilder;
use crate::api::{BuildContext, OperatorFactory};
use crate::operations::channels::ChannelFactory;
use crate::operations::channels::{RootChannelFactory, Sender};
use crate::operations::unary::SiblingBarrier;
use crate::operations::unary::{Unary, UnaryOperator};
use std::sync::Arc;

/// Creates a [`Unary`] transform instance. Consumed once per worker during the build step.
pub trait UnaryFactory<I, O>: Send + 'static {
    type Unary: Unary<I, O>;
    fn build_unary(self) -> Self::Unary;
}

/// An [`OperatorFactory`] that chains a head factory with a unary stage.
///
/// Type parameters:
/// - `I` — input type (what this operator reads from its channel)
/// - `O` — output type (what this operator sends downstream)
/// - `UF` — the [`UnaryFactory`] that creates the transform
/// - `C` — the [`ChannelFactory`] connecting the head to this operator
/// - `OP` — the head's [`OperatorFactory`] (upstream)
///
/// When [`build`](OperatorFactory::build) is called:
/// 1. The channel factory produces `(tx, rx)`
/// 2. The head is built with `tx` as its output sender
/// 3. A [`UnaryOperator`] is created with the `UF`'s transform, `rx`, and the downstream sender
pub struct UnaryOperatorFactory<
    I,
    O,
    UF: UnaryFactory<I, O>,
    C: ChannelFactory<I>,
    OP: OperatorFactory<I>,
> {
    head: OP,
    siblings_left: Arc<SiblingBarrier>,
    unary_factory: UF,
    channel_factory: C,
    _phantom: std::marker::PhantomData<fn(I) -> O>,
}

impl<I, O, UF: UnaryFactory<I, O>, C: ChannelFactory<I>, OP: OperatorFactory<I>>
    UnaryOperatorFactory<I, O, UF, C, OP>
{
    pub fn new(
        head: OP,
        unary_factory: UF,
        channel_factory: C,
        siblings_left: Arc<SiblingBarrier>,
    ) -> Self {
        Self {
            head,
            siblings_left,
            unary_factory,
            channel_factory,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I: 'static, O: 'static, UF: UnaryFactory<I, O>, C: ChannelFactory<I>, OP: OperatorFactory<I>>
    OperatorFactory<O> for UnaryOperatorFactory<I, O, UF, C, OP>
{
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<O>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let (tx, rx) = self.channel_factory.build();
        let chain = Box::new(self.head).build(Box::new(tx), context);
        chain.with(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }
}

/// Like [`UnaryOperatorFactory`], but for the root of a dataflow (no upstream head).
///
/// Uses a [`RootChannelFactory`] that only produces a receiver — the sender lives
/// externally (e.g. a shared [`Injector`](crossbeam_deque::Injector) queue that
/// distributes row group requests across workers).
pub struct RootUnaryOperatorFactory<I, O, UF: UnaryFactory<I, O>, C: RootChannelFactory<I>> {
    siblings_left: Arc<SiblingBarrier>,
    unary_factory: UF,
    channel_factory: C,
    _phantom: std::marker::PhantomData<fn(I) -> O>,
}

impl<I, O, UF: UnaryFactory<I, O>, C: RootChannelFactory<I>> RootUnaryOperatorFactory<I, O, UF, C> {
    pub fn new(unary_factory: UF, channel_factory: C, siblings_left: Arc<SiblingBarrier>) -> Self {
        Self {
            siblings_left,
            unary_factory,
            channel_factory,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I: 'static, O: 'static, UF: UnaryFactory<I, O>, C: RootChannelFactory<I>> OperatorFactory<O>
    for RootUnaryOperatorFactory<I, O, UF, C>
{
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<O>>,
        _context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let rx = self.channel_factory.build();
        OperatorGraphBuilder::root(Box::new(UnaryOperator::new(
            self.unary_factory.build_unary(),
            rx,
            sender,
            self.siblings_left,
        )))
    }
}
