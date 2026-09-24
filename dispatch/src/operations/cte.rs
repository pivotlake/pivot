//! A CTE: one chain of operators produces rows that several other chains, each
//! reading every row, consume.
//!
//! The producing chain (the CTE's definition) ends in a [`CteFanOut`], which
//! pushes each batch into one ordinary work-stealing channel per scan site. A
//! site reads its channel the way every stage reads its input: pop this
//! worker's queue, steal from same-node peers only once it is dry. So a batch
//! is consumed on the worker that produced it unless somebody else has run out
//! of work, and the sites' copies share its buffers.
//!
//! The graph carries an edge from the definition to each site, which is what
//! keeps a site from mistaking a lull for the end: a node cannot finish while
//! any of its publishers is still running, so a site finishes only once the
//! chain feeding it has. It also puts the sites downstream of the definition in
//! the leaf-to-root walk, so a worker consumes the batch it just produced
//! before producing another.

use crate::api::{BuildContext, OperatorFactory, OperatorGraphBuilder};
use crate::operations::channels::{ChannelFactory, Sender, StealableChannelFactory};
use crate::operations::in_memory::Forward;
use crate::operations::unary::SiblingBarrier;
use crate::operations::unary::UnaryOperator;
use arrow_array::RecordBatch;
use crossbeam_deque::Worker;
use std::rc::Rc;
use std::sync::Arc;

/// The tail of a CTE's definition chain: sends every batch to every scan site.
///
/// Cloning a `RecordBatch` bumps a reference count per column, so a site's copy
/// shares the producer's buffers rather than duplicating them.
pub struct CteFanOut {
    sites: Vec<Rc<Worker<RecordBatch>>>,
}

impl CteFanOut {
    pub(crate) fn new(sites: Vec<Rc<Worker<RecordBatch>>>) -> Self {
        Self { sites }
    }
}

impl Sender<RecordBatch> for CteFanOut {
    fn send(&mut self, batch: RecordBatch) -> crate::operations::channels::Result<()> {
        let Some((last, rest)) = self.sites.split_last_mut() else {
            return Ok(());
        };
        for site in rest {
            site.send(batch.clone())?;
        }
        last.send(batch)
    }
}

/// One scan site of a CTE: a root reading the channel it creates here.
///
/// The sending end is left in the [`BuildContext`] for the CTE's definition
/// chain, which an ancestor of every site builds once they all have.
pub struct CteScanFactory {
    pub channel_factory: StealableChannelFactory<RecordBatch>,
    pub cte_index: usize,
    pub siblings_left: Arc<SiblingBarrier>,
}

impl OperatorFactory<RecordBatch> for CteScanFactory {
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<RecordBatch>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let (tx, rx) = self.channel_factory.build();
        context.deposit_cte_sender(self.cte_index, tx);
        OperatorGraphBuilder::root(Box::new(UnaryOperator::new(
            Forward::default(),
            rx,
            sender,
            self.siblings_left,
        )))
        .awaiting_input(self.cte_index)
    }
}

/// A CTE: the body, plus the definition chain that feeds every scan site in it.
pub struct CteFactory {
    pub definition_head: Box<dyn OperatorFactory<RecordBatch>>,
    pub body_head: Box<dyn OperatorFactory<RecordBatch>>,
    pub cte_index: usize,
    pub sites: usize,
}

impl OperatorFactory<RecordBatch> for CteFactory {
    /// Build the body, then the definition that feeds it, as a side graph.
    ///
    /// The order is the point: a site only creates its channel when it is built,
    /// so the definition can be given the sending ends only once the whole body
    /// has been walked.
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<RecordBatch>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let body = self.body_head.build(sender, context);

        let sites = context.take_cte_senders(self.cte_index);
        assert_eq!(
            sites.len(),
            self.sites,
            "every scan site leaves the definition a sending end to fan out to"
        );
        let definition = self
            .definition_head
            .build(Box::new(CteFanOut::new(sites)), context);

        body.with_producer_side_graph(definition, self.cte_index)
    }
}
