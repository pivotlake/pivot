use crate::Identifier;
use crate::api::BuildContext;
use crate::data_flow::DataFlow;
use crate::operations::Operator;
use crate::stats::DataFlowStats;
use ahash::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, mpsc};
use thiserror::Error;

static ID: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

fn next_dataflow_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    PanicOnBuild(String),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// A build-time operator graph assembled by factories on a worker thread.
///
/// `output` identifies the leaf to which ordinary unary stages append. Disconnected
/// side graphs can be merged without changing that output, allowing one dataflow to
/// contain multiple roots while preserving the fluent pipeline's primary result path.
pub struct OperatorGraphBuilder {
    operators: Vec<Box<dyn Operator>>,
    edges: Vec<(Identifier, Identifier)>,
    roots: Vec<Identifier>,
    gates: HashMap<Identifier, Vec<Arc<AtomicBool>>>,
    /// Nodes whose publisher is built later, grouped by a tag both sides agree
    /// on. A node registers itself here when the stage feeding it lives in a
    /// graph that doesn't exist yet, and
    /// [`with_producer_side_graph`](Self::with_producer_side_graph) wires them
    /// up once it does.
    awaiting_input: HashMap<usize, Vec<Identifier>>,
    output: Identifier,
}

impl OperatorGraphBuilder {
    /// Start a graph with one root, which is also its designated output.
    pub fn root(operator: Box<dyn Operator>) -> Self {
        Self {
            operators: vec![operator],
            edges: Vec::new(),
            roots: vec![0],
            gates: HashMap::default(),
            awaiting_input: HashMap::default(),
            output: 0,
        }
    }

    /// Append an operator to the graph's designated output path.
    pub fn with(mut self, operator: Box<dyn Operator>) -> Self {
        let next = self.operators.len();
        self.edges.push((self.output, next));
        self.operators.push(operator);
        self.output = next;
        self
    }

    /// Prevent every current root from running until `gate` is published.
    ///
    /// Gates compose: when a graph is used as the input to nested joins, a root
    /// runs only after all gates attached to it have opened.
    pub fn gated_by(mut self, gate: Arc<AtomicBool>) -> Self {
        for &root in &self.roots {
            self.gates.entry(root).or_default().push(gate.clone());
        }
        self
    }

    /// Mark this graph's output as waiting for a publisher that some ancestor
    /// will supply, identified by `tag`.
    pub fn awaiting_input(mut self, tag: usize) -> Self {
        self.awaiting_input
            .entry(tag)
            .or_default()
            .push(self.output);
        self
    }

    /// Merge a disconnected side graph while preserving this graph's output path.
    pub fn with_side_graph(mut self, mut side: Self) -> Self {
        let offset = self.operators.len();
        self.operators.append(&mut side.operators);
        self.edges.extend(
            side.edges
                .into_iter()
                .map(|(publisher, subscriber)| (publisher + offset, subscriber + offset)),
        );
        self.roots
            .extend(side.roots.into_iter().map(|root| root + offset));
        self.gates.extend(
            side.gates
                .into_iter()
                .map(|(root, gates)| (root + offset, gates)),
        );
        for (tag, nodes) in side.awaiting_input {
            self.awaiting_input
                .entry(tag)
                .or_default()
                .extend(nodes.into_iter().map(|node| node + offset));
        }
        self
    }

    /// Merge a side graph whose output feeds every node registered under `tag`
    /// by [`awaiting_input`](Self::awaiting_input).
    ///
    /// Unlike [`with_side_graph`](Self::with_side_graph) the two halves end up
    /// connected, so the scheduler treats the merged side as upstream of those
    /// nodes: it runs them before making more of its output, and holds their
    /// finish until it has finished.
    pub fn with_producer_side_graph(self, side: Self, tag: usize) -> Self {
        let producer = self.operators.len() + side.output;
        let mut merged = self.with_side_graph(side);
        let subscribers = merged.awaiting_input.remove(&tag).unwrap_or_default();
        merged
            .edges
            .extend(subscribers.into_iter().map(|node| (producer, node)));
        merged
    }

    /// Convert this build-time graph into an executable `DataFlow`.
    pub fn into_data_flow(
        self,
        cancelled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<crate::data_flow::Error>,
        stats_tx: mpsc::Sender<DataFlowStats>,
        collect_stats: bool,
    ) -> DataFlow {
        DataFlow::new(
            next_dataflow_id(),
            cancelled,
            err_tx,
            self.operators,
            self.edges,
            self.gates,
            stats_tx,
            collect_stats,
        )
    }
}

/// A factory + output sender pair, ready to be sent to a worker thread.
///
/// Created by [`RecordBatchOperatorSpec::collect`](super::record_batch_operator::RecordBatchOperatorSpec::collect),
/// one per worker. The worker calls [`build`](DataFlowBuilder::build) to produce a `DataFlow`,
/// which triggers the recursive factory build chain on the worker thread.
pub struct DataFlowBuilder {
    /// A flag shared across all workers (and the DataFlowHandle) on whether to cancel this query
    cancelled: Arc<AtomicBool>,
    /// A sender for errors that may occur during running
    err_tx: mpsc::Sender<crate::data_flow::Error>,
    /// Where this worker's stats tally is shipped (used only when `collect_stats`)
    stats_tx: mpsc::Sender<DataFlowStats>,
    /// Whether the query opted into per-dataflow stats collection
    collect_stats: bool,
    /// Whether this dataflow is profiled: it runs exclusively (the worker pauses
    /// other dataflows) while a `perf` capture is in progress. Set when the
    /// launching dispatcher was marked (see [`with_profiling`](Self::with_profiling)).
    #[cfg(feature = "perf")]
    profiled: bool,
    /// How many of its node's following workers the receiving worker wakes
    /// (see [`waking_siblings`](Self::waking_siblings)).
    siblings_to_wake: usize,
    /// Builds the per-worker operator graph.
    build: Box<dyn FnOnce(&mut BuildContext) -> OperatorGraphBuilder + Send>,
}

impl DataFlowBuilder {
    pub fn new(
        build: Box<dyn FnOnce(&mut BuildContext) -> OperatorGraphBuilder + Send>,
        cancelled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<crate::data_flow::Error>,
        stats_tx: mpsc::Sender<DataFlowStats>,
        collect_stats: bool,
    ) -> Self {
        Self {
            cancelled,
            err_tx,
            stats_tx,
            collect_stats,
            #[cfg(feature = "perf")]
            profiled: false,
            siblings_to_wake: 0,
            build,
        }
    }

    /// Have the worker that receives this builder wake the next `siblings`
    /// workers of its node. A dataflow for part of the pool is dispatched by
    /// waking one worker per node, which wakes the rest of its node's share.
    pub fn waking_siblings(mut self, siblings: usize) -> Self {
        self.siblings_to_wake = siblings;
        self
    }

    /// How many of its node's following workers the receiving worker wakes.
    pub fn siblings_to_wake(&self) -> usize {
        self.siblings_to_wake
    }

    /// Mark this builder's dataflow profiled, so it runs exclusively while a
    /// `perf` capture is in progress.
    #[cfg(feature = "perf")]
    pub fn with_profiling(mut self, profiled: bool) -> Self {
        self.profiled = profiled;
        self
    }

    /// Build the full operator graph and convert it into an executable `DataFlow`.
    /// Called on the worker thread.
    pub fn build(self) -> Result<DataFlow> {
        let mut context = BuildContext::default();
        let graph = match catch_unwind(AssertUnwindSafe(|| (self.build)(&mut context))) {
            Ok(graph) => graph,
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| e.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                // A build failure must fail the whole query the way a runtime
                // failure does (see `DataFlow::bail_and_cancel`): peers that
                // built their piece of the graph hold output senders until the
                // query is cancelled, and the collector blocks until every
                // sender drops, so swallowing the error here hangs the caller.
                if self
                    .err_tx
                    .send(crate::data_flow::Error::Panic(msg.to_string()))
                    .is_err()
                {
                    tracing::warn!("Unable to send build error...");
                }
                self.cancelled.store(true, Ordering::Relaxed);
                crate::waker::waker_set().notify_all();
                return Err(Error::PanicOnBuild(msg.to_string()));
            }
        };

        let data_flow = graph.into_data_flow(
            self.cancelled,
            self.err_tx,
            self.stats_tx,
            self.collect_stats,
        );
        #[cfg(feature = "perf")]
        let data_flow = data_flow.with_profiling(self.profiled);
        Ok(data_flow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_flow::WorkStatus;
    use crate::io::{FsWriteRequest, HttpUploadRequest, OperatorIO};
    use crate::operations::{FinishStatus, Result};

    /// Stand-in for a real operator: these tests only assert on the shape of the
    /// graph the builder produces, never run it.
    struct InertOperator;

    impl Operator for InertOperator {
        fn run_cpu_work(&mut self, _io: &mut OperatorIO) -> Result<WorkStatus> {
            Ok(WorkStatus::Pending)
        }
        fn process_fs_write_response(&mut self, _request: FsWriteRequest) -> Result<()> {
            Ok(())
        }
        fn process_http_upload_response(&mut self, _request: HttpUploadRequest) -> Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> Result<FinishStatus> {
            Ok(FinishStatus::Done)
        }
    }

    fn root() -> OperatorGraphBuilder {
        OperatorGraphBuilder::root(Box::new(InertOperator))
    }

    #[test]
    fn nested_join_gates_all_probe_roots_but_not_its_build_root() {
        let inner_gate = Arc::new(AtomicBool::new(false));
        let outer_gate = Arc::new(AtomicBool::new(false));

        let inner_join = root()
            .gated_by(inner_gate.clone())
            .with(Box::new(InertOperator))
            .with_side_graph(root());
        let graph = inner_join
            .gated_by(outer_gate.clone())
            .with(Box::new(InertOperator))
            .with_side_graph(root());

        assert_eq!(graph.roots, vec![0, 2, 4]);
        let probe_gates = &graph.gates[&0];
        assert_eq!(probe_gates.len(), 2);
        assert!(Arc::ptr_eq(&probe_gates[0], &inner_gate));
        assert!(Arc::ptr_eq(&probe_gates[1], &outer_gate));
        assert!(Arc::ptr_eq(&graph.gates[&2][0], &outer_gate));
        assert!(!graph.gates.contains_key(&4));
    }
}
