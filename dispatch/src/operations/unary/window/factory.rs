//! Builds one worker's part of a window operator.

use std::sync::Arc;

use arrow_array::RecordBatch;

use super::consume::WindowConsumer;
use super::output::SharedBuckets;
use super::{WindowLayout, WindowSpec};
use crate::Topology;
use crate::api::BuildContext;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use crate::operations::{Built, BuiltUnary};

/// Builds one worker's [`WindowConsumer`].
pub struct WindowFactory {
    layout: Arc<WindowLayout>,
    shared: Arc<SharedBuckets>,
    worker_id: usize,
    topology: Topology,
}

impl WindowFactory {
    /// Creates one factory per worker, all sharing the coordination of one
    /// window operator.
    ///
    /// `arena_buffers` is how many ring buffers an arena of partition keys
    /// may grow to, the same bound GROUP BY's key arenas are given.
    pub fn create_for_workers(
        spec: WindowSpec,
        topology: Topology,
        arena_buffers: usize,
    ) -> Vec<WindowFactory> {
        assert!(
            !spec.functions.is_empty(),
            "a window operator computes at least one function"
        );
        if let Some(row_limit) = &spec.row_limit {
            assert!(
                row_limit.function_index < spec.functions.len(),
                "the row limit names function {} of {}",
                row_limit.function_index,
                spec.functions.len()
            );
        }
        let layout = Arc::new(WindowLayout::new(
            spec,
            topology.total_workers(),
            arena_buffers,
        ));
        let shared = Arc::new(SharedBuckets::new(layout.clone(), topology));
        (0..topology.total_workers())
            .map(|worker_id| WindowFactory {
                layout: layout.clone(),
                shared: shared.clone(),
                worker_id,
                topology,
            })
            .collect()
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for WindowFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, WindowConsumer>;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(PipelineBreaker::Consuming(WindowConsumer::new(
            self.layout,
            self.shared,
            self.worker_id,
            self.topology.node_of_worker(self.worker_id),
            self.topology.node_count,
        )))
    }
}
