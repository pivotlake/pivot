//! Factory for creating per-worker [`Group`] instances.
//!
//! [`GroupFactory::create_for_workers`] allocates the shared state once
//! (arena, hash state, injector, gather barrier) and produces one factory per
//! worker. Each factory is consumed by [`build_unary`](GroupFactory::build_unary)
//! to produce the actual [`Group`] operator.

use crate::GatherBarrier;
use crate::numa::Topology;
use crate::operations::UnaryFactory;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::RadixConfig;
use crate::operations::unary::group::hashtables::{
    AggregatedTableOutput, AggregationValue, KeyExtractor,
};
use crate::operations::unary::group::output::topk_pruning::{SharedBinTotals, TopKThreshold};
use crate::operations::unary::group::{AggregationSlot, Group, GroupLimit, PartitionJob};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Creates one [`Group`] operator per worker, with shared state wired up.
///
/// All workers share:
/// - A [`SharedArena`] for string key storage
/// - A [`RandomState`] so hashes are consistent across workers
/// - One [`Injector`] per NUMA node for work-stealing during the output phase
///   (a merge job reads one node's tables, so it queues on that node)
/// - A [`GatherBarrier`] collecting per-worker tables after consumption
pub struct GroupFactory<K: KeyExtractor, V: AggregationValue + ?Sized> {
    stage: Arc<GroupStage<K, V>>,
}

/// What every worker's [`Group`] of one stage starts from. A worker takes its
/// own copies of the key columns, value slots and key config when it builds,
/// on its own thread, so the thread compiling the plan clones one `Arc` per
/// worker instead of all of this.
struct GroupStage<K: KeyExtractor, V: AggregationValue + ?Sized> {
    key_arena: Arc<SharedArena>,
    value_arena: Arc<SharedArena>,
    key_cols: Vec<usize>,
    value_slots: Vec<AggregationSlot>,
    key_config: K::Config,
    output_limit: Option<GroupLimit>,
    count_only: bool,
    hash_state: RandomState,
    injectors: Arc<Vec<Injector<PartitionJob<K, V>>>>,
    partition_jobs_injected: Arc<AtomicBool>,
    gather: Arc<GatherBarrier<AggregatedTableOutput<K, V>>>,
    /// Radix scatter config with the per-worker bucket count sized for the pool
    /// (see [`get_scatter_bucket_count_for_worker`](super::get_scatter_bucket_count_for_worker)).
    radix: RadixConfig,
    /// Shared lower bound on a pushed top-k's k-th best group value, used by
    /// the merge phase to skip partitions that cannot reach it.
    topk_threshold: Arc<TopKThreshold>,
    /// Pool-wide bin totals every worker adds its own into at the gather barrier.
    shared_bin_totals: Arc<SharedBinTotals>,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> GroupFactory<K, V> {
    /// Create one factory per worker of `topology`, sharing the same arena,
    /// hash state, and synchronization primitives. `key_cols` are the GROUP BY
    /// column indices; `value_slots` configure the per-group aggregates.
    #[allow(clippy::too_many_arguments)]
    pub fn create_for_workers(
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        key_config: K::Config,
        output_limit: Option<GroupLimit>,
        count_only: bool,
        topology: Topology,
        buffers: usize,
    ) -> impl IntoIterator<Item = GroupFactory<K, V>> {
        let key_arena = SharedArena::new(buffers);
        let value_arena = SharedArena::new(buffers);
        let hash_state = RandomState::new();
        let injectors = Arc::new(
            (0..topology.node_count)
                .map(|_| Injector::new())
                .collect::<Vec<_>>(),
        );
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let gather = Arc::new(GatherBarrier::new(topology.total_workers()));
        let topk_threshold = Arc::new(TopKThreshold::for_limit(output_limit));
        let shared_bin_totals = Arc::new(SharedBinTotals::new(topology.node_count));
        let radix = RadixConfig {
            partitions: crate::operations::unary::group::get_scatter_bucket_count_for_worker(
                topology.total_workers(),
            ),
            ..RadixConfig::DEFAULT
        };

        let stage = Arc::new(GroupStage {
            key_arena,
            value_arena,
            key_cols,
            value_slots,
            key_config,
            output_limit,
            count_only,
            hash_state,
            injectors,
            partition_jobs_injected,
            gather,
            radix,
            topk_threshold,
            shared_bin_totals,
        });

        (0..topology.total_workers()).map(move |_| GroupFactory {
            stage: stage.clone(),
        })
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> UnaryFactory<RecordBatch, RecordBatch>
    for GroupFactory<K, V>
{
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Group<K, V>>;

    fn build_unary(self) -> PipelineBreaker<RecordBatch, RecordBatch, Group<K, V>> {
        let stage = self.stage;
        PipelineBreaker::Consuming(Group::new(
            stage.key_arena.clone(),
            stage.value_arena.clone(),
            stage.hash_state.clone(),
            stage.injectors.clone(),
            stage.key_cols.clone(),
            stage.value_slots.clone(),
            stage.key_config.clone(),
            stage.output_limit,
            stage.count_only,
            stage.gather.clone(),
            stage.partition_jobs_injected.clone(),
            stage.radix,
            stage.topk_threshold.clone(),
            stage.shared_bin_totals.clone(),
        ))
    }
}
