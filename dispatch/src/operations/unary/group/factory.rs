//! Factory for creating per-worker [`Group`] instances.
//!
//! [`GroupFactory::create_for_workers`] allocates the shared state once
//! (arena, hash state, injector, channel) and produces one factory per
//! worker. Only the first factory receives the channel receiver; the rest
//! get `None`. Each factory is consumed by [`build_unary`](GroupFactory::build_unary)
//! to produce the actual [`Group`] operator.

use crate::numa::Topology;
use crate::operations::UnaryFactory;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::RadixConfig;
use crate::operations::unary::group::hashtables::{
    AggregatedTableOutput, AggregationValue, KeyExtractor,
};
use crate::operations::unary::group::{AggregationSlot, Group, GroupLimit, PartitionJob};
use crate::operations::unary::pipeline_breaker::PipelineBreaker;
use ahash::RandomState;
use arrow_array::RecordBatch;
use crossbeam_deque::Injector;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};

/// Creates one [`Group`] operator per worker, with shared state wired up.
///
/// All workers share:
/// - A [`SharedArena`] for string key storage
/// - A [`RandomState`] so hashes are consistent across workers
/// - One [`Injector`] per NUMA node for work-stealing during the output phase
///   (a merge job reads one node's tables, so it queues on that node)
/// - An mpsc channel for collecting per-worker tables after consumption
///
/// Only the first worker receives the channel receiver; it will drain
/// the channel and inject partition jobs during the output phase.
pub struct GroupFactory<K: KeyExtractor, V: AggregationValue> {
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

    sender: mpsc::Sender<(usize, AggregatedTableOutput<K, V>)>,
    receiver: Option<mpsc::Receiver<(usize, AggregatedTableOutput<K, V>)>>,
}

impl<K: KeyExtractor, V: AggregationValue> GroupFactory<K, V> {
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
        let (tx, rx) = mpsc::channel::<_>();
        let mut rx_opt = Some(rx);

        (0..topology.total_workers()).map(move |_| GroupFactory {
            key_arena: key_arena.clone(),
            value_arena: value_arena.clone(),
            key_cols: key_cols.clone(),
            value_slots: value_slots.clone(),
            key_config: key_config.clone(),
            output_limit,
            count_only,
            hash_state: hash_state.clone(),
            injectors: injectors.clone(),
            partition_jobs_injected: partition_jobs_injected.clone(),
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl<K: KeyExtractor, V: AggregationValue> UnaryFactory<RecordBatch, RecordBatch>
    for GroupFactory<K, V>
{
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Group<K, V>>;

    fn build_unary(mut self) -> PipelineBreaker<RecordBatch, RecordBatch, Group<K, V>> {
        PipelineBreaker::Consuming(Group::new(
            self.key_arena,
            self.value_arena,
            self.hash_state,
            self.injectors,
            self.key_cols,
            self.value_slots,
            self.key_config,
            self.output_limit,
            self.count_only,
            self.sender,
            self.receiver.take(),
            self.partition_jobs_injected,
            RadixConfig::DEFAULT,
        ))
    }
}
