//! Factory for creating per-worker [`Group`] instances.
//!
//! [`GroupFactory::create_for_workers`] allocates the shared state once
//! (arena, hash state, injector, channel) and produces one factory per
//! worker. Only the first factory receives the channel receiver; the rest
//! get `None`. Each factory is consumed by [`build_unary`](GroupFactory::build_unary)
//! to produce the actual [`Group`] operator.

use crate::operations::UnaryFactory;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{KeyExtractor, PartitionBuffers, ValueExtractor};
use crate::operations::unary::group::{AggregationSlot, Group, PartitionJob};
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
/// - An [`Injector`] for work-stealing during the output phase
/// - An mpsc channel for collecting per-worker tables after consumption
///
/// Only the first worker receives the channel receiver; it will drain
/// the channel and inject partition jobs during the output phase.
pub struct GroupFactory<K: KeyExtractor, V: ValueExtractor> {
    shared_arena: Arc<SharedArena>,
    key_cols: Vec<usize>,
    value_slots: Vec<AggregationSlot>,
    top_k: Option<(usize, usize)>,
    hash_state: RandomState,
    injector: Arc<Injector<PartitionJob<K, V>>>,
    partition_jobs_injected: Arc<AtomicBool>,

    sender: mpsc::Sender<PartitionBuffers<K, V>>,
    receiver: Option<mpsc::Receiver<PartitionBuffers<K, V>>>,
}

impl<K: KeyExtractor, V: ValueExtractor> GroupFactory<K, V> {
    /// Create `worker_count` factories that share the same arena, hash state,
    /// and synchronization primitives. `key_cols` are the GROUP BY column
    /// indices; `value_slots` configure the per-group aggregates.
    pub fn create_for_workers(
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        top_k: Option<(usize, usize)>,
        worker_count: usize,
        buffers: usize,
    ) -> impl IntoIterator<Item = GroupFactory<K, V>> {
        let shared_arena = SharedArena::new(buffers);
        let hash_state = RandomState::new();
        let injector = Arc::new(Injector::new());
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<_>();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| GroupFactory {
            shared_arena: shared_arena.clone(),
            key_cols: key_cols.clone(),
            value_slots: value_slots.clone(),
            top_k,
            hash_state: hash_state.clone(),
            injector: injector.clone(),
            partition_jobs_injected: partition_jobs_injected.clone(),
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl<K: KeyExtractor, V: ValueExtractor> UnaryFactory<RecordBatch, RecordBatch>
    for GroupFactory<K, V>
{
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Group<K, V>>;

    fn build_unary(mut self) -> PipelineBreaker<RecordBatch, RecordBatch, Group<K, V>> {
        PipelineBreaker::Consuming(Group::new(
            self.shared_arena,
            self.hash_state,
            self.injector,
            self.key_cols,
            self.value_slots,
            self.top_k,
            self.sender,
            self.receiver.take(),
            self.partition_jobs_injected,
        ))
    }
}
