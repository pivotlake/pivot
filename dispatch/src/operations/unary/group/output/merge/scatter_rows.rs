//! Merging a partition's scatter rows that are in memory.

use crate::cpu_features::multitarget_kernel;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, HashTable, PartitionBuffers, PersistedKey, Prober, ScatterLayout,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::output::topk_pruning::{TopKCutoff, topk_admits};
use crate::operations::unary::group::values::ArityBody;

/// Merge a partition's scatter rows that are in memory, from every source
/// worker's buffers, into `table`. Scatter rows were not aggregated during
/// consume, so each is inserted once here. `table` grows out of `allocator`,
/// and `capacity` tracks its slot count as it does.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_scatter_rows<S: StoredKey, V: AggregationValue + ?Sized>(
    table: &mut HashTable<S::Persisted, V>,
    allocator: &mut SlabAllocator,
    capacity: &mut usize,
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) {
    // A merge partition may own several consecutive scatter buckets. Both
    // counts are powers of two, so the ranges divide evenly; the bucket
    // release relies on no two partitions sharing a bucket.
    if let Some(first) = buffers.first() {
        let scatter_bucket_count = first.bucket_count();
        assert!(
            scatter_bucket_count.is_multiple_of(num_partitions),
            "merge partitions ({num_partitions}) must evenly divide scatter buckets ({scatter_bucket_count})"
        );
    }
    // Dispatch once so every loop shares the specialized arity.
    V::dispatch_arity(
        V::storage_metadata(context),
        MergeScatterRows::<S, V> {
            table,
            allocator,
            capacity,
            partition,
            buffers,
            num_partitions,
            key_arena,
            context,
            cutoff,
        },
    )
}

/// State passed through arity dispatch for merging resident scatter rows.
struct MergeScatterRows<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    table: &'a mut HashTable<S::Persisted, V>,
    allocator: &'a mut SlabAllocator,
    capacity: &'a mut usize,
    partition: usize,
    buffers: &'a [PartitionBuffers<S::Persisted, V>],
    num_partitions: usize,
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
    cutoff: Option<TopKCutoff<'a>>,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<()> for MergeScatterRows<'_, S, V> {
    #[inline(always)]
    fn run<const N: usize>(self) {
        let MergeScatterRows {
            table,
            allocator,
            capacity,
            partition,
            buffers,
            num_partitions,
            key_arena,
            context,
            cutoff,
        } = self;
        scatter_rows_kernel::<N, S, V>(
            table,
            allocator,
            capacity,
            partition,
            buffers,
            num_partitions,
            key_arena,
            context,
            cutoff,
        )
    }
}

multitarget_kernel! {
    /// The scatter-row merge, cloned per CPU tier: it is the bandwidth-bound
    /// part of a group-by's output.
    #[allow(clippy::too_many_arguments)]
    fn scatter_rows_kernel[const N: usize, S: StoredKey, V: AggregationValue + ?Sized]::[N, S, V](
        table: &mut HashTable<<S as StoredKey>::Persisted, V>,
        allocator: &mut SlabAllocator,
        capacity: &mut usize,
        partition: usize,
        buffers: &[PartitionBuffers<<S as StoredKey>::Persisted, V>],
        num_partitions: usize,
        key_arena: &SharedArena,
        context: &V::SharedContext,
        cutoff: Option<TopKCutoff<'_>>
    ) {
        let mut target = Prober::new::<N>(table);
        const SCATTER_PREFETCH_AHEAD: usize = 16;
        // Prefetch strings unconditionally and fixed-width keys only when the
        // target is large enough for probes to be cold.
        const TARGET_PREFETCH_MIN_SLOTS: usize = 16384;
        let merge_prefetch =
            <S::Persisted as PersistedKey>::HAS_BLOB || *capacity > TARGET_PREFETCH_MIN_SLOTS;
        // All scatter buckets for this signature share one row layout.
        let scatter_layout = ScatterLayout::<V>::new::<S::Persisted, N>(context);
        for worker_buffers in buffers {
            for bucket in worker_buffers.bucket_range(partition, num_partitions) {
                if merge_prefetch {
                    worker_buffers.with_bucket(bucket, |rows| {
                        rows.for_each_prefetched::<SCATTER_PREFETCH_AHEAD>(
                            scatter_layout,
                            |hash, key, stored, ahead| {
                                if !topk_admits(cutoff, hash) {
                                    return;
                                }
                                if let Some((ahead_hash, ahead_key)) = ahead {
                                    ahead_key.prefetch_blob(key_arena);
                                    target.prefetch(ahead_hash);
                                }
                                target.merge_scatter_row::<S>(allocator, capacity, key_arena, context, hash, key, stored);
                            },
                        )
                    });
                } else {
                    worker_buffers.with_bucket(bucket, |rows| {
                        rows.for_each(scatter_layout, |hash, key, stored| {
                            if !topk_admits(cutoff, hash) {
                                return;
                            }
                            target.merge_scatter_row::<S>(allocator, capacity, key_arena, context, hash, key, stored);
                        })
                    });
                }
            }
        }
    }
}
