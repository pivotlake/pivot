//! Merging a partition's rows that were spilled, evicted, and read back.

use crate::cpu_features::multitarget_kernel;
use crate::io::ReadData;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, HashTable, PartitionRows, Prober, ScatterLayout,
};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::output::topk_pruning::{TopKCutoff, topk_admits};
use crate::operations::unary::group::values::ArityBody;

/// Merge a partition's rows that one read brought back from disk into `table`.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_read_rows<S: StoredKey, V: AggregationValue + ?Sized>(
    table: &mut HashTable<S::Persisted, V>,
    allocator: &mut SlabAllocator,
    capacity: &mut usize,
    rows: &[ReadData],
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) {
    V::dispatch_arity(
        V::storage_metadata(context),
        MergeReadRows::<S, V> {
            table,
            allocator,
            capacity,
            rows,
            key_arena,
            context,
            cutoff,
        },
    )
}

/// State passed through arity dispatch for merging rows read back from disk.
struct MergeReadRows<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    table: &'a mut HashTable<S::Persisted, V>,
    allocator: &'a mut SlabAllocator,
    capacity: &'a mut usize,
    rows: &'a [ReadData],
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
    cutoff: Option<TopKCutoff<'a>>,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<()> for MergeReadRows<'_, S, V> {
    #[inline(always)]
    fn run<const N: usize>(self) {
        let MergeReadRows {
            table,
            allocator,
            capacity,
            rows,
            key_arena,
            context,
            cutoff,
        } = self;
        read_rows_kernel::<N, S, V>(table, allocator, capacity, rows, key_arena, context, cutoff)
    }
}

multitarget_kernel! {
    /// The read-back merge, cloned per CPU tier like the other two: rows read
    /// from disk go through the same probe and fold.
    #[allow(clippy::too_many_arguments)]
    fn read_rows_kernel[const N: usize, S: StoredKey, V: AggregationValue + ?Sized]::[N, S, V](
        table: &mut HashTable<<S as StoredKey>::Persisted, V>,
        allocator: &mut SlabAllocator,
        capacity: &mut usize,
        rows: &[ReadData],
        key_arena: &SharedArena,
        context: &V::SharedContext,
        cutoff: Option<TopKCutoff<'_>>
    ) {
        let mut target = Prober::new::<N>(table);
        let scatter_layout = ScatterLayout::<V>::new::<S::Persisted, N>(context);
        // A row read back from disk may come in two pieces, or in a piece
        // not aligned for a row; one aligned row of scratch takes those.
        let mut row_scratch =
            allocator.get_aligned_slab(scatter_layout.stride(), scatter_layout.align(), false);
        PartitionRows::<S::Persisted, V>::for_each_loaded(
            scatter_layout,
            rows,
            row_scratch.as_mut_slice(),
            |hash, key, stored| {
                if !topk_admits(cutoff, hash) {
                    return;
                }
                target.merge_scatter_row::<S>(allocator, capacity, key_arena, context, hash, key, stored);
            },
        );
    }
}
