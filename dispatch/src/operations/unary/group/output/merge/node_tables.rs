//! Merging one partition's per-node results into the final table.

use super::tables::resize_if_needed;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, HashTable, MAX_LOAD_FACTOR,
};
use crate::operations::unary::group::keys::StoredKey;

/// Merges one partition's node-local results.
///
/// The largest input is reused when it can hold the combined upper bound.
/// Otherwise a new table is allocated at the estimated partition capacity.
pub fn merge_node_aggregated_tables<S: StoredKey, V: AggregationValue + ?Sized>(
    mut node_tables: Vec<HashTable<S::Persisted, V>>,
    partition_capacity: usize,
    partition_bits: u32,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> HashTable<S::Persisted, V> {
    let mut allocator = SlabAllocator::new(true);
    let total: usize = node_tables.iter().map(|table| table.len()).sum();
    let largest_table_index = node_tables
        .iter()
        .enumerate()
        .max_by_key(|(_, table)| table.len())
        .map(|(index, _)| index)
        .expect("a partition always has at least one node_table");
    let mut target =
        if total as f64 <= node_tables[largest_table_index].capacity() as f64 * MAX_LOAD_FACTOR {
            node_tables.swap_remove(largest_table_index)
        } else {
            let capacity = partition_capacity.max(DEFAULT_CAPACITY);
            <HashTable<S::Persisted, V>>::new(&mut allocator, capacity, partition_bits, context)
        };
    let mut prober = target.prober();
    for node_table in &node_tables {
        for entry in node_table.iter(0) {
            prober.merge_from::<true, _>(
                entry.hash,
                S::resolve_persisted(key_arena, *entry.key),
                entry.stored,
                context,
            );
            resize_if_needed::<S::Persisted, V>(&mut allocator, &mut prober);
        }
    }
    // The prober's borrow of `target` ends at its last use above.
    target
}
