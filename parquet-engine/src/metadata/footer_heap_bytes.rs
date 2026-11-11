//! An estimate of the memory a file's footer holds once read: what a cache
//! of row groups counts each file for.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Schema};

use crate::types::metadata::{
    ColumnChunkMeta, FileLeafStatistics, FileStatistics, RowGroupMetadata,
};

/// An estimate of the memory `row_groups`, those one footer read produced
/// for a file, hold between them, the list of them included: every
/// allocation at the size an allocator serves it at. A footer read builds
/// one open of the file, one schema and one set of statistics, which every
/// row group shares, so they are counted once.
pub fn estimate_footer_heap_bytes(row_groups: &[Arc<RowGroupMetadata>]) -> u64 {
    let own_bytes = round_to_allocation(std::mem::size_of_val(row_groups))
        + row_groups
            .iter()
            .map(|row_group| {
                estimate_shared_bytes(size_of::<RowGroupMetadata>())
                    + round_to_allocation(
                        row_group.columns.capacity() * size_of::<ColumnChunkMeta>(),
                    )
                    + estimate_shared_bytes(size_of::<AtomicUsize>())
            })
            .sum::<usize>();
    let shared_bytes = row_groups.first().map_or(0, |row_group| {
        row_group.open_file.heap_bytes()
            + estimate_schema_bytes(&row_group.schema)
            + estimate_statistics_bytes(&row_group.statistics)
    });
    (own_bytes + shared_bytes) as u64
}

/// What a shared value of `bytes` takes: it is allocated with its two
/// reference counts.
fn estimate_shared_bytes(bytes: usize) -> usize {
    round_to_allocation(bytes + size_of::<[usize; 2]>())
}

fn estimate_schema_bytes(schema: &Schema) -> usize {
    estimate_shared_bytes(size_of::<Schema>())
        + estimate_fields_bytes(schema.fields())
        + estimate_string_map_bytes(schema.metadata())
}

/// What a list of fields takes: the shared slice of them and each field.
fn estimate_fields_bytes(fields: &[Arc<Field>]) -> usize {
    estimate_shared_bytes(std::mem::size_of_val(fields))
        + fields
            .iter()
            .map(|field| estimate_field_bytes(field))
            .sum::<usize>()
}

fn estimate_field_bytes(field: &Field) -> usize {
    let nested_bytes = match field.data_type() {
        DataType::Struct(fields) => estimate_fields_bytes(fields),
        DataType::List(element)
        | DataType::LargeList(element)
        | DataType::ListView(element)
        | DataType::LargeListView(element)
        | DataType::FixedSizeList(element, _)
        | DataType::Map(element, _) => estimate_field_bytes(element),
        _ => 0,
    };
    estimate_shared_bytes(size_of::<Field>())
        + round_to_allocation(field.name().len())
        + estimate_string_map_bytes(field.metadata())
        + nested_bytes
}

/// What a map of strings takes: its table, which keeps a control byte per
/// slot and fills seven slots in eight, and each string.
fn estimate_string_map_bytes(map: &std::collections::HashMap<String, String>) -> usize {
    if map.capacity() == 0 {
        return 0;
    }
    let slots = (map.capacity() * 8).div_ceil(7).next_power_of_two();
    round_to_allocation(slots * (size_of::<(String, String)>() + 1))
        + map
            .iter()
            .map(|(key, value)| round_to_allocation(key.len()) + round_to_allocation(value.len()))
            .sum::<usize>()
}

/// What a file's statistics take: per leaf, the two arrays of bounds, the
/// two lists of whether each bound is exact, and the two lists of counts.
fn estimate_statistics_bytes(statistics: &FileStatistics) -> usize {
    estimate_shared_bytes(size_of::<FileStatistics>())
        + round_to_allocation(statistics.capacity() * size_of::<Option<FileLeafStatistics>>())
        + statistics
            .iter()
            .flatten()
            .map(|leaf| {
                let bounds_bytes: usize = [&leaf.min, &leaf.max]
                    .into_iter()
                    .flatten()
                    .map(estimate_bounds_bytes)
                    .sum();
                bounds_bytes
                    + round_to_allocation(leaf.min_is_exact.capacity())
                    + round_to_allocation(leaf.max_is_exact.capacity())
                    + round_to_allocation(leaf.null_counts.capacity() * size_of::<Option<i64>>())
                    + round_to_allocation(
                        leaf.distinct_counts.capacity() * size_of::<Option<i64>>(),
                    )
            })
            .sum::<usize>()
}

/// What `bounds`, a leaf's bounds across a file's row groups, take: the
/// shared array with its buffers' bytes, and the shared record each buffer
/// keeps of its allocation.
fn estimate_bounds_bytes(bounds: &ArrayRef) -> usize {
    /// What a buffer's record of its allocation takes, shared.
    const BUFFER_RECORD_BYTES: usize = 64;

    let data = bounds.to_data();
    let buffers = data.buffers().len() + usize::from(data.nulls().is_some());
    bounds.get_array_memory_size() + size_of::<[usize; 2]>() + buffers * BUFFER_RECORD_BYTES
}

/// What an allocation of `bytes` takes: an allocator serves a request from
/// a size class, of which there are four to each doubling of the size, and
/// none below sixteen bytes but eight.
fn round_to_allocation(bytes: usize) -> usize {
    match bytes {
        0 => 0,
        1..=8 => 8,
        9..=16 => 16,
        _ => {
            let doubling = (bytes - 1).ilog2();
            bytes.next_multiple_of(1 << doubling.saturating_sub(2).max(4))
        }
    }
}
