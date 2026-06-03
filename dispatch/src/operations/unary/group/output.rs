//! Assembles a merged partition table into an output [`RecordBatch`].
//!
//! This is the one place the key and value sides meet on the output path: the
//! [`KeyExtractor`] emits the leading key column(s) and the [`ValueExtractor`]
//! the trailing value column(s), and a single combinator zips them — so neither
//! extractor has to know about the other.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};

use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{Table, TableStorage};
use crate::operations::unary::group::key_extractions::{KeyColumns, KeyExtractor};
use crate::operations::unary::group::value_extractions::{ValueColumns, ValueExtractor};

/// A heap entry for top-k selection, ordered solely by the aggregate `sort`
/// scalar. Ties compare equal, which is fine: an `ORDER BY <slot> DESC LIMIT k`
/// is indifferent to the order within a tied group.
struct TopK<P, Val> {
    sort: i64,
    key: P,
    value: Val,
}

impl<P, Val> PartialEq for TopK<P, Val> {
    fn eq(&self, other: &Self) -> bool {
        self.sort == other.sort
    }
}
impl<P, Val> Eq for TopK<P, Val> {}
impl<P, Val> PartialOrd for TopK<P, Val> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<P, Val> Ord for TopK<P, Val> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort.cmp(&other.sort)
    }
}

/// Select this partition's top-`limit` groups by `V::sort_key(value, slot)`.
///
/// Top-k is decomposable across partitions, so emitting only the local top-k
/// (instead of every group) lets a downstream `ORDER BY … DESC LIMIT` discard
/// nothing it would otherwise have to materialise — at ~100M groups that is the
/// difference between emitting ~100M rows and emitting `limit` of them.
fn top_k_rows<K, V, S>(
    table: &Table<K, V, S>,
    slot: usize,
    limit: usize,
) -> Vec<(K::Persisted, V::Value)>
where
    K: KeyExtractor,
    V: ValueExtractor,
    S: TableStorage<K, V>,
{
    // Size-`limit` min-heap (via `Reverse`) keyed by the sort scalar; keeps the
    // `limit` largest entries seen.
    let mut heap: BinaryHeap<Reverse<TopK<K::Persisted, V::Value>>> =
        BinaryHeap::with_capacity(limit + 1);
    for entry in table.iter(0) {
        let sort = V::sort_key(entry.value(), slot);
        if heap.len() < limit {
            heap.push(Reverse(TopK {
                sort,
                key: *entry.key(),
                value: *entry.value(),
            }));
        } else if sort > heap.peek().unwrap().0.sort {
            heap.pop();
            heap.push(Reverse(TopK {
                sort,
                key: *entry.key(),
                value: *entry.value(),
            }));
        }
    }
    heap.into_iter()
        .map(|Reverse(e)| (e.key, e.value))
        .collect()
}

/// Convert a completed partition table into an Arrow `RecordBatch` of key
/// columns followed by value columns.
///
/// `top_k` is `Some((value_slot, limit))` when this group directly feeds an
/// `ORDER BY <value_slot> DESC LIMIT limit`; only this partition's top-`limit`
/// rows are emitted in that case. The two modes are separate functions so the
/// common all-groups loop optimises without the top-k heap machinery in scope.
pub(crate) fn build_record_batch<K, V, S>(
    table: Table<K, V, S>,
    arena: &Arc<SharedArena>,
    top_k: Option<(usize, usize)>,
) -> Result<RecordBatch, ArrowError>
where
    K: KeyExtractor,
    V: ValueExtractor,
    S: TableStorage<K, V>,
{
    match top_k {
        Some((slot, limit)) if limit < table.len() => build_top_k::<K, V, S>(&table, arena, slot, limit),
        _ => build_all::<K, V, S>(&table, arena),
    }
}

/// Emit every group, key columns followed by value columns.
fn build_all<K, V, S>(table: &Table<K, V, S>, arena: &Arc<SharedArena>) -> Result<RecordBatch, ArrowError>
where
    K: KeyExtractor,
    V: ValueExtractor,
    S: TableStorage<K, V>,
{
    let mut keys = K::Columns::with_capacity(table.len());
    let mut values = V::Columns::with_capacity(table.len());
    for entry in table.iter(0) {
        keys.push(entry.key());
        values.push(entry.value());
    }
    assemble::<K, V>(keys, values, arena)
}

/// Emit only the top-`limit` groups by `V::sort_key(value, slot)`.
fn build_top_k<K, V, S>(
    table: &Table<K, V, S>,
    arena: &Arc<SharedArena>,
    slot: usize,
    limit: usize,
) -> Result<RecordBatch, ArrowError>
where
    K: KeyExtractor,
    V: ValueExtractor,
    S: TableStorage<K, V>,
{
    let rows = top_k_rows::<K, V, S>(table, slot, limit);
    let mut keys = K::Columns::with_capacity(rows.len());
    let mut values = V::Columns::with_capacity(rows.len());
    for (key, value) in &rows {
        keys.push(key);
        values.push(value);
    }
    assemble::<K, V>(keys, values, arena)
}

/// Concatenate the key and value columns into a single `RecordBatch`.
fn assemble<K, V>(
    keys: K::Columns,
    values: V::Columns,
    arena: &Arc<SharedArena>,
) -> Result<RecordBatch, ArrowError>
where
    K: KeyExtractor,
    V: ValueExtractor,
{
    let (mut fields, mut columns) = keys.finish(arena);
    let (value_fields, value_columns) = values.finish();
    fields.extend(value_fields);
    columns.extend(value_columns);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}
