//! Assembles a merged partition table into output [`RecordBatch`]es.
//!
//! This is the one place the key and value sides meet on the output path: the
//! [`KeyExtractor`] emits the leading key column(s) and the [`ValueExtractor`]
//! the trailing value column(s), and a single combinator zips them — so neither
//! extractor has to know about the other.
//!
//! Columns are built into engine slab memory (see [`crate::arrays`]) via the
//! per-column builders, so output buffers stay on our pre-faulted, accounted
//! memory. Because a slab is at most 2MB, the partition is emitted in
//! [`OUTPUT_CHUNK_ROWS`]-row chunks — one `RecordBatch` per chunk.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;

use crate::memory::{BUFFER_SIZE, SlabAllocator};
use crate::operations::channels::Sender;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{Table, TableStorage};
use crate::operations::unary::group::keys::{KeyColumns, KeyExtractor};
use crate::operations::unary::group::values::{ValueColumns, ValueExtractor};

use super::Result;

/// Rows per output chunk. Each chunk's columns are built into single 2MB slabs,
/// so the widest column (a `u128` key = 16 bytes) bounds this: `chunk * 16 <= 2MB`.
/// Kept at the maximum so a typical partition is a single chunk — chunking only
/// kicks in for partitions with more groups than fit one slab. (A separate, much
/// smaller value would re-wrap the whole arena once per chunk for string keys.)
const OUTPUT_CHUNK_ROWS: usize = BUFFER_SIZE / 16;

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
/// nothing it would otherwise have to materialise — at very large group counts that is the
/// difference between emitting every group and emitting `limit` of them.
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

/// Build a single chunk's key+value columns into one `RecordBatch` and send it.
fn emit<K, V, Snd>(
    keys: K::Columns,
    values: V::Columns,
    arena: &Arc<SharedArena>,
    sender: &mut Snd,
) -> Result<()>
where
    K: KeyExtractor,
    V: ValueExtractor,
    Snd: Sender<RecordBatch>,
{
    let (mut fields, mut columns) = keys.finish(arena);
    let (value_fields, value_columns) = values.finish();
    fields.extend(value_fields);
    columns.extend(value_columns);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    sender.send(batch)?;
    Ok(())
}

/// Convert a completed partition table into output `RecordBatch`es (one per
/// [`OUTPUT_CHUNK_ROWS`]-row chunk), built into `allocator`'s slab memory.
///
/// `top_k` is `Some((value_slot, limit))` when this group directly feeds an
/// `ORDER BY <value_slot> DESC LIMIT limit`; only this partition's top-`limit`
/// rows are emitted in that case.
pub(crate) fn build_and_send<K, V, S, Snd>(
    table: Table<K, V, S>,
    arena: &Arc<SharedArena>,
    allocator: &mut SlabAllocator,
    top_k: Option<(usize, usize)>,
    sender: &mut Snd,
) -> Result<()>
where
    K: KeyExtractor,
    V: ValueExtractor,
    S: TableStorage<K, V>,
    Snd: Sender<RecordBatch>,
{
    match top_k {
        Some((slot, limit)) if limit < table.len() => {
            let rows = top_k_rows::<K, V, S>(&table, slot, limit);
            for chunk in rows.chunks(OUTPUT_CHUNK_ROWS) {
                let mut keys = K::Columns::with_capacity(allocator, chunk.len());
                let mut values = V::Columns::with_capacity(allocator, chunk.len());
                for (key, value) in chunk {
                    keys.push(key);
                    values.push(value);
                }
                emit::<K, V, Snd>(keys, values, arena, sender)?;
            }
        }
        _ => {
            let mut entries = table.iter(0);
            let mut remaining = table.len();
            while remaining > 0 {
                let chunk = remaining.min(OUTPUT_CHUNK_ROWS);
                let mut keys = K::Columns::with_capacity(allocator, chunk);
                let mut values = V::Columns::with_capacity(allocator, chunk);
                for _ in 0..chunk {
                    let entry = entries.next().expect("iterator yields table.len() entries");
                    keys.push(entry.key());
                    values.push(entry.value());
                }
                emit::<K, V, Snd>(keys, values, arena, sender)?;
                remaining -= chunk;
            }
        }
    }
    Ok(())
}
