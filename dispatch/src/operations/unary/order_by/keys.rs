//! How ORDER BY compares two rows by the sort key.
//!
//! Every step of the sort — sorting one batch, checking whether rows already
//! arrive in order, planning where to split a merge, and the merge walk
//! itself — reduces to one question: of two rows, which sorts first? The
//! [`KeyOrdering`] trait answers it for a pair of runs (chunked sequences of
//! record batches): `left` positions index the left run's chunks and `right`
//! positions the right run's. Comparing rows within a single run passes that
//! run as both sides.
//!
//! Three implementations, picked by [`select_key_ordering`] from what the key
//! actually is, fastest first:
//!
//! - [`FixedWidthKeyOrdering`]: one null-free fixed-width column (integers,
//!   dates, timestamps, 64-bit decimals, floats). Each value reads as a `u64`
//!   whose unsigned order equals the column's order (sign flip for signed
//!   integers, the IEEE total-order flip for floats), so a comparison is two
//!   loads and an integer compare.
//! - [`ViewBytesKeyOrdering`]: one null-free byte-view column (strings,
//!   binary). A comparison resolves each view to its bytes and memcmps.
//! - [`ComparatorKeyOrdering`]: everything else, meaning several key columns,
//!   nulls, or wider types. Arrow comparators are built per pair of chunks
//!   and cached by pair, and a comparison walks the key columns until one
//!   differs. A k-way merge heap interleaves chunk pairs, so the cache keeps
//!   every pair the walk revisits instead of only the latest one.

use std::cmp::Ordering;
use std::collections::HashMap;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_ord::ord::{DynComparator, make_comparator};
use arrow_schema::{ArrowError, DataType, SortOptions};

use crate::operations::unary::order_by_limit::OrderBy;

/// A row's address within one run: which chunk, and which row of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RunRow {
    pub(super) chunk: usize,
    pub(super) row: usize,
}

/// Orders rows of a left run against rows of a right run by the sort key.
/// Implementations may cache per-chunk state, which is why comparing takes
/// `&mut self`.
pub(super) trait KeyOrdering {
    /// How the left run's row at `left` orders against the right run's row at
    /// `right`. `Equal` means either row may be emitted first.
    fn compare(&mut self, left: RunRow, right: RunRow) -> Ordering;
}

/// The [`KeyOrdering`] chosen for a pair of runs. Callers match and hand the
/// concrete ordering to the generic sort/merge routines, so the fast paths
/// compare without a virtual call per row.
pub(super) enum SelectedKeyOrdering<'chunks> {
    FixedWidth(FixedWidthKeyOrdering<'chunks>),
    ViewBytes(ViewBytesKeyOrdering<'chunks>),
    General(ComparatorKeyOrdering<'chunks>),
}

/// Dispatches to the selected key representation while keeping the comparison
/// loop monomorphized.
macro_rules! with_key_ordering {
    ($selected:expr, |$ordering:ident| $body:expr) => {
        match $selected {
            SelectedKeyOrdering::FixedWidth(mut $ordering) => $body,
            SelectedKeyOrdering::ViewBytes(mut $ordering) => $body,
            SelectedKeyOrdering::General(mut $ordering) => $body,
        }
    };
}
pub(super) use with_key_ordering;

/// Pick the fastest ordering the key allows for these chunks (see the module
/// docs). The chunks must all share the schema the key columns index into.
pub(super) fn select_key_ordering<'chunks>(
    order_by: &[OrderBy],
    left_chunks: &'chunks [RecordBatch],
    right_chunks: &'chunks [RecordBatch],
) -> Result<SelectedKeyOrdering<'chunks>, ArrowError> {
    if let [single_key] = order_by {
        let key_column = single_key.column_idx();
        let null_free = left_chunks
            .iter()
            .chain(right_chunks)
            .all(|chunk| chunk.column(key_column).null_count() == 0);
        if null_free {
            let data_type = data_type_of(key_column, left_chunks, right_chunks);
            if let Some(read_key) = normalized_key_reader(&data_type) {
                return Ok(SelectedKeyOrdering::FixedWidth(FixedWidthKeyOrdering {
                    left: fixed_width_chunks(key_column, left_chunks),
                    right: fixed_width_chunks(key_column, right_chunks),
                    read_key,
                    descending: single_key.descending(),
                }));
            }
            if matches!(data_type, DataType::Utf8View | DataType::BinaryView) {
                return Ok(SelectedKeyOrdering::ViewBytes(ViewBytesKeyOrdering {
                    left: view_chunks(key_column, left_chunks),
                    right: view_chunks(key_column, right_chunks),
                    descending: single_key.descending(),
                }));
            }
        }
    }
    Ok(SelectedKeyOrdering::General(ComparatorKeyOrdering {
        order_by: order_by.to_vec(),
        left_chunks,
        right_chunks,
        comparators_by_chunks: HashMap::new(),
    }))
}

/// The key column's type; either side may be empty, but never both.
fn data_type_of(key_column: usize, left: &[RecordBatch], right: &[RecordBatch]) -> DataType {
    left.iter()
        .chain(right)
        .next()
        .expect("an ordering is only selected over at least one chunk")
        .column(key_column)
        .data_type()
        .clone()
}

/// One null-free fixed-width key column: rows compare as normalized `u64`s
/// read straight from each chunk's values buffer.
pub(super) struct FixedWidthKeyOrdering<'chunks> {
    left: Vec<FixedWidthChunk<'chunks>>,
    right: Vec<FixedWidthChunk<'chunks>>,
    read_key: fn(*const u8, usize) -> u64,
    descending: bool,
}

/// One chunk's key values: the raw pointer to its first value. The borrow it
/// was taken under rides along, keeping the chunk alive.
struct FixedWidthChunk<'chunks> {
    first_value: *const u8,
    _keeps_chunk_alive: std::marker::PhantomData<&'chunks RecordBatch>,
}

fn fixed_width_chunks<'chunks>(
    key_column: usize,
    chunks: &'chunks [RecordBatch],
) -> Vec<FixedWidthChunk<'chunks>> {
    chunks
        .iter()
        .map(|chunk| {
            let data = chunk.column(key_column).to_data();
            let width = data
                .data_type()
                .primitive_width()
                .expect("selected for a fixed-width key");
            // SAFETY (of the `add`): buffer 0 holds `offset + len` values, so
            // the offset stays within its allocation.
            let first_value = unsafe { data.buffers()[0].as_ptr().add(data.offset() * width) };
            FixedWidthChunk {
                first_value,
                _keeps_chunk_alive: std::marker::PhantomData,
            }
        })
        .collect()
}

impl KeyOrdering for FixedWidthKeyOrdering<'_> {
    fn compare(&mut self, left: RunRow, right: RunRow) -> Ordering {
        let read_key = self.read_key;
        let left_key = read_key(self.left[left.chunk].first_value, left.row);
        let right_key = read_key(self.right[right.chunk].first_value, right.row);
        let ascending = left_key.cmp(&right_key);
        if self.descending {
            ascending.reverse()
        } else {
            ascending
        }
    }
}

/// The reader turning row `row` of a chunk's values into a `u64` whose
/// unsigned order equals the column's sort order, or `None` for a type with
/// no such form.
fn normalized_key_reader(data_type: &DataType) -> Option<fn(*const u8, usize) -> u64> {
    const SIGN: u64 = 1 << 63;

    // SAFETY of every reader: the caller hands the chunk's values pointer and
    // an in-bounds row of a null-free fixed-width column. Unaligned reads
    // tolerate sliced offsets.
    fn read_signed<T: Copy + Into<i64>>(values: *const u8, row: usize) -> u64 {
        let value: T = unsafe { (values as *const T).add(row).read_unaligned() };
        (value.into() as u64) ^ SIGN
    }
    fn read_unsigned<T: Copy + Into<u64>>(values: *const u8, row: usize) -> u64 {
        let value: T = unsafe { (values as *const T).add(row).read_unaligned() };
        value.into()
    }
    // The IEEE total-order flip: negatives (sign bit set) invert entirely, so
    // more-negative orders lower; the rest just set the sign bit to sit above
    // every negative. Matches the total order Arrow's sort applies, NaN at
    // the ends included.
    fn read_f64(values: *const u8, row: usize) -> u64 {
        let bits: u64 = unsafe { (values as *const u64).add(row).read_unaligned() };
        if bits & SIGN != 0 { !bits } else { bits | SIGN }
    }
    fn read_f32(values: *const u8, row: usize) -> u64 {
        let bits: u32 = unsafe { (values as *const u32).add(row).read_unaligned() };
        let flipped = if bits & (1 << 31) != 0 {
            !bits
        } else {
            bits | (1 << 31)
        };
        flipped as u64
    }

    match data_type {
        DataType::Int64
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Time64(_)
        | DataType::Duration(_)
        | DataType::Decimal64(_, _) => Some(read_signed::<i64>),
        DataType::Int32 | DataType::Date32 | DataType::Time32(_) => Some(read_signed::<i32>),
        DataType::Int16 => Some(read_signed::<i16>),
        DataType::Int8 => Some(read_signed::<i8>),
        DataType::UInt64 => Some(read_unsigned::<u64>),
        DataType::UInt32 => Some(read_unsigned::<u32>),
        DataType::UInt16 => Some(read_unsigned::<u16>),
        DataType::UInt8 => Some(read_unsigned::<u8>),
        DataType::Float64 => Some(read_f64),
        DataType::Float32 => Some(read_f32),
        _ => None,
    }
}

/// One null-free byte-view key column: rows compare by their value bytes.
pub(super) struct ViewBytesKeyOrdering<'chunks> {
    left: Vec<ViewChunk<'chunks>>,
    right: Vec<ViewChunk<'chunks>>,
    descending: bool,
}

/// One chunk's view column: the 16-byte views and the data buffers the
/// out-of-line values live in.
struct ViewChunk<'chunks> {
    views: *const u128,
    data_buffers: &'chunks [arrow_buffer::Buffer],
}

/// A byte-view value up to this length lives inside its own view.
const INLINE_VIEW_LEN: u32 = 12;

fn view_chunks<'chunks>(
    key_column: usize,
    chunks: &'chunks [RecordBatch],
) -> Vec<ViewChunk<'chunks>> {
    chunks
        .iter()
        .map(|chunk| {
            let column = chunk.column(key_column);
            let (views, data_buffers) = match column.data_type() {
                DataType::Utf8View => {
                    let array = as_string_view(column);
                    (array.views().as_ptr(), array.data_buffers())
                }
                DataType::BinaryView => {
                    let array = as_binary_view(column);
                    (array.views().as_ptr(), array.data_buffers())
                }
                _ => unreachable!("selected for a byte-view key"),
            };
            ViewChunk {
                views,
                data_buffers,
            }
        })
        .collect()
}

fn as_string_view(column: &ArrayRef) -> &arrow_array::StringViewArray {
    column
        .as_any()
        .downcast_ref()
        .expect("the column's data type said Utf8View")
}

fn as_binary_view(column: &ArrayRef) -> &arrow_array::BinaryViewArray {
    column
        .as_any()
        .downcast_ref()
        .expect("the column's data type said BinaryView")
}

/// The bytes of the value at `row`: inline within the view itself for short
/// values, in the data buffer the view names otherwise.
fn view_value_bytes<'chunks>(chunk: &ViewChunk<'chunks>, row: usize) -> &'chunks [u8] {
    // SAFETY: the row is in bounds and views are 16 bytes each; the inline
    // bytes are read out of the view itself through a pointer into the views
    // buffer, which the chunk keeps alive.
    unsafe {
        let view_ptr = chunk.views.add(row);
        let view = view_ptr.read_unaligned();
        let length = view as u32;
        if length <= INLINE_VIEW_LEN {
            std::slice::from_raw_parts((view_ptr as *const u8).add(4), length as usize)
        } else {
            let buffer = (view >> 64) as u32 as usize;
            let offset = (view >> 96) as u32 as usize;
            &chunk.data_buffers[buffer][offset..offset + length as usize]
        }
    }
}

impl KeyOrdering for ViewBytesKeyOrdering<'_> {
    fn compare(&mut self, left: RunRow, right: RunRow) -> Ordering {
        let left_bytes = view_value_bytes(&self.left[left.chunk], left.row);
        let right_bytes = view_value_bytes(&self.right[right.chunk], right.row);
        let ascending = left_bytes.cmp(right_bytes);
        if self.descending {
            ascending.reverse()
        } else {
            ascending
        }
    }
}

/// How many `(left chunk, right chunk)` comparator sets
/// [`ComparatorKeyOrdering`] retains before starting over. A k-way merge heap
/// revisits a small working set of pairs, so the bound exists only to keep a
/// merge over thousands of chunks from holding comparators for every pair it
/// ever touched.
const MAX_CACHED_COMPARATOR_PAIRS: usize = 1024;

/// The general fallback: Arrow comparators per key column, built for each pair
/// of chunks as it is first compared and cached. A k-way merge interleaves
/// many chunk pairs, so the cache is a map rather than the last pair alone.
pub(super) struct ComparatorKeyOrdering<'chunks> {
    order_by: Vec<OrderBy>,
    left_chunks: &'chunks [RecordBatch],
    right_chunks: &'chunks [RecordBatch],
    /// One comparator per key column, in key order, per chunk pair compared.
    comparators_by_chunks: HashMap<(usize, usize), Vec<DynComparator>>,
}

impl ComparatorKeyOrdering<'_> {
    fn build_comparators(&self, left_chunk: usize, right_chunk: usize) -> Vec<DynComparator> {
        self.order_by
            .iter()
            .map(|key| {
                make_comparator(
                    self.left_chunks[left_chunk]
                        .column(key.column_idx())
                        .as_ref(),
                    self.right_chunks[right_chunk]
                        .column(key.column_idx())
                        .as_ref(),
                    SortOptions {
                        descending: key.descending(),
                        nulls_first: key.nulls_first(),
                    },
                )
                .expect("both chunks share the schema the key was planned against")
            })
            .collect()
    }
}

impl KeyOrdering for ComparatorKeyOrdering<'_> {
    fn compare(&mut self, left: RunRow, right: RunRow) -> Ordering {
        let chunk_pair = (left.chunk, right.chunk);
        if !self.comparators_by_chunks.contains_key(&chunk_pair) {
            if self.comparators_by_chunks.len() >= MAX_CACHED_COMPARATOR_PAIRS {
                self.comparators_by_chunks.clear();
            }
            let comparators = self.build_comparators(left.chunk, right.chunk);
            self.comparators_by_chunks.insert(chunk_pair, comparators);
        }
        for comparator in &self.comparators_by_chunks[&chunk_pair] {
            let ordering = comparator(left.row, right.row);
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    }
}
