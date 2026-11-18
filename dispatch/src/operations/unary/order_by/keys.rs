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
//! Four implementations, picked by [`select_key_ordering`] from what the key
//! actually is, fastest first:
//!
//! - [`FixedWidthKeyOrdering`]: one null-free fixed-width column (integers,
//!   dates, timestamps, 64-bit decimals, floats). Each value reads as a `u64`
//!   whose unsigned order equals the column's order (sign flip for signed
//!   integers, the IEEE total-order flip for floats), so a comparison is two
//!   loads and an integer compare.
//! - [`ViewBytesKeyOrdering`]: one null-free byte-view column (strings,
//!   binary). A comparison resolves each view to its bytes and memcmps.
//! - [`ColumnwiseKeyOrdering`]: any number of columns, with or without
//!   nulls, each fixed-width, a 128-bit decimal, a boolean or a byte view. A
//!   comparison walks the key columns until one differs, reading each
//!   column's values straight from its buffers.
//! - [`ComparatorKeyOrdering`]: everything else, meaning a key with a column
//!   of an interval or nested type. Arrow comparators are built per pair of chunks
//!   and cached by pair, and a comparison walks the key columns until one
//!   differs. A k-way merge heap interleaves chunk pairs, so the cache keeps
//!   every pair the walk revisits instead of only the latest one.

use std::cmp::Ordering;
use std::collections::HashMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType,
    DurationMillisecondType, DurationNanosecondType, DurationSecondType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_ord::ord::{DynComparator, make_comparator};
use arrow_schema::{ArrowError, DataType, SortOptions, TimeUnit};
use std::borrow::Borrow;

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
    Columnwise(ColumnwiseKeyOrdering<'chunks>),
    General(ComparatorKeyOrdering<'chunks>),
}

/// Dispatches to the selected key representation while keeping the comparison
/// loop monomorphized.
macro_rules! with_key_ordering {
    ($selected:expr, |$ordering:ident| $body:expr) => {
        match $selected {
            SelectedKeyOrdering::FixedWidth(mut $ordering) => $body,
            SelectedKeyOrdering::ViewBytes(mut $ordering) => $body,
            SelectedKeyOrdering::Columnwise(mut $ordering) => $body,
            SelectedKeyOrdering::General(mut $ordering) => $body,
        }
    };
}
pub(super) use with_key_ordering;

/// Pick the fastest ordering the key allows for these chunks (see the module
/// docs). The chunks must all share the schema the key columns index into.
///
/// The chunks may be owned batches or references to them, so a caller that
/// orders batches it only borrows does not have to clone them.
pub(super) fn select_key_ordering<'chunks, Batch: Borrow<RecordBatch>>(
    order_by: &[OrderBy],
    left_chunks: &'chunks [Batch],
    right_chunks: &'chunks [Batch],
) -> Result<SelectedKeyOrdering<'chunks>, ArrowError> {
    let left_chunks: Vec<&'chunks RecordBatch> = left_chunks.iter().map(Borrow::borrow).collect();
    let right_chunks: Vec<&'chunks RecordBatch> = right_chunks.iter().map(Borrow::borrow).collect();
    let (left_chunks, right_chunks) = (left_chunks.as_slice(), right_chunks.as_slice());
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
    if let Some(columnwise) = ColumnwiseKeyOrdering::try_new(order_by, left_chunks, right_chunks) {
        return Ok(SelectedKeyOrdering::Columnwise(columnwise));
    }
    Ok(SelectedKeyOrdering::General(ComparatorKeyOrdering {
        order_by: order_by.to_vec(),
        left_chunks: left_chunks.to_vec(),
        right_chunks: right_chunks.to_vec(),
        comparators_by_chunks: HashMap::new(),
    }))
}

/// The key column's type; either side may be empty, but never both.
fn data_type_of(key_column: usize, left: &[&RecordBatch], right: &[&RecordBatch]) -> DataType {
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
    chunks: &[&'chunks RecordBatch],
) -> Vec<FixedWidthChunk<'chunks>> {
    chunks
        .iter()
        .map(|chunk| FixedWidthChunk {
            first_value: FixedWidthChunk::find_first_value(chunk.column(key_column).as_ref()),
            _keeps_chunk_alive: std::marker::PhantomData,
        })
        .collect()
}

impl FixedWidthChunk<'_> {
    const SIGN: u64 = 1 << 63;

    // SAFETY of every reader: the caller hands the chunk's values pointer and
    // an in-bounds row of a null-free fixed-width column. Unaligned reads
    // tolerate sliced offsets.

    /// Row `row` of signed integers at `values`, as a `u64` that orders like
    /// the integers do.
    #[inline(always)]
    fn read_signed<T: Copy + Into<i64>>(values: *const u8, row: usize) -> u64 {
        let value: T = unsafe { (values as *const T).add(row).read_unaligned() };
        (value.into() as u64) ^ Self::SIGN
    }

    /// Row `row` of unsigned integers at `values`, widened to a `u64`.
    #[inline(always)]
    fn read_unsigned<T: Copy + Into<u64>>(values: *const u8, row: usize) -> u64 {
        let value: T = unsafe { (values as *const T).add(row).read_unaligned() };
        value.into()
    }

    /// Row `row` of 64-bit floats at `values`, as a `u64` in the IEEE total
    /// order: negatives (sign bit set) invert entirely, so more-negative
    /// orders lower, and the rest set the sign bit to sit above every
    /// negative. Matches the total order Arrow's sort applies, NaN at the
    /// ends included.
    #[inline(always)]
    fn read_f64(values: *const u8, row: usize) -> u64 {
        let bits: u64 = unsafe { (values as *const u64).add(row).read_unaligned() };
        if bits & Self::SIGN != 0 {
            !bits
        } else {
            bits | Self::SIGN
        }
    }

    /// Row `row` of 32-bit floats at `values`, ordered like
    /// [`read_f64`](Self::read_f64).
    #[inline(always)]
    fn read_f32(values: *const u8, row: usize) -> u64 {
        let bits: u32 = unsafe { (values as *const u32).add(row).read_unaligned() };
        let flipped = if bits & (1 << 31) != 0 {
            !bits
        } else {
            bits | (1 << 31)
        };
        flipped as u64
    }

    /// The address of the first value of a fixed-width column.
    fn find_first_value(column: &dyn Array) -> *const u8 {
        macro_rules! first_value_of {
            ($($data_type:pat => $primitive:ty,)+) => {
                match column.data_type() {
                    $($data_type => column.as_primitive::<$primitive>().values().as_ptr() as *const u8,)+
                    other => unreachable!("{other} is not a fixed-width key type"),
                }
            };
        }
        first_value_of! {
            DataType::Int8 => Int8Type,
            DataType::Int16 => Int16Type,
            DataType::Int32 => Int32Type,
            DataType::Int64 => Int64Type,
            DataType::UInt8 => UInt8Type,
            DataType::UInt16 => UInt16Type,
            DataType::UInt32 => UInt32Type,
            DataType::UInt64 => UInt64Type,
            DataType::Float32 => Float32Type,
            DataType::Float64 => Float64Type,
            DataType::Date32 => Date32Type,
            DataType::Date64 => Date64Type,
            DataType::Timestamp(TimeUnit::Second, _) => TimestampSecondType,
            DataType::Timestamp(TimeUnit::Millisecond, _) => TimestampMillisecondType,
            DataType::Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecondType,
            DataType::Timestamp(TimeUnit::Nanosecond, _) => TimestampNanosecondType,
            DataType::Time32(TimeUnit::Second) => Time32SecondType,
            DataType::Time32(TimeUnit::Millisecond) => Time32MillisecondType,
            DataType::Time64(TimeUnit::Microsecond) => Time64MicrosecondType,
            DataType::Time64(TimeUnit::Nanosecond) => Time64NanosecondType,
            DataType::Duration(TimeUnit::Second) => DurationSecondType,
            DataType::Duration(TimeUnit::Millisecond) => DurationMillisecondType,
            DataType::Duration(TimeUnit::Microsecond) => DurationMicrosecondType,
            DataType::Duration(TimeUnit::Nanosecond) => DurationNanosecondType,
            DataType::Decimal32(_, _) => Decimal32Type,
            DataType::Decimal64(_, _) => Decimal64Type,
            DataType::Decimal128(_, _) => Decimal128Type,
        }
    }
}

/// Evaluates `$with_reader` with `$read_key` bound to the reader that turns
/// a row of a chunk's values into a `u64` whose unsigned order equals the sort
/// order of a `$data_type` column, or `$without_reader` for a type with no
/// such form. Each type gets its own copy of `$with_reader`, so a loop
/// written there reads its values without a call per row.
macro_rules! match_normalized_key_reader {
    ($data_type:expr, |$read_key:ident| $with_reader:expr, $without_reader:expr) => {
        match $data_type {
            DataType::Int64
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time64(_)
            | DataType::Duration(_)
            | DataType::Decimal64(_, _) => {
                let $read_key = FixedWidthChunk::read_signed::<i64>;
                $with_reader
            }
            DataType::Int32
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Decimal32(_, _) => {
                let $read_key = FixedWidthChunk::read_signed::<i32>;
                $with_reader
            }
            DataType::Int16 => {
                let $read_key = FixedWidthChunk::read_signed::<i16>;
                $with_reader
            }
            DataType::Int8 => {
                let $read_key = FixedWidthChunk::read_signed::<i8>;
                $with_reader
            }
            DataType::UInt64 => {
                let $read_key = FixedWidthChunk::read_unsigned::<u64>;
                $with_reader
            }
            DataType::UInt32 => {
                let $read_key = FixedWidthChunk::read_unsigned::<u32>;
                $with_reader
            }
            DataType::UInt16 => {
                let $read_key = FixedWidthChunk::read_unsigned::<u16>;
                $with_reader
            }
            DataType::UInt8 => {
                let $read_key = FixedWidthChunk::read_unsigned::<u8>;
                $with_reader
            }
            DataType::Float64 => {
                let $read_key = FixedWidthChunk::read_f64;
                $with_reader
            }
            DataType::Float32 => {
                let $read_key = FixedWidthChunk::read_f32;
                $with_reader
            }
            _ => $without_reader,
        }
    };
}

/// A row's position with the leading columns of its sort key packed into one
/// integer, so that sorting compares integers held next to each other
/// instead of reading the key columns for every comparison.
#[derive(Clone, Copy)]
pub(super) struct PackedRow {
    /// The packed key columns. A lower value sorts first.
    pub(super) key: (u64, u64),
    pub(super) chunk: u32,
    pub(super) row: u32,
}

/// Packs the leading columns of a sort key into [`PackedRow`]s.
///
/// A column can be packed when it is fixed-width, at most 64 bits wide and
/// has no nulls. Columns are packed in key order for as long as they fit in
/// 128 bits together, each as an unsigned integer that orders like the
/// column does (inverted for a descending column), so comparing two packed
/// keys compares those columns in key order.
pub(super) struct PackedKeyPrefix<'chunks> {
    columns: Vec<PackedColumn<'chunks>>,
    /// Whether every key column is packed. When not, rows with equal packed
    /// keys still have to be compared on the full key.
    pub(super) covers_whole_key: bool,
}

struct PackedColumn<'chunks> {
    chunks: Vec<FixedWidthChunk<'chunks>>,
    data_type: DataType,
    /// How many bits of the packed key the column takes.
    bits: u32,
    /// Subtracted from the value the column's reader returns to bring it
    /// within `bits` bits. A signed column narrower than 64 bits reads as a
    /// value centred on 2^63, and this moves its lowest value to zero.
    bias: u64,
    descending: bool,
}

impl<'chunks> PackedKeyPrefix<'chunks> {
    /// `None` when the key's first column cannot be packed.
    pub(super) fn try_new<Batch: Borrow<RecordBatch>>(
        order_by: &[OrderBy],
        chunks: &'chunks [Batch],
    ) -> Option<Self> {
        let chunks: Vec<&'chunks RecordBatch> = chunks.iter().map(Borrow::borrow).collect();
        let chunks = chunks.as_slice();
        let mut columns = Vec::new();
        let mut used_bits = 0;
        for key in order_by {
            let key_column = key.column_idx();
            let data_type = chunks.first()?.column(key_column).data_type();
            let (bits, bias) = match data_type {
                DataType::Int8 => (8, (1 << 63) - (1 << 7)),
                DataType::Int16 => (16, (1 << 63) - (1 << 15)),
                DataType::Int32
                | DataType::Date32
                | DataType::Time32(_)
                | DataType::Decimal32(_, _) => (32, (1 << 63) - (1 << 31)),
                DataType::UInt8 => (8, 0),
                DataType::UInt16 => (16, 0),
                DataType::UInt32 | DataType::Float32 => (32, 0),
                _ => (64, 0),
            };
            let has_nulls = chunks
                .iter()
                .any(|chunk| chunk.column(key_column).null_count() > 0);
            if normalized_key_reader(data_type).is_none() || has_nulls || used_bits + bits > 128 {
                break;
            }
            used_bits += bits;
            columns.push(PackedColumn {
                chunks: fixed_width_chunks(key_column, chunks),
                data_type: data_type.clone(),
                bits,
                bias,
                descending: key.descending(),
            });
        }
        if columns.is_empty() {
            return None;
        }
        Some(Self {
            covers_whole_key: columns.len() == order_by.len(),
            columns,
        })
    }

    /// Replaces `packed_rows` with one entry per position of `rows`, each a
    /// `(chunk, row)` pair. The keys are filled a column at a time, so each
    /// column is read by a loop written for its type.
    pub(super) fn pack_rows(&self, rows: &[(u32, u32)], packed_rows: &mut Vec<PackedRow>) {
        packed_rows.clear();
        packed_rows.extend(rows.iter().map(|&(chunk, row)| PackedRow {
            key: (0, 0),
            chunk,
            row,
        }));
        for column in &self.columns {
            column.append_to_keys(packed_rows);
        }
    }
}

impl PackedColumn<'_> {
    /// Moves the key of every row of `packed_rows` up by this column's bits
    /// and puts the row's value of this column in the bits that frees.
    fn append_to_keys(&self, packed_rows: &mut [PackedRow]) {
        match_normalized_key_reader!(
            &self.data_type,
            |read_key| self.append_values_to_keys(packed_rows, read_key),
            unreachable!("a packed column is of a type with a normalized key reader")
        )
    }

    #[inline(always)]
    fn append_values_to_keys(
        &self,
        packed_rows: &mut [PackedRow],
        read_key: impl Fn(*const u8, usize) -> u64,
    ) {
        let highest_value = u64::MAX >> (64 - self.bits);
        for packed_row in packed_rows {
            let first_value = self.chunks[packed_row.chunk as usize].first_value;
            let value = read_key(first_value, packed_row.row as usize) - self.bias;
            let value = if self.descending {
                highest_value - value
            } else {
                value
            };
            let key = (u128::from(packed_row.key.0) << 64) | u128::from(packed_row.key.1);
            let key = (key << self.bits) | u128::from(value);
            packed_row.key = ((key >> 64) as u64, key as u64);
        }
    }
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
    match_normalized_key_reader!(
        data_type,
        |read_key| Some(read_key as fn(*const u8, usize) -> u64),
        None
    )
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
    chunks: &[&'chunks RecordBatch],
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

/// Any number of key columns, each fixed-width or a byte view, with or
/// without nulls. Rows compare one key column at a time until a column
/// differs.
pub(super) struct ColumnwiseKeyOrdering<'chunks> {
    columns: Vec<KeyColumn<'chunks>>,
}

/// One key column of a [`ColumnwiseKeyOrdering`], bound to every chunk of
/// both runs.
struct KeyColumn<'chunks> {
    values: KeyColumnValues<'chunks>,
    /// Each chunk's validity, `None` for a chunk without nulls. Only read
    /// when `has_nulls` is set.
    left_validity: Vec<Option<&'chunks NullBuffer>>,
    right_validity: Vec<Option<&'chunks NullBuffer>>,
    /// Whether any chunk of either run holds a null in this column. A column
    /// without nulls skips the validity checks altogether.
    has_nulls: bool,
    descending: bool,
    nulls_first: bool,
}

/// How one key column's values are read, per chunk of each run.
enum KeyColumnValues<'chunks> {
    FixedWidth {
        left: Vec<FixedWidthChunk<'chunks>>,
        right: Vec<FixedWidthChunk<'chunks>>,
        read_key: fn(*const u8, usize) -> u64,
    },
    /// A 128-bit decimal, whose values are signed 128-bit integers.
    Int128 {
        left: Vec<FixedWidthChunk<'chunks>>,
        right: Vec<FixedWidthChunk<'chunks>>,
    },
    /// A boolean, stored one bit per value. False orders before true.
    Boolean {
        left: Vec<&'chunks BooleanBuffer>,
        right: Vec<&'chunks BooleanBuffer>,
    },
    ViewBytes {
        left: Vec<ViewChunk<'chunks>>,
        right: Vec<ViewChunk<'chunks>>,
    },
}

impl FixedWidthChunk<'_> {
    /// Reads row `row` of a chunk of 128-bit integers.
    #[inline(always)]
    fn read_int128(&self, row: usize) -> i128 {
        // SAFETY: the chunk's pointer is the first value of a 16-byte-wide
        // column and the row is in bounds. Unaligned reads tolerate sliced
        // offsets.
        unsafe { (self.first_value as *const i128).add(row).read_unaligned() }
    }
}

impl<'chunks> KeyColumn<'chunks> {
    /// The values of a boolean key column, per chunk.
    fn boolean_chunks(
        key_column: usize,
        chunks: &[&'chunks RecordBatch],
    ) -> Vec<&'chunks BooleanBuffer> {
        chunks
            .iter()
            .map(|chunk| chunk.column(key_column).as_boolean().values())
            .collect()
    }

    /// The validity of a key column, per chunk. `None` for a chunk without
    /// nulls.
    fn chunk_validities(
        key_column: usize,
        chunks: &[&'chunks RecordBatch],
    ) -> Vec<Option<&'chunks NullBuffer>> {
        chunks
            .iter()
            .map(|chunk| {
                let column = chunk.column(key_column);
                column.nulls().filter(|_| column.null_count() > 0)
            })
            .collect()
    }
}

impl<'chunks> ColumnwiseKeyOrdering<'chunks> {
    /// `None` when some key column is of a type this ordering cannot read.
    fn try_new(
        order_by: &[OrderBy],
        left_chunks: &[&'chunks RecordBatch],
        right_chunks: &[&'chunks RecordBatch],
    ) -> Option<Self> {
        let columns = order_by
            .iter()
            .map(|key| {
                let key_column = key.column_idx();
                let data_type = data_type_of(key_column, left_chunks, right_chunks);
                let values = if let Some(read_key) = normalized_key_reader(&data_type) {
                    KeyColumnValues::FixedWidth {
                        left: fixed_width_chunks(key_column, left_chunks),
                        right: fixed_width_chunks(key_column, right_chunks),
                        read_key,
                    }
                } else {
                    match data_type {
                        DataType::Utf8View | DataType::BinaryView => KeyColumnValues::ViewBytes {
                            left: view_chunks(key_column, left_chunks),
                            right: view_chunks(key_column, right_chunks),
                        },
                        DataType::Decimal128(_, _) => KeyColumnValues::Int128 {
                            left: fixed_width_chunks(key_column, left_chunks),
                            right: fixed_width_chunks(key_column, right_chunks),
                        },
                        DataType::Boolean => KeyColumnValues::Boolean {
                            left: KeyColumn::boolean_chunks(key_column, left_chunks),
                            right: KeyColumn::boolean_chunks(key_column, right_chunks),
                        },
                        _ => return None,
                    }
                };
                let left_validity = KeyColumn::chunk_validities(key_column, left_chunks);
                let right_validity = KeyColumn::chunk_validities(key_column, right_chunks);
                let has_nulls = left_validity
                    .iter()
                    .chain(&right_validity)
                    .any(Option::is_some);
                Some(KeyColumn {
                    values,
                    left_validity,
                    right_validity,
                    has_nulls,
                    descending: key.descending(),
                    nulls_first: key.nulls_first(),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { columns })
    }
}

impl KeyOrdering for ColumnwiseKeyOrdering<'_> {
    fn compare(&mut self, left: RunRow, right: RunRow) -> Ordering {
        for column in &self.columns {
            if column.has_nulls {
                let left_valid = column.left_validity[left.chunk]
                    .is_none_or(|validity| validity.is_valid(left.row));
                let right_valid = column.right_validity[right.chunk]
                    .is_none_or(|validity| validity.is_valid(right.row));
                match (left_valid, right_valid) {
                    (true, true) => {}
                    // Two nulls tie on this column; the next column decides.
                    (false, false) => continue,
                    // Where nulls go does not depend on the sort direction.
                    (false, true) if column.nulls_first => return Ordering::Less,
                    (false, true) => return Ordering::Greater,
                    (true, false) if column.nulls_first => return Ordering::Greater,
                    (true, false) => return Ordering::Less,
                }
            }
            let ascending = match &column.values {
                KeyColumnValues::FixedWidth {
                    left: left_chunks,
                    right: right_chunks,
                    read_key,
                } => {
                    let left_key = read_key(left_chunks[left.chunk].first_value, left.row);
                    let right_key = read_key(right_chunks[right.chunk].first_value, right.row);
                    left_key.cmp(&right_key)
                }
                KeyColumnValues::Int128 {
                    left: left_chunks,
                    right: right_chunks,
                } => {
                    let left_key = left_chunks[left.chunk].read_int128(left.row);
                    let right_key = right_chunks[right.chunk].read_int128(right.row);
                    left_key.cmp(&right_key)
                }
                KeyColumnValues::Boolean {
                    left: left_chunks,
                    right: right_chunks,
                } => {
                    let left_key = left_chunks[left.chunk].value(left.row);
                    let right_key = right_chunks[right.chunk].value(right.row);
                    left_key.cmp(&right_key)
                }
                KeyColumnValues::ViewBytes {
                    left: left_chunks,
                    right: right_chunks,
                } => {
                    let left_bytes = view_value_bytes(&left_chunks[left.chunk], left.row);
                    let right_bytes = view_value_bytes(&right_chunks[right.chunk], right.row);
                    left_bytes.cmp(right_bytes)
                }
            };
            match ascending {
                Ordering::Equal => {}
                _ if column.descending => return ascending.reverse(),
                _ => return ascending,
            }
        }
        Ordering::Equal
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
    left_chunks: Vec<&'chunks RecordBatch>,
    right_chunks: Vec<&'chunks RecordBatch>,
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
