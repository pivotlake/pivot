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

/// The leading columns of a row's sort key as one value whose own order is
/// the sort order: a lower value sorts first. A sort that holds these values
/// next to each other compares them without reading the key columns again.
///
/// There is one type per width, and a sort is compiled for each, so that it
/// moves no more bytes per row than the key columns need. An unsigned
/// integer holds up to 64 bits of key columns, and [`Packed`] holds more.
pub(super) trait SortKey: Copy + Ord {
    /// How many bits of key columns the value holds.
    const BITS: u32;
    /// The value before any column is appended.
    const EMPTY: Self;

    /// Moves the value up by `bits` bits and puts `column_value`, which fits
    /// in `bits` bits, in the bits that frees. A column appended earlier
    /// therefore decides a comparison before one appended later.
    fn append_column(&mut self, bits: u32, column_value: u64);

    /// [`append_column`](Self::append_column) for a column of up to 128
    /// bits, which goes in as its high part followed by its low 64 bits.
    #[inline(always)]
    fn append_wide_column(&mut self, bits: u32, column_value: u128) {
        // The same column is appended to every row, so which of the two
        // cases runs is the same from row to row.
        if bits > u64::BITS {
            self.append_column(bits - u64::BITS, (column_value >> u64::BITS) as u64);
            self.append_column(u64::BITS, column_value as u64);
        } else {
            self.append_column(bits, column_value as u64);
        }
    }

    /// Sets the lowest `bits` bits to zero, which are the columns appended
    /// last.
    fn clear_lowest_bits(&mut self, bits: u32);
}

macro_rules! impl_sort_key_for_integer {
    ($($integer:ty),*) => {
        $(impl SortKey for $integer {
            const BITS: u32 = <$integer>::BITS;
            const EMPTY: Self = 0;

            #[inline(always)]
            fn append_column(&mut self, bits: u32, column_value: u64) {
                // A column of 64 bits replaces the whole value, and moving a
                // 64-bit integer up by 64 bits is not defined.
                *self = if bits == u64::BITS {
                    column_value as $integer
                } else {
                    ((u64::from(*self) << bits) | column_value) as $integer
                };
            }

            #[inline(always)]
            fn clear_lowest_bits(&mut self, bits: u32) {
                // Moving an integer by its whole width is not defined.
                *self = if bits >= <$integer>::BITS {
                    0
                } else {
                    (*self >> bits) << bits
                };
            }
        })*
    };
}

impl_sort_key_for_integer!(u8, u16, u32, u64);

/// A [`SortKey`] of `WORDS` 64-bit words, for key columns that need more
/// than 64 bits together. The first word holds the highest bits, so the
/// words compare in the order the columns were appended.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Packed<const WORDS: usize>([u64; WORDS]);

impl<const WORDS: usize> Ord for Packed<WORDS> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        for word in 0..WORDS {
            if self.0[word] != other.0[word] {
                return self.0[word].cmp(&other.0[word]);
            }
        }
        Ordering::Equal
    }
}

impl<const WORDS: usize> PartialOrd for Packed<WORDS> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    /// Whether this key is lower, which is the one question a sort asks.
    /// It stops at the first word that differs. That is nearly always the
    /// first word, which holds the leading key column, so the branch is one
    /// the processor predicts, and the words after it are not compared.
    /// Working the answer out from all the words without a branch measured
    /// slower.
    #[inline(always)]
    fn lt(&self, other: &Self) -> bool {
        for word in 0..WORDS - 1 {
            if self.0[word] != other.0[word] {
                return self.0[word] < other.0[word];
            }
        }
        self.0[WORDS - 1] < other.0[WORDS - 1]
    }
}

impl<const WORDS: usize> SortKey for Packed<WORDS> {
    const BITS: u32 = 64 * WORDS as u32;
    const EMPTY: Self = Self([0; WORDS]);

    #[inline(always)]
    fn append_column(&mut self, bits: u32, column_value: u64) {
        // The same column is appended to every row, so which of the two
        // cases runs is the same from row to row.
        if bits == u64::BITS {
            // A column of 64 bits moves every word up by one.
            for word in 0..WORDS - 1 {
                self.0[word] = self.0[word + 1];
            }
            self.0[WORDS - 1] = column_value;
        } else {
            // Each word takes its own bits moved up and, below them, the top
            // bits of the word after it, which has not been changed yet.
            for word in 0..WORDS - 1 {
                self.0[word] = (self.0[word] << bits) | (self.0[word + 1] >> (u64::BITS - bits));
            }
            self.0[WORDS - 1] = (self.0[WORDS - 1] << bits) | column_value;
        }
    }

    fn clear_lowest_bits(&mut self, bits: u32) {
        // The last word holds the lowest bits.
        let whole_words = ((bits / u64::BITS) as usize).min(WORDS);
        for word in WORDS - whole_words..WORDS {
            self.0[word] = 0;
        }
        let remaining_bits = bits % u64::BITS;
        if whole_words < WORDS && remaining_bits > 0 {
            let word = WORDS - 1 - whole_words;
            self.0[word] = (self.0[word] >> remaining_bits) << remaining_bits;
        }
    }
}

/// The most bits of key columns a [`SortKey`] holds.
const MAX_PACKED_BITS: u32 = Packed::<4>::BITS;

/// A row's position with its [`SortKey`], which is what a sort moves around.
#[derive(Clone, Copy)]
pub(super) struct KeyedRow<Key> {
    pub(super) key: Key,
    pub(super) chunk: u32,
    pub(super) row: u32,
}

/// How many bits the length of a value takes in a key that holds the first
/// [`INLINE_VIEW_LEN`] bytes of each value. It holds every length up to
/// `INLINE_VIEW_LEN`, and [`LONGER_VALUE_LENGTH`] for all longer ones.
const VALUE_PREFIX_LENGTH_BITS: u32 = 4;

/// The length written to a key for a value longer than [`INLINE_VIEW_LEN`]
/// bytes, of which the key holds only the first bytes.
const LONGER_VALUE_LENGTH: u32 = INLINE_VIEW_LEN + 1;

/// Packs the leading columns of a sort key into the [`SortKey`]s of
/// [`KeyedRow`]s.
///
/// A column can be packed when it has no nulls and is either fixed-width and
/// at most 64 bits wide, or a byte view (a string or binary column). Columns
/// are packed in key order, each as an unsigned integer that orders like the
/// column does (inverted for a descending column), so comparing two packed
/// keys compares those columns in key order.
///
/// How many bits a byte-view column takes depends on how long its values
/// are. So every sort measures the rows it was given and lays its keys out
/// for them (see [`PackedKeyLayout`]). The keys of one sort are never
/// compared with the keys of another, so two sorts over the same batches may
/// lay their keys out differently.
pub(super) struct PackedKeyPrefix<'chunks> {
    /// The key's leading columns that can be packed, in key order.
    columns: Vec<PackedColumn<'chunks>>,
    /// How many columns the whole sort key has.
    key_column_count: usize,
}

struct PackedColumn<'chunks> {
    values: PackedColumnValues<'chunks>,
    descending: bool,
}

/// Where a packed column's values are read from, per chunk.
enum PackedColumnValues<'chunks> {
    FixedWidth {
        chunks: Vec<FixedWidthChunk<'chunks>>,
        data_type: DataType,
        /// How many bits of the packed key the column takes.
        bits: u32,
        /// Subtracted from the value the column's reader returns to bring it
        /// within `bits` bits. A signed column narrower than 64 bits reads as
        /// a value centred on 2^63, and this moves its lowest value to zero.
        bias: u64,
    },
    ViewBytes {
        chunks: Vec<ViewChunk<'chunks>>,
    },
}

/// How one sort lays the packed columns out in its keys.
pub(super) struct PackedKeyLayout {
    /// The layout of each column the keys hold, in key order. These are the
    /// leading columns of the [`PackedKeyPrefix`] that fit in the widest
    /// [`SortKey`] together.
    columns: Vec<PackedColumnLayout>,
    /// How many bits the packed columns take together.
    pub(super) bits: u32,
}

/// How one sort writes one packed column into its keys.
#[derive(Clone, Copy)]
enum PackedColumnLayout {
    /// A fixed-width column, as the unsigned integer its reader returns.
    FixedWidth { bits: u32 },
    /// A byte-view column whose longest value among the sorted rows is
    /// `longest_length` bytes, which is at most [`INLINE_VIEW_LEN`]. A key
    /// holds the whole value: its bytes, with zeros after the last one, and
    /// then its length. Two values whose bytes differ only in zeros at the
    /// end get the same bytes in the key, and the length puts the shorter
    /// one first.
    WholeValues { longest_length: u32 },
    /// A byte-view column with a value longer than [`INLINE_VIEW_LEN`] bytes
    /// among the sorted rows. A key holds the first `INLINE_VIEW_LEN` bytes
    /// of the value and then its length, written as [`LONGER_VALUE_LENGTH`]
    /// when the value is longer than that. A shorter value is whole in its
    /// key. Two longer values with the same first bytes get the same key
    /// even when they differ further on, so such rows are compared on the
    /// full key (see [`PackedKeyPrefix::pack_rows`]).
    ValuePrefixes,
}

impl PackedColumnLayout {
    /// How many bits of a key the column takes.
    fn count_bits(self) -> u32 {
        match self {
            PackedColumnLayout::FixedWidth { bits } => bits,
            PackedColumnLayout::WholeValues { longest_length } => {
                let length_bits = u32::BITS - longest_length.leading_zeros();
                8 * longest_length + length_bits
            }
            PackedColumnLayout::ValuePrefixes => 8 * INLINE_VIEW_LEN + VALUE_PREFIX_LENGTH_BITS,
        }
    }
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
        for key in order_by {
            let key_column = key.column_idx();
            let data_type = chunks.first()?.column(key_column).data_type();
            let has_nulls = chunks
                .iter()
                .any(|chunk| chunk.column(key_column).null_count() > 0);
            if has_nulls {
                break;
            }
            let values = if normalized_key_reader(data_type).is_some() {
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
                PackedColumnValues::FixedWidth {
                    chunks: fixed_width_chunks(key_column, chunks),
                    data_type: data_type.clone(),
                    bits,
                    bias,
                }
            } else if matches!(data_type, DataType::Utf8View | DataType::BinaryView) {
                PackedColumnValues::ViewBytes {
                    chunks: view_chunks(key_column, chunks),
                }
            } else {
                break;
            };
            columns.push(PackedColumn {
                values,
                descending: key.descending(),
            });
        }
        if columns.is_empty() {
            return None;
        }
        Some(Self {
            columns,
            key_column_count: order_by.len(),
        })
    }

    /// Lays the keys out for a sort of `rows`, each a `(chunk, row)` pair.
    ///
    /// A byte-view column is laid out for the longest value it has among
    /// `rows`, which this reads from every row's view. Columns are taken in
    /// key order for as long as they fit in the widest [`SortKey`] together.
    pub(super) fn plan_layout(&self, rows: &[(u32, u32)]) -> PackedKeyLayout {
        let mut columns = Vec::with_capacity(self.columns.len());
        let mut used_bits = 0;
        for column in &self.columns {
            let column_layout = match &column.values {
                PackedColumnValues::FixedWidth { bits, .. } => {
                    PackedColumnLayout::FixedWidth { bits: *bits }
                }
                PackedColumnValues::ViewBytes { chunks } => {
                    let mut longest_length = 0;
                    for &(chunk, row) in rows {
                        // SAFETY: the row is in bounds of its chunk's views.
                        // A view starts with its value's length.
                        let view = unsafe {
                            chunks[chunk as usize]
                                .views
                                .add(row as usize)
                                .read_unaligned()
                        };
                        longest_length = longest_length.max(view as u32);
                    }
                    if longest_length <= INLINE_VIEW_LEN {
                        PackedColumnLayout::WholeValues { longest_length }
                    } else {
                        PackedColumnLayout::ValuePrefixes
                    }
                }
            };
            if used_bits + column_layout.count_bits() > MAX_PACKED_BITS {
                break;
            }
            used_bits += column_layout.count_bits();
            columns.push(column_layout);
        }
        PackedKeyLayout {
            columns,
            bits: used_bits,
        }
    }

    /// Replaces `keyed_rows` with one entry per position of `rows`, each a
    /// `(chunk, row)` pair, with keys laid out as `layout` says. `layout`
    /// must have been planned for these `rows`, and `Key` must hold at least
    /// its [`bits`](PackedKeyLayout::bits) bits. The keys are filled a
    /// column at a time, so each column is read by a loop written for its
    /// type.
    ///
    /// Returns whether the keys alone order the rows. They do not when a key
    /// column is left out of the keys, or when some key holds only the first
    /// bytes of its value. Rows with equal keys then still have to be
    /// compared on the full key.
    ///
    /// A key that holds only the first bytes of a value has every column
    /// after that value set to zero. Two rows that agree up to those first
    /// bytes therefore get equal keys, whatever their later columns hold,
    /// and the full key decides between them.
    pub(super) fn pack_rows<Key: SortKey>(
        &self,
        layout: &PackedKeyLayout,
        rows: &[(u32, u32)],
        keyed_rows: &mut Vec<KeyedRow<Key>>,
    ) -> bool {
        assert!(layout.bits <= Key::BITS);
        keyed_rows.clear();
        keyed_rows.extend(rows.iter().map(|&(chunk, row)| KeyedRow {
            key: Key::EMPTY,
            chunk,
            row,
        }));
        // Each entry whose key holds only the first bytes of a value, with
        // how many bits the later columns take in the key.
        let mut entries_holding_a_value_prefix = Vec::new();
        let mut bits_after_column = layout.bits;
        for (column, &column_layout) in self.columns.iter().zip(&layout.columns) {
            bits_after_column -= column_layout.count_bits();
            match (&column.values, column_layout) {
                (
                    PackedColumnValues::FixedWidth {
                        chunks,
                        data_type,
                        bits,
                        bias,
                    },
                    PackedColumnLayout::FixedWidth { .. },
                ) => match_normalized_key_reader!(
                    data_type,
                    |read_key| column
                        .append_fixed_width_values(chunks, *bits, *bias, keyed_rows, read_key),
                    unreachable!("a fixed-width packed column has a normalized key reader")
                ),
                (
                    PackedColumnValues::ViewBytes { chunks },
                    PackedColumnLayout::WholeValues { longest_length },
                ) => column.append_whole_values(
                    chunks,
                    longest_length,
                    column_layout.count_bits(),
                    keyed_rows,
                ),
                (PackedColumnValues::ViewBytes { chunks }, PackedColumnLayout::ValuePrefixes) => {
                    column.append_value_prefixes(
                        chunks,
                        bits_after_column,
                        keyed_rows,
                        &mut entries_holding_a_value_prefix,
                    )
                }
                _ => unreachable!("the layout was planned for these columns"),
            }
        }
        for &(entry, bits_of_later_columns) in &entries_holding_a_value_prefix {
            keyed_rows[entry]
                .key
                .clear_lowest_bits(bits_of_later_columns);
        }
        layout.columns.len() == self.key_column_count && entries_holding_a_value_prefix.is_empty()
    }
}

impl PackedColumn<'_> {
    /// Appends the value this fixed-width column has in every row of
    /// `keyed_rows` to the row's key.
    #[inline(always)]
    fn append_fixed_width_values<Key: SortKey>(
        &self,
        chunks: &[FixedWidthChunk],
        bits: u32,
        bias: u64,
        keyed_rows: &mut [KeyedRow<Key>],
        read_key: impl Fn(*const u8, usize) -> u64,
    ) {
        let highest_value = u64::MAX >> (u64::BITS - bits);
        for keyed_row in keyed_rows {
            let first_value = chunks[keyed_row.chunk as usize].first_value;
            let value = read_key(first_value, keyed_row.row as usize) - bias;
            let value = if self.descending {
                highest_value - value
            } else {
                value
            };
            keyed_row.key.append_column(bits, value);
        }
    }

    /// Appends the whole value this byte-view column has in every row of
    /// `keyed_rows` to the row's key, as `bits` bits: the value's bytes in
    /// `longest_length` bytes, then its length. No value is longer than
    /// `longest_length`, which is at most [`INLINE_VIEW_LEN`].
    fn append_whole_values<Key: SortKey>(
        &self,
        chunks: &[ViewChunk],
        longest_length: u32,
        bits: u32,
        keyed_rows: &mut [KeyedRow<Key>],
    ) {
        if longest_length == 0 {
            // Every value is empty, and the column takes no bits.
            return;
        }
        let length_bits = bits - 8 * longest_length;
        let highest_value = u128::MAX >> (u128::BITS - bits);
        for keyed_row in keyed_rows {
            // SAFETY: the row is in bounds of its chunk's views.
            let view = unsafe {
                chunks[keyed_row.chunk as usize]
                    .views
                    .add(keyed_row.row as usize)
                    .read_unaligned()
            };
            let length = view as u32;
            let bytes = ViewChunk::read_inline_bytes(view, length)
                >> (8 * (INLINE_VIEW_LEN - longest_length));
            let value = (bytes << length_bits) | u128::from(length);
            let value = if self.descending {
                highest_value - value
            } else {
                value
            };
            keyed_row.key.append_wide_column(bits, value);
        }
    }

    /// Appends the first [`INLINE_VIEW_LEN`] bytes of the value this
    /// byte-view column has in every row of `keyed_rows` to the row's key,
    /// followed by the value's length.
    ///
    /// A value longer than that is not whole in its key. Its length is
    /// written as [`LONGER_VALUE_LENGTH`], and its entry is added to
    /// `entries_holding_a_value_prefix` together with `bits_after_column`,
    /// the number of bits the later columns take in the key.
    fn append_value_prefixes<Key: SortKey>(
        &self,
        chunks: &[ViewChunk],
        bits_after_column: u32,
        keyed_rows: &mut [KeyedRow<Key>],
        entries_holding_a_value_prefix: &mut Vec<(usize, u32)>,
    ) {
        let bits = 8 * INLINE_VIEW_LEN + VALUE_PREFIX_LENGTH_BITS;
        let highest_value = u128::MAX >> (u128::BITS - bits);
        for (entry, keyed_row) in keyed_rows.iter_mut().enumerate() {
            let chunk = &chunks[keyed_row.chunk as usize];
            // SAFETY: the row is in bounds of its chunk's views.
            let view = unsafe { chunk.views.add(keyed_row.row as usize).read_unaligned() };
            let length = view as u32;
            let (bytes, written_length) = if length <= INLINE_VIEW_LEN {
                (ViewChunk::read_inline_bytes(view, length), length)
            } else {
                // A longer value lives in the data buffer its view names, at
                // the offset the view gives.
                let buffer = (view >> 64) as u32 as usize;
                let offset = (view >> 96) as u32 as usize;
                let first_bytes =
                    &chunk.data_buffers[buffer][offset..offset + INLINE_VIEW_LEN as usize];
                let mut number_bytes = [0; 16];
                number_bytes[4..].copy_from_slice(first_bytes);
                entries_holding_a_value_prefix.push((entry, bits_after_column));
                (u128::from_be_bytes(number_bytes), LONGER_VALUE_LENGTH)
            };
            let value = (bytes << VALUE_PREFIX_LENGTH_BITS) | u128::from(written_length);
            let value = if self.descending {
                highest_value - value
            } else {
                value
            };
            keyed_row.key.append_wide_column(bits, value);
        }
    }
}

impl KeyOrdering for SelectedKeyOrdering<'_> {
    /// Compares through whichever ordering was selected. The choice is made
    /// again on every call, so this is for comparisons that are rare, such
    /// as those between rows whose packed keys are equal.
    fn compare(&mut self, left: RunRow, right: RunRow) -> Ordering {
        match self {
            SelectedKeyOrdering::FixedWidth(ordering) => ordering.compare(left, right),
            SelectedKeyOrdering::ViewBytes(ordering) => ordering.compare(left, right),
            SelectedKeyOrdering::Columnwise(ordering) => ordering.compare(left, right),
            SelectedKeyOrdering::General(ordering) => ordering.compare(left, right),
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

impl ViewChunk<'_> {
    /// The bytes of a value that lives inside `view`, as a 96-bit number
    /// that orders like the bytes do: the first byte is the highest 8 bits,
    /// and the bits after the value's last byte are zero. `length` is the
    /// value's length, at most [`INLINE_VIEW_LEN`].
    #[inline(always)]
    fn read_inline_bytes(view: u128, length: u32) -> u128 {
        // The 12 bytes after the view's 4-byte length hold the value. Read
        // as a little-endian number the first byte is the lowest, so the
        // bytes are reversed.
        let bytes = (view >> 32).swap_bytes() >> 32;
        // Zero whatever the view holds after the value's last byte.
        let bits_after_value = 8 * (INLINE_VIEW_LEN - length);
        (bytes >> bits_after_value) << bits_after_value
    }
}

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
