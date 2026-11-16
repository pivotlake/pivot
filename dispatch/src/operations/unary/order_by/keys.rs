//! How ORDER BY compares two rows by the sort key.
//!
//! [`KeyOrdering`] compares a row of one run with a row of another by
//! reading the key columns, one column at a time until one differs. A run is
//! a chunked sequence of record batches: `left` positions index the left
//! run's chunks and `right` positions the right run's. Comparing rows within
//! a single run passes that run as both sides. It reads fixed-width columns,
//! 128-bit decimals, booleans, strings and binary values, with or without
//! nulls, and no other type can be a sort key.
//!
//! A sort avoids comparing through the columns where it can.
//! [`PackedKeyPrefix`] packs the leading key columns of every row into one
//! [`SortKey`], and the sort compares those. The [`KeyOrdering`] is left
//! with what the packed keys cannot answer: whether rows already arrive in
//! order, where to split a merge, and which of two rows with equal packed
//! keys sorts first.

use std::cmp::Ordering;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType,
    DurationMillisecondType, DurationNanosecondType, DurationSecondType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, RecordBatch, StringArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use std::borrow::Borrow;

use crate::operations::unary::order_by_limit::OrderBy;

/// A row's address within one run: which chunk, and which row of it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct RunRow {
    pub(super) chunk: usize,
    pub(super) row: usize,
}

impl From<(u32, u32)> for RunRow {
    /// The `(chunk, row)` pair a sort holds a row's position as.
    fn from((chunk, row): (u32, u32)) -> Self {
        RunRow {
            chunk: chunk as usize,
            row: row as usize,
        }
    }
}

/// The key column's type; either side may be empty, but never both.
fn data_type_of(key_column: usize, left: &[&RecordBatch], right: &[&RecordBatch]) -> DataType {
    left.iter()
        .chain(right)
        .next()
        .expect("an ordering is only built over at least one chunk")
        .column(key_column)
        .data_type()
        .clone()
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
    /// the integers do and is no wider than they are: the lowest integer
    /// reads as zero.
    #[inline(always)]
    fn read_signed<T: Copy + Into<i64>>(values: *const u8, row: usize) -> u64 {
        let value: T = unsafe { (values as *const T).add(row).read_unaligned() };
        let lowest_integer_magnitude = 1 << (8 * size_of::<T>() - 1);
        (value.into() as u64).wrapping_add(lowest_integer_magnitude)
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
/// moves no more bytes per row than the key columns need. A `u32` holds up
/// to 32 bits of key columns, and a [`Packed`] holds 64 bits per word.
/// Nothing narrower than a `u32` is worth having: a sort entry also holds a
/// row's position in two `u32`s, so a narrower key would not make it smaller.
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

impl SortKey for u32 {
    const BITS: u32 = u32::BITS;
    const EMPTY: Self = 0;

    // Both are worked out in 64 bits, where moving a value by all 32 bits of
    // a `u32` is defined.

    #[inline(always)]
    fn append_column(&mut self, bits: u32, column_value: u64) {
        *self = ((u64::from(*self) << bits) | column_value) as u32;
    }

    #[inline(always)]
    fn clear_lowest_bits(&mut self, bits: u32) {
        *self = ((u64::from(*self) >> bits) << bits) as u32;
    }
}

/// A [`SortKey`] of `WORDS` 64-bit words. The first word holds the highest
/// bits, so the words compare in the order the columns were appended.
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
/// A fixed-width column is packed as the integer its reader returns. A
/// byte-view column is packed as the leading bytes of each value, with zeros
/// after the value's last byte, followed by the value's length. Two values
/// whose bytes differ only in zeros at the end get the same bytes in the
/// key, and the length puts the shorter one first.
///
/// How many bits a byte-view column takes depends on how long its values
/// are. So every sort measures the rows it was given and lays its keys out
/// for them (see [`plan_column_bits`](Self::plan_column_bits)). The keys of
/// one sort are never compared with the keys of another, so two sorts over
/// the same batches may lay their keys out differently.
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
    },
    ViewBytes {
        chunks: Vec<ViewChunk<'chunks>>,
    },
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
                PackedColumnValues::FixedWidth {
                    chunks: fixed_width_chunks(key_column, chunks),
                    data_type: data_type.clone(),
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

    /// How many bits each packed column takes in the keys of a sort of
    /// `rows`, each a `(chunk, row)` pair, in key order. Columns are taken
    /// for as long as they fit in the widest [`SortKey`] together, so the
    /// list can be shorter than the columns that can be packed.
    ///
    /// A fixed-width column takes its width. A byte-view column takes 8 bits
    /// for each byte of the longest value it has among `rows`, which this
    /// reads from every row's view, and after them the bits that value's
    /// length needs. When the longest value is over [`INLINE_VIEW_LEN`]
    /// bytes, the column takes the bits of `INLINE_VIEW_LEN` bytes and of
    /// [`LONGER_VALUE_LENGTH`], and a key holds only the first bytes of such
    /// a value.
    pub(super) fn plan_column_bits(&self, rows: &[(u32, u32)]) -> Vec<u32> {
        let mut column_bits = Vec::with_capacity(self.columns.len());
        let mut used_bits = 0;
        for column in &self.columns {
            let bits = match &column.values {
                PackedColumnValues::FixedWidth { data_type, .. } => {
                    let bytes = data_type
                        .primitive_width()
                        .expect("a fixed-width packed column is of a primitive type");
                    8 * bytes as u32
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
                    let longest_written_length = longest_length.min(LONGER_VALUE_LENGTH);
                    let length_bits = u32::BITS - longest_written_length.leading_zeros();
                    8 * longest_length.min(INLINE_VIEW_LEN) + length_bits
                }
            };
            if used_bits + bits > MAX_PACKED_BITS {
                break;
            }
            used_bits += bits;
            column_bits.push(bits);
        }
        column_bits
    }

    /// Replaces `keyed_rows` with one entry per position of `rows`, each a
    /// `(chunk, row)` pair, with keys that give column `i` the low
    /// `column_bits[i]` bits left after the columns before it.
    /// `column_bits` must be what
    /// [`plan_column_bits`](Self::plan_column_bits) returned for these
    /// `rows`, and `Key` must hold at least their sum. The keys are filled a
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
        column_bits: &[u32],
        rows: &[(u32, u32)],
        keyed_rows: &mut Vec<KeyedRow<Key>>,
    ) -> bool {
        let key_bits: u32 = column_bits.iter().sum();
        assert!(key_bits <= Key::BITS);
        keyed_rows.clear();
        keyed_rows.extend(rows.iter().map(|&(chunk, row)| KeyedRow {
            key: Key::EMPTY,
            chunk,
            row,
        }));
        // Each entry whose key holds only the first bytes of a value, with
        // how many bits the later columns take in the key.
        let mut entries_holding_a_value_prefix = Vec::new();
        let mut bits_after_column = key_bits;
        for (column, &bits) in self.columns.iter().zip(column_bits) {
            if bits == 0 {
                // A byte-view column whose values are all empty.
                continue;
            }
            bits_after_column -= bits;
            // A descending column's values are inverted within its bits, so
            // that a higher value gets a lower key.
            let descending_mask = if column.descending {
                u128::MAX >> (u128::BITS - bits)
            } else {
                0
            };
            match &column.values {
                PackedColumnValues::FixedWidth { chunks, data_type } => {
                    match_normalized_key_reader!(
                        data_type,
                        |read_key| for keyed_row in keyed_rows.iter_mut() {
                            let first_value = chunks[keyed_row.chunk as usize].first_value;
                            let value = read_key(first_value, keyed_row.row as usize);
                            keyed_row
                                .key
                                .append_column(bits, value ^ descending_mask as u64);
                        },
                        unreachable!("a fixed-width packed column has a normalized key reader")
                    )
                }
                PackedColumnValues::ViewBytes { chunks } => ViewChunk::append_values_to_keys(
                    chunks,
                    bits,
                    descending_mask,
                    bits_after_column,
                    keyed_rows,
                    &mut entries_holding_a_value_prefix,
                ),
            }
        }
        for &(entry, bits_of_later_columns) in &entries_holding_a_value_prefix {
            keyed_rows[entry]
                .key
                .clear_lowest_bits(bits_of_later_columns);
        }
        column_bits.len() == self.key_column_count && entries_holding_a_value_prefix.is_empty()
    }
}

impl ViewChunk<'_> {
    /// Appends the value a byte-view column has in every row of `keyed_rows`
    /// to the row's key, as `bits` bits: the value's leading bytes, then its
    /// length. `chunks` are the column's chunks, and `descending_mask`
    /// inverts the value of a descending column.
    ///
    /// A value longer than [`INLINE_VIEW_LEN`] bytes is not whole in its
    /// key. Its length is written as [`LONGER_VALUE_LENGTH`], and its entry
    /// is added to `entries_holding_a_value_prefix` together with
    /// `bits_after_column`, the number of bits the later columns take in the
    /// key.
    fn append_values_to_keys<Key: SortKey>(
        chunks: &[Self],
        bits: u32,
        descending_mask: u128,
        bits_after_column: u32,
        keyed_rows: &mut [KeyedRow<Key>],
        entries_holding_a_value_prefix: &mut Vec<(usize, u32)>,
    ) {
        // The column's bits are 8 for each value byte a key holds and fewer
        // than 8 for the length, so the two come apart again here.
        let value_bytes = bits / 8;
        let length_bits = bits % 8;
        if value_bytes == INLINE_VIEW_LEN {
            // A key that holds all `INLINE_VIEW_LEN` bytes always has the
            // same layout. Passing it as constants compiles the loop a
            // second time with fixed shifts, which is measurably faster for
            // long values than shifting by amounts read at run time.
            Self::append_leading_bytes_to_keys(
                chunks,
                INLINE_VIEW_LEN,
                u32::BITS - LONGER_VALUE_LENGTH.leading_zeros(),
                descending_mask,
                bits_after_column,
                keyed_rows,
                entries_holding_a_value_prefix,
            )
        } else {
            Self::append_leading_bytes_to_keys(
                chunks,
                value_bytes,
                length_bits,
                descending_mask,
                bits_after_column,
                keyed_rows,
                entries_holding_a_value_prefix,
            )
        }
    }

    /// Appends to every key the first `value_bytes` bytes of its row's
    /// value, then the value's length in `length_bits` bits. The rest is as
    /// [`append_values_to_keys`](Self::append_values_to_keys) describes.
    #[inline(always)]
    fn append_leading_bytes_to_keys<Key: SortKey>(
        chunks: &[Self],
        value_bytes: u32,
        length_bits: u32,
        descending_mask: u128,
        bits_after_column: u32,
        keyed_rows: &mut [KeyedRow<Key>],
        entries_holding_a_value_prefix: &mut Vec<(usize, u32)>,
    ) {
        let bits = 8 * value_bytes + length_bits;
        let bits_of_bytes_left_out = 8 * (INLINE_VIEW_LEN - value_bytes);
        for (entry, keyed_row) in keyed_rows.iter_mut().enumerate() {
            let chunk = &chunks[keyed_row.chunk as usize];
            // SAFETY: the row is in bounds of its chunk's views.
            let view = unsafe { chunk.views.add(keyed_row.row as usize).read_unaligned() };
            let length = view as u32;
            let (bytes, written_length) = if length <= INLINE_VIEW_LEN {
                (Self::read_inline_bytes(view, length), length)
            } else {
                let first_bytes = chunk.read_out_of_line_bytes(view, INLINE_VIEW_LEN as usize);
                let mut number_bytes = [0; 16];
                number_bytes[4..].copy_from_slice(first_bytes);
                entries_holding_a_value_prefix.push((entry, bits_after_column));
                (u128::from_be_bytes(number_bytes), LONGER_VALUE_LENGTH)
            };
            let value =
                ((bytes >> bits_of_bytes_left_out) << length_bits) | u128::from(written_length);
            keyed_row
                .key
                .append_wide_column(bits, value ^ descending_mask);
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

/// One chunk's view column: the 16-byte views and the data buffers the
/// out-of-line values live in.
struct ViewChunk<'chunks> {
    views: *const u128,
    data_buffers: &'chunks [arrow_buffer::Buffer],
}

/// A byte-view value up to this length lives inside its own view.
const INLINE_VIEW_LEN: u32 = 12;

impl<'chunks> ViewChunk<'chunks> {
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

    /// The first `length` bytes of the value `view` describes. The value is
    /// longer than [`INLINE_VIEW_LEN`], so it lives in the data buffer its
    /// view names, at the offset the view gives.
    #[inline(always)]
    fn read_out_of_line_bytes(&self, view: u128, length: usize) -> &'chunks [u8] {
        let buffer = (view >> 64) as u32 as usize;
        let offset = (view >> 96) as u32 as usize;
        &self.data_buffers[buffer][offset..offset + length]
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
                    let array = column.as_string_view();
                    (array.views().as_ptr(), array.data_buffers())
                }
                DataType::BinaryView => {
                    let array = column.as_binary_view();
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
            chunk.read_out_of_line_bytes(view, length as usize)
        }
    }
}

/// Orders rows of a left run against rows of a right run by the sort key,
/// one key column at a time until a column differs (see the module docs).
pub(super) struct KeyOrdering<'chunks> {
    columns: Vec<KeyColumn<'chunks>>,
}

/// One key column of a [`KeyOrdering`], bound to every chunk of both runs.
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
    /// A string column that holds its values back to back in one buffer,
    /// with an offset per value, where a view column holds a view per value.
    OffsetStrings {
        left: Vec<&'chunks StringArray>,
        right: Vec<&'chunks StringArray>,
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

    /// The arrays of a string key column that holds offsets, per chunk.
    fn offset_string_chunks(
        key_column: usize,
        chunks: &[&'chunks RecordBatch],
    ) -> Vec<&'chunks StringArray> {
        chunks
            .iter()
            .map(|chunk| chunk.column(key_column).as_string())
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

impl<'chunks> KeyOrdering<'chunks> {
    /// Binds the key columns of `order_by` in the chunks of both runs. The
    /// chunks must all share the schema the key columns index into, and at
    /// least one side must have a chunk.
    ///
    /// The chunks may be owned batches or references to them, so a caller
    /// that orders batches it only borrows does not have to clone them.
    ///
    /// Fails when a key column is of a type that cannot be a sort key, such
    /// as a struct, a list or an interval.
    pub(super) fn try_new<Batch: Borrow<RecordBatch>>(
        order_by: &[OrderBy],
        left_chunks: &'chunks [Batch],
        right_chunks: &'chunks [Batch],
    ) -> Result<Self, ArrowError> {
        let left_chunks: Vec<&'chunks RecordBatch> =
            left_chunks.iter().map(Borrow::borrow).collect();
        let right_chunks: Vec<&'chunks RecordBatch> =
            right_chunks.iter().map(Borrow::borrow).collect();
        let (left_chunks, right_chunks) = (left_chunks.as_slice(), right_chunks.as_slice());
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
                        DataType::Utf8 => KeyColumnValues::OffsetStrings {
                            left: KeyColumn::offset_string_chunks(key_column, left_chunks),
                            right: KeyColumn::offset_string_chunks(key_column, right_chunks),
                        },
                        other => {
                            return Err(ArrowError::NotYetImplemented(format!(
                                "rows cannot be ordered by a column of type {other}; \
                                 cast it to a number, a string or another type with an order"
                            )));
                        }
                    }
                };
                let left_validity = KeyColumn::chunk_validities(key_column, left_chunks);
                let right_validity = KeyColumn::chunk_validities(key_column, right_chunks);
                let has_nulls = left_validity
                    .iter()
                    .chain(&right_validity)
                    .any(Option::is_some);
                Ok(KeyColumn {
                    values,
                    left_validity,
                    right_validity,
                    has_nulls,
                    descending: key.descending(),
                    nulls_first: key.nulls_first(),
                })
            })
            .collect::<Result<Vec<_>, ArrowError>>()?;
        Ok(Self { columns })
    }

    /// How the left run's row at `left` orders against the right run's row
    /// at `right`. `Equal` means either row may be emitted first.
    #[inline]
    pub(super) fn compare(&self, left: RunRow, right: RunRow) -> Ordering {
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
                KeyColumnValues::OffsetStrings {
                    left: left_chunks,
                    right: right_chunks,
                } => {
                    let left_string = left_chunks[left.chunk].value(left.row);
                    let right_string = right_chunks[right.chunk].value(right.row);
                    left_string.cmp(right_string)
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
