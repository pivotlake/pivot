//! Accumulates rows of many batches into full-size batches.
//!
//! Two callers share one mechanism. A selective operator that forwarded each
//! input batch's survivors directly would emit a runt batch per input batch, and
//! every downstream operator would pay its per-batch costs at that rate. The
//! write path has the mirror-image problem, reshaping `RECORD_BATCH_SIZE` inputs
//! into row groups of a hundred thousand rows. Both append rows as they arrive
//! and take a batch out once enough of them accumulated.
//!
//! A column's values land in slab buffers wherever its type allows, which is the
//! same ring-owned memory the rest of the dataflow runs on, so emitting costs no
//! general-heap round trip and each slab returns to the pool when the downstream
//! consumer drops the batch. How that is done depends on the type, so each
//! strategy is a [`ColumnAccumulator`] of its own: [`fixed_width`] copies raw
//! values, [`view`] gathers the view structs, and [`structs`] accumulates a
//! struct's children. A type none of them covers falls to [`concatenated`],
//! which holds slices of the input and concatenates them through Arrow's own
//! allocator, so those columns are the exception to both the slab memory above
//! and the ownership below.
//!
//! Two things are decided at construction, because they are what the two callers
//! differ on:
//!
//! - **Capacity.** A slab is [`BUFFER_SIZE`] and every buffer handed to Arrow is
//!   exactly one of them, so an accumulator holds at most what a slab holds for
//!   its widest column ([`calculate_row_capacity`]). A caller asking for more gets that
//!   instead, which for the write path is where a row group is cut.
//! - **[`ValueStorage`]**, which decides whether the emitted batch keeps its
//!   inputs alive. A coalescing operator's batch is consumed immediately
//!   downstream, so its view columns keep pointing at the source batches' data
//!   buffers and nothing is copied. An accumulation held for a long time (a row
//!   group waiting for its file to finish encoding) must not pin those buffers,
//!   so it copies the values into blocks of its own instead.

mod column;
mod concatenated;
mod fixed_width;
mod structs;
mod validity;
mod view;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, Fields, SchemaRef};

use crate::RECORD_BATCH_SIZE;
use crate::memory::{BUFFER_SIZE, SlabAllocator};
pub use column::ChunkedColumn;
use column::ColumnAccumulator;
use concatenated::ConcatenatedColumn;
use fixed_width::FixedWidthColumn;
use structs::StructColumn;
use view::ViewColumn;

/// How many rows a coalescing accumulator holds. One full batch to emit, plus
/// room for an input batch that lands on top of an almost-full accumulator.
const COALESCING_CAPACITY: usize = 2 * RECORD_BATCH_SIZE;

/// Whether an accumulation keeps the batches its rows came from alive.
///
/// This only decides anything for a column whose values live outside its own
/// array, which today means the byte-view columns. A fixed-width value is copied
/// either way, and a column on the [`concatenated`] path holds slices of its
/// input under both settings, so a schema containing one keeps those batches
/// alive whatever is asked for here.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ValueStorage {
    /// Reference the source's data buffers rather than copying their bytes. The
    /// emitted batch keeps every source batch it drew a value from alive, which
    /// is free and harmless when that batch is consumed straight away.
    RetainSourceBuffers,
    /// Copy each value into the accumulator's own blocks, so the emitted batch
    /// holds nothing of its inputs and they can be dropped as soon as they have
    /// been appended. The blocks are slab memory, except for a single value too
    /// large for any slab (see [`view`]).
    CopyValues,
}

/// Accumulates rows of many batches and hands them out as one batch (see the
/// module docs). Rows arrive via
/// [`append_batch_by_indices`](Self::append_batch_by_indices),
/// [`append_from_batches`](Self::append_from_batches), or
/// [`append_range`](Self::append_range); [`take_batch`](Self::take_batch) hands
/// out the accumulated rows and resets.
pub struct BatchAccumulator {
    schema: SchemaRef,
    columns: Vec<Box<dyn ColumnAccumulator>>,
    capacity: usize,
    len: usize,
}

impl BatchAccumulator {
    /// An accumulator for an operator whose output is consumed straight away:
    /// [`COALESCING_CAPACITY`] rows, whose view columns reference the source
    /// batches' data buffers rather than copying their values
    /// ([`ValueStorage::RetainSourceBuffers`]).
    pub fn retaining_source_buffers(schema: SchemaRef, allocator: &mut SlabAllocator) -> Self {
        Self::new(
            schema,
            COALESCING_CAPACITY,
            ValueStorage::RetainSourceBuffers,
            allocator,
        )
    }

    /// An accumulator for rows that outlive the batches they came from, such as
    /// a Parquet row group held until its file finishes encoding. Its values are
    /// copied ([`ValueStorage::CopyValues`]), so the batch it hands out holds no
    /// reference to its inputs and they can be dropped as they are appended. A
    /// column on the [`concatenated`] path is the exception, and keeps its
    /// inputs alive until the batch is taken.
    ///
    /// It holds `rows` rows, or as many as a slab holds for the widest column
    /// where that is fewer (see [`capacity`](Self::capacity)).
    pub fn copying_values(schema: SchemaRef, rows: usize, allocator: &mut SlabAllocator) -> Self {
        Self::new(schema, rows, ValueStorage::CopyValues, allocator)
    }

    fn new(
        schema: SchemaRef,
        rows: usize,
        storage: ValueStorage,
        allocator: &mut SlabAllocator,
    ) -> Self {
        let capacity = rows.max(1).min(calculate_row_capacity(schema.fields()));
        let columns = schema
            .fields()
            .iter()
            .map(|field| create_column_accumulator(field.data_type(), capacity, storage, allocator))
            .collect();
        Self {
            schema,
            columns,
            capacity,
            len: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rows accumulated so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Rows this accumulator can hold before it must be taken. Not always what
    /// the caller asked for, see [`copying_values`](Self::copying_values).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The schema this accumulator coalesces into. A batch of a different schema
    /// cannot be appended (its columns would not concatenate), so a caller
    /// checks this and flushes first.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Whether a full batch has accumulated, which is what a coalescing caller
    /// emits on. Not the same as being at [`capacity`](Self::capacity): the
    /// accumulator has room past this point, which is what lets a whole input
    /// batch land on an almost-full one.
    pub fn has_full_batch(&self) -> bool {
        self.len >= RECORD_BATCH_SIZE
    }

    /// Append the rows of `batch` at `indices`, which must be ascending row
    /// positions, in bounds, and no more than the remaining
    /// [`capacity`](Self::capacity). The batch must match the accumulator's
    /// schema.
    ///
    /// Each column copies one row per index instead of detecting and copying
    /// consecutive runs. Selective filters usually produce runs of only one or
    /// two rows, so detecting runs adds branches without reducing copying, while
    /// independent loads let the CPU overlap source cache misses.
    ///
    /// `allocator` is only drawn on by an accumulator that copies values.
    pub fn append_batch_by_indices(
        &mut self,
        batch: &RecordBatch,
        indices: &[u32],
        allocator: &mut SlabAllocator,
    ) {
        debug_assert!(self.len + indices.len() <= self.capacity);
        for (accumulator, column) in self.columns.iter_mut().zip(batch.columns()) {
            accumulator.append_from_indices(column, indices, self.len, allocator);
        }
        self.len += indices.len();
    }

    /// Append rows stored across many batches at the encoded `ids`
    /// (`batch << shift | row`), which must be no more than the remaining
    /// [`capacity`](Self::capacity). `columns` holds one prepared
    /// [`ChunkedColumn`] per accumulator column; their schema must match the
    /// accumulator's.
    pub fn append_from_batches(
        &mut self,
        columns: &[ChunkedColumn],
        shift: u32,
        ids: &[u32],
        allocator: &mut SlabAllocator,
    ) {
        debug_assert!(self.len + ids.len() <= self.capacity);
        for (accumulator, column) in self.columns.iter_mut().zip(columns) {
            accumulator.append_from_batches(column, ids, shift, self.len, allocator);
        }
        self.len += ids.len();
    }

    /// Append `len` consecutive rows of `batch` from `start`, which must fit in
    /// the remaining [`capacity`](Self::capacity). This is the bulk form of
    /// [`append_batch_by_indices`](Self::append_batch_by_indices), for a caller taking whole
    /// batches. The rows are contiguous in the source as well as the
    /// destination, so each column copies in one run instead of a value at a
    /// time.
    pub fn append_range(
        &mut self,
        batch: &RecordBatch,
        start: usize,
        len: usize,
        allocator: &mut SlabAllocator,
    ) {
        debug_assert!(self.len + len <= self.capacity);
        for (accumulator, column) in self.columns.iter_mut().zip(batch.columns()) {
            accumulator.append_from_range(column, start, len, self.len, allocator);
        }
        self.len += len;
    }

    /// Emit the accumulated rows as one batch and reset.
    pub fn take_batch(&mut self, allocator: &mut SlabAllocator) -> Result<RecordBatch, ArrowError> {
        let len = self.len;
        let columns = self
            .columns
            .iter_mut()
            .map(|accumulator| accumulator.take_array(len, allocator))
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        self.len = 0;
        // Row-count options so a column-less accumulation (a join side whose
        // columns all live on the other side) still emits its row count.
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(len));
        RecordBatch::try_new_with_options(self.schema.clone(), columns, &options)
    }
}

/// The accumulator a column of this type is built with: raw values where the
/// type has a fixed width, views where it is a byte-view column, children where
/// it is a struct, and slices to concatenate for anything else.
fn create_column_accumulator(
    data_type: &DataType,
    capacity: usize,
    storage: ValueStorage,
    allocator: &mut SlabAllocator,
) -> Box<dyn ColumnAccumulator> {
    match data_type {
        DataType::Utf8View | DataType::BinaryView => {
            Box::new(ViewColumn::new(data_type, capacity, storage, allocator))
        }
        DataType::Struct(fields) => {
            Box::new(StructColumn::new(fields, capacity, storage, allocator))
        }
        other => match other.primitive_width() {
            Some(width @ (1 | 2 | 4 | 8 | 16)) => {
                Box::new(FixedWidthColumn::new(data_type, width, capacity, allocator))
            }
            _ => Box::new(ConcatenatedColumn::new()),
        },
    }
}

/// Rows of `fields` a single slab holds, which is what bounds an accumulator:
/// every buffer handed to Arrow is one slab, and a column's values buffer is
/// `rows * width`. A struct's bound is its narrowest child's, a view column's is
/// set by its 16-byte views rather than by the bytes they point at, and a column
/// with no slab behind it (the [`ConcatenatedColumn`] path) bounds nothing.
fn calculate_row_capacity(fields: &Fields) -> usize {
    fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Utf8View | DataType::BinaryView => BUFFER_SIZE / size_of::<u128>(),
            DataType::Struct(children) => calculate_row_capacity(children),
            other => match other.primitive_width() {
                Some(width @ (1 | 2 | 4 | 8 | 16)) => BUFFER_SIZE / width,
                _ => usize::MAX,
            },
        })
        .min()
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow::compute::{concat_batches, filter_record_batch};
    use arrow_array::cast::AsArray;
    use arrow_array::{
        BinaryViewArray, BooleanArray, Date32Array, Int64Array, StringViewArray, StructArray,
    };
    use arrow_buffer::{Buffer, NullBuffer};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn three_column_batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8View, false),
                Field::new("day", DataType::Date32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringViewArray::from(vec![
                    "aa",
                    "a longer string that spills out of the view",
                    "cc",
                    "dd",
                    "a second long string kept by the second run",
                ])),
                Arc::new(Date32Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap()
    }

    fn indices_of(mask: &BooleanArray) -> Vec<u32> {
        mask.values().set_indices_u32().collect()
    }

    #[test]
    fn coalesces_batches() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mask = BooleanArray::from(vec![false, true, true, false, true]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::retaining_source_buffers(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &indices_of(&mask), &mut allocator);
        let all: Vec<u32> = (0..batch.num_rows() as u32).collect();
        accumulator.append_batch_by_indices(&batch, &all, &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = concat_batches(
            &batch.schema(),
            &[filter_record_batch(&batch, &mask).unwrap(), batch.clone()],
        )
        .unwrap();
        assert_eq!(emitted, expected);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn keeps_null_rows() {
        init_test_free_pool(4);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("name", DataType::Utf8View, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    None,
                    Some(3),
                    None,
                    Some(5),
                ])),
                Arc::new(StringViewArray::from(vec![
                    None,
                    Some("a longer string that spills out of the view"),
                    Some("cc"),
                    None,
                    Some("ee"),
                ])),
            ],
        )
        .unwrap();
        let mask = BooleanArray::from(vec![true, true, false, true, true]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::retaining_source_buffers(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &indices_of(&mask), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(emitted, expected);
    }

    /// A `Utf8View` batch whose rows are `(buffer index, value)` pairs, each
    /// value living at offset 0 of the data buffer it names.
    fn view_batch(buffers: Vec<Buffer>, rows: &[(u32, &str)]) -> RecordBatch {
        let views: Vec<u128> = rows
            .iter()
            .map(|(buffer_index, value)| {
                let prefix = u32::from_le_bytes(value.as_bytes()[..4].try_into().unwrap());
                // A view of a value longer than 12 bytes: length, the first
                // four bytes, the data buffer it points into, and last the
                // offset within that buffer, which is 0 for every value here.
                (value.len() as u128) | (prefix as u128) << 32 | (*buffer_index as u128) << 64
            })
            .collect();
        let column = StringViewArray::try_new(views.into(), buffers, None).unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "name",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(column)],
        )
        .unwrap()
    }

    /// Two batches can hold the same first data buffer and the same buffer
    /// count while differing in the rest, so a source is only the one already
    /// accumulated when every one of its buffers matches.
    #[test]
    fn appends_sources_that_share_a_first_data_buffer() {
        init_test_free_pool(4);
        let shared = Buffer::from_slice_ref("a value both sources hold".as_bytes());
        let first = view_batch(
            vec![
                shared.clone(),
                Buffer::from_slice_ref("a value only the first source holds".as_bytes()),
            ],
            &[
                (0, "a value both sources hold"),
                (1, "a value only the first source holds"),
            ],
        );
        let second = view_batch(
            vec![
                shared.clone(),
                Buffer::from_slice_ref("a value only the second source holds".as_bytes()),
            ],
            &[
                (0, "a value both sources hold"),
                (1, "a value only the second source holds"),
            ],
        );
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::retaining_source_buffers(first.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&first, &[0, 1], &mut allocator);
        accumulator.append_batch_by_indices(&second, &[0, 1], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = concat_batches(&first.schema(), &[first, second]).unwrap();
        assert_eq!(emitted, expected);
    }

    #[test]
    fn resets_between_emits() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::retaining_source_buffers(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &[1, 2], &mut allocator);
        let first = accumulator.take_batch(&mut allocator).unwrap();
        accumulator.append_batch_by_indices(&batch, &[4], &mut allocator);
        let second = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(first, batch.slice(1, 2));
        assert_eq!(second, batch.slice(4, 1));
    }

    /// A contiguous append takes the same rows as the equivalent selection, so
    /// the bulk path and the gather agree.
    #[test]
    fn appending_a_range_matches_appending_its_positions() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mut allocator = SlabAllocator::new(false);
        let mut ranged = BatchAccumulator::retaining_source_buffers(batch.schema(), &mut allocator);
        let mut selected =
            BatchAccumulator::retaining_source_buffers(batch.schema(), &mut allocator);

        ranged.append_range(&batch, 1, 3, &mut allocator);
        selected.append_batch_by_indices(&batch, &[1, 2, 3], &mut allocator);

        assert_eq!(
            ranged.take_batch(&mut allocator).unwrap(),
            selected.take_batch(&mut allocator).unwrap()
        );
    }

    /// A copying accumulator holds the same rows without referencing the
    /// source's data buffers at all, which is what lets its inputs be dropped.
    #[test]
    fn a_copying_accumulator_keeps_none_of_the_sources_buffers() {
        init_test_free_pool(8);
        let batch = three_column_batch();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::copying_values(batch.schema(), RECORD_BATCH_SIZE, &mut allocator);

        accumulator.append_range(&batch, 0, batch.num_rows(), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch);
        let source: Vec<*const u8> = batch
            .column(1)
            .as_string_view()
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_ptr())
            .collect();
        let held: Vec<*const u8> = emitted
            .column(1)
            .as_string_view()
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_ptr())
            .collect();
        assert!(
            held.iter().all(|buffer| !source.contains(buffer)),
            "a copying accumulator must not hold a source data buffer"
        );
    }

    /// A value larger than a slab cannot be stored in one, so it gets a buffer
    /// of its own rather than a reference to the source's.
    #[test]
    fn a_copying_accumulator_takes_a_value_larger_than_a_slab() {
        init_test_free_pool(8);
        let value = "v".repeat(BUFFER_SIZE + 1024);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "name",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(StringViewArray::from(vec![value.as_str()]))],
        )
        .unwrap();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::copying_values(batch.schema(), RECORD_BATCH_SIZE, &mut allocator);

        accumulator.append_range(&batch, 0, 1, &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted.column(0).as_string_view().value(0), value);
    }

    /// A struct column accumulates through its children rather than falling to
    /// the concatenating path, nulls on the struct itself included. This is the
    /// shape of a variant column.
    #[test]
    fn a_struct_column_accumulates_through_its_children() {
        init_test_free_pool(8);
        let fields = Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]);
        let column = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(BinaryViewArray::from(vec![
                    b"m0".as_slice(),
                    b"m1".as_slice(),
                    b"m2".as_slice(),
                ])),
                Arc::new(BinaryViewArray::from(vec![
                    Some(b"a document body well past twelve bytes".as_slice()),
                    None,
                    Some(b"v2".as_slice()),
                ])),
            ],
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "doc",
                DataType::Struct(fields),
                true,
            )])),
            vec![Arc::new(column)],
        )
        .unwrap();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator =
            BatchAccumulator::copying_values(batch.schema(), RECORD_BATCH_SIZE, &mut allocator);

        accumulator.append_range(&batch, 0, 3, &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch);
    }

    /// The capacity is what a slab holds for the widest column, however many
    /// rows the caller asks for: every buffer handed to Arrow is one slab.
    #[test]
    fn capacity_is_capped_at_what_one_slab_holds() {
        init_test_free_pool(8);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8View, false),
        ]));
        let mut allocator = SlabAllocator::new(false);

        let accumulator = BatchAccumulator::copying_values(schema, usize::MAX, &mut allocator);

        // The views (16 bytes a row) are the widest of the two columns.
        assert_eq!(accumulator.capacity(), BUFFER_SIZE / size_of::<u128>());
    }
}
