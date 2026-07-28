//! Accumulates selected rows of many batches into full-size batches.
//!
//! A selective operator that forwarded each input batch's survivors directly
//! would emit a runt batch per input batch, and every downstream operator
//! would pay its per-batch costs at that rate. [`BatchAccumulator`] appends
//! the survivors and emits a batch only once a full batch's worth of rows
//! accumulated (plus a tail flush at stream end).
//!
//! All output values land in slab buffers, the same ring-owned memory the
//! rest of the dataflow runs on, so emitting costs no general-heap round
//! trip; each slab returns to the pool when the downstream consumer drops the
//! batch. Fixed-width columns gather raw values, view columns gather the view
//! structs and share the underlying data buffers, and every other type
//! appends as zero-copy slices concatenated on emit.

use arrow::array::ArrayData;
use arrow::compute::kernels::concat::concat;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_schema::{ArrowError, DataType, SchemaRef};

use crate::RECORD_BATCH_SIZE;
use crate::arrays::slab_into_buffer;
use crate::memory::{SlabAllocator, SlabBuffer};

/// How many rows an accumulator can hold: an emit threshold of one full batch
/// plus one input batch that lands on top of an almost-full accumulator.
const ACCUMULATOR_CAPACITY: usize = 2 * RECORD_BATCH_SIZE;

/// Accumulates the selected rows of many batches and emits them as full-size
/// batches (see the module docs). Rows arrive via [`append`](Self::append) as
/// a batch plus ascending selected row positions; [`take_batch`](Self::take_batch)
/// hands out an emitted batch once [`should_emit`](Self::should_emit) says
/// enough rows accumulated (and once more at stream end for the tail).
pub struct BatchAccumulator {
    schema: SchemaRef,
    columns: Vec<ColumnAccumulator>,
    len: usize,
}

impl BatchAccumulator {
    pub fn new(schema: SchemaRef, allocator: &mut SlabAllocator) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|field| match field.data_type() {
                DataType::Utf8View | DataType::BinaryView => ColumnAccumulator::Views {
                    slab: allocator.create_slab_buffer(ACCUMULATOR_CAPACITY, false),
                    buffers: Vec::new(),
                    validity: ValidityMask::new(),
                    last_source: None,
                },
                dt => match dt.primitive_width() {
                    Some(width @ (1 | 2 | 4 | 8 | 16)) => ColumnAccumulator::Fixed {
                        width,
                        slab: allocator.create_slab_buffer(
                            (ACCUMULATOR_CAPACITY * width).div_ceil(size_of::<u128>()),
                            false,
                        ),
                        validity: ValidityMask::new(),
                    },
                    _ => ColumnAccumulator::General { parts: Vec::new() },
                },
            })
            .collect();
        Self {
            schema,
            columns,
            len: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The schema this accumulator coalesces into. A batch of a different schema
    /// cannot be appended (its columns would not concatenate), so a caller
    /// checks this and flushes first.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn should_emit(&self) -> bool {
        self.len >= RECORD_BATCH_SIZE
    }

    /// Append the rows of `batch` at `indices` (ascending row positions).
    /// The batch must match the accumulator's schema and the indices must be
    /// in bounds and at most [`RECORD_BATCH_SIZE`] long. A flat index gather
    /// rather than run ranges: at selective masks the runs are a row or two,
    /// and independent gather iterations let the CPU overlap the source
    /// cache misses.
    pub fn append(&mut self, batch: &RecordBatch, indices: &[u32]) {
        debug_assert!(self.len + indices.len() <= ACCUMULATOR_CAPACITY);
        for (accumulator, column) in self.columns.iter_mut().zip(batch.columns()) {
            accumulator.append(column, indices, self.len);
        }
        self.len += indices.len();
    }

    /// Emit the accumulated rows as one batch and reset.
    pub fn take_batch(&mut self, allocator: &mut SlabAllocator) -> Result<RecordBatch, ArrowError> {
        let len = self.len;
        let columns = self
            .schema
            .fields()
            .iter()
            .zip(self.columns.iter_mut())
            .map(|(field, accumulator)| accumulator.take_array(field.data_type(), len, allocator))
            .collect::<Result<Vec<_>, _>>()?;
        self.len = 0;
        // Row-count options so a column-less accumulation (a join side whose
        // columns all live on the other side) still emits its row count.
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(len));
        RecordBatch::try_new_with_options(self.schema.clone(), columns, &options)
    }
}

enum ColumnAccumulator {
    /// Raw values of a fixed-width column, appended by byte copy. The slab
    /// is u128-backed so the buffer start is aligned for any fixed-width
    /// value type Arrow reads through it.
    Fixed {
        width: usize,
        slab: SlabBuffer<u128>,
        validity: ValidityMask,
    },
    /// View structs of a view column. Appended views referencing data
    /// buffers are rebased onto the accumulated buffer list. Consecutive
    /// appends from the same source (a join draining against one build
    /// payload) reuse the source's base instead of re-adding its buffers.
    Views {
        slab: SlabBuffer<u128>,
        buffers: Vec<Buffer>,
        validity: ValidityMask,
        /// Where the previous append's source data buffers were rebased onto,
        /// as their (start, length) range in `buffers`.
        last_source: Option<(usize, usize)>,
    },
    /// Any other column type (booleans, offset strings, nested types,
    /// dictionaries, overly wide primitives): kept as zero-copy slices and
    /// concatenated on emit, so every type the engine produces is supported —
    /// the variants above are only fast paths.
    General { parts: Vec<ArrayRef> },
}

impl ColumnAccumulator {
    fn append(&mut self, column: &ArrayRef, indices: &[u32], at: usize) {
        match self {
            ColumnAccumulator::Fixed {
                width,
                slab,
                validity,
            } => {
                let width = *width;
                let data = column.to_data();
                validity.append(data.nulls(), indices, at);
                // SAFETY: the source holds `offset + len` values, the indices
                // are in-bounds rows, and the slab has capacity for `at` plus
                // the appended rows (checked by the caller).
                unsafe {
                    let src = data.buffers()[0].as_ptr().add(data.offset() * width);
                    let dst = (slab.ptr_at_index(0) as *mut u8).add(at * width);
                    match width {
                        1 => append_gather::<u8>(src, dst, indices),
                        2 => append_gather::<u16>(src, dst, indices),
                        4 => append_gather::<u32>(src, dst, indices),
                        8 => append_gather::<u64>(src, dst, indices),
                        16 => append_gather::<u128>(src, dst, indices),
                        _ => unreachable!("Fixed is only built for these widths"),
                    }
                }
            }
            ColumnAccumulator::Views {
                slab,
                buffers,
                validity,
                last_source,
            } => {
                // Borrow the views and data buffers through the concrete
                // array type: `to_data` would clone (and then drop) an Arc
                // per source data buffer on every append, which a source
                // reused across appends turns into an atomics storm.
                let (views, source_buffers, nulls) = match column.data_type() {
                    DataType::Utf8View => {
                        let array = column.as_string_view();
                        (array.views(), array.data_buffers(), array.nulls())
                    }
                    DataType::BinaryView => {
                        let array = column.as_binary_view();
                        (array.views(), array.data_buffers(), array.nulls())
                    }
                    _ => unreachable!("Views is only built for view types"),
                };
                validity.append(nulls, indices, at);
                let base = match *last_source {
                    Some((start, count))
                        if same_buffers(&buffers[start..start + count], source_buffers) =>
                    {
                        start as u128
                    }
                    _ => {
                        let start = buffers.len();
                        buffers.extend(source_buffers.iter().cloned());
                        *last_source = Some((start, source_buffers.len()));
                        start as u128
                    }
                };
                // SAFETY: as for Fixed; views are 16 bytes each and the
                // ScalarBuffer slice already accounts for the array offset.
                unsafe {
                    let src = views.as_ptr();
                    let dst = slab.ptr_at_index(at);
                    if base == 0 {
                        // Nothing accumulated references a data buffer yet,
                        // so incoming views keep their buffer indices and copy
                        // verbatim. All-inline columns stay on this path for
                        // every batch.
                        append_gather::<u128>(src as *const u8, dst as *mut u8, indices);
                    } else {
                        let mut dst = dst;
                        for &row in indices {
                            let mut view = src.add(row as usize).read_unaligned();
                            // A view longer than 12 bytes points into a data
                            // buffer; rebase its buffer index onto the
                            // accumulated buffer list.
                            if view as u32 > 12 {
                                view += base << 64;
                            }
                            dst.write(view);
                            dst = dst.add(1);
                        }
                    }
                }
            }
            ColumnAccumulator::General { parts } => {
                let mut run_start: Option<(usize, usize)> = None;
                for &row in indices {
                    let row = row as usize;
                    match run_start {
                        Some((start, end)) if row == end => run_start = Some((start, end + 1)),
                        Some((start, end)) => {
                            parts.push(column.slice(start, end - start));
                            run_start = Some((row, row + 1));
                        }
                        None => run_start = Some((row, row + 1)),
                    }
                }
                if let Some((start, end)) = run_start {
                    parts.push(column.slice(start, end - start));
                }
            }
        }
    }

    fn take_array(
        &mut self,
        data_type: &DataType,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        match self {
            ColumnAccumulator::Fixed {
                width,
                slab,
                validity,
            } => {
                let width = *width;
                let fresh = allocator.create_slab_buffer(
                    (ACCUMULATOR_CAPACITY * width).div_ceil(size_of::<u128>()),
                    false,
                );
                let slab = std::mem::replace(slab, fresh);
                let buffer = slab_into_buffer(slab, len * width);
                let out = ArrayData::builder(data_type.clone())
                    .len(len)
                    .add_buffer(buffer)
                    .nulls(validity.take(len));
                // SAFETY: a fixed-width array is a single values buffer plus
                // optional validity, and both were copied verbatim.
                Ok(make_array(unsafe { out.build_unchecked() }))
            }
            ColumnAccumulator::Views {
                slab,
                buffers,
                validity,
                last_source,
            } => {
                *last_source = None;
                let fresh = allocator.create_slab_buffer(ACCUMULATOR_CAPACITY, false);
                let slab = std::mem::replace(slab, fresh);
                let views = slab_into_buffer(slab, len * size_of::<u128>());
                let out = ArrayData::builder(data_type.clone())
                    .len(len)
                    .add_buffer(views)
                    .add_buffers(std::mem::take(buffers))
                    .nulls(validity.take(len));
                // SAFETY: every appended view was rebased onto the buffer
                // list emitted with it, so all view references stay valid;
                // views of null rows hold whatever bytes the source held,
                // which the validity marks as not a value.
                Ok(make_array(unsafe { out.build_unchecked() }))
            }
            ColumnAccumulator::General { parts } => {
                let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
                let out = concat(&refs)?;
                parts.clear();
                Ok(out)
            }
        }
    }
}

/// Validity bits for the accumulated rows of one column, one bit per row.
/// All-ones until a null actually arrives, so null-free streams never touch
/// it beyond one flag check per append.
struct ValidityMask {
    words: Vec<u64>,
    any_null: bool,
}

impl ValidityMask {
    fn new() -> Self {
        Self {
            words: vec![u64::MAX; ACCUMULATOR_CAPACITY.div_ceil(64)],
            any_null: false,
        }
    }

    /// Record the validity of the appended rows: `indices` rows of a column
    /// whose null buffer is `nulls`, landing at accumulated position `at`.
    fn append(&mut self, nulls: Option<&NullBuffer>, indices: &[u32], at: usize) {
        let Some(nulls) = nulls else {
            return;
        };
        for (offset, &row) in indices.iter().enumerate() {
            if !nulls.is_valid(row as usize) {
                let position = at + offset;
                self.words[position / 64] &= !(1 << (position % 64));
                self.any_null = true;
            }
        }
    }

    /// The accumulated rows' null buffer (`None` when every row is valid),
    /// resetting for the next batch.
    fn take(&mut self, len: usize) -> Option<NullBuffer> {
        if !self.any_null {
            return None;
        }
        let bits = BooleanBuffer::new(Buffer::from_slice_ref(&self.words), 0, len);
        self.words.fill(u64::MAX);
        self.any_null = false;
        Some(NullBuffer::new(bits))
    }
}

/// Whether `accumulated` holds exactly the buffers of `source`.
///
/// Comparing by address is sound because `accumulated` holds a clone of every
/// buffer it names: those clones keep the allocations alive, so an address
/// that still matches cannot have been freed and handed to a different
/// buffer in the meantime.
fn same_buffers(accumulated: &[Buffer], source: &[Buffer]) -> bool {
    accumulated.len() == source.len()
        && accumulated.iter().zip(source).all(|(held, incoming)| {
            held.as_ptr() == incoming.as_ptr() && held.len() == incoming.len()
        })
}

/// Gather `indices` rows from `src` to `dst`, as elements of `T`. A flat
/// loop with independent iterations, so the CPU overlaps the source cache
/// misses of many gathers.
///
/// # Safety
/// `src` must hold every indexed row, `dst` must have room for
/// `indices.len()` elements, and both must be valid for unaligned `T`
/// access.
unsafe fn append_gather<T: Copy>(src: *const u8, dst: *mut u8, indices: &[u32]) {
    let src = src as *const T;
    let dst = dst as *mut T;
    unsafe {
        for (out_idx, &row) in indices.iter().enumerate() {
            dst.add(out_idx)
                .write_unaligned(src.add(row as usize).read_unaligned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow::compute::{concat_batches, filter_record_batch};
    use arrow_array::{BooleanArray, Date32Array, Int64Array, StringViewArray};
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
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append(&batch, &indices_of(&mask));
        let all: Vec<u32> = (0..batch.num_rows() as u32).collect();
        accumulator.append(&batch, &all);
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
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append(&batch, &indices_of(&mask));
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
        let mut accumulator = BatchAccumulator::new(first.schema(), &mut allocator);

        accumulator.append(&first, &[0, 1]);
        accumulator.append(&second, &[0, 1]);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = concat_batches(&first.schema(), &[first, second]).unwrap();
        assert_eq!(emitted, expected);
    }

    #[test]
    fn resets_between_emits() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append(&batch, &[1, 2]);
        let first = accumulator.take_batch(&mut allocator).unwrap();
        accumulator.append(&batch, &[4]);
        let second = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(first, batch.slice(1, 2));
        assert_eq!(second, batch.slice(4, 1));
    }
}
