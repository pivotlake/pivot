//! Takes selected rows of one column into a slab-backed array.

use arrow::array::ArrayData;
use arrow::compute::take as arrow_take;
use arrow_array::{Array, ArrayRef, UInt32Array, make_array};
use arrow_buffer::Buffer;
use arrow_schema::{ArrowError, DataType};

use crate::arrays::slab_into_buffer;
use crate::memory::SlabAllocator;

/// Take the rows of `column` at `indices` (ascending) into a new array.
///
/// Null-free fixed-width columns gather raw values into a slab buffer, the
/// same ring-owned memory the rest of the dataflow runs on; view columns
/// gather the view structs and share the underlying data buffers. Anything
/// else (columns with nulls, dictionaries, nested types) goes through Arrow's
/// take kernel instead.
pub fn take(
    allocator: &mut SlabAllocator,
    column: &ArrayRef,
    indices: &[u32],
) -> Result<ArrayRef, ArrowError> {
    let data = column.to_data();
    if data.null_count() > 0 {
        return take_via_arrow(column, indices);
    }
    match data.data_type() {
        DataType::Utf8View | DataType::BinaryView => {
            let views = gather_indices::<u128>(allocator, &data, indices);
            let out = ArrayData::builder(data.data_type().clone())
                .len(indices.len())
                .add_buffer(views)
                .add_buffers(data.buffers()[1..].to_vec());
            // SAFETY: the views are copied verbatim and reference the same
            // data buffers as the source array, so every view stays valid.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
        dt => {
            let values = match dt.primitive_width() {
                Some(1) => gather_indices::<u8>(allocator, &data, indices),
                Some(2) => gather_indices::<u16>(allocator, &data, indices),
                Some(4) => gather_indices::<u32>(allocator, &data, indices),
                Some(8) => gather_indices::<u64>(allocator, &data, indices),
                Some(16) => gather_indices::<u128>(allocator, &data, indices),
                _ => return take_via_arrow(column, indices),
            };
            let out = ArrayData::builder(dt.clone())
                .len(indices.len())
                .add_buffer(values);
            // SAFETY: a null-free fixed-width array is a single values
            // buffer, and the values are copied verbatim.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
    }
}

fn gather_indices<T: Copy>(
    allocator: &mut SlabAllocator,
    data: &ArrayData,
    indices: &[u32],
) -> Buffer {
    let slab = allocator.create_slab_buffer::<T>(indices.len(), false);
    // SAFETY: buffer 0 holds `offset + len` values of `T`, every index is a
    // valid row of the source array, and the slab holds `indices.len()`
    // slots. Unaligned reads tolerate sliced offsets.
    unsafe {
        let src = (data.buffers()[0].as_ptr() as *const T).add(data.offset());
        let dst = slab.ptr_at_index(0);
        for (out_idx, &row) in indices.iter().enumerate() {
            dst.add(out_idx)
                .write(src.add(row as usize).read_unaligned());
        }
    }
    slab_into_buffer(slab, indices.len() * size_of::<T>())
}

fn take_via_arrow(column: &ArrayRef, indices: &[u32]) -> Result<ArrayRef, ArrowError> {
    arrow_take(column, &UInt32Array::from(indices.to_vec()), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow::compute::filter_record_batch;
    use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn take_matches_arrow_filter() {
        init_test_free_pool(4);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8View, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringViewArray::from(vec![
                    "aa",
                    "a longer string that spills out of the view",
                    "cc",
                    "dd",
                    "ee",
                ])),
            ],
        )
        .unwrap();
        let mask = BooleanArray::from(vec![true, false, true, false, true]);
        let mut allocator = SlabAllocator::new(false);

        let taken = take(&mut allocator, batch.column(1), &[0, 2, 4]).unwrap();

        let expected = filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(&taken, expected.column(1));
    }
}
