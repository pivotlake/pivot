//! The decoder of a leaf the file has no chunk for: a declared column the file
//! predates. It reads nothing and emits a NULL per row.

use super::{BuiltDictionary, LeafDecoder, Result, SharedDictionary};
use crate::thrift::headers::DictionaryPageHeader;
use crate::types::page::DecompressedPage;
use arrow_array::{ArrayRef, Scalar, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use bytes::Bytes;
use dispatch::arrays::{SlabColumn, ValidityBuilder};
use dispatch::memory::SlabAllocator;

/// Emits an all-NULL array of `data_type` for whatever row count is asked of
/// it, carved from the batch's slabs like every other decoder's output. The
/// one stand-in page the indexer produces for an absent column arrives through
/// [`insert_page`](LeafDecoder::insert_page) and is discarded: the decoder's
/// supply of rows is unbounded, so it never holds a batch back.
pub struct AbsentLeafDecoder {
    data_type: DataType,
}

impl AbsentLeafDecoder {
    pub fn new(data_type: DataType) -> Self {
        Self { data_type }
    }
}

impl LeafDecoder for AbsentLeafDecoder {
    fn available(&self) -> usize {
        usize::MAX
    }

    fn insert_page(&mut self, _page: DecompressedPage) {}

    fn build_dictionary(
        &self,
        _header: DictionaryPageHeader,
        _data: Vec<Bytes>,
        _allocator: &mut SlabAllocator,
    ) -> BuiltDictionary {
        unreachable!("an absent leaf has no dictionary page")
    }

    fn adopt_dictionary(&mut self, _dictionary: SharedDictionary) {
        unreachable!("an absent leaf has no dictionary page")
    }

    fn restart_at_page(&mut self, _page_idx: usize) {}

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        Ok(null_array_on_slabs(allocator, &self.data_type, size))
    }

    /// The column holds no value, so no constant can ever match; the query's
    /// own `Filter` rejects every row, which needs no help from here.
    fn set_eq_constant(&mut self, _value: &Scalar<ArrayRef>) {}
}

/// An all-NULL array of `data_type` with `len` rows whose memory comes from
/// `allocator`'s slabs. Nothing but the validity bitmap is ever read from an
/// all-NULL array, and every layout accepts zeroed buffers of the right
/// lengths (a zero offset is an empty string, a zero view an empty inline
/// one), so the array is a zeroed bitmap plus zeroed buffers of the layout's
/// shape.
fn null_array_on_slabs(
    allocator: &mut SlabAllocator,
    data_type: &DataType,
    len: usize,
) -> ArrayRef {
    let mut validity = ValidityBuilder::with_capacity(allocator, len);
    validity.append_n(len, false);
    let nulls = NullBuffer::new(BooleanBuffer::new(validity.into_buffer(), 0, len));
    let buffers = match data_type {
        DataType::Boolean => vec![zeroed_buffer(allocator, len.div_ceil(8))],
        DataType::Utf8 | DataType::Binary => {
            vec![
                zeroed_buffer(allocator, (len + 1) * 4),
                Buffer::from(Vec::<u8>::new()),
            ]
        }
        DataType::LargeUtf8 | DataType::LargeBinary => {
            vec![
                zeroed_buffer(allocator, (len + 1) * 8),
                Buffer::from(Vec::<u8>::new()),
            ]
        }
        DataType::Utf8View | DataType::BinaryView => vec![zeroed_buffer(allocator, len * 16)],
        fixed_width => {
            // A nested column is decoded leaf by leaf, and every leaf type is
            // fixed-width, boolean or bytes.
            let width = fixed_width
                .primitive_width()
                .unwrap_or_else(|| panic!("no all-NULL layout for {fixed_width}"));
            vec![zeroed_buffer(allocator, len * width)]
        }
    };
    let data = ArrayData::builder(data_type.clone())
        .len(len)
        .nulls(Some(nulls))
        .buffers(buffers)
        .build()
        .expect("zeroed buffers of the layout's lengths form a valid array");
    make_array(data)
}

/// `bytes` zero bytes on a slab, aligned for any fixed-width value.
fn zeroed_buffer(allocator: &mut SlabAllocator, bytes: usize) -> Buffer {
    let mut column = SlabColumn::<u128>::with_capacity(allocator, bytes.div_ceil(16));
    column.spare_mut(bytes.div_ceil(16)).fill(0);
    column.into_buffer().slice_with_length(0, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_schema::TimeUnit;
    use dispatch::memory::init_test_free_pool;

    fn read_nulls(allocator: &mut SlabAllocator, data_type: DataType, len: usize) -> ArrayRef {
        AbsentLeafDecoder::new(data_type)
            .read(allocator, len)
            .unwrap()
    }

    #[test]
    fn every_layout_reads_as_all_null_rows() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let types = [
            DataType::Boolean,
            DataType::Int32,
            DataType::Int64,
            DataType::Float64,
            DataType::Date32,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Decimal128(38, 0),
            DataType::Utf8,
            DataType::LargeBinary,
            DataType::Utf8View,
            DataType::BinaryView,
        ];

        for data_type in types {
            let array = read_nulls(&mut allocator, data_type.clone(), 37);

            assert_eq!(array.data_type(), &data_type);
            assert_eq!(array.len(), 37);
            assert_eq!(array.null_count(), 37, "{data_type}");
        }
    }

    #[test]
    fn an_empty_batch_is_an_empty_array() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);

        let array = read_nulls(&mut allocator, DataType::Utf8View, 0);

        assert_eq!(array.len(), 0);
    }
}
