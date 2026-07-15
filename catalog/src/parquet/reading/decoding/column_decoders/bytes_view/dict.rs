//! [`ViewDict`] — a pre-parsed string dictionary for RLE-dictionary-encoded
//! columns.
//!
//! The dictionary page contains plain-encoded (length-prefixed) strings.
//! [`DictFactory`] walks those strings — handling cross-buffer boundaries —
//! and builds a `Vec<u128>` of Arrow views so that each RLE index can be
//! resolved to a view in O(1).

use crate::parquet::reading::decoding::column_decoders::Dict;
use crate::parquet::reading::decoding::column_decoders::bytes_view::views_builder::ViewsBuilder;
use arrow_array::builder::make_view;
use arrow_array::types::ByteViewType;
use arrow_array::{
    Array, ArrayRef, BooleanArray, GenericByteViewArray, RecordBatch, Scalar, StringViewArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, ScalarBuffer};
use bytes::Bytes;
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};
use std::marker::PhantomData;

/// Internal signal for cross-buffer boundary conditions during dictionary
/// parsing.
enum Error {
    /// The 4-byte length prefix straddles a buffer boundary.
    Len,
    /// The string body (of the given length) straddles a buffer boundary.
    String(u32),
}

/// Parses plain-encoded strings from scattered buffers into a [`ViewDict`].
pub(crate) struct DictFactory {
    data: Vec<Bytes>,
    buffers: Vec<Buffer>,
    position: ReaderPosition,
    views: Vec<u128>,
}

impl DictFactory {
    pub fn new(data: Vec<Bytes>, total_entries: usize) -> Self {
        Self {
            buffers: data.iter().map(|b| Buffer::from(b.clone())).collect(),
            data,
            position: Default::default(),
            views: Vec::with_capacity(total_entries),
        }
    }

    /// Fast path: decode entries entirely within the current buffer.
    ///
    /// Returns `Err(Error::Len)` if a length prefix is split, or
    /// `Err(Error::String(len))` if a string body is split.
    fn decode_entries_for_current_buffer(&mut self) -> Result<(), Error> {
        let buffer = &self.buffers[self.position.buffer_index];
        while self.position.offset < buffer.len() {
            if self.position.offset + 4 > buffer.len() {
                return Err(Error::Len);
            }

            let len_bytes: [u8; 4] = unsafe {
                buffer
                    .get_unchecked(self.position.offset..self.position.offset + 4)
                    .try_into()
                    .unwrap()
            };
            self.position.offset += 4;

            let len = u32::from_le_bytes(len_bytes);
            if self.position.offset + len as usize > buffer.len() {
                return Err(Error::String(len));
            }

            self.views.push(make_view(
                &buffer.as_ref()[self.position.offset..self.position.offset + len as usize],
                self.position.buffer_index as u32,
                self.position.offset as u32,
            ));
            self.position.offset += len as usize;
        }

        Ok(())
    }

    /// Slow path: reads a string that spans a buffer boundary, allocates a
    /// new contiguous block for it, and records the view.
    fn read_string_across_boundaries(&mut self, length: usize) {
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        let bytes = reader.read_bytes(length);
        self.views
            .push(make_view(bytes.as_ref(), self.buffers.len() as u32, 0));
        self.buffers.push(Buffer::from(bytes));
    }

    /// Consumes the factory, parsing all dictionary entries and returning the
    /// finished [`ViewDict`].
    pub fn create_dict<V: ByteViewType>(mut self) -> ViewDict<V> {
        while self.position.buffer_index < self.data.len() {
            match self.decode_entries_for_current_buffer() {
                Ok(_) => {
                    self.position.buffer_index += 1;
                    self.position.offset = 0;
                }
                Err(Error::Len) => {
                    let length = {
                        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
                        reader.read_u32_le()
                    };
                    self.read_string_across_boundaries(length as usize);
                }
                Err(Error::String(length)) => {
                    self.read_string_across_boundaries(length as usize);
                }
            }
        }

        ViewDict {
            data: self.buffers,
            views: self.views,
            phantom: PhantomData,
        }
    }
}

/// Pre-parsed byte-array dictionary for O(1) view lookups by RLE index.
///
/// `views[i]` is the 128-bit Arrow view for dictionary entry `i`. `data`
/// holds the backing blocks referenced by non-inline views. `V` names the
/// flavour of [`ViewsBuilder`] the entries feed into (string or binary).
pub struct ViewDict<V: ByteViewType> {
    /// Backing data blocks (original page buffers + any cross-boundary copies).
    data: Vec<Buffer>,
    /// One 128-bit view per dictionary entry.
    views: Vec<u128>,
    phantom: PhantomData<V>,
}

impl<V: ByteViewType> ViewDict<V> {
    #[inline(always)]
    pub fn view(&self, idx: usize) -> u128 {
        self.views[idx]
    }

    /// The dictionary's entries as an Arrow array, so entry bytes can be
    /// read through Arrow's own view decoding.
    fn as_arrow_array(&self) -> GenericByteViewArray<V> {
        GenericByteViewArray::<V>::new(
            ScalarBuffer::from(self.views.clone()),
            self.data.clone(),
            None,
        )
    }

    #[cfg(test)]
    fn get_str(&self, idx: usize) -> String {
        use dispatch::env::MAX_INLINE_STRING_VIEW;
        let view = self.views[idx];
        let bytes = view.to_le_bytes();
        let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        if len <= MAX_INLINE_STRING_VIEW {
            String::from_utf8(bytes[4..4 + len].to_vec()).unwrap()
        } else {
            let block_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            let offset = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
            String::from_utf8(self.data[block_id][offset..offset + len].to_vec()).unwrap()
        }
    }
}

impl<V: ByteViewType> Dict for ViewDict<V> {
    type Builder = ViewsBuilder<V>;
    type Item = u128;
    type EqConstant = Vec<u8>;

    /// The scalar matches when its array is a single, valid `StringViewArray`
    /// element; anything else yields `None`.
    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<Vec<u8>> {
        let (arr, _) = arrow_array::Datum::get(scalar);
        let strings = arr.as_any().downcast_ref::<StringViewArray>()?;
        (strings.len() == 1 && strings.is_valid(0)).then(|| strings.value(0).as_bytes().to_vec())
    }

    fn new(data: Vec<Bytes>, size: usize, _allocator: &mut SlabAllocator) -> Self {
        DictFactory::new(data, size).create_dict()
    }

    #[inline(always)]
    fn entry(&self, idx: usize) -> Self::Item {
        self.view(idx)
    }

    fn register_onto(&self, builder: &mut Self::Builder) {
        for (i, buffer) in self.data.iter().enumerate() {
            if builder.append_block(buffer.clone()) != i as u32 {
                panic!("Unexpected id");
            }
        }
    }

    fn filter_record_batch_by_const(
        &self,
        batch: RecordBatch,
        column: usize,
        needle: &Vec<u8>,
    ) -> RecordBatch {
        // Every row of a fully dictionary-encoded chunk copies its dictionary
        // entry's 128-bit view verbatim, so once we know which entry holds
        // `needle`, "row equals needle" becomes "row view equals that entry's
        // view" - an integer comparison instead of a string comparison.
        let entries = self.as_arrow_array();
        let needle_entry = GenericByteViewArray::<V>::new(
            ScalarBuffer::from(vec![make_view(needle, 0, 0)]),
            vec![Buffer::from(needle.as_slice())],
            None,
        );
        let matches = arrow_ord::cmp::eq(&entries, &Scalar::new(&needle_entry))
            .expect("the needle entry is built as the entries' own type");
        // A unique entry must hold the needle: with duplicates, matching rows
        // may carry either view and no single view decides the equality. When
        // no entry holds it, the query's `Filter` drops the rows instead.
        if matches.true_count() != 1 {
            return batch;
        }
        let matching_entry = matches
            .values()
            .set_indices()
            .next()
            .expect("true_count is one");
        let matching_view = self.views[matching_entry];

        let views = batch
            .column(column)
            .as_any()
            .downcast_ref::<GenericByteViewArray<V>>()
            .expect("the pushed constant only installs on this column's own type")
            .views();
        let keep = BooleanBuffer::collect_bool(views.len(), |i| views[i] == matching_view);
        if keep.count_set_bits() == keep.len() {
            return batch;
        }
        arrow_select::filter::filter_record_batch(&batch, &BooleanArray::new(keep, None))
            .expect("the mask is built to the batch's row count")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::StringViewType;
    use bytes::Bytes;

    fn encode_plain(strings: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();
        for s in strings {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        data
    }

    fn decode_entries(data: Vec<Bytes>, num_entries: usize) -> ViewDict<StringViewType> {
        DictFactory::new(data, num_entries).create_dict()
    }

    #[test]
    fn test_single_buffer() {
        let data = vec![Bytes::from(encode_plain(&["aa", "bb", "cc"]))];
        let dict = decode_entries(data, 3);

        assert_eq!(dict.views.len(), 3);
        assert_eq!(dict.get_str(0), "aa");
        assert_eq!(dict.get_str(1), "bb");
        assert_eq!(dict.get_str(2), "cc");
    }

    #[test]
    fn test_two_clean_buffers() {
        let data = vec![
            Bytes::from(encode_plain(&["aa"])),
            Bytes::from(encode_plain(&["bb"])),
        ];
        let dict = decode_entries(data, 2);

        assert_eq!(dict.views.len(), 2);
        assert_eq!(dict.get_str(0), "aa");
        assert_eq!(dict.get_str(1), "bb");
    }

    #[test]
    fn test_string_body_crosses_boundary() {
        let all = encode_plain(&["aa", "hello"]);
        let data = vec![
            Bytes::from(all[..10].to_vec()),
            Bytes::from(all[10..].to_vec()),
        ];
        let dict = decode_entries(data, 2);

        assert_eq!(dict.views.len(), 2);
        assert_eq!(dict.get_str(0), "aa");
        assert_eq!(dict.get_str(1), "hello");
    }

    #[test]
    fn test_length_prefix_crosses_boundary() {
        let all = encode_plain(&["aa", "bb"]);
        let data = vec![
            Bytes::from(all[..8].to_vec()),
            Bytes::from(all[8..].to_vec()),
        ];
        let dict = decode_entries(data, 2);

        assert_eq!(dict.views.len(), 2);
        assert_eq!(dict.get_str(0), "aa");
        assert_eq!(dict.get_str(1), "bb");
    }

    #[test]
    fn test_three_clean_buffers() {
        let data = vec![
            Bytes::from(encode_plain(&["aa"])),
            Bytes::from(encode_plain(&["bb"])),
            Bytes::from(encode_plain(&["cc"])),
        ];
        let dict = decode_entries(data, 3);

        assert_eq!(dict.views.len(), 3);
        assert_eq!(dict.get_str(0), "aa");
        assert_eq!(dict.get_str(1), "bb");
        assert_eq!(dict.get_str(2), "cc");
    }
}
