//! [`ViewDict`] — a pre-parsed string dictionary for RLE-dictionary-encoded
//! columns.
//!
//! The dictionary page contains plain-encoded (length-prefixed) strings.
//! [`DictFactory`] walks those strings — handling cross-buffer boundaries —
//! and builds a `Vec<u128>` of Arrow views so that each RLE index can be
//! resolved to a view in O(1).

use crate::reading::decoding::leaf_decoders::bytes_view::plain_page_decoder::{
    WALK_PREFETCH_BYTES, cached_page_views, prefetch_line, read_u32_le_at, view_at,
};
use crate::reading::decoding::leaf_decoders::{CachedPage, Dict, DictFromBytes, DictFromVecBytes};
use arrow_array::builder::make_view;
use arrow_array::types::ByteViewType;
use arrow_array::{
    Array, ArrayRef, BooleanArray, GenericByteViewArray, RecordBatch, Scalar, StringViewArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, ScalarBuffer};
use bytes::Bytes;
use dispatch::arrays::ViewBuilder;
use dispatch::env::MAX_INLINE_STRING_VIEW;
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
        let block_id = self.position.buffer_index as u32;
        let buffer: &[u8] = self.buffers[self.position.buffer_index].as_ref();
        let end = buffer.len();
        let mut offset = self.position.offset;
        while offset < end {
            if offset + 4 > end {
                self.position.offset = offset;
                return Err(Error::Len);
            }
            if offset + WALK_PREFETCH_BYTES < end {
                prefetch_line(unsafe { buffer.as_ptr().add(offset + WALK_PREFETCH_BYTES) });
            }
            // SAFETY: `offset + 4 <= end` was just checked.
            let len = unsafe { read_u32_le_at(buffer, offset) };
            let start = offset + 4;
            let stop = start + len as usize;
            if stop > end {
                self.position.offset = start;
                return Err(Error::String(len));
            }
            // SAFETY: the entry's bytes lie inside `buffer`.
            self.views
                .push(unsafe { view_at(buffer, start, len, block_id) });
            offset = stop;
        }
        self.position.offset = offset;

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
/// flavour of [`ViewBuilder`] the entries feed into (string or binary).
pub struct ViewDict<V: ByteViewType> {
    /// Backing data blocks (original page buffers + any cross-boundary copies).
    data: Vec<Buffer>,
    /// One 128-bit view per dictionary entry.
    views: Vec<u128>,
    phantom: PhantomData<fn() -> V>,
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
    type Builder = ViewBuilder<V>;
    type Item = u128;
    type EqConstant = Vec<u8>;

    /// The scalar matches when its array is a single, valid `StringViewArray`
    /// element; anything else yields `None`.
    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<Vec<u8>> {
        let (arr, _) = arrow_array::Datum::get(scalar);
        let strings = arr.as_any().downcast_ref::<StringViewArray>()?;
        (strings.len() == 1 && strings.is_valid(0)).then(|| strings.value(0).as_bytes().to_vec())
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    #[inline(always)]
    fn entry(&self, idx: usize) -> Self::Item {
        self.view(idx)
    }

    #[inline(always)]
    unsafe fn entry_unchecked(&self, idx: usize) -> Self::Item {
        unsafe { *self.views.get_unchecked(idx) }
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

// Views are `Vec<u128>`-backed either way (unlike a fixed-width dictionary,
// the view count doesn't shrink with the page's byte size, so one contiguous
// buffer doesn't bound them); the split only decides what `DictFactory` has
// to handle. A single buffer can't straddle, so entries parse on the fast
// path throughout.
impl<V: ByteViewType> DictFromBytes for ViewDict<V> {
    fn new_from_bytes(
        data: Bytes,
        size: usize,
        _allocator: &mut SlabAllocator,
        cached_page: Option<CachedPage>,
    ) -> Self {
        Self::build(vec![data], size, cached_page)
    }
}

impl<V: ByteViewType> DictFromVecBytes for ViewDict<V> {
    fn new_from_vec_bytes(
        data: Vec<Bytes>,
        size: usize,
        _allocator: &mut SlabAllocator,
        cached_page: Option<CachedPage>,
    ) -> Self {
        Self::build(data, size, cached_page)
    }
}

impl<V: ByteViewType> ViewDict<V> {
    /// The dictionary of `size` plain-encoded entries in `data`: from the
    /// entries' views cached beside the page when the page has them, taken
    /// the first time it is built, so a row group opened again copies views
    /// instead of walking every entry; from the entries otherwise.
    fn build(data: Vec<Bytes>, size: usize, cached_page: Option<CachedPage>) -> Self {
        let Some(mut views) =
            cached_page.and_then(|page| cached_page_views(&data, ReaderPosition::default(), &page))
        else {
            return DictFactory::new(data, size).create_dict();
        };
        let mut buffers: Vec<Buffer> = data.iter().map(|b| Buffer::from(b.clone())).collect();
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            let view = views.next_view();
            let len = view as u32;
            if len as usize <= MAX_INLINE_STRING_VIEW {
                entries.push(view);
                continue;
            }
            let (buffer, offset) = views.locate((view >> 96) as u32 as usize);
            if offset + len as usize <= data[buffer].len() {
                entries.push(
                    (view & u64::MAX as u128)
                        | ((buffer as u128) << 64)
                        | ((offset as u32 as u128) << 96),
                );
            } else {
                // An entry split between two buffers gets a buffer of its
                // own, as the walk gives it.
                let mut position = ReaderPosition {
                    buffer_index: buffer,
                    offset,
                };
                let bytes = MultiBufferReader::new(&data, &mut position).read_bytes(len as usize);
                entries.push(make_view(bytes.as_ref(), buffers.len() as u32, 0));
                buffers.push(Buffer::from(bytes));
            }
        }
        ViewDict {
            data: buffers,
            views: entries,
            phantom: PhantomData,
        }
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
    fn a_dictionary_built_again_from_cached_views_has_the_same_entries() {
        use dispatch::io::{LocalFile, OpenFile};
        use dispatch::memory::{BlockKey, init_test_free_pool};
        init_test_free_pool(8);
        let all = encode_plain(&["aa", "a value long enough to need a buffer", "bb", "cc"]);
        let data = vec![
            Bytes::from(all[..30].to_vec()),
            Bytes::from(all[30..].to_vec()),
        ];
        let file = LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
        let page = CachedPage {
            key: BlockKey {
                open_file: OpenFile::Local(file),
                offset: 4096,
                len: all.len(),
            },
            values: 4,
        };
        let mut allocator = SlabAllocator::new(true);
        let first = ViewDict::<StringViewType>::new_from_vec_bytes(
            data.clone(),
            4,
            &mut allocator,
            Some(page.clone()),
        );

        let again =
            ViewDict::<StringViewType>::new_from_vec_bytes(data, 4, &mut allocator, Some(page));

        let entries = |dict: &ViewDict<StringViewType>| {
            (0..4)
                .map(|i| dict.get_str(i).to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(entries(&again), entries(&first));
        assert_eq!(
            entries(&again),
            ["aa", "a value long enough to need a buffer", "bb", "cc"]
        );
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
