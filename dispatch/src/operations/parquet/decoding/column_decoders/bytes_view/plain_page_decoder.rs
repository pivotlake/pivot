//! [`PlainPageDecoder`] — reads length-prefixed byte arrays from plain-encoded
//! Parquet pages into a [`ViewsBuilder`].
//!
//! Each value on disk is a 4-byte little-endian length followed by that many
//! bytes of payload. The decoder has a fast path that stays within a single
//! underlying buffer (zero-copy view creation) and two fallback paths for when
//! the length prefix or the string body straddles a buffer boundary.

use super::super::ArrayBuilder;
use crate::env::MAX_INLINE_STRING_VIEW;
use crate::memory::{MultiBufferReader, ReaderPosition};
use crate::operations::parquet::decoding::column_decoders::DecodePlain;
use crate::operations::parquet::decoding::column_decoders::bytes_view::views_builder::ViewsBuilder;
use arrow_array::builder::make_view;
use arrow_buffer::Buffer;
use bytes::Bytes;
use thiserror::Error;

/// Internal error signalling that the current buffer was exhausted mid-value.
#[derive(Debug, Error)]
pub enum Error {
    /// The 4-byte length prefix straddles a buffer boundary.
    #[error("")]
    Len,
    /// The string body (of the given length) straddles a buffer boundary.
    #[error("")]
    ReadStr(u32),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Reads plain-encoded (length-prefixed) byte arrays from scattered buffers,
/// producing string views.
///
/// Holds both the raw `Bytes` buffers (for cross-boundary reads via
/// [`MultiBufferReader`]) and Arrow `Buffer` copies (for zero-copy view
/// block registration).
pub struct PlainPageDecoder {
    data: Vec<Bytes>,
    /// Arrow `Buffer` wrappers over `data`, registered as view blocks.
    buffers: Vec<Buffer>,
    position: ReaderPosition,
}

impl PlainPageDecoder {
    /// Reads a string of `len` bytes that spans a buffer boundary, creates a
    /// new data block for it, and appends the view.
    #[inline(always)]
    fn append_view_across_boundaries(
        &mut self,
        output: &mut ViewsBuilder,
        len: u32,
    ) -> Result<(), Error> {
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        let bytes = reader.read_bytes(len as usize);
        if len > MAX_INLINE_STRING_VIEW as u32 {
            let id = output.append_block(Buffer::from(bytes));
            unsafe {
                output.append_view_unchecked(id, 0, len);
            }
        } else {
            unsafe { output.append_raw_view_unchecked(&make_view(bytes.as_ref(), 0, 0)) };
        };
        Ok(())
    }

    /// Fast path: reads values entirely within the current buffer.
    ///
    /// Returns `Err(Error::Len)` if the length prefix is split across buffers,
    /// or `Err(Error::ReadStr(len))` if the string body is split.
    fn read_from_current_buffer(
        &mut self,
        output: &mut ViewsBuilder,
        size: usize,
    ) -> Result<(), Error> {
        let bytes = &self.data[self.position.buffer_index];
        let block_id = output.append_block(self.buffers[self.position.buffer_index].clone());

        let buf: &[u8] = bytes.as_ref();
        let mut read = 0;

        while self.position.offset < bytes.len() && read != size {
            if self.position.offset + 4 > bytes.len() {
                return Err(Error::Len);
            }
            let len_bytes: [u8; 4] = unsafe {
                buf.get_unchecked(self.position.offset..self.position.offset + 4)
                    .try_into()
                    .unwrap()
            };
            let len = u32::from_le_bytes(len_bytes);

            let start_offset = self.position.offset + 4;
            let end_offset = start_offset + len as usize;

            if end_offset > buf.len() {
                self.position.offset = start_offset;
                return Err(Error::ReadStr(len));
            }

            unsafe {
                output.append_view_unchecked(block_id, start_offset as u32, len);
            }
            self.position.offset = end_offset;

            read += 1;
        }

        Ok(())
    }

    /// Skip values within the current buffer. Returns the number of values
    /// fully skipped and an optional error if a value straddled the boundary.
    fn skip_within_buffer(&mut self, size: usize) -> (usize, Option<Error>) {
        let bytes = &self.data[self.position.buffer_index];

        let buf: &[u8] = bytes.as_ref();
        let mut skipped = 0;

        while self.position.offset < bytes.len() && skipped != size {
            if self.position.offset + 4 > bytes.len() {
                return (skipped, Some(Error::Len));
            }
            let len_bytes: [u8; 4] = unsafe {
                buf.get_unchecked(self.position.offset..self.position.offset + 4)
                    .try_into()
                    .unwrap()
            };
            let len = u32::from_le_bytes(len_bytes);

            let start_offset = self.position.offset + 4;
            let end_offset = start_offset + len as usize;

            if end_offset > buf.len() {
                self.position.offset = start_offset;
                return (skipped, Some(Error::ReadStr(len)));
            }

            self.position.offset = end_offset;
            skipped += 1;
        }

        (skipped, None)
    }
}

impl DecodePlain for PlainPageDecoder {
    type Builder = ViewsBuilder;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self {
        Self {
            buffers: data.iter().map(|b| Buffer::from(b.clone())).collect(),
            data,
            position,
        }
    }

    fn read(&mut self, builder: &mut Self::Builder, size: usize) {
        let target_size = builder.len() + size;
        while builder.len() < target_size {
            if self.position.offset >= self.data[self.position.buffer_index].len() {
                self.position.buffer_index += 1;
                self.position.offset = 0;
            }

            match self.read_from_current_buffer(builder, target_size - builder.len()) {
                Ok(()) => {}
                Err(Error::ReadStr(len)) => {
                    self.append_view_across_boundaries(builder, len).unwrap();
                }
                Err(Error::Len) => {
                    let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
                    let len = reader.read_u32_le();
                    self.append_view_across_boundaries(builder, len).unwrap();
                }
            }
        }
    }

    fn skip(&mut self, mut size: usize) {
        while size > 0 {
            if self.position.offset >= self.data[self.position.buffer_index].len() {
                self.position.buffer_index += 1;
                self.position.offset = 0;
            }

            let (skipped, err) = self.skip_within_buffer(size);
            size -= skipped;
            match err {
                None => {}
                Some(Error::ReadStr(len)) => {
                    let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
                    reader.skip(len as usize);
                    size -= 1;
                }
                Some(Error::Len) => {
                    let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
                    let len = reader.read_u32_le();
                    reader.skip(len as usize);
                    size -= 1;
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SlabAllocator;
    use crate::memory::init_test_free_pool;
    use crate::operations::parquet::decoding::column_decoders::bytes_view::views_builder::ViewsBuilder;
    use crate::operations::parquet::decoding::column_decoders::{ArrayBuilder, DecodePlain};
    use arrow_array::{Array, StringViewArray};
    use bytes::Bytes;

    fn encode_plain(strings: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();
        for s in strings {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        data
    }

    fn make_data(buffers: Vec<Vec<u8>>) -> Vec<Bytes> {
        buffers.into_iter().map(Bytes::from).collect()
    }

    fn extract(buf: ViewsBuilder) -> Vec<String> {
        let arr = buf.into_array(None);
        let sv = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        (0..sv.len()).map(|i| sv.value(i).to_string()).collect()
    }

    #[test]
    fn test_single_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["hi", "bye", "ok"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 3);

        dec.read(&mut out, 3);

        assert_eq!(extract(out), vec!["hi", "bye", "ok"]);
    }

    #[test]
    fn test_partial_read() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["a", "b", "c"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());

        let mut out = ViewsBuilder::with_capacity(&mut allocator, 3);
        dec.read(&mut out, 2);
        assert_eq!(out.len(), 2);

        // Read remaining value
        dec.read(&mut out, 1);
        assert_eq!(extract(out), vec!["a", "b", "c"]);
    }

    /// String body straddles buffer boundary (ReadStr path).
    /// "ab" fits in buffer 0, "hello" length fits but body is split.
    #[test]
    fn test_string_body_crosses_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let all = encode_plain(&["ab", "hello"]);
        let data = make_data(vec![all[..12].to_vec(), all[12..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 2);

        dec.read(&mut out, 2);

        assert_eq!(extract(out), vec!["ab", "hello"]);
    }

    /// Length prefix straddles buffer boundary (Len path).
    /// "ab" fits in buffer 0, the 4-byte length of "cd" is split.
    #[test]
    fn test_length_prefix_crosses_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let all = encode_plain(&["ab", "cd"]);
        let data = make_data(vec![all[..8].to_vec(), all[8..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 2);

        dec.read(&mut out, 2);

        assert_eq!(extract(out), vec!["ab", "cd"]);
    }

    /// Two buffers, each containing complete encoded strings (no straddling).
    #[test]
    fn test_two_clean_buffers() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["ab"]), encode_plain(&["cd"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 2);

        dec.read(&mut out, 2);

        assert_eq!(extract(out), vec!["ab", "cd"]);
    }

    /// Regression: skip with a string body crossing a buffer boundary must
    /// correctly count the boundary value and the values skipped before it.
    /// Without the fix, size is never decremented for either, causing
    /// subsequent reads to return the wrong values.
    #[test]
    fn test_skip_with_body_crossing_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // "ab"(6 bytes) + "hello"(9 bytes) + "v1"(6) + "v2"(6) + "v3"(6) + "v4"(6) = 39 bytes
        // Split at 12: "hello" body straddles buffer boundary (ReadStr path).
        let all = encode_plain(&["ab", "hello", "v1", "v2", "v3", "v4"]);
        let data = make_data(vec![all[..12].to_vec(), all[12..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());

        // Skip "ab" and "hello"
        DecodePlain::skip(&mut dec, 2);

        // Read the next 2 — should be "v1", "v2"
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 2);
        dec.read(&mut out, 2);
        assert_eq!(extract(out), vec!["v1", "v2"]);
    }

    /// Regression: skip with a length prefix crossing a buffer boundary.
    #[test]
    fn test_skip_with_len_crossing_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // "ab"(6 bytes) + "cd"(6 bytes) + "v1"(6) + "v2"(6) + "v3"(6) + "v4"(6) = 36 bytes
        // Split at 8: "cd" length prefix straddles buffer boundary (Len path).
        //   Buffer 0 = "ab"(6) + 2 bytes of "cd" length = 8 bytes
        let all = encode_plain(&["ab", "cd", "v1", "v2", "v3", "v4"]);
        let data = make_data(vec![all[..8].to_vec(), all[8..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());

        // Skip "ab" and "cd"
        DecodePlain::skip(&mut dec, 2);

        // Read the next 2 — should be "v1", "v2"
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 2);
        dec.read(&mut out, 2);
        assert_eq!(extract(out), vec!["v1", "v2"]);
    }

    /// Regression: read_from_current_buffer must receive the remaining count,
    /// not the original size. Otherwise, after a boundary error the next call
    /// reads too many values from the subsequent buffer.
    #[test]
    fn test_read_does_not_over_read_after_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // Layout: "a"(5) "b"(5) "hello"(9) "c"(5) "d"(5) "e"(5) "f"(5) = 39 bytes
        // Split at 15: "hello" body straddles (ReadStr path).
        // After boundary: builder has 3 values, needs 1 more.
        // Bug: read_from_current_buffer(builder, 4) reads 4 from next buffer → 7 total.
        let all = encode_plain(&["a", "b", "hello", "c", "d", "e", "f"]);
        let data = make_data(vec![all[..15].to_vec(), all[15..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 4);

        dec.read(&mut out, 4);

        assert_eq!(out.len(), 4, "should read exactly 4 values, not more");
        assert_eq!(extract(out), vec!["a", "b", "hello", "c"]);
    }

    /// Multiple values read before a boundary error — remaining must account
    /// for both the values read inside push_for_single_buffer AND the
    /// cross-boundary value handled by append_view_across_boundaries.
    #[test]
    fn test_remaining_accurate_across_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let all = encode_plain(&["x", "y", "hello"]);
        let data = make_data(vec![all[..15].to_vec(), all[15..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default());
        let mut out = ViewsBuilder::with_capacity(&mut allocator, 3);

        dec.read(&mut out, 3);

        assert_eq!(extract(out), vec!["x", "y", "hello"]);
    }
}
