//! [`PlainPageDecoder`] — reads length-prefixed byte arrays from plain-encoded
//! Parquet pages into a [`ViewBuilder`].
//!
//! Each value on disk is a 4-byte little-endian length followed by that many
//! bytes of payload. The decoder has a fast path that stays within a single
//! underlying buffer (zero-copy view creation) and two fallback paths for when
//! the length prefix or the string body straddles a buffer boundary.

use super::super::ArrayBuilder;
use crate::reading::decoding::leaf_decoders::bytes_view::delta_length_page_decoder::DeltaLengthPageDecoder;
use crate::reading::decoding::leaf_decoders::{CachedPage, DecodePlain};
use arrow_array::builder::make_view;
use arrow_array::types::ByteViewType;
use arrow_buffer::Buffer;
use bytes::Bytes;
use dispatch::arrays::ViewBuilder;
use dispatch::env::MAX_INLINE_STRING_VIEW;
use dispatch::memory::{BlockKey, MultiBufferReader, ReaderPosition, memory_ctx};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use thiserror::Error;

/// How far ahead of the walk each string's prefetch reaches. Each string's
/// length load depends on the previous string's end, so the chain runs at
/// the latency of wherever the next line sits; a few lines ahead keeps that
/// in L1 while the hardware prefetcher stages further out.
pub(crate) const WALK_PREFETCH_BYTES: usize = 2048;

/// Prefetches the cache line at `ptr` into L1. A hint with no memory effect.
#[inline(always)]
pub(crate) fn prefetch_line(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::pointers_in_nomem_asm_block)]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// Reads the little-endian `u32` at `offset`.
///
/// # Safety
///
/// `offset + 4 <= buf.len()`.
#[inline(always)]
pub(crate) unsafe fn read_u32_le_at(buf: &[u8], offset: usize) -> u32 {
    unsafe {
        u32::from_le_bytes(std::ptr::read_unaligned(
            buf.as_ptr().add(offset) as *const [u8; 4]
        ))
    }
}

/// The Arrow byte view of a string longer than the inline size: its length,
/// first four bytes, block and offset.
///
/// # Safety
///
/// `start + 4 <= buf.len()` and `len as usize > MAX_INLINE_STRING_VIEW`.
#[inline(always)]
pub(crate) unsafe fn long_view_at(buf: &[u8], start: usize, len: u32, block_id: u32) -> u128 {
    let prefix = unsafe { read_u32_le_at(buf, start) };
    (len as u128)
        | ((prefix as u128) << 32)
        | ((block_id as u128) << 64)
        | ((start as u32 as u128) << 96)
}

/// The Arrow byte view of a string that inlines: its bytes behind its length,
/// loaded as one 16-byte word and masked when the buffer has 16 bytes left at
/// `start`, and assembled byte by byte only at the buffer's tail.
///
/// # Safety
///
/// `start + len <= buf.len()` and `len as usize <= MAX_INLINE_STRING_VIEW`.
#[inline(always)]
pub(crate) unsafe fn inline_view_at(buf: &[u8], start: usize, len: u32) -> u128 {
    unsafe {
        if start + 16 <= buf.len() {
            let word = u128::from_le_bytes(std::ptr::read_unaligned(
                buf.as_ptr().add(start) as *const [u8; 16]
            ));
            let kept = if len == 0 {
                0
            } else {
                word & (u128::MAX >> (128 - 8 * len as usize))
            };
            (len as u128) | (kept << 32)
        } else {
            make_view(buf.get_unchecked(start..start + len as usize), 0, 0)
        }
    }
}

/// The Arrow byte view of the `len` bytes at `start` in block `block_id`,
/// matching [`make_view`] without its call: [`long_view_at`] past the inline
/// size, [`inline_view_at`] otherwise.
///
/// # Safety
///
/// `start + len <= buf.len()`.
#[inline(always)]
pub(crate) unsafe fn view_at(buf: &[u8], start: usize, len: u32, block_id: u32) -> u128 {
    unsafe {
        if len as usize > MAX_INLINE_STRING_VIEW {
            long_view_at(buf, start, len, block_id)
        } else {
            inline_view_at(buf, start, len)
        }
    }
}

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
/// producing views for a string or binary [`ViewBuilder`] (per `V`).
///
/// Holds both the raw `Bytes` buffers (for cross-boundary reads via
/// [`MultiBufferReader`]) and Arrow `Buffer` copies (for zero-copy view
/// block registration).
pub struct PlainPageDecoder<V: ByteViewType> {
    data: Vec<Bytes>,
    /// Arrow `Buffer` wrappers over `data`, registered as view blocks.
    buffers: Vec<Buffer>,
    position: ReaderPosition,
    /// The page's views, when the decompressed cache holds them: values are
    /// then copied from these instead of walked out of the page.
    views: Option<PageViews>,
    phantom: PhantomData<V>,
}

/// Marks a views block's key in the decompressed cache: the page's own key
/// with this bit set in its offset. No file offset reaches it, so a views
/// block never collides with a page, and a walk over a column chunk's byte
/// range never meets one.
const VIEWS_KEY_BIT: usize = 1 << 62;

/// A plain string page's values as Arrow byte views, one 16-byte view per
/// value in page order: a short value's view as Arrow inlines it, a long
/// value's with its buffer index unset and its offset counted from the
/// page's first byte.
///
/// Reading a plain page means walking its length prefixes, each one found
/// only by reading the value before it, and that walk touches every byte of
/// every value whether the query looks at them or not. The views are taken
/// once, the first time the page is decoded, and kept in the decompressed
/// cache beside the page. A later read copies a view per value, and a
/// value's bytes are touched only by what actually reads them, such as a
/// filter on the rows an earlier condition kept, or never, for a length or
/// emptiness test. Being relative to the page, the views stay valid however
/// the page is laid out in the cache.
pub(crate) struct PageViews {
    /// The views' bytes as the cache holds them, split anywhere across slots.
    pieces: Vec<Bytes>,
    /// The piece holding the next view, and that view's offset in it.
    piece: usize,
    offset: usize,
    /// Where each of the page's buffers starts within the page.
    buffer_starts: Vec<usize>,
    /// The page buffer the last long value was found in; values come in page
    /// order, so the search for the next one starts here.
    buffer: usize,
}

impl PageViews {
    /// The next value's view.
    #[inline(always)]
    pub(crate) fn next_view(&mut self) -> u128 {
        let piece = &self.pieces[self.piece];
        if self.offset + 16 <= piece.len() {
            // SAFETY: the 16 bytes at `offset` lie inside `piece`.
            let view = unsafe {
                u128::from_le_bytes(std::ptr::read_unaligned(
                    piece.as_ptr().add(self.offset) as *const [u8; 16]
                ))
            };
            self.offset += 16;
            if self.offset == piece.len() {
                self.piece += 1;
                self.offset = 0;
            }
            return view;
        }
        let mut bytes = [0u8; 16];
        let mut filled = 0;
        while filled < 16 {
            let piece = &self.pieces[self.piece];
            let take = (piece.len() - self.offset).min(16 - filled);
            bytes[filled..filled + take].copy_from_slice(&piece[self.offset..self.offset + take]);
            filled += take;
            self.offset += take;
            if self.offset == piece.len() {
                self.piece += 1;
                self.offset = 0;
            }
        }
        u128::from_le_bytes(bytes)
    }

    /// Steps past `count` views.
    fn skip(&mut self, count: usize) {
        let mut remaining = count * 16;
        while remaining > 0 {
            let take = (self.pieces[self.piece].len() - self.offset).min(remaining);
            remaining -= take;
            self.offset += take;
            if self.offset == self.pieces[self.piece].len() {
                self.piece += 1;
                self.offset = 0;
            }
        }
    }

    /// The page buffer holding the byte at `page_offset`, and the offset of
    /// that byte within it.
    #[inline(always)]
    pub(crate) fn locate(&mut self, page_offset: usize) -> (usize, usize) {
        while self.buffer + 1 < self.buffer_starts.len()
            && self.buffer_starts[self.buffer + 1] <= page_offset
        {
            self.buffer += 1;
        }
        while self.buffer_starts[self.buffer] > page_offset {
            self.buffer -= 1;
        }
        (self.buffer, page_offset - self.buffer_starts[self.buffer])
    }
}

/// The views of the `values` plain-encoded values starting at `position`, in
/// [`PageViews`]' page-relative form, as little-endian bytes.
fn page_relative_views(
    data: &[Bytes],
    buffer_starts: &[usize],
    mut position: ReaderPosition,
    values: usize,
) -> Vec<u8> {
    let mut views = Vec::with_capacity(values * 16);
    let mut reader = MultiBufferReader::new(data, &mut position);
    for _ in 0..values {
        let len = reader.read_u32_le();
        let start = reader.position();
        let page_offset = buffer_starts[start.buffer_index] + start.offset;
        let view = if len as usize > MAX_INLINE_STRING_VIEW {
            let mut prefix = [0u8; 4];
            reader.read_into_unfixed_slice(&mut prefix);
            reader.skip(len as usize - 4);
            (len as u128)
                | ((u32::from_le_bytes(prefix) as u128) << 32)
                | ((page_offset as u32 as u128) << 96)
        } else {
            let mut bytes = [0u8; MAX_INLINE_STRING_VIEW];
            reader.read_into_unfixed_slice(&mut bytes[..len as usize]);
            make_view(&bytes[..len as usize], 0, 0)
        };
        views.extend_from_slice(&view.to_le_bytes());
    }
    views
}

/// The cached views of `page`, taken from its data and filed in the
/// decompressed cache when they are not there yet.
pub(crate) fn cached_page_views(
    data: &[Bytes],
    position: ReaderPosition,
    page: &CachedPage,
) -> Option<PageViews> {
    let cache = memory_ctx().decompressed_cache();
    if !cache.is_enabled() || page.values == 0 {
        return None;
    }
    let mut buffer_starts = Vec::with_capacity(data.len());
    let mut start = 0;
    for buffer in data {
        buffer_starts.push(start);
        start += buffer.len();
    }
    let key = BlockKey {
        open_file: page.key.open_file.clone(),
        offset: page.key.offset | VIEWS_KEY_BIT,
        len: page.key.len,
    };
    let pieces = match cache.get(&key) {
        Some(pieces) => pieces,
        None => {
            let views = page_relative_views(data, &buffer_starts, position, page.values);
            let mut reservation = cache.reserve(&key, views.len());
            let mut copied = 0;
            for slice in reservation.as_mut_slices() {
                let len = slice.len().min(views.len() - copied);
                slice[..len].copy_from_slice(&views[copied..copied + len]);
                copied += len;
            }
            cache.insert(key, Vec::new(), reservation, Arc::new(AtomicUsize::new(0)))
        }
    };
    Some(PageViews {
        pieces,
        piece: 0,
        offset: 0,
        buffer_starts,
        buffer: 0,
    })
}

impl<V: ByteViewType> PlainPageDecoder<V> {
    /// Appends the next `size` values from the page's cached views: a short
    /// value's view as it is, a long one's pointed at the page buffer that
    /// holds it, registered as a block of `output` the first time this call
    /// needs it. A long value split between two buffers is copied into a
    /// block of its own, as the walk does.
    fn read_views(&mut self, output: &mut ViewBuilder<V>, size: usize) {
        let views = self.views.as_mut().expect("read_views needs cached views");
        let mut blocks: Vec<Option<u32>> = vec![None; self.data.len()];
        for _ in 0..size {
            let view = views.next_view();
            let len = view as u32;
            if len as usize <= MAX_INLINE_STRING_VIEW {
                // SAFETY: the view was built from this page's value bytes.
                unsafe { output.append_raw_view_unchecked(&view) };
                continue;
            }
            let (buffer, offset) = views.locate((view >> 96) as u32 as usize);
            if offset + len as usize <= self.data[buffer].len() {
                let block = *blocks[buffer]
                    .get_or_insert_with(|| output.append_block(self.buffers[buffer].clone()));
                let view = (view & u64::MAX as u128)
                    | ((block as u128) << 64)
                    | ((offset as u32 as u128) << 96);
                // SAFETY: the value's `len` bytes lie at `offset` in the block.
                unsafe { output.append_raw_view_unchecked(&view) };
            } else {
                let mut position = ReaderPosition {
                    buffer_index: buffer,
                    offset,
                };
                let bytes =
                    MultiBufferReader::new(&self.data, &mut position).read_bytes(len as usize);
                let id = output.append_block(Buffer::from(bytes));
                // SAFETY: the block holds exactly the value's `len` bytes.
                unsafe { output.append_view_unchecked(id, 0, len) };
            }
        }
    }

    /// Reads a string of `len` bytes that spans a buffer boundary, creates a
    /// new data block for it, and appends the view.
    #[inline(always)]
    fn append_view_across_boundaries(
        &mut self,
        output: &mut ViewBuilder<V>,
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
    ///
    /// The loop is the per-string cost of every plain string column, so it
    /// walks the buffer by raw offset and builds each view in place: a long
    /// string's view is its length, its first four bytes, and where it sits;
    /// a short one is its bytes, taken with one 16-byte load when the buffer
    /// has 16 bytes left and masked down to the length.
    ///
    /// The buffer is registered as a data block only once a value too long to
    /// inline needs it. A run of short values registers nothing, so an array
    /// whose values all inline names no page at all and keeps none alive.
    fn read_from_current_buffer(
        &mut self,
        output: &mut ViewBuilder<V>,
        size: usize,
    ) -> Result<(), Error> {
        let bytes = &self.data[self.position.buffer_index];
        let mut block_id: Option<u32> = None;

        let buf: &[u8] = bytes.as_ref();
        let end = buf.len();
        let mut offset = self.position.offset;
        let mut read = 0;

        while offset < end && read != size {
            if offset + 4 > end {
                self.position.offset = offset;
                return Err(Error::Len);
            }
            if offset + WALK_PREFETCH_BYTES < end {
                prefetch_line(unsafe { buf.as_ptr().add(offset + WALK_PREFETCH_BYTES) });
            }
            // SAFETY: `offset + 4 <= end` was just checked.
            let len = unsafe { read_u32_le_at(buf, offset) };
            let start = offset + 4;
            let stop = start + len as usize;
            if stop > end {
                self.position.offset = start;
                return Err(Error::ReadStr(len));
            }
            // SAFETY: the string's bytes lie inside `buf`.
            let view = if len as usize > MAX_INLINE_STRING_VIEW {
                let block = *block_id.get_or_insert_with(|| {
                    output.append_block(self.buffers[self.position.buffer_index].clone())
                });
                unsafe { long_view_at(buf, start, len, block) }
            } else {
                unsafe { inline_view_at(buf, start, len) }
            };
            unsafe { output.append_raw_view_unchecked(&view) };
            offset = stop;
            read += 1;
        }
        self.position.offset = offset;

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

impl<V: ByteViewType> DecodePlain for PlainPageDecoder<V> {
    type Builder = ViewBuilder<V>;
    type Delta = DeltaLengthPageDecoder<V>;

    fn new(data: Vec<Bytes>, position: ReaderPosition, cached_page: Option<CachedPage>) -> Self {
        let views = cached_page.and_then(|page| cached_page_views(&data, position, &page));
        Self {
            buffers: data.iter().map(|b| Buffer::from(b.clone())).collect(),
            data,
            position,
            views,
            phantom: PhantomData,
        }
    }

    fn read(&mut self, builder: &mut Self::Builder, size: usize) {
        if self.views.is_some() {
            self.read_views(builder, size);
            return;
        }
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
        if let Some(views) = self.views.as_mut() {
            views.skip(size);
            return;
        }
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

    /// The in-place view matches `make_view` for every inline length, both
    /// with a full 16-byte word available and at the end of the buffer, and
    /// for a long string.
    #[test]
    fn view_at_matches_make_view() {
        let mut buf: Vec<u8> = (0..40u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(5))
            .collect();
        buf.extend_from_slice(b"a long string well past twelve bytes");

        for len in 0..=12u32 {
            let middle = unsafe { view_at(&buf, 3, len, 7) };
            let tail_start = buf.len() - len as usize;
            let tail = unsafe { view_at(&buf, tail_start, len, 7) };

            assert_eq!(
                middle,
                make_view(&buf[3..3 + len as usize], 7, 3),
                "len {len}"
            );
            assert_eq!(
                tail,
                make_view(&buf[tail_start..], 7, tail_start as u32),
                "len {len} at the buffer end"
            );
        }
        let long = unsafe { view_at(&buf, 40, 36, 9) };
        assert_eq!(long, make_view(&buf[40..76], 9, 40));
    }
    use crate::reading::decoding::leaf_decoders::{ArrayBuilder, DecodePlain};
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, StringViewArray};
    use bytes::Bytes;
    use dispatch::arrays::ViewBuilder;
    use dispatch::memory::SlabAllocator;
    use dispatch::memory::init_test_free_pool;

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

    fn extract(buf: ViewBuilder<StringViewType>) -> Vec<String> {
        let arr = buf.into_array(None);
        let sv = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
        (0..sv.len()).map(|i| sv.value(i).to_string()).collect()
    }

    #[test]
    fn test_single_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["hi", "bye", "ok"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 3);

        dec.read(&mut out, 3);

        assert_eq!(extract(out), vec!["hi", "bye", "ok"]);
    }

    #[test]
    fn test_partial_read() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["a", "b", "c"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);

        let mut out = ViewBuilder::with_capacity(&mut allocator, 3);
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
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 2);

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
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 2);

        dec.read(&mut out, 2);

        assert_eq!(extract(out), vec!["ab", "cd"]);
    }

    /// Two buffers, each containing complete encoded strings (no straddling).
    #[test]
    fn test_two_clean_buffers() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["ab"]), encode_plain(&["cd"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 2);

        dec.read(&mut out, 2);

        assert_eq!(extract(out), vec!["ab", "cd"]);
    }

    /// Regression: skip with a string body crossing a buffer boundary must
    /// correctly count the boundary value and the values skipped before it.
    #[test]
    fn test_skip_with_body_crossing_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // "ab"(6 bytes) + "hello"(9 bytes) + "v1"(6) + "v2"(6) + "v3"(6) + "v4"(6) = 39 bytes
        // Split at 12: "hello" body straddles buffer boundary (ReadStr path).
        let all = encode_plain(&["ab", "hello", "v1", "v2", "v3", "v4"]);
        let data = make_data(vec![all[..12].to_vec(), all[12..].to_vec()]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);

        // Skip "ab" and "hello"
        DecodePlain::skip(&mut dec, 2);

        // Read the next 2 — should be "v1", "v2"
        let mut out = ViewBuilder::with_capacity(&mut allocator, 2);
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
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);

        // Skip "ab" and "cd"
        DecodePlain::skip(&mut dec, 2);

        // Read the next 2 — should be "v1", "v2"
        let mut out = ViewBuilder::with_capacity(&mut allocator, 2);
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
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 4);

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
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 3);

        dec.read(&mut out, 3);

        assert_eq!(extract(out), vec!["x", "y", "hello"]);
    }

    fn data_buffer_count(builder: ViewBuilder<StringViewType>) -> usize {
        builder
            .into_array(None)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .data_buffers()
            .len()
    }

    /// A page consumed as many runs (how nulls and pushed-down filter masks
    /// drive the decoder) registers its buffer once, not once per run.
    #[test]
    fn reading_a_page_in_runs_registers_its_buffer_once() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let long = "a value longer than twelve bytes";
        let data = make_data(vec![encode_plain(&[long; 6])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 6);

        for _ in 0..6 {
            dec.read(&mut out, 1);
        }

        assert_eq!(data_buffer_count(out), 1);
    }

    /// Values that inline into their views need no data block, so a page of
    /// short values leaves the array with no buffers to keep alive.
    #[test]
    fn short_values_register_no_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&["short", "values", "only"])]);
        let mut dec = PlainPageDecoder::new(data, ReaderPosition::default(), None);
        let mut out = ViewBuilder::with_capacity(&mut allocator, 3);

        dec.read(&mut out, 3);

        assert_eq!(data_buffer_count(out), 0);
    }

    fn cached_page(offset: usize, values: usize) -> CachedPage {
        CachedPage {
            key: BlockKey {
                open_file: crate::test_utils::dummy_row_group().open_file.clone(),
                offset,
                len: 100,
            },
            values,
        }
    }

    const MIXED: [&str; 5] = [
        "hi",
        "a value long enough to live outside its view",
        "",
        "another value well past the inline size",
        "end",
    ];

    /// The second decoder of the page finds the views the first one cached,
    /// over a page split across buffers mid-value.
    #[test]
    fn a_page_read_again_decodes_the_same_values() {
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let all = encode_plain(&MIXED);
        let data = make_data(vec![all[..30].to_vec(), all[30..].to_vec()]);
        let mut first = PlainPageDecoder::new(
            data.clone(),
            ReaderPosition::default(),
            Some(cached_page(4096, MIXED.len())),
        );
        let mut first_out = ViewBuilder::with_capacity(&mut allocator, MIXED.len());
        first.read(&mut first_out, MIXED.len());
        let mut second = PlainPageDecoder::new(
            data,
            ReaderPosition::default(),
            Some(cached_page(4096, MIXED.len())),
        );
        let mut second_out = ViewBuilder::with_capacity(&mut allocator, MIXED.len());

        second.read(&mut second_out, MIXED.len());

        assert_eq!(extract(first_out), MIXED);
        assert_eq!(extract(second_out), MIXED);
    }

    #[test]
    fn a_cached_page_skips_and_reads_in_runs() {
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let data = make_data(vec![encode_plain(&MIXED)]);
        let mut dec = PlainPageDecoder::new(
            data,
            ReaderPosition::default(),
            Some(cached_page(8192, MIXED.len())),
        );
        let mut out = ViewBuilder::with_capacity(&mut allocator, 3);

        dec.skip(1);
        dec.read(&mut out, 2);
        dec.skip(1);
        dec.read(&mut out, 1);

        assert_eq!(extract(out), vec![MIXED[1], MIXED[2], MIXED[4]]);
    }
}
