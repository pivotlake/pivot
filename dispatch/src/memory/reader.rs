//! A cursor for reading across a list of [`Bytes`] buffers as a contiguous
//! byte stream.
//!
//! Data often arrives as a sequence of separate buffers (ring-buffer slabs,
//! network chunks, file pages, etc.). [`MultiBufferReader`] presents them
//! as a single stream, handling cross-buffer reads transparently so callers
//! can read typed values (u8, u32, i64, varint, byte slices) without
//! worrying about which buffer a value falls in or whether it straddles two.
//!
//! A [`ReaderPosition`] tracks the current buffer index and byte offset.
//! It can be saved and passed to a new [`MultiBufferReader`] to resume
//! reading from where a previous reader left off.

use bytes::Bytes;
use std::io::Read;
use std::ptr::copy_nonoverlapping;

/// Saved position within a [`MultiBufferReader`].
///
/// Can be stored externally and passed back to [`MultiBufferReader::new`]
/// to resume reading from where a previous reader left off.
#[derive(Default, Debug, Copy, Clone)]
pub struct ReaderPosition {
    pub buffer_index: usize,
    pub offset: usize,
}

/// A cursor for reading across scattered `Bytes` buffers as a contiguous byte stream.
///
/// Holds a direct slice reference (`cur`) to the remaining bytes in the current buffer,
/// avoiding repeated indexing into `data[buf_idx]` on every read.
pub struct MultiBufferReader<'a, 'b> {
    buffers: &'a [Bytes],
    cur: &'a [u8],
    position: &'b mut ReaderPosition,
}

impl<'a, 'b> MultiBufferReader<'a, 'b> {
    /// Create a reader starting at `position` within `buffers`.
    pub fn new(buffers: &'a [Bytes], position: &'b mut ReaderPosition) -> Self {
        Self {
            buffers,
            cur: if position.buffer_index < buffers.len() {
                &buffers[position.buffer_index]
            } else {
                &[]
            },
            position,
        }
    }

    /// Move to the start of the next buffer in the list.
    fn advance_buffer(&mut self) {
        self.position.buffer_index += 1;
        self.position.offset = 0;
        self.cur = &self.buffers[self.position.buffer_index];
    }

    /// The unread portion of the current buffer.
    #[inline(always)]
    fn current_slice(&self) -> &[u8] {
        &self.cur[self.position.offset..]
    }

    /// Bytes remaining in the current buffer (not the total stream).
    pub fn remaining_in_cur(&self) -> usize {
        self.cur.len().saturating_sub(self.position.offset)
    }

    /// The current read position (buffer index + offset).
    pub fn position(&self) -> &ReaderPosition {
        self.position
    }

    /// Read exactly `N` bytes into a fixed-size array, crossing buffer
    /// boundaries if needed.
    #[inline(always)]
    fn read_fixed_slice<const N: usize>(&mut self) -> [u8; N] {
        let mut slice: [u8; N] = [0u8; N];

        let copy_from_cur = std::cmp::min(self.remaining_in_cur(), N);
        unsafe {
            copy_nonoverlapping(
                self.current_slice().as_ptr(),
                slice.as_mut_ptr(),
                copy_from_cur,
            )
        };
        self.position.offset += copy_from_cur;

        let remaining = N - copy_from_cur;
        if remaining > 0 {
            self.advance_buffer();
            slice[copy_from_cur..].copy_from_slice(&self.current_slice()[..remaining]);
            self.position.offset = remaining;
        }
        slice
    }

    /// Read `s.len()` bytes into the provided slice, crossing buffer
    /// boundaries if needed.
    #[inline(always)]
    fn read_into_unfixed_slice(&mut self, s: &mut [u8]) {
        let copy_from_cur = std::cmp::min(self.remaining_in_cur(), s.len());
        unsafe {
            copy_nonoverlapping(self.current_slice().as_ptr(), s.as_mut_ptr(), copy_from_cur)
        };
        self.position.offset += copy_from_cur;

        let remaining = s.len() - copy_from_cur;
        if remaining > 0 {
            self.advance_buffer();
            s[copy_from_cur..].copy_from_slice(&self.current_slice()[..remaining]);
            self.position.offset = remaining;
        }
    }

    /// Extract `size` bytes as zero-copy [`Bytes`] slices.
    ///
    /// Returns one `Bytes` per underlying buffer touched. Unlike
    /// [`read_bytes`](Self::read_bytes) this avoids copying — each returned
    /// `Bytes` shares the reference count with the source buffer.
    pub fn copy_out_buffers(&mut self, size: usize) -> Vec<Bytes> {
        let mut remaining = size;
        let mut bytes = vec![];

        while remaining > self.remaining_in_cur() {
            let buffer = &self.buffers[self.position.buffer_index];
            bytes.push(buffer.slice(self.position.offset..));
            remaining -= self.current_slice().len();
            self.advance_buffer();
        }

        if remaining != 0 {
            let buffer = &self.buffers[self.position.buffer_index];
            bytes.push(buffer.slice(self.position.offset..self.position.offset + remaining));
            self.position.offset += remaining;
        }

        bytes
    }

    /// Advance the cursor by `size` bytes without reading, crossing buffer
    /// boundaries as needed.
    pub fn skip(&mut self, size: usize) {
        let skip_in_cur = std::cmp::min(self.remaining_in_cur(), size);
        self.position.offset += skip_in_cur;

        let mut remaining = size - skip_in_cur;
        while remaining > 0 {
            self.advance_buffer();
            if self.cur.len() < remaining {
                remaining -= self.cur.len();
            } else {
                self.position.offset = remaining;
                return;
            }
        }
    }

    /// Read a single byte.
    #[inline(always)]
    pub fn read_u8(&mut self) -> u8 {
        self.read_fixed_slice::<1>()[0]
    }

    /// Read a little-endian u32.
    #[inline]
    pub fn read_u32_le(&mut self) -> u32 {
        u32::from_le_bytes(self.read_fixed_slice::<4>())
    }

    /// Read a little-endian i32.
    #[inline(always)]
    pub fn read_i32_le(&mut self) -> i32 {
        self.read_u32_le() as i32
    }

    /// Read a little-endian i64.
    #[inline]
    pub fn read_i64_le(&mut self) -> i64 {
        i64::from_le_bytes(self.read_fixed_slice::<8>())
    }

    /// Read `len` bytes into a new `Vec`, crossing buffer boundaries if needed.
    pub fn read_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut buffer = Box::new_uninit_slice(len);
        // SAFETY: read_into_unfixed_slice writes exactly `len` bytes,
        // and u8 has no invalid bit patterns.
        self.read_into_unfixed_slice(unsafe {
            &mut *(buffer.as_mut() as *mut [std::mem::MaybeUninit<u8>] as *mut [u8])
        });
        unsafe { buffer.assume_init() }.into_vec()
    }

    /// Read an unsigned LEB128 varint.
    #[inline]
    pub fn read_varint(&mut self) -> u32 {
        // Fast path: first byte has continuation bit clear (value < 128).
        let first = self.read_u8();
        if first & 0x80 == 0 {
            return first as u32;
        }

        let mut result = (first & 0x7F) as u32;
        let mut shift = 7;
        loop {
            let byte = self.read_u8();
            result |= ((byte & 0x7F) as u32) << shift;
            if byte & 0x80 == 0 {
                return result;
            }
            shift += 7;
        }
    }
}

impl<'a, 'b> Read for MultiBufferReader<'a, 'b> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read_into_unfixed_slice(buf);
        Ok(buf.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_buffer() {
        let data = vec![Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        assert_eq!(reader.remaining_in_cur(), 8);
        assert_eq!(reader.read_u8(), 1);
        assert_eq!(reader.remaining_in_cur(), 7);
        assert_eq!(reader.read_u32_le(), u32::from_le_bytes([2, 3, 4, 5]));
        assert_eq!(reader.remaining_in_cur(), 3);
    }

    #[test]
    fn test_cross_buffer_u32() {
        let data = vec![Bytes::from(vec![1u8, 2]), Bytes::from(vec![3u8, 4, 5])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.read_u8(); // consume 1 byte, leaving [2] [3, 4, 5]
        let val = reader.read_u32_le();
        assert_eq!(val, u32::from_le_bytes([2, 3, 4, 5]));
    }

    #[test]
    fn test_read_bytes_cross_boundary() {
        let data = vec![Bytes::from(vec![10u8, 20]), Bytes::from(vec![30u8, 40, 50])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.read_u8(); // consume 10
        let bytes = reader.read_bytes(3); // should read [20, 30, 40]
        assert_eq!(&bytes[..], &[20, 30, 40]);
        assert_eq!(reader.remaining_in_cur(), 1);
    }

    #[test]
    fn test_resume_at_position() {
        let data = vec![Bytes::from(vec![1u8, 2, 3]), Bytes::from(vec![4u8, 5, 6])];
        let mut pos = ReaderPosition::default();
        // Read 4 bytes to advance into second buffer
        {
            let mut reader = MultiBufferReader::new(&data, &mut pos);
            for _ in 0..4 {
                reader.read_u8();
            }
        }
        // Resume from saved position
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_u8(), 5);
    }

    #[test]
    fn test_resume_at_exact_buffer_boundary() {
        let data = vec![Bytes::from(vec![1u8, 2, 3]), Bytes::from(vec![4u8, 5])];
        let mut pos = ReaderPosition::default();
        // Read exactly 3 bytes (all of first buffer)
        {
            let mut reader = MultiBufferReader::new(&data, &mut pos);
            for _ in 0..3 {
                reader.read_u8();
            }
        }
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_u8(), 4);
    }

    #[test]
    fn test_varint_single_byte() {
        let data = vec![Bytes::from(vec![42u8])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_varint(), 42);
    }

    #[test]
    fn test_varint_multi_byte() {
        // varint encoding of 300: 0xAC 0x02
        let data = vec![Bytes::from(vec![0xACu8, 0x02])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_varint(), 300);
    }

    #[test]
    fn test_empty() {
        let data: Vec<Bytes> = vec![];
        let mut pos = ReaderPosition::default();
        let reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.remaining_in_cur(), 0);
    }

    #[test]
    fn test_read_i32_le_positive() {
        let data = vec![Bytes::from(vec![0x01, 0x00, 0x00, 0x00])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_i32_le(), 1);
    }

    #[test]
    fn test_read_i32_le_negative() {
        let val: i32 = -42;
        let data = vec![Bytes::from(val.to_le_bytes().to_vec())];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_i32_le(), -42);
    }

    #[test]
    fn test_read_i64_le() {
        let val: i64 = -123456789;
        let data = vec![Bytes::from(val.to_le_bytes().to_vec())];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_i64_le(), -123456789);
    }

    #[test]
    fn test_read_i64_le_cross_buffer() {
        let bytes = (-99i64).to_le_bytes();
        let data = vec![
            Bytes::from(bytes[..3].to_vec()),
            Bytes::from(bytes[3..].to_vec()),
        ];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_i64_le(), -99);
    }

    #[test]
    fn test_read_bytes_single_buffer() {
        let data = vec![Bytes::from(vec![10, 20, 30, 40, 50])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_bytes(3), vec![10, 20, 30]);
        assert_eq!(reader.remaining_in_cur(), 2);
    }

    #[test]
    fn test_copy_out_buffers_single_buffer() {
        let data = vec![Bytes::from(vec![1, 2, 3, 4, 5])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.read_u8(); // skip 1
        let out = reader.copy_out_buffers(3);
        assert_eq!(out.len(), 1);
        assert_eq!(&out[0][..], &[2, 3, 4]);
        assert_eq!(reader.remaining_in_cur(), 1);
    }

    #[test]
    fn test_copy_out_buffers_cross_boundary() {
        let data = vec![Bytes::from(vec![1, 2]), Bytes::from(vec![3, 4, 5])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        let out = reader.copy_out_buffers(4);
        // first chunk is entire first buffer [1,2], second chunk is [3,4]
        assert_eq!(out.len(), 2);
        assert_eq!(&out[0][..], &[1, 2]);
        assert_eq!(&out[1][..], &[3, 4]);
        assert_eq!(reader.remaining_in_cur(), 1);
    }

    #[test]
    fn test_copy_out_buffers_exact_boundary() {
        let data = vec![Bytes::from(vec![1, 2]), Bytes::from(vec![3, 4])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        let out = reader.copy_out_buffers(2);
        assert_eq!(out.len(), 1);
        assert_eq!(&out[0][..], &[1, 2]);
    }

    #[test]
    fn test_read_trait() {
        let data = vec![Bytes::from(vec![1, 2]), Bytes::from(vec![3, 4, 5])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        let mut buf = [0u8; 4];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 4);
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn test_varint_cross_buffer() {
        // varint encoding of 300: 0xAC 0x02, split across buffers
        let data = vec![Bytes::from(vec![0xAC]), Bytes::from(vec![0x02])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_varint(), 300);
    }

    #[test]
    fn test_varint_max_single_byte() {
        let data = vec![Bytes::from(vec![0x7F])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);
        assert_eq!(reader.read_varint(), 127);
    }

    // -- skip --

    /// Skip within a single buffer, then read the next byte.
    #[test]
    fn test_skip_within_single_buffer() {
        let data = vec![Bytes::from(vec![1, 2, 3, 4, 5])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.skip(3);

        assert_eq!(reader.read_u8(), 4);
    }

    /// Skip zero bytes — position must not change.
    #[test]
    fn test_skip_zero() {
        let data = vec![Bytes::from(vec![1, 2, 3])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.skip(0);

        assert_eq!(reader.read_u8(), 1);
    }

    /// Skip exactly to the end of the current buffer.
    #[test]
    fn test_skip_exact_buffer_end() {
        let data = vec![Bytes::from(vec![1, 2]), Bytes::from(vec![3, 4])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.skip(2);

        assert_eq!(reader.read_u8(), 3);
    }

    /// Skip across one buffer boundary.
    #[test]
    fn test_skip_cross_buffer() {
        let data = vec![Bytes::from(vec![1, 2]), Bytes::from(vec![3, 4, 5])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.skip(3);

        assert_eq!(reader.read_u8(), 4);
    }

    /// Skip spanning more than two buffers.
    #[test]
    fn test_skip_multiple_buffers() {
        let data = vec![
            Bytes::from(vec![1, 2]),
            Bytes::from(vec![3, 4]),
            Bytes::from(vec![5, 6, 7]),
        ];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        reader.skip(5);

        assert_eq!(reader.read_u8(), 6);
    }

    /// Read some bytes, skip some, then read again.
    #[test]
    fn test_read_skip_read() {
        let data = vec![Bytes::from(vec![10, 20, 30, 40, 50])];
        let mut pos = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut pos);

        assert_eq!(reader.read_u8(), 10);
        reader.skip(2);
        assert_eq!(reader.read_u8(), 40);
    }
}
