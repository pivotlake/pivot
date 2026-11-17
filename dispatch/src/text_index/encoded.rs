//! The encoded form of an index piece: bytes in slabs, written front to back
//! and read back the same way.
//!
//! ```text
//!   EncodedWriter ──▶ │ slab │ slab │ … │ slab │ ──▶ EncodedReader
//!                     as many as the bytes need
//! ```
//!
//! Numbers are written as varints: seven bits to a byte, lowest bits first,
//! with the top bit of a byte saying another byte follows. A small number
//! takes one byte. A signed number is folded first so that a small negative
//! one is small too (0, -1, 1, -2, … become 0, 1, 2, 3, …).
//!
//! What is read may come from a file, so every read checks what it finds
//! and answers with an [`Error`] rather than trusting it.

use crate::memory::{Slab, SlabAllocator, SlabVec};
use crate::text_index::Error;

/// Appends bytes to slabs taken from an allocator.
pub(super) struct EncodedWriter<'a> {
    allocator: &'a mut SlabAllocator,
    bytes: SlabVec<u8>,
}

impl<'a> EncodedWriter<'a> {
    pub fn new(allocator: &'a mut SlabAllocator) -> Self {
        Self {
            allocator,
            bytes: SlabVec::new(),
        }
    }

    /// How many bytes [`write_varint`](Self::write_varint) takes for `value`.
    pub fn count_varint_bytes(value: u64) -> usize {
        let significant_bits = (u64::BITS - value.leading_zeros()).max(1);
        significant_bits.div_ceil(7) as usize
    }

    pub fn write_byte(&mut self, byte: u8) {
        self.bytes.push(&mut *self.allocator, byte);
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_byte(byte);
        }
    }

    pub fn write_varint(&mut self, value: u64) {
        let mut rest = value;
        while rest >= 0x80 {
            self.write_byte(rest as u8 | 0x80);
            rest >>= 7;
        }
        self.write_byte(rest as u8);
    }

    pub fn write_signed_varint(&mut self, value: i64) {
        self.write_varint(((value << 1) ^ (value >> 63)) as u64);
    }

    /// The slabs holding everything written, in order.
    pub fn into_slabs(self) -> Vec<Slab> {
        self.bytes.into_slabs()
    }
}

/// Reads the bytes of encoded slabs front to back.
pub(super) struct EncodedReader<'a> {
    /// The slabs not yet started.
    slabs: &'a [Slab],
    /// The unread bytes of the slab being read.
    current: &'a [u8],
}

impl<'a> EncodedReader<'a> {
    pub fn new(slabs: &'a [Slab]) -> Self {
        Self {
            slabs,
            current: &[],
        }
    }

    /// Move on to the next slab that holds a byte. `false` when none does.
    fn advance_to_unread_bytes(&mut self) -> bool {
        while self.current.is_empty() {
            let Some((next, later)) = self.slabs.split_first() else {
                return false;
            };
            self.current = next.as_slice();
            self.slabs = later;
        }
        true
    }

    pub fn read_byte(&mut self) -> Result<u8, Error> {
        if !self.advance_to_unread_bytes() {
            return Err(Error::Corrupt("the bytes end too early"));
        }
        let byte = self.current[0];
        self.current = &self.current[1..];
        Ok(byte)
    }

    /// Read `len` bytes onto the end of `into`.
    pub fn read_bytes_into(&mut self, len: usize, into: &mut Vec<u8>) -> Result<(), Error> {
        let mut left = len;
        while left > 0 {
            if !self.advance_to_unread_bytes() {
                return Err(Error::Corrupt("the bytes end too early"));
            }
            let (taken, rest) = self.current.split_at(left.min(self.current.len()));
            into.extend_from_slice(taken);
            self.current = rest;
            left -= taken.len();
        }
        Ok(())
    }

    pub fn read_varint(&mut self) -> Result<u64, Error> {
        let mut value = 0u64;
        for shift in (0..u64::BITS).step_by(7) {
            let byte = self.read_byte()?;
            let bits = (byte & 0x7f) as u64;
            if bits << shift >> shift != bits {
                return Err(Error::Corrupt("a number does not fit 64 bits"));
            }
            value |= bits << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Error::Corrupt("a number does not fit 64 bits"))
    }

    pub fn read_signed_varint(&mut self) -> Result<i64, Error> {
        let folded = self.read_varint()?;
        Ok((folded >> 1) as i64 ^ -((folded & 1) as i64))
    }

    /// Read a number that is a length or a position in memory or in a file.
    pub fn read_len(&mut self) -> Result<usize, Error> {
        usize::try_from(self.read_varint()?)
            .map_err(|_| Error::Corrupt("a length does not fit this machine"))
    }

    /// Check that everything was read.
    pub fn finish(mut self) -> Result<(), Error> {
        if self.advance_to_unread_bytes() {
            return Err(Error::Corrupt("bytes follow the end"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    #[test]
    fn numbers_read_back_as_written() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let unsigned = [0, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
        let signed = [0, -1, 1, -64, 64, i64::MIN, i64::MAX];
        let mut writer = EncodedWriter::new(&mut allocator);
        for value in unsigned {
            writer.write_varint(value);
        }
        for value in signed {
            writer.write_signed_varint(value);
        }
        let encoded = writer.into_slabs();

        let mut reader = EncodedReader::new(&encoded);
        let read_unsigned = unsigned.map(|_| reader.read_varint().unwrap());
        let read_signed = signed.map(|_| reader.read_signed_varint().unwrap());

        assert_eq!(read_unsigned, unsigned);
        assert_eq!(read_signed, signed);
        assert!(reader.finish().is_ok());
    }

    #[test]
    fn a_number_takes_the_bytes_counted_for_it() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);

        let lens = [0, 127, 128, 16_383, 16_384, u64::MAX].map(|value| {
            let mut writer = EncodedWriter::new(&mut allocator);
            writer.write_varint(value);
            let written: usize = writer.into_slabs().iter().map(Slab::size).sum();
            (written, EncodedWriter::count_varint_bytes(value))
        });

        assert_eq!(lens, [(1, 1), (1, 1), (2, 2), (2, 2), (3, 3), (10, 10)]);
    }

    #[test]
    fn bytes_written_across_slabs_read_back_in_order() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let written: Vec<u8> = (0..100_000u32).map(|index| index as u8).collect();
        let mut writer = EncodedWriter::new(&mut allocator);
        writer.write_bytes(&written);
        let encoded = writer.into_slabs();

        let mut read = Vec::new();
        let mut reader = EncodedReader::new(&encoded);
        reader.read_bytes_into(written.len(), &mut read).unwrap();

        assert!(encoded.len() > 1);
        assert_eq!(read, written);
    }

    #[test]
    fn reading_past_the_last_byte_is_an_error() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut writer = EncodedWriter::new(&mut allocator);
        writer.write_byte(0x80); // a number whose next byte is missing
        let encoded = writer.into_slabs();

        let read = EncodedReader::new(&encoded).read_varint();

        assert!(read.is_err());
    }

    #[test]
    fn a_number_wider_than_64_bits_is_an_error() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut writer = EncodedWriter::new(&mut allocator);
        writer.write_bytes(&[0xff; 9]);
        writer.write_byte(0x02);
        let encoded = writer.into_slabs();

        let read = EncodedReader::new(&encoded).read_varint();

        assert!(read.is_err());
    }
}
