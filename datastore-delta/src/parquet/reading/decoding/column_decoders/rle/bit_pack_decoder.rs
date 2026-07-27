//! Bit-pack decoder for Parquet's RLE/bit-packed hybrid encoding.
//!
//! Parquet dictionary-encoded columns store row indices as bit-packed groups of
//! 8 values, each value occupying `bit_width` bits. This module provides
//! [`BitPackDecoder`], a high-throughput decoder that unpacks those values into
//! `u32`s using a four-phase pipeline:
//!
//! 1. **Drain partial** — emit leftover values from the previous group.
//! 2. **Bulk ×64** — unpack 8 groups (64 values) per iteration via
//!    [`unpack8`], which loads two `u128` halves and extracts values with
//!    compile-time-constant shifts.
//! 3. **Tail step-down** — handle 32, 16, or 8 remaining values.
//! 4. **New partial** — unpack one more group, emit what's needed, stash
//!    the rest in `partial` for the next call.
//!
//! The entire pipeline is const-specialised per `bit_width` (0–32) via a
//! [`dispatch!`] jump table, so every shift and mask is a compile-time
//! constant.

/// Data ended mid-group — caller needs to supply more bytes.
/// Contains the number of values already written to output.
#[derive(Debug)]
pub struct BitDecoderOverflow(pub usize);

/// Decodes bit-packed u32 indices from a contiguous byte slice.
///
/// See the [module docs](self) for the decoding pipeline.
pub struct BitPackDecoder<'a> {
    data: &'a [u8],
    /// Current byte offset into `data`.
    pos: usize,
    bit_width: u8,
    /// Up to 7 leftover values from the last unpacked group.
    partial: [u32; 7],
    /// How many values in `partial` are still unconsumed.
    partial_count: u8,
}

impl<'a> BitPackDecoder<'a> {
    /// Creates a decoder over `data` starting at byte `pos`.
    ///
    /// `partial` and `partial_count` carry leftover values from a previous
    /// decode call (pass zeroes for a fresh start).
    pub fn new(
        data: &'a [u8],
        pos: usize,
        bit_width: u8,
        partial: [u32; 7],
        partial_count: u8,
    ) -> Self {
        debug_assert!(bit_width <= 32);
        Self {
            data,
            pos,
            bit_width,
            partial,
            partial_count,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn partial_count(&self) -> u8 {
        self.partial_count
    }

    pub fn partial_values(&self) -> &[u32; 7] {
        &self.partial
    }

    /// Unpacks values into `output`, returning the number written.
    ///
    /// Returns `Err(BitDecoderOverflow(n))` if the data runs out mid-group,
    /// where `n` is the number of values successfully written before the
    /// overflow. The caller should supply the remaining bytes from the next
    /// buffer and retry.
    pub fn decode(&mut self, output: &mut [u32]) -> Result<usize, BitDecoderOverflow> {
        dispatch!(
            self,
            output,
            self.bit_width,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        )
    }
}

const fn mask(bw: usize) -> u32 {
    if bw >= 32 { u32::MAX } else { (1u32 << bw) - 1 }
}

/// Bytes needed to cover half a group (4 values × bw bits).
const fn half_bytes(bw: usize) -> usize {
    (4 * bw).div_ceil(8)
}

/// Unpack 8 values from `data[pos..]`. Advances `pos` by BW.
///
/// Loads two u128 halves: lo covers vals 0-3, hi covers vals 4-7.
/// All shift amounts are compile-time constants.
#[inline(always)]
unsafe fn unpack8<const BW: usize>(data: &[u8], pos: &mut usize, out: *mut u32) {
    const fn hb(bw: usize) -> usize {
        (4 * bw) % 8
    }

    unsafe {
        let ptr = data.as_ptr().add(*pos);
        *pos += BW;

        let lo = {
            let mut v = 0u128;
            std::ptr::copy_nonoverlapping(ptr, &mut v as *mut u128 as *mut u8, half_bytes(BW));
            v
        };
        let hi = {
            let mut v = 0u128;
            std::ptr::copy_nonoverlapping(
                ptr.add((4 * BW) / 8),
                &mut v as *mut u128 as *mut u8,
                half_bytes(BW),
            );
            v
        };

        *out.add(0) = (lo as u32) & mask(BW);
        *out.add(1) = ((lo >> BW) as u32) & mask(BW);
        *out.add(2) = ((lo >> (2 * BW)) as u32) & mask(BW);
        *out.add(3) = ((lo >> (3 * BW)) as u32) & mask(BW);
        *out.add(4) = ((hi >> hb(BW)) as u32) & mask(BW);
        *out.add(5) = ((hi >> (hb(BW) + BW)) as u32) & mask(BW);
        *out.add(6) = ((hi >> (hb(BW) + 2 * BW)) as u32) & mask(BW);
        *out.add(7) = ((hi >> (hb(BW) + 3 * BW)) as u32) & mask(BW);
    }
}

/// Generic pipeline — calls const-specialized `unpack8<BW>`.
fn decode_inner<const BW: usize>(
    data: &[u8],
    pos: &mut usize,
    partial: &mut [u32; 7],
    partial_count: &mut u8,
    output: &mut [u32],
) -> Result<usize, BitDecoderOverflow> {
    if BW == 0 {
        output.fill(0);
        return Ok(output.len());
    }

    let out = output.as_mut_ptr();
    let mut i = 0;

    // Phase 1: drain partial
    while i < output.len() && *partial_count > 0 {
        unsafe { *out.add(i) = partial[7 - *partial_count as usize] };
        *partial_count -= 1;
        i += 1;
    }

    // Phase 2: bulk — 64 values per iteration
    unsafe {
        while i + 64 <= output.len() && *pos + 8 * BW <= data.len() {
            let p = out.add(i);
            unpack8::<BW>(data, pos, p);
            unpack8::<BW>(data, pos, p.add(8));
            unpack8::<BW>(data, pos, p.add(16));
            unpack8::<BW>(data, pos, p.add(24));
            unpack8::<BW>(data, pos, p.add(32));
            unpack8::<BW>(data, pos, p.add(40));
            unpack8::<BW>(data, pos, p.add(48));
            unpack8::<BW>(data, pos, p.add(56));
            i += 64;
        }

        // Phase 3: tail — step down 32, 16, 8
        if i + 32 <= output.len() && *pos + 4 * BW <= data.len() {
            let p = out.add(i);
            unpack8::<BW>(data, pos, p);
            unpack8::<BW>(data, pos, p.add(8));
            unpack8::<BW>(data, pos, p.add(16));
            unpack8::<BW>(data, pos, p.add(24));
            i += 32;
        }
        if i + 16 <= output.len() && *pos + 2 * BW <= data.len() {
            let p = out.add(i);
            unpack8::<BW>(data, pos, p);
            unpack8::<BW>(data, pos, p.add(8));
            i += 16;
        }
        if i + 8 <= output.len() && *pos + BW <= data.len() {
            unpack8::<BW>(data, pos, out.add(i));
            i += 8;
        }
    }

    // Phase 4: partial remainder
    if i < output.len() {
        if *pos + BW > data.len() {
            return Err(BitDecoderOverflow(i));
        }
        let mut scratch = [0u32; 8];
        unsafe { unpack8::<BW>(data, pos, scratch.as_mut_ptr()) };
        let take = output.len() - i;
        #[allow(clippy::needless_range_loop)]
        unsafe {
            for j in 0..take {
                *out.add(i + j) = scratch[j];
            }
        }
        i += take;
        let remaining = 8 - take;
        *partial_count = remaining as u8;
        for j in 0..remaining {
            partial[7 - remaining + j] = scratch[take + j];
        }
    }

    Ok(i)
}

macro_rules! dispatch {
    ($self:expr, $output:expr, $bw:expr, [$($n:literal),*]) => {
        match $bw {
            $($n => decode_inner::<$n>($self.data, &mut $self.pos, &mut $self.partial, &mut $self.partial_count, $output),)*
            _ => unreachable!(),
        }
    };
}
use dispatch;

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(values: &[u32], bit_width: u8) -> Vec<u8> {
        let total_bits = values.len() * bit_width as usize;
        let mut bytes = vec![0u8; total_bits.div_ceil(8)];
        for (i, &val) in values.iter().enumerate() {
            let bit_pos = i * bit_width as usize;
            for b in 0..bit_width as usize {
                if (val >> b) & 1 == 1 {
                    bytes[(bit_pos + b) / 8] |= 1 << ((bit_pos + b) % 8);
                }
            }
        }
        bytes
    }

    fn decode_all(data: &[u8], bit_width: u8, count: usize) -> Vec<u32> {
        let mut out = vec![0u32; count];
        let mut dec = BitPackDecoder::new(data, 0, bit_width, [0; 7], 0);
        let n = dec.decode(&mut out).unwrap();
        assert_eq!(n, count);
        out
    }

    #[test]
    fn test_bw1() {
        let values: Vec<u32> = (0..16).map(|i| i & 1).collect();
        assert_eq!(decode_all(&encode(&values, 1), 1, 16), values);
    }

    #[test]
    fn test_bw2() {
        let values: Vec<u32> = (0..8).map(|i| i % 4).collect();
        assert_eq!(decode_all(&encode(&values, 2), 2, 8), values);
    }

    #[test]
    fn test_bw3() {
        let values: Vec<u32> = (0..8).map(|i| i % 8).collect();
        assert_eq!(decode_all(&encode(&values, 3), 3, 8), values);
    }

    #[test]
    fn test_bw4() {
        let values: Vec<u32> = (0..16).map(|i| i % 16).collect();
        assert_eq!(decode_all(&encode(&values, 4), 4, 16), values);
    }

    #[test]
    fn test_bw8() {
        let values: Vec<u32> = (0..24).map(|i| i % 256).collect();
        assert_eq!(decode_all(&encode(&values, 8), 8, 24), values);
    }

    #[test]
    fn test_bw16() {
        let values: Vec<u32> = (0..8).map(|i| i * 1000).collect();
        assert_eq!(decode_all(&encode(&values, 16), 16, 8), values);
    }

    #[test]
    fn test_bw20() {
        let values: Vec<u32> = (0..8).map(|i| i * 100_000).collect();
        assert_eq!(decode_all(&encode(&values, 20), 20, 8), values);
    }

    #[test]
    fn test_bw32() {
        let values: Vec<u32> = (0..8).map(|i| i * 500_000_000 + 1).collect();
        assert_eq!(decode_all(&encode(&values, 32), 32, 8), values);
    }

    #[test]
    fn test_partial_across_calls() {
        let values: Vec<u32> = (0..24).map(|i| i % 4).collect();
        let data = encode(&values, 2);
        let mut dec = BitPackDecoder::new(&data, 0, 2, [0; 7], 0);
        let mut result = Vec::new();
        for chunk in [5, 5, 5, 5, 4] {
            let mut buf = vec![0u32; chunk];
            let n = dec.decode(&mut buf).unwrap();
            assert_eq!(n, chunk);
            result.extend_from_slice(&buf);
        }
        assert_eq!(result, values);
    }

    #[test]
    fn test_bulk_path() {
        let values: Vec<u32> = (0..256).map(|i| i % 4).collect();
        assert_eq!(decode_all(&encode(&values, 2), 2, 256), values);
    }

    #[test]
    fn test_tail_stepdown() {
        let values: Vec<u32> = (0..120).map(|i| i % 16).collect();
        assert_eq!(decode_all(&encode(&values, 4), 4, 120), values);
    }

    #[test]
    fn test_scalar_only() {
        let full_group = vec![1u32, 2, 3, 0, 0, 0, 0, 0];
        let data = encode(&full_group, 2);
        let mut dec = BitPackDecoder::new(&data, 0, 2, [0; 7], 0);
        let mut buf = vec![0u32; 3];
        assert_eq!(dec.decode(&mut buf).unwrap(), 3);
        assert_eq!(buf, vec![1, 2, 3]);
    }

    #[test]
    fn test_partial_resume() {
        let values: Vec<u32> = (0..8).collect();
        let data = encode(&values, 4);
        let mut dec = BitPackDecoder::new(&data, 0, 4, [0; 7], 0);

        let mut buf = vec![0u32; 5];
        assert_eq!(dec.decode(&mut buf).unwrap(), 5);
        assert_eq!(buf, &values[..5]);
        assert_eq!(dec.partial_count, 3);

        let mut buf = vec![0u32; 3];
        assert_eq!(dec.decode(&mut buf).unwrap(), 3);
        assert_eq!(buf, &values[5..8]);
        assert_eq!(dec.partial_count, 0);
    }

    #[test]
    fn test_overflow_error() {
        let values: Vec<u32> = (0..8).collect();
        let data = encode(&values, 4);
        let mut dec = BitPackDecoder::new(&data, 0, 4, [0; 7], 0);
        let mut buf = vec![0u32; 16];
        let err = dec.decode(&mut buf).unwrap_err();
        assert_eq!(err.0, 8);
        assert_eq!(&buf[..8], &values[..]);
    }

    #[test]
    fn test_overflow_after_partial() {
        let values: Vec<u32> = (0..8).collect();
        let data = encode(&values, 4);
        let mut dec = BitPackDecoder::new(&data, 0, 4, [0; 7], 0);
        let mut buf = vec![0u32; 5];
        dec.decode(&mut buf).unwrap();
        let mut buf = vec![0u32; 10];
        let err = dec.decode(&mut buf).unwrap_err();
        assert_eq!(err.0, 3);
        assert_eq!(&buf[..3], &values[5..8]);
    }
}
