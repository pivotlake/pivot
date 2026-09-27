//! RLE/bit-packed hybrid decoder for Parquet dictionary indices.
//!
//! Dictionary-encoded columns store row values as integer indices into a
//! dictionary. These indices are encoded with Parquet's RLE/bit-packed hybrid
//! scheme: a stream of *runs*, each either:
//!
//! - **RLE** — a single value repeated `count` times.
//! - **Bit-packed** — `num_groups × 8` values packed at `bit_width` bits each,
//!   decoded by [`BitPackDecoder`].
//!
//! [`RleDecoder`] is the stateful entry point: it lazily parses run headers and
//! delegates to [`Run::read_into`] / [`Run::skip`]. Partially consumed runs are
//! saved and resumed on the next call, so the decoder can be driven in
//! arbitrarily sized chunks.

use crate::reading::decoding::leaf_decoders::{ArrayBuilder, Dict};
use bytes::Bytes;
use dispatch::memory::{MultiBufferReader, ReaderPosition};

mod bit_pack_decoder;
pub use bit_pack_decoder::{BitDecoderOverflow, BitPackDecoder};

/// Writes `dict`'s entry for each key into `dest` (same length).
///
/// Kept out of line: with its own small frame the lookup loop unrolls,
/// while inlined into the decode body it competes for registers with
/// everything else and stays scalar.
///
/// # Safety
///
/// Every key must be `< dict.len()`.
#[inline(never)]
unsafe fn gather_entries<D: Dict>(dict: &D, keys: &[u32], dest: &mut [D::Item]) {
    for (d, &k) in dest.iter_mut().zip(keys) {
        *d = unsafe { dict.entry_unchecked(k as usize) };
    }
}

/// A single run in the RLE/bit-packed stream.
pub enum Run {
    Rle {
        value: u32,
        count: usize,
    },
    BitPacked {
        remaining_in_run: usize,
        partial: [u32; 7],
        partial_count: u8,
    },
}

/// Reads one group of 8 bit-packed values that straddles a buffer boundary.
///
/// Uses [`MultiBufferReader`] to gather `bit_width` bytes across buffers,
/// decodes the full group, and returns all 8 values.
#[inline(always)]
fn decode_cross_boundary_group(
    data: &[Bytes],
    bit_width: u8,
    position: &mut ReaderPosition,
) -> [u32; 8] {
    let mut reader = MultiBufferReader::new(data, position);
    let mut group_bytes = [0u8; 32];
    for b in group_bytes.iter_mut().take(bit_width as usize) {
        *b = reader.read_u8();
    }
    let mut group = [0u32; 8];
    BitPackDecoder::new(&group_bytes[..bit_width as usize], 0, bit_width, [0; 7], 0)
        .decode(&mut group)
        .unwrap();
    group
}

/// Builds partial state from the unused tail of a decoded group.
///
/// `partial[7-count..7]` holds the values, matching the layout expected by
/// [`BitPackDecoder`].
#[inline(always)]
fn partial_from_group(group: &[u32; 8], used: usize) -> ([u32; 7], u8) {
    let count = (8 - used) as u8;
    let mut partial = [0u32; 7];
    for j in 0..count as usize {
        partial[7 - count as usize + j] = group[used + j];
    }
    (partial, count)
}

/// Decodes up to `limit` values from a bit-packed run, processing them in
/// chunks of up to 1024 via the `on_chunk` callback.
///
/// Handles data spanning multiple [`Bytes`] buffers: when the current buffer
/// runs out mid-group, [`decode_cross_boundary_group`] stitches the group
/// across the boundary. Any unused values from that group are carried forward
/// as partial state on the new [`BitPackDecoder`].
///
/// # Returns
///
/// `(values_read, leftover_run)` — the count of values consumed and an
/// optional [`Run::BitPacked`] for the unconsumed tail of the run.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn decode_bitpacked(
    remaining_in_run: usize,
    partial: [u32; 7],
    partial_count: u8,
    scratch: &mut [u32; 1024],
    data: &[Bytes],
    bit_width: u8,
    position: &mut ReaderPosition,
    limit: usize,
    mut on_chunk: impl FnMut(&mut [u32; 1024], usize),
) -> (usize, Option<Run>) {
    let to_read = remaining_in_run.min(limit);
    let mut remaining = to_read;

    let mut decoder = BitPackDecoder::new(
        &data[position.buffer_index],
        position.offset,
        bit_width,
        partial,
        partial_count,
    );

    while remaining > 0 {
        let chunk = remaining.min(1024);
        let count = match decoder.decode(&mut scratch[..chunk]) {
            Ok(n) => n,
            Err(BitDecoderOverflow(n)) => {
                position.offset = decoder.pos();
                let group = decode_cross_boundary_group(data, bit_width, position);

                let take = (chunk - n).min(8);
                scratch[n..n + take].copy_from_slice(&group[..take]);

                // Carry unused group values as partial state so they're
                // emitted on the next iteration.
                let (new_partial, new_partial_count) = partial_from_group(&group, take);
                decoder = BitPackDecoder::new(
                    &data[position.buffer_index],
                    position.offset,
                    bit_width,
                    new_partial,
                    new_partial_count,
                );
                n + take
            }
        };

        on_chunk(scratch, count);
        remaining -= count;
    }

    position.offset = decoder.pos();

    let leftover = if remaining_in_run > limit {
        Some(Run::BitPacked {
            remaining_in_run: remaining_in_run - limit,
            partial: *decoder.partial_values(),
            partial_count: decoder.partial_count(),
        })
    } else {
        None
    };

    (to_read, leftover)
}

impl Run {
    /// Parses the next run header from `reader` and returns the run.
    ///
    /// The header's LSB distinguishes RLE (0) from bit-packed (1). For RLE
    /// runs the repeated value follows; for bit-packed runs only the group
    /// count is recorded — actual decoding happens in [`read_into`](Self::read_into).
    pub fn parse_from_header(bit_width: u8, reader: &mut MultiBufferReader) -> Self {
        let header = reader.read_varint();
        if header & 1 == 0 {
            // RLE run
            let count = (header >> 1) as usize;
            let byte_width = byte_width(bit_width);
            let mut value: u32 = 0;
            for i in 0..byte_width {
                value |= (reader.read_u8() as u32) << (i * 8);
            }
            Self::Rle { value, count }
        } else {
            // Bit-packed run
            let num_groups = (header >> 1) as usize;
            Self::BitPacked {
                remaining_in_run: num_groups * 8,
                partial: [0; 7],
                partial_count: 0,
            }
        }
    }

    /// Decodes up to `limit` values from this run, looking up each index in
    /// `dict` and pushing the result into `builder`.
    ///
    /// Returns `Some(run)` with the remaining portion if the run was not
    /// fully consumed, or `None` if it's exhausted.
    #[allow(clippy::too_many_arguments)]
    pub fn read_into<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
        self,
        builder: &mut B,
        scratch: &mut [u32; 1024],
        data: &[Bytes],
        bit_width: u8,
        position: &mut ReaderPosition,
        dict: &D,
        limit: usize,
    ) -> Option<Self> {
        match self {
            Run::Rle { value, mut count } => {
                let emit = count.min(limit);
                let entry = dict.entry(value as usize);
                builder.push(&entry, emit);
                count -= emit;
                if count > 0 {
                    Some(Run::Rle { value, count })
                } else {
                    None
                }
            }
            Run::BitPacked {
                remaining_in_run,
                partial,
                partial_count,
            } => {
                // Keys are bounded by the bit width; when the dictionary is
                // at least that large no key can be out of range. Otherwise
                // one max-scan per chunk (auto-vectorized) checks the whole
                // chunk up front: slab-backed dictionaries index with no
                // per-element bounds check, so out-of-range keys must be
                // impossible by the time the lookup loop runs.
                let keys_in_range = (1u64 << bit_width) as usize <= dict.len();
                let (_, leftover) = decode_bitpacked(
                    remaining_in_run,
                    partial,
                    partial_count,
                    scratch,
                    data,
                    bit_width,
                    position,
                    limit,
                    |scratch, count| {
                        let keys = &scratch[..count];
                        if !keys_in_range {
                            let max = keys.iter().copied().max().unwrap_or(0);
                            assert!(
                                (max as usize) < dict.len(),
                                "dictionary key {max} out of range ({} entries)",
                                dict.len()
                            );
                        }
                        let dest = builder.spare_mut(count);
                        // SAFETY: every key is < dict.len() - by the bit-width
                        // bound or the max-scan assert above.
                        unsafe { gather_entries(dict, keys, dest) };
                    },
                );
                leftover
            }
        }
    }

    /// Advances past up to `amount` values without writing them.
    ///
    /// Returns `(skipped, remaining_run)` — the number of values actually
    /// skipped and the leftover run (if any).
    pub fn skip(
        self,
        data: &[Bytes],
        bit_width: u8,
        position: &mut ReaderPosition,
        amount: usize,
    ) -> (usize, Option<Run>) {
        match self {
            Run::Rle { value, mut count } => {
                let emit = count.min(amount);
                count -= emit;
                if count > 0 {
                    (emit, Some(Run::Rle { value, count }))
                } else {
                    (emit, None)
                }
            }
            Run::BitPacked {
                remaining_in_run,
                partial,
                mut partial_count,
            } => {
                let skipped = amount.min(remaining_in_run);
                let mut left = skipped;
                // Values already unpacked into the partial group sit at the
                // end of the array, so consuming them from the front only
                // shortens the count.
                let from_partial = (partial_count as usize).min(left);
                partial_count -= from_partial as u8;
                left -= from_partial;
                // Whole groups are `bit_width` bytes each and need no
                // unpacking to step over.
                let groups = left / 8;
                if groups > 0 {
                    MultiBufferReader::new(data, position).skip(groups * bit_width as usize);
                    left -= groups * 8;
                }
                // The values past the last whole group come from one more
                // group, unpacked so its unconsumed tail carries over.
                let (partial, partial_count) = if left > 0 {
                    let group = decode_cross_boundary_group(data, bit_width, position);
                    partial_from_group(&group, left)
                } else {
                    (partial, partial_count)
                };
                let leftover = (remaining_in_run > skipped).then_some(Run::BitPacked {
                    remaining_in_run: remaining_in_run - skipped,
                    partial,
                    partial_count,
                });
                (skipped, leftover)
            }
        }
    }
}

/// Reads the ULEB128 value starting at `offset`, returning it and its length
/// in bytes, or `None` when it does not end inside `bytes`.
#[inline(always)]
fn read_varint_at(bytes: &[u8], offset: usize) -> Option<(u32, usize)> {
    let first = *bytes.get(offset)?;
    if first & 0x80 == 0 {
        return Some((first as u32, 1));
    }
    let mut value = (first & 0x7f) as u32;
    for i in 1..5 {
        let byte = *bytes.get(offset + i)?;
        value |= ((byte & 0x7f) as u32) << (7 * i);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Minimum number of bytes needed to hold a value of `bit_width` bits.
const fn byte_width(bit_width: u8) -> u8 {
    bit_width.div_ceil(8)
}

/// Stateful decoder for Parquet's RLE/bit-packed hybrid encoding.
///
/// Lazily parses [`Run`] headers on demand and supports incremental
/// consumption — partially consumed runs are stashed in `run` and resumed on
/// the next `read` or `skip` call.
pub struct RleDecoder {
    position: ReaderPosition,
    bit_width: u8,
    data: Vec<Bytes>,
    /// Scratch buffer for bit-pack decoding (avoids per-call allocation).
    buffer: Box<[u32; 1024]>,
    /// The in-progress run, if any.
    run: Option<Run>,
}

impl RleDecoder {
    pub fn new(data: Vec<Bytes>, position: ReaderPosition, bit_width: u8) -> Self {
        Self {
            position,
            bit_width,
            data,
            buffer: Box::new([0; 1024]),
            run: None,
        }
    }

    /// Returns the current in-progress run, or parses the next one from the
    /// stream.
    #[inline(always)]
    fn get_or_set_next_run(&mut self) -> Run {
        match self.run.take() {
            Some(r) => r,
            None => Run::parse_from_header(
                self.bit_width,
                &mut MultiBufferReader::new(&self.data, &mut self.position),
            ),
        }
    }

    /// Advances past `size` values without producing output.
    pub fn skip(&mut self, mut size: usize) {
        while size > 0 {
            if self.run.is_none() {
                size = self.skip_whole_runs(size);
                if size == 0 {
                    break;
                }
            }
            let run = self.get_or_set_next_run();
            let (skipped, run) = run.skip(&self.data, self.bit_width, &mut self.position, size);
            self.run = run;
            size -= skipped;
        }
    }

    /// Steps over every run that ends within the next `size` values, reading
    /// only run headers, and returns how many values are left to skip.
    ///
    /// A decoder repositioning deep into a page steps over thousands of runs,
    /// so this walks the headers straight off the current buffer. It stops at
    /// the first run that reaches past `size` or past the buffer, which the
    /// general path then takes apart.
    fn skip_whole_runs(&mut self, mut size: usize) -> usize {
        let Some(buffer) = self.data.get(self.position.buffer_index) else {
            return size;
        };
        let bytes: &[u8] = buffer;
        let bit_width = self.bit_width as usize;
        let value_bytes = byte_width(self.bit_width) as usize;
        let mut offset = self.position.offset;
        while size > 0 {
            let Some((header, header_len)) = read_varint_at(bytes, offset) else {
                break;
            };
            let groups = (header >> 1) as usize;
            let (values, run_bytes) = if header & 1 == 0 {
                (groups, value_bytes)
            } else {
                (groups * 8, groups * bit_width)
            };
            let end = offset + header_len + run_bytes;
            if values > size || end > bytes.len() {
                break;
            }
            offset = end;
            size -= values;
        }
        self.position.offset = offset;
        size
    }

    /// Decodes `size` values, looking up each index in `dict` and pushing
    /// the result into `builder`.
    pub fn read<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
        &mut self,
        builder: &mut B,
        dict: &D,
        size: usize,
    ) {
        let target = builder.len() + size;
        while builder.len() < target {
            let size = target - builder.len();
            let run = self.get_or_set_next_run();
            self.run = run.read_into(
                builder,
                &mut self.buffer,
                &self.data,
                self.bit_width,
                &mut self.position,
                dict,
                size,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::ArrayBuilder;
    use crate::reading::decoding::leaf_decoders::Dict;
    use crate::reading::decoding::leaf_decoders::bytes_view::dict::{DictFactory, ViewDict};
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, StringViewArray};
    use bytes::Bytes;
    use dispatch::arrays::ViewBuilder;
    use dispatch::memory::SlabAllocator;
    use dispatch::memory::init_test_free_pool;

    fn encode_plain_strings(strings: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();
        for s in strings {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        data
    }

    fn make_dict(entries: &[&str]) -> ViewDict<StringViewType> {
        let data = vec![Bytes::from(encode_plain_strings(entries))];
        DictFactory::new(data, entries.len()).create_dict()
    }

    fn make_data(data: Vec<u8>) -> Vec<Bytes> {
        vec![Bytes::from(data)]
    }

    fn make_data_multi_buffer(buffers: Vec<Vec<u8>>) -> Vec<Bytes> {
        buffers.into_iter().map(Bytes::from).collect()
    }

    fn extract_strings(builder: ViewBuilder<StringViewType>) -> Vec<String> {
        let array = builder.into_array(None);
        let sv = array.as_any().downcast_ref::<StringViewArray>().unwrap();
        (0..sv.len()).map(|i| sv.value(i).to_string()).collect()
    }

    fn new_decoder(data: Vec<u8>, bit_width: u8) -> RleDecoder {
        RleDecoder::new(make_data(data), ReaderPosition::default(), bit_width)
    }

    fn push_all(
        allocator: &mut SlabAllocator,
        decoder: &mut RleDecoder,
        dict: &ViewDict<StringViewType>,
        size: usize,
    ) -> Vec<String> {
        let mut buf = ViewBuilder::with_capacity(allocator, size);
        dict.register_onto(&mut buf);
        decoder.read(&mut buf, dict, size);
        extract_strings(buf)
    }

    #[test]
    fn test_rle_run() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC"]);
        let mut dec = new_decoder(vec![10, 2], 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 5);

        assert_eq!(result, vec!["CC"; 5]);
    }

    #[test]
    fn test_rle_incremental() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["X", "Y"]);
        let mut dec = new_decoder(vec![12, 1], 1);

        let r1 = push_all(&mut allocator, &mut dec, &dict, 2);
        let r2 = push_all(&mut allocator, &mut dec, &dict, 2);
        let r3 = push_all(&mut allocator, &mut dec, &dict, 2);

        assert_eq!(r1, vec!["Y"; 2]);
        assert_eq!(r2, vec!["Y"; 2]);
        assert_eq!(r3, vec!["Y"; 2]);
    }

    #[test]
    fn test_bitpacked_single_group() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![3, 0xE4, 0xE4], 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 8);

        assert_eq!(result, vec!["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn test_bitpacked_incremental() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![3, 0xE4, 0xE4], 2);

        let r1 = push_all(&mut allocator, &mut dec, &dict, 3);
        let r2 = push_all(&mut allocator, &mut dec, &dict, 5);

        assert_eq!(r1, vec!["AA", "BB", "CC"]);
        assert_eq!(r2, vec!["DD", "AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn test_rle_then_bitpacked() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![4, 0, 3, 0xE4, 0xE4], 2);

        let r1 = push_all(&mut allocator, &mut dec, &dict, 2);
        assert_eq!(r1, vec!["AA", "AA"]);

        let r2 = push_all(&mut allocator, &mut dec, &dict, 8);
        assert_eq!(r2, vec!["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn test_bitpacked_cross_buffer_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let data = make_data_multi_buffer(vec![vec![3], vec![0xE4, 0xE4]]);
        let mut dec = RleDecoder::new(data, ReaderPosition::default(), 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 8);

        assert_eq!(result, vec!["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
    }

    /// Skipping into the middle of a bit-packed run leaves the decoder at the
    /// right value, whether the skip ends inside a group or on a group edge,
    /// and whether it starts with unpacked values pending.
    #[test]
    fn skip_lands_inside_bitpacked_run() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        // Header 0x07 = 3 bit-packed groups of AA BB CC DD AA BB CC DD.
        let mut dec = new_decoder(vec![0x07, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4], 2);

        let head = push_all(&mut allocator, &mut dec, &dict, 3);
        dec.skip(11);
        let middle = push_all(&mut allocator, &mut dec, &dict, 2);
        dec.skip(4);
        let tail = push_all(&mut allocator, &mut dec, &dict, 4);

        assert_eq!(head, vec!["AA", "BB", "CC"]);
        assert_eq!(middle, vec!["CC", "DD"]);
        assert_eq!(tail, vec!["AA", "BB", "CC", "DD"]);
    }

    /// Skipping any number of values over a stream of mixed RLE and
    /// bit-packed runs, whole or split across buffers, leaves the decoder
    /// where decoding through them would.
    #[test]
    fn skip_over_many_runs_matches_decoding_through_them() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        // Repeats of: 1 bit-packed group (AA BB CC DD AA BB CC DD), then an
        // RLE run of 5 x CC.
        let segment = [0x03, 0xE4, 0xE4, 0x0A, 0x02];
        let stream: Vec<u8> = segment
            .iter()
            .copied()
            .cycle()
            .take(segment.len() * 20)
            .collect();
        let total = 13 * 20;
        let mut reference = new_decoder(stream.clone(), 2);
        let expected = push_all(&mut allocator, &mut reference, &dict, total);

        for split in [stream.len(), 7, 38] {
            for skip in [0, 7, 8, 13, 100, 101, 251] {
                let data = make_data_multi_buffer(vec![
                    stream[..split].to_vec(),
                    stream[split..].to_vec(),
                ]);
                let mut dec = RleDecoder::new(data, ReaderPosition::default(), 2);

                dec.skip(skip);
                let rest = push_all(&mut allocator, &mut dec, &dict, total - skip);

                assert_eq!(rest, expected[skip..], "split {split} skip {skip}");
            }
        }
    }

    /// Regression: after handling a buffer-boundary overflow, the decoder must
    /// not re-enter the overflow path on subsequent successful decodes.
    #[test]
    fn test_bitpacked_cross_buffer_with_remaining_groups() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        // Header 0x07 = 3 bit-packed groups (24 values). Split so group 1
        // straddles the buffer boundary, forcing an overflow. Groups 2-3
        // are fully in buffer 1.
        let data =
            make_data_multi_buffer(vec![vec![0x07, 0xE4], vec![0xE4, 0xE4, 0xE4, 0xE4, 0xE4]]);
        let mut dec = RleDecoder::new(data, ReaderPosition::default(), 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 24);

        let expected: Vec<&str> = ["AA", "BB", "CC", "DD"]
            .iter()
            .copied()
            .cycle()
            .take(24)
            .collect();
        assert_eq!(result, expected);
    }
}
