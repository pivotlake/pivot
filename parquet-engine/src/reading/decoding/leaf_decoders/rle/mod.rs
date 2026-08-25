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
use bit_pack_decoder::unpack8;
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
        scratch: &mut [u32; 1024],
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
                partial_count,
            } => {
                let (skipped, leftover) = decode_bitpacked(
                    remaining_in_run,
                    partial,
                    partial_count,
                    scratch,
                    data,
                    bit_width,
                    position,
                    amount,
                    |_, _| {},
                );
                (skipped, leftover)
            }
        }
    }
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
            let run = self.get_or_set_next_run();
            let (skipped, run) = run.skip(
                &mut self.buffer,
                &self.data,
                self.bit_width,
                &mut self.position,
                size,
            );
            self.run = run;
            size -= skipped;
        }
    }

    /// Decodes `size` values, looking up each index in `dict` and pushing
    /// the result into `builder`.
    ///
    /// A page held in one buffer (the common case) decodes through
    /// [`read_contiguous`](Self::read_contiguous); a page scattered across
    /// buffers, and any run left over from an earlier call, go through the
    /// general run-at-a-time path.
    pub fn read<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
        &mut self,
        builder: &mut B,
        dict: &D,
        size: usize,
    ) {
        let target = builder.len() + size;
        if let Some(run) = self.run.take() {
            self.run = run.read_into(
                builder,
                &mut self.buffer,
                &self.data,
                self.bit_width,
                &mut self.position,
                dict,
                target - builder.len(),
            );
        }
        if builder.len() < target && self.data.len() == 1 {
            self.read_contiguous(builder, dict, target);
        }
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

    /// Decodes from the single buffer holding this page until `builder` holds
    /// `target` values, with no pending run on entry.
    ///
    /// Sorted or clustered data encodes as a long alternation of short RLE runs
    /// and one- or two-group bit-packed runs, so the general path's per-run
    /// work (a reader over the buffer list, a run value passed in and out,
    /// a decoder set up per run, a chunk callback and two calls per chunk)
    /// outweighs the decoding itself. This loop reads headers straight off the
    /// slice, fills RLE runs in place, and unpacks bit-packed groups into the
    /// builder as it goes. Bit widths up to 16 (every low-cardinality
    /// dictionary) get a monomorphized group unpack; wider ones keep the bulk
    /// unpacker, whose long runs amortize its call.
    fn read_contiguous<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
        &mut self,
        builder: &mut B,
        dict: &D,
        target: usize,
    ) {
        macro_rules! specialized {
            ($($n:literal),*) => {
                match self.bit_width {
                    $($n => self.read_contiguous_with::<$n, I, B, D>(builder, dict, target),)*
                    _ => self.read_contiguous_with::<0, I, B, D>(builder, dict, target),
                }
            };
        }
        specialized!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    }

    /// [`read_contiguous`](Self::read_contiguous) for one bit width. `BW` is
    /// the width when the group unpack is specialized for it, and zero for the
    /// generic path (bit width zero, or wider than the specialized set).
    #[inline(never)]
    fn read_contiguous_with<
        const BW: usize,
        I,
        B: ArrayBuilder<Element = I>,
        D: Dict<Builder = B, Item = I>,
    >(
        &mut self,
        builder: &mut B,
        dict: &D,
        target: usize,
    ) {
        debug_assert!(self.run.is_none() && self.position.buffer_index == 0);
        let bit_width = self.bit_width as usize;
        let buf: &[u8] = self.data[0].as_ref();
        let mut pos = self.position.offset;
        let keys_in_range = (1u64 << bit_width) as usize <= dict.len();
        while builder.len() < target {
            let remaining = target - builder.len();
            let header = read_varint(buf, &mut pos);
            if header & 1 == 0 {
                let count = (header >> 1) as usize;
                let value = read_rle_value(buf, &mut pos, byte_width(self.bit_width));
                let emit = count.min(remaining);
                builder.push(&dict.entry(value as usize), emit);
                if emit < count {
                    self.run = Some(Run::Rle {
                        value,
                        count: count - emit,
                    });
                }
                continue;
            }
            let groups = (header >> 1) as usize;
            let values = groups * 8;
            let bytes = groups * bit_width;
            assert!(
                pos + bytes <= buf.len(),
                "bit-packed run of {groups} groups at {pos} runs past the page ({} bytes)",
                buf.len()
            );
            let take = values.min(remaining);
            let full_groups = take / 8;
            let (partial, partial_count) = if BW != 0 {
                let dest = builder.spare_mut(full_groups * 8);
                let mut group = [0u32; 8];
                for chunk in dest.chunks_exact_mut(8) {
                    // SAFETY: the whole run was checked to lie inside `buf`.
                    unsafe { unpack8::<BW>(buf, &mut pos, group.as_mut_ptr()) };
                    if !keys_in_range {
                        check_keys(&group, dict.len());
                    }
                    for (slot, &key) in chunk.iter_mut().zip(&group) {
                        // SAFETY: keys are below the dictionary length by the
                        // bit-width bound or the check above.
                        *slot = unsafe { dict.entry_unchecked(key as usize) };
                    }
                }
                let rest = take - full_groups * 8;
                if rest > 0 {
                    unsafe { unpack8::<BW>(buf, &mut pos, group.as_mut_ptr()) };
                    if !keys_in_range {
                        check_keys(&group, dict.len());
                    }
                    for (slot, &key) in builder.spare_mut(rest).iter_mut().zip(&group) {
                        *slot = unsafe { dict.entry_unchecked(key as usize) };
                    }
                    partial_from_group(&group, rest)
                } else {
                    ([0; 7], 0)
                }
            } else {
                let mut decoder = BitPackDecoder::new(buf, pos, self.bit_width, [0; 7], 0);
                let mut left = take;
                while left > 0 {
                    let chunk = left.min(self.buffer.len());
                    let count = decoder
                        .decode(&mut self.buffer[..chunk])
                        .expect("the run was checked to lie inside the buffer");
                    let keys = &self.buffer[..count];
                    if !keys_in_range {
                        check_keys(keys, dict.len());
                    }
                    let dest = builder.spare_mut(count);
                    // SAFETY: every key is below the dictionary length.
                    unsafe { gather_entries(dict, keys, dest) };
                    left -= count;
                }
                pos = decoder.pos();
                (*decoder.partial_values(), decoder.partial_count())
            };
            if take < values {
                self.run = Some(Run::BitPacked {
                    remaining_in_run: values - take,
                    partial,
                    partial_count,
                });
            }
        }
        self.position.offset = pos;
    }
}

/// Reads a ULEB128 varint from `buf` at `pos`, advancing past it.
#[inline(always)]
fn read_varint(buf: &[u8], pos: &mut usize) -> u32 {
    let first = buf[*pos];
    *pos += 1;
    if first & 0x80 == 0 {
        return first as u32;
    }
    let mut result = (first & 0x7F) as u32;
    let mut shift = 7;
    loop {
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return result;
        }
        shift += 7;
    }
}

/// Reads an RLE run's repeated value: `byte_width` little-endian bytes.
#[inline(always)]
fn read_rle_value(buf: &[u8], pos: &mut usize, byte_width: u8) -> u32 {
    let mut value = 0u32;
    for i in 0..byte_width as usize {
        value |= (buf[*pos + i] as u32) << (8 * i);
    }
    *pos += byte_width as usize;
    value
}

/// Panics when any key reaches `dict_len`, so the unchecked lookups that
/// follow stay in bounds for a dictionary smaller than its bit width allows.
#[inline(always)]
fn check_keys(keys: &[u32], dict_len: usize) {
    let max = keys.iter().copied().max().unwrap_or(0);
    assert!(
        (max as usize) < dict_len,
        "dictionary key {max} out of range ({dict_len} entries)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::ArrayBuilder;
    use crate::reading::decoding::leaf_decoders::Dict;
    use crate::reading::decoding::leaf_decoders::bytes_view::dict::{DictFactory, ViewDict};
    use crate::reading::decoding::leaf_decoders::bytes_view::views_builder::ViewsBuilder;
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, StringViewArray};
    use bytes::Bytes;
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

    fn extract_strings(builder: ViewsBuilder<StringViewType>) -> Vec<String> {
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
        let mut buf = ViewsBuilder::with_capacity(allocator, size);
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

    /// A fragmented stream (short RLE runs between one- and two-group
    /// bit-packed runs) decodes the same values whether it is read in one
    /// call or in pieces that end inside runs and groups.
    #[test]
    fn fragmented_stream_reads_the_same_in_any_piece_sizes() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        // RLE 5 x DD, 2 groups of AA BB CC DD .., RLE 3 x BB, 1 group.
        let stream = vec![10, 3, 0x05, 0xE4, 0xE4, 0xE4, 0xE4, 6, 1, 0x03, 0xE4, 0xE4];
        let mut whole = new_decoder(stream.clone(), 2);
        let mut pieces = new_decoder(stream, 2);

        let all = push_all(&mut allocator, &mut whole, &dict, 32);
        let mut in_pieces = Vec::new();
        for size in [2, 5, 4, 6, 3, 9, 3] {
            in_pieces.extend(push_all(&mut allocator, &mut pieces, &dict, size));
        }

        let mut expected: Vec<&str> = vec!["DD"; 5];
        expected.extend(["AA", "BB", "CC", "DD"].iter().copied().cycle().take(16));
        expected.extend(["BB"; 3]);
        expected.extend(["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
        assert_eq!(all, expected);
        assert_eq!(in_pieces, expected);
    }

    /// The single-buffer path agrees with the scattered-buffer path on random
    /// run streams at every bit width, including widths past the specialized
    /// set and dictionaries smaller than the width allows.
    #[test]
    fn contiguous_and_scattered_paths_agree() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        for bit_width in 1..=20u8 {
            let dict_len = ((1usize << bit_width) - next(3) as usize).max(1);
            let entries: Vec<String> = (0..dict_len).map(|i| format!("v{i}")).collect();
            let entry_refs: Vec<&str> = entries.iter().map(String::as_str).collect();
            let dict = make_dict(&entry_refs);
            let mut stream = Vec::new();
            let mut total = 0usize;
            for _ in 0..12 {
                if next(2) == 0 {
                    let count = 1 + next(20) as usize;
                    let value = next(dict_len as u64) as u32;
                    stream.push((count << 1) as u8);
                    stream.extend(&value.to_le_bytes()[..byte_width(bit_width) as usize]);
                    total += count;
                } else {
                    let groups = 1 + next(3) as usize;
                    stream.push(((groups << 1) | 1) as u8);
                    let mut bits = 0u128;
                    let mut filled = 0;
                    for _ in 0..groups * 8 {
                        bits |= (next(dict_len as u64) as u128) << filled;
                        filled += bit_width as usize;
                        while filled >= 8 {
                            stream.push(bits as u8);
                            bits >>= 8;
                            filled -= 8;
                        }
                    }
                    total += groups * 8;
                }
            }
            let split = stream.len() / 2;
            let scattered =
                make_data_multi_buffer(vec![stream[..split].to_vec(), stream[split..].to_vec()]);
            let mut contiguous = new_decoder(stream, bit_width);
            let mut reference = RleDecoder::new(scattered, ReaderPosition::default(), bit_width);

            let mut got = Vec::new();
            let mut want = Vec::new();
            let mut left = total;
            while left > 0 {
                let size = left.min(1 + next(13) as usize);
                got.extend(push_all(&mut allocator, &mut contiguous, &dict, size));
                want.extend(push_all(&mut allocator, &mut reference, &dict, size));
                left -= size;
            }
            assert_eq!(got, want, "bit width {bit_width}");
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
