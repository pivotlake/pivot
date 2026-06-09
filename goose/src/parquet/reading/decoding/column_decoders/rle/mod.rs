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

use crate::parquet::reading::decoding::column_decoders::{ArrayBuilder, Dict};
use bytes::Bytes;
use dispatch::memory::{MultiBufferReader, ReaderPosition};

mod bit_pack_decoder;
pub use bit_pack_decoder::{BitDecoderOverflow, BitPackDecoder};

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
                        let dest = builder.spare_mut(count);
                        for i in 0..count {
                            dest[i] = dict.entry(scratch[i] as usize);
                        }
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
    use crate::parquet::reading::decoding::column_decoders::ArrayBuilder;
    use crate::parquet::reading::decoding::column_decoders::Dict;
    use crate::parquet::reading::decoding::column_decoders::bytes_view::dict::{
        DictFactory, ViewDict,
    };
    use crate::parquet::reading::decoding::column_decoders::bytes_view::views_builder::ViewsBuilder;
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

    fn make_dict(entries: &[&str]) -> ViewDict {
        let data = vec![Bytes::from(encode_plain_strings(entries))];
        DictFactory::new(data, entries.len()).create_dict()
    }

    fn make_data(data: Vec<u8>) -> Vec<Bytes> {
        vec![Bytes::from(data)]
    }

    fn make_data_multi_buffer(buffers: Vec<Vec<u8>>) -> Vec<Bytes> {
        buffers.into_iter().map(Bytes::from).collect()
    }

    fn extract_strings(builder: ViewsBuilder) -> Vec<String> {
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
        dict: &ViewDict,
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

    /// Regression: after handling a buffer-boundary overflow, the decoder must
    /// not re-enter the overflow path on subsequent successful decodes.
    /// With the old code the stale `overflowed` flag caused a spurious
    /// `push_from_bitpack_overflow` call that read past the end of the data.
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
