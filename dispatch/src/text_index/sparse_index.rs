//! The sparse index: for a token, which dict index it would be in.
//!
//! One entry per dict index of the file: the smallest token that dict index
//! holds, and where the dict index is. The dict indexes hold the file's
//! tokens in ascending order, so a token belongs to the last dict index whose
//! smallest token is not above it:
//!
//! ```text
//!   smallest token   dict index at
//!   ──────────────   ─────────────
//!   "apple"          0 .. 4000
//!   "date"           4000 .. 9000      ◀── "kiwi": "date" ≤ "kiwi" < "melon"
//!   "melon"          9000 .. 12000
//! ```
//!
//! In memory it is the smallest tokens as [`SortedTokens`] and, at each
//! token's index, the range of its dict index: 16 bytes a dict index beside
//! the token's own bytes.
//!
//! Encoded:
//!
//! ```text
//!   dict index count                                        varint
//!   bytes of all the smallest tokens together               varint
//!   per dict index:
//!     its smallest token                                    as sorted tokens write one
//!     where it starts, less where the one before it ended   signed varint
//!     its length                                            varint
//! ```
//!
//! Dict indexes written one after the other each start where the one before
//! ended, so the start usually takes one byte.

use crate::io::FileRange;
use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, Slab, SlabAllocator, SlabSource, SlabVec};
use crate::text_index::Error;
use crate::text_index::carve_order::CarveOrder;
use crate::text_index::encoded::{EncodedReader, EncodedWriter};
use crate::text_index::sorted_tokens::{SortedTokens, SortedTokensBuilder, VECS_OF_SORTED_TOKENS};

/// The number, in the sparse index's carve order, of the vec of dict index
/// ranges.
const DICT_INDEX_RANGES_VEC: u8 = VECS_OF_SORTED_TOKENS;

/// Where one dict index is in the index file, in 12 bytes: a dict index is
/// never 4 GiB long, so its length fits 32 bits.
#[derive(Clone, Copy)]
#[repr(C, packed(4))]
struct DictIndexRange {
    offset: u64,
    len: u32,
}

impl DictIndexRange {
    fn to_file_range(self) -> FileRange {
        FileRange::new(self.offset as usize, self.len as usize)
    }
}

/// An index file's sparse index, all of it in memory for as long as this is
/// held.
pub struct SparseIndex {
    smallest_tokens: SortedTokens,
    /// At the index of each smallest token, where its dict index is.
    dict_index_ranges: SlabVec<DictIndexRange>,
}

/// What a cache keeps beside the slabs of a sparse index built into it.
pub struct SparseIndexMeta {
    dict_index_count: usize,
    token_bytes_len: usize,
    carve_order: CarveOrder,
}

impl SparseIndex {
    /// How many dict indexes the file has.
    pub fn count_dict_indexes(&self) -> usize {
        self.smallest_tokens.count_tokens()
    }

    /// Where the dict index is that holds `token` if the file holds it at
    /// all. `None` when `token` sorts before every token of the file.
    pub fn find_dict_index(&self, token: &[u8]) -> Option<FileRange> {
        let index = self.smallest_tokens.find_last_at_or_below(token)?;
        Some(self.dict_index_ranges.get(index).to_file_range())
    }

    /// The sparse index in its encoded form, in slabs from `allocator`.
    pub fn encode(&self, allocator: &mut SlabAllocator) -> Vec<Slab> {
        let mut writer = EncodedWriter::new(allocator);
        writer.write_varint(self.smallest_tokens.count_tokens() as u64);
        writer.write_varint(self.smallest_tokens.count_token_bytes() as u64);
        let mut end_of_range_before = 0;
        self.smallest_tokens
            .encode_each(&mut writer, |writer, index| {
                let range = self.dict_index_ranges.get(index).to_file_range();
                writer.write_signed_varint(range.offset as i64 - end_of_range_before as i64);
                writer.write_varint(range.len as u64);
                end_of_range_before = range.offset + range.len;
            });
        writer.into_slabs()
    }

    /// Gather the sparse index `encoded` holds, with memory from `source`.
    /// `encoded` is the bytes [`encode`](Self::encode) writes, in whatever
    /// pieces they come: the slabs it wrote, or the buffers of a read. Fails
    /// when they are anything else.
    pub fn decode(
        encoded: &[impl AsRef<[u8]>],
        source: &mut impl SlabSource,
    ) -> Result<SparseIndexBuilder, Error> {
        let mut reader = EncodedReader::new(encoded);
        let dict_index_count = reader.read_len()?;
        let token_bytes_len = reader.read_len()?;
        let mut builder = SparseIndexBuilder {
            smallest_tokens: SortedTokensBuilder::with_exact_lens(
                dict_index_count,
                token_bytes_len,
            ),
            dict_index_ranges: SlabVec::with_exact_len(dict_index_count),
            carve_order: CarveOrder::default(),
        };
        let mut end_of_range_before = 0i64;
        for _ in 0..dict_index_count {
            builder
                .smallest_tokens
                .decode_token(&mut reader, source, &mut builder.carve_order)?;
            let offset = end_of_range_before
                .checked_add(reader.read_signed_varint()?)
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or(Error::Corrupt("a dict index starts outside any file"))?;
            let len = u32::try_from(reader.read_varint()?)
                .map_err(|_| Error::Corrupt("a dict index is longer than 4 GiB"))?;
            end_of_range_before = offset
                .checked_add(len as u64)
                .and_then(|end| i64::try_from(end).ok())
                .ok_or(Error::Corrupt("a dict index ends outside any file"))?;
            builder.carve_order.push(
                DICT_INDEX_RANGES_VEC,
                &mut builder.dict_index_ranges,
                source,
                DictIndexRange { offset, len },
            );
        }
        if builder.smallest_tokens.count_token_bytes() != token_bytes_len {
            return Err(Error::Corrupt(
                "the tokens take another number of bytes than it says",
            ));
        }
        reader.finish()?;
        Ok(builder)
    }
}

impl Cached for SparseIndex {
    type Meta = SparseIndexMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        let [token_ends, token_bytes, dict_index_ranges] = meta.carve_order.deal_slabs(slabs);
        Self {
            smallest_tokens: SortedTokens::from_chunks(
                token_ends,
                token_bytes,
                meta.dict_index_count,
                meta.token_bytes_len,
            ),
            dict_index_ranges: SlabVec::from_chunks(dict_index_ranges, meta.dict_index_count),
        }
    }
}

/// Gathers a file's dict indexes, in the order they hold the tokens, into a
/// sparse index.
#[derive(Default)]
pub struct SparseIndexBuilder {
    smallest_tokens: SortedTokensBuilder,
    dict_index_ranges: SlabVec<DictIndexRange>,
    carve_order: CarveOrder,
}

impl SparseIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the next dict index: its smallest token, which must sort after the
    /// one added before, and where it is. The memory comes from `source`.
    pub fn push(
        &mut self,
        source: &mut impl SlabSource,
        smallest_token: &[u8],
        dict_index: FileRange,
    ) {
        let len = u32::try_from(dict_index.len).expect("a dict index is shorter than 4 GiB");
        self.smallest_tokens
            .push(source, &mut self.carve_order, smallest_token);
        self.carve_order.push(
            DICT_INDEX_RANGES_VEC,
            &mut self.dict_index_ranges,
            source,
            DictIndexRange {
                offset: dict_index.offset as u64,
                len,
            },
        );
    }

    /// The sparse index, ready to be looked up where it was gathered.
    pub fn finish(self) -> SparseIndex {
        SparseIndex {
            smallest_tokens: self.smallest_tokens.finish(),
            dict_index_ranges: self.dict_index_ranges,
        }
    }

    /// What the cache whose fill the sparse index was gathered into keeps
    /// beside it.
    pub fn into_meta(self) -> SparseIndexMeta {
        SparseIndexMeta {
            dict_index_count: self.smallest_tokens.count_tokens(),
            token_bytes_len: self.smallest_tokens.count_token_bytes(),
            carve_order: self.carve_order,
        }
    }
}
