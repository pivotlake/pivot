//! [`TokenRanges`]: tokens in ascending order, each with a range of the index
//! file. What the sparse index and a dict index are both made of.
//!
//! ```text
//!   entries      SlabVec<(token end, file range)>     one per token, fixed size
//!   token bytes  SlabVec<u8>                          the tokens back to back
//!
//!   entries      │ 5, range │ 9, range │ 14, range │
//!   token bytes  │ a p p l e│ d a t e  │ m e l o n │
//!                0          5          9           14
//! ```
//!
//! A token's bytes run from the previous token's end to its own, wherever
//! the chunks of the token bytes happen to be cut, so a token may lie across
//! two chunks and is compared in place a chunk's part at a time. Finding a
//! token is a binary search over the entries, which also lie in as many
//! chunks as they need.
//!
//! Encoded, the tokens are a count and then one record per token, in order:
//!
//! ```text
//!   token count                                              varint
//!   per token:
//!     bytes shared with the start of the token before it     varint
//!     bytes after those                                      varint, then the bytes
//!     range offset, less where the range before it ended     signed varint
//!     range length                                           varint
//! ```
//!
//! Neighbouring tokens of a sorted run start alike, and ranges written one
//! after the other each start where the one before ended, so most records
//! are a few bytes.

use crate::io::FileRange;
use crate::memory::{FrozenSlab, Slab, SlabAllocator, SlabSource, SlabVec};
use crate::text_index::Error;
use crate::text_index::encoded::{EncodedReader, EncodedWriter};
use std::cmp::Ordering;

/// What is kept for one token.
#[derive(Clone, Copy)]
struct TokenEntry {
    /// Where the token ends in the token bytes. It starts where the token
    /// before it ends.
    token_end: usize,
    range: FileRange,
}

/// Tokens in ascending order, each with a range of the index file, all in
/// memory for as long as this is held.
pub(super) struct TokenRanges {
    entries: SlabVec<TokenEntry>,
    token_bytes: SlabVec<u8>,
}

/// What a cache keeps beside the slabs of tokens built into it: what putting
/// the two vecs back together over the slabs needs.
pub struct TokenRangesMeta {
    token_count: usize,
    token_bytes_len: usize,
    /// For each slab the build carved, in order: whether it is a chunk of the
    /// entries rather than of the token bytes.
    chunk_holds_entries: Vec<bool>,
}

impl TokenRanges {
    /// The tokens a build left in `slabs`, the slabs it carved in order.
    pub fn from_slabs(meta: &TokenRangesMeta, slabs: Vec<FrozenSlab>) -> Self {
        assert_eq!(
            slabs.len(),
            meta.chunk_holds_entries.len(),
            "the tokens were built into another number of slabs"
        );
        let (mut entry_chunks, mut token_byte_chunks) = (Vec::new(), Vec::new());
        for (slab, &holds_entries) in slabs.into_iter().zip(&meta.chunk_holds_entries) {
            if holds_entries {
                entry_chunks.push(slab);
            } else {
                token_byte_chunks.push(slab);
            }
        }
        Self {
            entries: SlabVec::from_chunks(entry_chunks, meta.token_count),
            token_bytes: SlabVec::from_chunks(token_byte_chunks, meta.token_bytes_len),
        }
    }

    /// The tokens in their encoded form, in slabs from `allocator`.
    pub fn encode(&self, allocator: &mut SlabAllocator) -> Vec<Slab> {
        let mut writer = EncodedWriter::new(allocator);
        writer.write_varint(self.entries.len() as u64);
        let (mut token, mut previous_token) = (Vec::new(), Vec::new());
        let (mut token_start, mut previous_range_end) = (0, 0);
        self.entries.for_each(|entry| {
            token.clear();
            token.extend(
                (token_start..entry.token_end).map(|position| self.token_bytes.get(position)),
            );
            let shared_len = token
                .iter()
                .zip(&previous_token)
                .take_while(|(byte, previous_byte)| byte == previous_byte)
                .count();
            writer.write_varint(shared_len as u64);
            writer.write_varint((token.len() - shared_len) as u64);
            writer.write_bytes(&token[shared_len..]);
            writer.write_signed_varint(entry.range.offset as i64 - previous_range_end as i64);
            writer.write_varint(entry.range.len as u64);

            token_start = entry.token_end;
            previous_range_end = entry.range.offset + entry.range.len;
            std::mem::swap(&mut token, &mut previous_token);
        });
        writer.into_slabs()
    }

    /// Gather the tokens `encoded` holds, with memory from `source`. Fails
    /// when `encoded` is not what [`encode`](Self::encode) writes.
    pub fn decode(
        encoded: &[Slab],
        source: &mut impl SlabSource,
    ) -> Result<TokenRangesBuilder, Error> {
        let mut reader = EncodedReader::new(encoded);
        let mut builder = TokenRangesBuilder::default();
        let token_count = reader.read_varint()?;
        let mut token = Vec::new();
        let mut previous_range_end = 0i64;
        for _ in 0..token_count {
            let shared_len = reader.read_len()?;
            if shared_len > token.len() {
                return Err(Error::Corrupt(
                    "a token shares more bytes than the token before it has",
                ));
            }
            token.truncate(shared_len);
            let rest_len = reader.read_len()?;
            reader.read_bytes_into(rest_len, &mut token)?;
            if builder.token_count > 0 && token <= builder.previous_token {
                return Err(Error::Corrupt("the tokens are not in ascending order"));
            }

            let offset = previous_range_end
                .checked_add(reader.read_signed_varint()?)
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or(Error::Corrupt("a range starts outside any file"))?;
            let len = reader.read_len()?;
            previous_range_end = offset
                .checked_add(len)
                .and_then(|end| i64::try_from(end).ok())
                .ok_or(Error::Corrupt("a range ends outside any file"))?;
            builder.push(source, &token, FileRange::new(offset, len));
        }
        reader.finish()?;
        Ok(builder)
    }

    pub fn count_tokens(&self) -> usize {
        self.entries.len()
    }

    /// The range of the last token that is not above `token`. `None` when
    /// `token` sorts before every token here.
    pub fn find_range_at_or_below(&self, token: &[u8]) -> Option<FileRange> {
        let index = self.count_tokens_at_or_below(token).checked_sub(1)?;
        Some(self.entries.get(index).range)
    }

    /// The range of `token` itself. `None` when it is not one of the tokens.
    pub fn find_range_of(&self, token: &[u8]) -> Option<FileRange> {
        let index = self.count_tokens_at_or_below(token).checked_sub(1)?;
        let entry = self.entries.get(index);
        (self.compare_token(index, entry, token) == Ordering::Equal).then_some(entry.range)
    }

    fn count_tokens_at_or_below(&self, token: &[u8]) -> usize {
        self.entries.find_partition_point(|index, entry| {
            self.compare_token(index, entry, token) != Ordering::Greater
        })
    }

    /// How the token of `entry`, the entry at `index`, compares to `token`.
    fn compare_token(&self, index: usize, entry: TokenEntry, token: &[u8]) -> Ordering {
        let token_start = match index {
            0 => 0,
            _ => self.entries.get(index - 1).token_end,
        };
        self.token_bytes
            .compare_range(token_start..entry.token_end, token)
    }
}

/// Gathers tokens, which must arrive in ascending order, into slabs.
#[derive(Default)]
pub(super) struct TokenRangesBuilder {
    entries: SlabVec<TokenEntry>,
    token_bytes: SlabVec<u8>,
    token_count: usize,
    token_bytes_len: usize,
    previous_token: Vec<u8>,
    chunk_holds_entries: Vec<bool>,
}

impl TokenRangesBuilder {
    /// Add `token`, which must sort after every token added so far, with its
    /// range. The memory comes from `source`.
    pub fn push(&mut self, source: &mut impl SlabSource, token: &[u8], range: FileRange) {
        assert!(
            self.token_count == 0 || self.previous_token.as_slice() < token,
            "tokens must be added in ascending order, each once"
        );
        let token_byte_chunks = self.token_bytes.count_chunks();
        for &byte in token {
            self.token_bytes.push(source, byte);
        }
        let carved_for_token_bytes = self.token_bytes.count_chunks() - token_byte_chunks;
        self.chunk_holds_entries
            .extend(std::iter::repeat_n(false, carved_for_token_bytes));
        self.token_bytes_len += token.len();

        let entry_chunks = self.entries.count_chunks();
        self.entries.push(
            source,
            TokenEntry {
                token_end: self.token_bytes_len,
                range,
            },
        );
        let carved_for_entries = self.entries.count_chunks() - entry_chunks;
        self.chunk_holds_entries
            .extend(std::iter::repeat_n(true, carved_for_entries));
        self.token_count += 1;

        self.previous_token.clear();
        self.previous_token.extend_from_slice(token);
    }

    /// The tokens, ready to be looked up where they were gathered.
    pub fn finish(self) -> TokenRanges {
        TokenRanges {
            entries: self.entries,
            token_bytes: self.token_bytes,
        }
    }

    /// What the cache whose fill the tokens were gathered into keeps beside
    /// them.
    pub fn into_meta(self) -> TokenRangesMeta {
        TokenRangesMeta {
            token_count: self.token_count,
            token_bytes_len: self.token_bytes_len,
            chunk_holds_entries: self.chunk_holds_entries,
        }
    }
}
