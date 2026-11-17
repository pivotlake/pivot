//! [`SortedTokens`]: a run of tokens in ascending order, each found by binary
//! search. The sparse index and a dict index are both a run of sorted tokens
//! with something kept for each.
//!
//! ```text
//!   token ends    SlabVec<u32>   │ 5 │ 9 │ 14 │
//!   token bytes   SlabVec<u8>    │ a p p l e │ d a t e │ m e l o n │
//!                                0           5         9           14
//! ```
//!
//! The tokens lie back to back in the token bytes, and a token's bytes run
//! from where the token before it ends to its own end. A search probes the
//! token ends, four bytes a token, and compares the probed token's bytes in
//! place. Both vecs lie in as many chunks as they need: a token cut in two by
//! the end of a chunk is compared a part at a time.
//!
//! A token is known here by its index in the run. What a piece keeps for a
//! token (where a dict index is, where the token's rows are) is in a vec of
//! the piece's own, at the same index.
//!
//! Encoded, a token is written as what it adds to the token before it, since
//! neighbours in a sorted run start alike:
//!
//! ```text
//!   bytes shared with the start of the token before it     varint
//!   bytes after those                                      varint, then the bytes
//! ```

use crate::memory::{FrozenSlab, SlabSource, SlabVec};
use crate::text_index::Error;
use crate::text_index::carve_order::CarveOrder;
use crate::text_index::encoded::{EncodedReader, EncodedWriter};
use std::cmp::Ordering;

/// The sorted tokens' two vecs are vecs 0 and 1 in the carve order of the
/// piece they belong to. The piece numbers its own vecs from
/// [`VECS_OF_SORTED_TOKENS`] on.
const TOKEN_ENDS_VEC: u8 = 0;
const TOKEN_BYTES_VEC: u8 = 1;
pub(super) const VECS_OF_SORTED_TOKENS: u8 = 2;

/// A run of tokens in ascending order, all in memory for as long as this is
/// held.
pub(super) struct SortedTokens {
    /// Where each token ends in the token bytes.
    token_ends: SlabVec<u32>,
    token_bytes: SlabVec<u8>,
}

impl SortedTokens {
    /// The tokens over the chunks of their two vecs, as a build left them.
    pub fn from_chunks(
        token_ends: Vec<FrozenSlab>,
        token_bytes: Vec<FrozenSlab>,
        token_count: usize,
        token_bytes_len: usize,
    ) -> Self {
        Self {
            token_ends: SlabVec::from_chunks(token_ends, token_count),
            token_bytes: SlabVec::from_chunks(token_bytes, token_bytes_len),
        }
    }

    pub fn count_tokens(&self) -> usize {
        self.token_ends.len()
    }

    /// The bytes of every token together.
    pub fn count_token_bytes(&self) -> usize {
        self.token_bytes.len()
    }

    /// The index of the last token that is not above `token`. `None` when
    /// `token` sorts before every token of the run.
    pub fn find_last_at_or_below(&self, token: &[u8]) -> Option<usize> {
        let tokens_at_or_below = self
            .token_ends
            .find_partition_point(0..self.count_tokens(), |index, token_end| {
                self.compare_token(index, token_end, token) != Ordering::Greater
            });
        tokens_at_or_below.checked_sub(1)
    }

    /// The index of `token` itself. `None` when it is not in the run.
    pub fn find(&self, token: &[u8]) -> Option<usize> {
        let index = self.find_last_at_or_below(token)?;
        let token_end = self.token_ends.get(index);
        (self.compare_token(index, token_end, token) == Ordering::Equal).then_some(index)
    }

    /// How the token at `index`, which ends at `token_end`, compares to
    /// `token`.
    fn compare_token(&self, index: usize, token_end: u32, token: &[u8]) -> Ordering {
        let token_start = match index {
            0 => 0,
            _ => self.token_ends.get(index - 1),
        };
        self.token_bytes
            .compare_range(token_start as usize..token_end as usize, token)
    }

    /// Write every token, in order, in its encoded form. `write_kept` is
    /// called right after each token, with the token's index, to write what
    /// the piece keeps for it.
    pub fn encode_each(
        &self,
        writer: &mut EncodedWriter<'_>,
        mut write_kept: impl FnMut(&mut EncodedWriter<'_>, usize),
    ) {
        let (mut token, mut token_before) = (Vec::new(), Vec::new());
        let (mut token_start, mut index) = (0, 0);
        self.token_ends.for_each(|token_end| {
            token.clear();
            self.token_bytes
                .copy_range_into(token_start as usize..token_end as usize, &mut token);
            let shared_len = token
                .iter()
                .zip(&token_before)
                .take_while(|(byte, byte_before)| byte == byte_before)
                .count();
            writer.write_varint(shared_len as u64);
            writer.write_varint((token.len() - shared_len) as u64);
            writer.write_bytes(&token[shared_len..]);
            write_kept(writer, index);

            token_start = token_end;
            index += 1;
            std::mem::swap(&mut token, &mut token_before);
        });
    }
}

/// Gathers tokens, which must arrive in ascending order, into slabs.
pub(super) struct SortedTokensBuilder {
    token_ends: SlabVec<u32>,
    token_bytes: SlabVec<u8>,
    token_count: usize,
    token_bytes_len: usize,
    /// The most tokens, and the most bytes of tokens, the builder takes: what
    /// it was created for, or what a 32-bit token end can say.
    most_tokens: usize,
    most_token_bytes: usize,
    /// The token added last.
    token_before: Vec<u8>,
    /// While decoding a token: its bytes after those it shares with the
    /// token before it.
    rest_of_token: Vec<u8>,
}

impl Default for SortedTokensBuilder {
    fn default() -> Self {
        Self {
            token_ends: SlabVec::new(),
            token_bytes: SlabVec::new(),
            token_count: 0,
            token_bytes_len: 0,
            most_tokens: usize::MAX,
            most_token_bytes: u32::MAX as usize,
            token_before: Vec::new(),
            rest_of_token: Vec::new(),
        }
    }
}

impl SortedTokensBuilder {
    /// A builder for exactly `token_count` tokens of `token_bytes_len` bytes
    /// together, which then carves no more memory than they take.
    pub fn with_exact_lens(token_count: usize, token_bytes_len: usize) -> Self {
        Self {
            token_ends: SlabVec::with_exact_len(token_count),
            token_bytes: SlabVec::with_exact_len(token_bytes_len),
            most_tokens: token_count,
            most_token_bytes: token_bytes_len.min(u32::MAX as usize),
            ..Self::default()
        }
    }

    pub fn count_tokens(&self) -> usize {
        self.token_count
    }

    /// The bytes of every token added so far together.
    pub fn count_token_bytes(&self) -> usize {
        self.token_bytes_len
    }

    /// Whether one more token, of `token_len` bytes, still fits.
    fn has_room_for(&self, token_len: usize) -> bool {
        self.token_count < self.most_tokens
            && token_len <= self.most_token_bytes - self.token_bytes_len
    }

    /// Add `token`, which must sort after every token added so far. The
    /// memory comes from `source`, and `carve_order` is that of the piece the
    /// tokens belong to.
    pub fn push(
        &mut self,
        source: &mut impl SlabSource,
        carve_order: &mut CarveOrder,
        token: &[u8],
    ) {
        assert!(
            self.token_count == 0 || self.token_before.as_slice() < token,
            "tokens must be added in ascending order, each once"
        );
        assert!(
            self.has_room_for(token.len()),
            "more tokens than the run can hold"
        );
        carve_order.extend_from_slice(TOKEN_BYTES_VEC, &mut self.token_bytes, source, token);
        self.token_bytes_len += token.len();
        carve_order.push(
            TOKEN_ENDS_VEC,
            &mut self.token_ends,
            source,
            self.token_bytes_len as u32,
        );
        self.token_count += 1;
        self.token_before.clear();
        self.token_before.extend_from_slice(token);
    }

    /// Read the next token of an encoded run from `reader` and add it. Fails,
    /// adding nothing, when what is read is not a token that may follow the
    /// ones added so far.
    pub fn decode_token(
        &mut self,
        reader: &mut EncodedReader<'_>,
        source: &mut impl SlabSource,
        carve_order: &mut CarveOrder,
    ) -> Result<(), Error> {
        let shared_len = reader.read_len()?;
        if shared_len > self.token_before.len() {
            return Err(Error::Corrupt(
                "a token shares more bytes than the token before it has",
            ));
        }
        let rest_len = reader.read_len()?;
        self.rest_of_token.clear();
        reader.read_bytes_into(rest_len, &mut self.rest_of_token)?;
        // The two tokens start with the same `shared_len` bytes, so they
        // compare as what follows those does.
        let rest_of_token_before = &self.token_before[shared_len..];
        if self.token_count > 0 && self.rest_of_token.as_slice() <= rest_of_token_before {
            return Err(Error::Corrupt("the tokens are not in ascending order"));
        }
        if !self.has_room_for(shared_len + rest_len) {
            return Err(Error::Corrupt("there are more tokens than it says"));
        }

        self.token_before.truncate(shared_len);
        self.token_before.extend_from_slice(&self.rest_of_token);
        carve_order.extend_from_slice(
            TOKEN_BYTES_VEC,
            &mut self.token_bytes,
            source,
            &self.token_before,
        );
        self.token_bytes_len += self.token_before.len();
        carve_order.push(
            TOKEN_ENDS_VEC,
            &mut self.token_ends,
            source,
            self.token_bytes_len as u32,
        );
        self.token_count += 1;
        Ok(())
    }

    /// The tokens, ready to be looked up where they were gathered.
    pub fn finish(self) -> SortedTokens {
        SortedTokens {
            token_ends: self.token_ends,
            token_bytes: self.token_bytes,
        }
    }
}
