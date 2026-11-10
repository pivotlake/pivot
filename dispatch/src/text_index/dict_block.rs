//! Dictionary blocks: the index's tokens in sorted order, each with where its
//! rows are.
//!
//! The dictionary is cut into blocks of [`TOKENS_PER_BLOCK`] tokens so a
//! lookup reads one block instead of the whole dictionary. A block exists in
//! two forms.
//!
//! In the file it is compact. Neighbouring sorted tokens share prefixes, so
//! each token stores only how much of the previous one it repeats:
//!
//! ```text
//!   token count u32
//!   per token:  shared prefix len │ suffix len │ suffix │ row count │ rows
//!               varint            │ varint     │ bytes  │ varint    │
//!
//!   rows, for a token in at most MAX_INLINE_ROWS rows:  the rows themselves, as varint deltas
//!   rows, otherwise:                                    posting list offset │ len   (varints)
//! ```
//!
//! In the cache ([`DictBlock`]) it is laid out for binary search, in one
//! aligned slab:
//!
//! ```text
//!   ┌────────┬──────────────────────┬──────────────┬─────────────┬─────────────┐
//!   │ header │ entries              │ token ends   │ inline rows │ token bytes │
//!   │ 16 B   │ 16 B × tokens        │ u32 × tokens │ u32 × n     │             │
//!   └────────┴──────────────────────┴──────────────┴─────────────┴─────────────┘
//!   entry = (location u64, len u32, row count u32)
//!   location is a posting list's file offset, or the first inline row's index
//! ```

use crate::io::FileRange;
use crate::memory::cache::Cached;
use crate::memory::{FrozenSlab, SlabSource};
use crate::text_index::Error;
use crate::text_index::format::{
    ByteSink, read_u32, read_u64, read_varint, take_bytes, write_varint,
};

/// Tokens in every dictionary block but the last.
pub const TOKENS_PER_BLOCK: usize = 1024;
/// A token in at most this many rows keeps them in its dictionary entry, so
/// finding them costs no read beyond the block.
pub const MAX_INLINE_ROWS: usize = 8;

const IMAGE_HEADER_LEN: usize = 16;
const IMAGE_ENTRY_LEN: usize = 16;

/// Where in the index file a token's posting list is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PostingsLocation {
    pub offset: u64,
    pub len: u32,
    pub row_count: u32,
}

impl PostingsLocation {
    pub fn range(&self) -> FileRange {
        FileRange::new(self.offset as usize, self.len as usize)
    }
}

/// The rows of a token being written to a block.
pub enum TokenRows<'a> {
    /// At most [`MAX_INLINE_ROWS`] rows, ascending, kept in the block.
    Inline(&'a [u32]),
    List(PostingsLocation),
}

/// Accumulates one block's tokens, which must arrive in ascending order, and
/// writes its file form.
#[derive(Default)]
pub struct DictBlockEncoder {
    token_count: usize,
    first_token: Vec<u8>,
    previous_token: Vec<u8>,
    entries: Vec<u8>,
}

impl DictBlockEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// The block's smallest token. Meaningful once a token has been pushed.
    pub fn first_token(&self) -> &[u8] {
        &self.first_token
    }

    pub fn push(&mut self, token: &[u8], rows: TokenRows<'_>) {
        debug_assert!(self.token_count == 0 || self.previous_token.as_slice() < token);
        if self.token_count == 0 {
            self.first_token.clear();
            self.first_token.extend_from_slice(token);
        }
        let shared_len = self
            .previous_token
            .iter()
            .zip(token)
            .take_while(|(previous, current)| previous == current)
            .count();
        write_varint(&mut self.entries, shared_len as u64);
        write_varint(&mut self.entries, (token.len() - shared_len) as u64);
        self.entries.write_bytes(&token[shared_len..]);
        match rows {
            TokenRows::Inline(rows) => {
                assert!(
                    (1..=MAX_INLINE_ROWS).contains(&rows.len()),
                    "{} rows cannot be inlined",
                    rows.len()
                );
                write_varint(&mut self.entries, rows.len() as u64);
                let mut previous_row = 0;
                for &row in rows {
                    write_varint(&mut self.entries, (row - previous_row) as u64);
                    previous_row = row;
                }
            }
            TokenRows::List(location) => {
                assert!(
                    location.row_count as usize > MAX_INLINE_ROWS,
                    "a {} row token must be inlined",
                    location.row_count
                );
                write_varint(&mut self.entries, location.row_count as u64);
                write_varint(&mut self.entries, location.offset);
                write_varint(&mut self.entries, location.len as u64);
            }
        }
        self.previous_token.clear();
        self.previous_token.extend_from_slice(token);
        self.token_count += 1;
    }

    /// Append the block's file form to `sink` and start an empty block.
    pub fn finish(&mut self, sink: &mut impl ByteSink) {
        sink.write_bytes(&(self.token_count as u32).to_le_bytes());
        sink.write_bytes(&self.entries);
        self.token_count = 0;
        self.previous_token.clear();
        self.entries.clear();
    }
}

/// A dictionary block laid out for searching, pinned in the cache for as long
/// as this is held.
pub struct DictBlock {
    image: FrozenSlab,
}

/// What the cache keeps beside a block's bytes: nothing, the bytes describe
/// themselves.
pub struct DictBlockMeta;

impl Cached for DictBlock {
    type Meta = DictBlockMeta;

    fn from_slabs(_meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        let image = slabs
            .into_iter()
            .next()
            .expect("a dictionary block is one slab");
        Self { image }
    }
}

/// What a block says about one of its tokens.
#[derive(Clone, Copy, Debug)]
pub enum TokenEntry<'a> {
    /// The token's rows, ascending.
    Inline(InlineRows<'a>),
    List(PostingsLocation),
}

/// The rows a dictionary entry holds itself, ascending.
#[derive(Clone, Copy, Debug)]
pub struct InlineRows<'a> {
    encoded: &'a [u8],
}

impl InlineRows<'_> {
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.encoded
            .as_chunks::<4>()
            .0
            .iter()
            .map(|row| u32::from_le_bytes(*row))
    }
}

impl DictBlock {
    /// Decode the block's file form `encoded` into one slab carved from
    /// `slabs`.
    pub fn build(encoded: &[u8], slabs: &mut impl SlabSource) -> Result<DictBlockMeta, Error> {
        let image = decode_image(encoded)?;
        let mut slab = slabs.get_aligned_slab(image.len(), 8, false);
        slab.as_mut_slice().copy_from_slice(&image);
        Ok(DictBlockMeta)
    }

    pub fn token_count(&self) -> usize {
        read_u32(self.image.as_slice(), 0) as usize
    }

    /// What the block holds for `token`, if it holds the token at all.
    pub fn find(&self, token: &[u8]) -> Option<TokenEntry<'_>> {
        let (mut low, mut high) = (0, self.token_count());
        while low < high {
            let middle = low + (high - low) / 2;
            match self.token(middle).cmp(token) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return Some(self.entry(middle)),
            }
        }
        None
    }

    fn inline_row_count(&self) -> usize {
        read_u32(self.image.as_slice(), 4) as usize
    }

    fn token_ends_offset(&self) -> usize {
        IMAGE_HEADER_LEN + IMAGE_ENTRY_LEN * self.token_count()
    }

    fn inline_rows_offset(&self) -> usize {
        self.token_ends_offset() + 4 * self.token_count()
    }

    fn token_bytes_offset(&self) -> usize {
        self.inline_rows_offset() + 4 * self.inline_row_count()
    }

    fn token_end(&self, index: usize) -> usize {
        read_u32(self.image.as_slice(), self.token_ends_offset() + 4 * index) as usize
    }

    fn token(&self, index: usize) -> &[u8] {
        let start = if index == 0 {
            0
        } else {
            self.token_end(index - 1)
        };
        let bytes = &self.image.as_slice()[self.token_bytes_offset()..];
        &bytes[start..self.token_end(index)]
    }

    fn entry(&self, index: usize) -> TokenEntry<'_> {
        let image = self.image.as_slice();
        let entry_offset = IMAGE_HEADER_LEN + IMAGE_ENTRY_LEN * index;
        let location = read_u64(image, entry_offset);
        let len = read_u32(image, entry_offset + 8);
        let row_count = read_u32(image, entry_offset + 12);
        if row_count as usize > MAX_INLINE_ROWS {
            return TokenEntry::List(PostingsLocation {
                offset: location,
                len,
                row_count,
            });
        }
        let first_row = self.inline_rows_offset() + 4 * location as usize;
        TokenEntry::Inline(InlineRows {
            encoded: &image[first_row..first_row + 4 * row_count as usize],
        })
    }
}

/// Decode a block's file form into the bytes of its searchable layout.
fn decode_image(encoded: &[u8]) -> Result<Vec<u8>, Error> {
    let mut rest = encoded;
    let token_count = read_u32(take_bytes(&mut rest, 4)?, 0) as usize;
    let mut entries = Vec::with_capacity(IMAGE_ENTRY_LEN * token_count);
    let mut token_ends = Vec::with_capacity(4 * token_count);
    let mut inline_rows = Vec::new();
    let mut token_bytes: Vec<u8> = Vec::with_capacity(encoded.len());
    let mut previous_token_start = 0;

    for _ in 0..token_count {
        let shared_len = read_varint(&mut rest)? as usize;
        let suffix_len = read_varint(&mut rest)? as usize;
        let suffix = take_bytes(&mut rest, suffix_len)?;
        let token_start = token_bytes.len();
        if shared_len > token_start - previous_token_start {
            return Err(Error::corrupt("token shares more than the previous token"));
        }
        token_bytes.extend_from_within(previous_token_start..previous_token_start + shared_len);
        token_bytes.extend_from_slice(suffix);
        token_ends.extend_from_slice(&(token_bytes.len() as u32).to_le_bytes());
        previous_token_start = token_start;

        let row_count = read_varint(&mut rest)?;
        if row_count == 0 || row_count > u32::MAX as u64 {
            return Err(Error::corrupt("token row count is invalid"));
        }
        let (location, len) = if row_count as usize > MAX_INLINE_ROWS {
            let offset = read_varint(&mut rest)?;
            let len = read_varint(&mut rest)?;
            if len > u32::MAX as u64 {
                return Err(Error::corrupt("posting list length is invalid"));
            }
            (offset, len as u32)
        } else {
            let first_inline_row = inline_rows.len() / 4;
            let mut row = 0u64;
            for _ in 0..row_count {
                row += read_varint(&mut rest)?;
                if row > u32::MAX as u64 {
                    return Err(Error::corrupt("inline row is out of range"));
                }
                inline_rows.extend_from_slice(&(row as u32).to_le_bytes());
            }
            (first_inline_row as u64, 0)
        };
        entries.extend_from_slice(&location.to_le_bytes());
        entries.extend_from_slice(&len.to_le_bytes());
        entries.extend_from_slice(&(row_count as u32).to_le_bytes());
    }
    if !rest.is_empty() {
        return Err(Error::corrupt("dictionary block has trailing bytes"));
    }

    let mut image = Vec::with_capacity(
        IMAGE_HEADER_LEN + entries.len() + token_ends.len() + inline_rows.len() + token_bytes.len(),
    );
    image.extend_from_slice(&(token_count as u32).to_le_bytes());
    image.extend_from_slice(&((inline_rows.len() / 4) as u32).to_le_bytes());
    image.extend_from_slice(&(token_bytes.len() as u32).to_le_bytes());
    image.extend_from_slice(&0u32.to_le_bytes());
    image.extend_from_slice(&entries);
    image.extend_from_slice(&token_ends);
    image.extend_from_slice(&inline_rows);
    image.extend_from_slice(&token_bytes);
    Ok(image)
}
