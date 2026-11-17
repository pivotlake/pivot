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
//! slab:
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
use crate::memory::cache::{CacheFill, Cached};
use crate::memory::{BUFFER_SIZE, FrozenSlab, SlabAllocator};
use crate::text_index::Error;
use crate::text_index::format::{ByteSink, FragmentReader, read_u32, read_u64, write_varint};
use crate::text_index::slab_stream::SlabStream;
use crate::text_index::tokenizer::{MAX_INDEXED_TOKEN_LEN, TokenBuffer};
use bytes::Bytes;

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
    first_token: TokenBuffer,
    previous_token: TokenBuffer,
    /// The tokens' entries in file form, held on the ring until the block is
    /// written: the posting lists of a block's tokens go into the file before
    /// the block does.
    entries: SlabStream,
}

impl DictBlockEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// The smallest token of the block being gathered or, right after
    /// [`finish`](Self::finish), of the block just written.
    pub fn first_token(&self) -> &[u8] {
        self.first_token.as_slice()
    }

    /// Add `token` to the block, holding its entry in a chunk from
    /// `allocator`.
    pub fn push(&mut self, allocator: &mut SlabAllocator, token: &[u8], rows: TokenRows<'_>) {
        debug_assert!(self.token_count == 0 || self.previous_token.as_slice() < token);
        if self.token_count == 0 {
            self.first_token.set(token);
        }
        let shared_len = self
            .previous_token
            .as_slice()
            .iter()
            .zip(token)
            .take_while(|(previous, current)| previous == current)
            .count();
        let mut entries = self.entries.write_with(allocator);
        write_varint(&mut entries, shared_len as u64);
        write_varint(&mut entries, (token.len() - shared_len) as u64);
        entries.write_bytes(&token[shared_len..]);
        match rows {
            TokenRows::Inline(rows) => {
                assert!(
                    (1..=MAX_INLINE_ROWS).contains(&rows.len()),
                    "{} rows cannot be inlined",
                    rows.len()
                );
                write_varint(&mut entries, rows.len() as u64);
                let mut previous_row = 0;
                for &row in rows {
                    write_varint(&mut entries, (row - previous_row) as u64);
                    previous_row = row;
                }
            }
            TokenRows::List(location) => {
                assert!(
                    location.row_count as usize > MAX_INLINE_ROWS,
                    "a {} row token must be inlined",
                    location.row_count
                );
                write_varint(&mut entries, location.row_count as u64);
                write_varint(&mut entries, location.offset);
                write_varint(&mut entries, location.len as u64);
            }
        }
        self.previous_token.set(token);
        self.token_count += 1;
    }

    /// Append the block's file form to `sink` and start an empty block.
    pub fn finish(&mut self, sink: &mut impl ByteSink) {
        sink.write_bytes(&(self.token_count as u32).to_le_bytes());
        for chunk in self.entries.written_chunks() {
            sink.write_bytes(chunk);
        }
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

    /// The rows, copied out of the block so they can outlive its pin.
    pub fn copy_rows(&self) -> CopiedInlineRows {
        let mut copied = CopiedInlineRows {
            rows: [0; MAX_INLINE_ROWS],
            row_count: 0,
        };
        for row in self.iter() {
            copied.rows[copied.row_count] = row;
            copied.row_count += 1;
        }
        copied
    }
}

/// The rows of a dictionary entry, held in place: there are at most
/// [`MAX_INLINE_ROWS`] of them.
#[derive(Clone, Copy, Debug)]
pub struct CopiedInlineRows {
    rows: [u32; MAX_INLINE_ROWS],
    row_count: usize,
}

impl CopiedInlineRows {
    /// The rows, ascending.
    pub fn as_slice(&self) -> &[u32] {
        &self.rows[..self.row_count]
    }
}

impl DictBlock {
    /// Decode the block's file form, which `encoded` holds (the fragments of
    /// one read, in order), into one slab carved from `fill`.
    ///
    /// The file form is read twice: once to check it and learn how large each
    /// part of the searchable layout is, and once to write the parts straight
    /// into the slab, so the block is never held anywhere else.
    pub fn build(encoded: &[Bytes], fill: &mut CacheFill<'_>) -> Result<DictBlockMeta, Error> {
        let mut rest = FragmentReader::new(encoded);
        let mut token_count = [0u8; 4];
        rest.read_exact(&mut token_count)?;
        let token_count = u32::from_le_bytes(token_count) as usize;

        let mut unmeasured = rest;
        let (mut inline_row_count, mut token_bytes_len, mut previous_token_len) = (0, 0, 0);
        for _ in 0..token_count {
            let token = EncodedToken::read(&mut unmeasured)?;
            if token.shared_len > previous_token_len {
                return Err(Error::corrupt("token shares more than the previous token"));
            }
            previous_token_len = token.shared_len + token.suffix_len;
            token_bytes_len += previous_token_len;
            if let EncodedRows::Inline(_) = token.rows {
                inline_row_count += token.row_count as usize;
            }
        }
        if unmeasured.remaining() != 0 {
            return Err(Error::corrupt("dictionary block has trailing bytes"));
        }
        let entries_len = IMAGE_ENTRY_LEN * token_count;
        let token_ends_len = 4 * token_count;
        let inline_rows_len = 4 * inline_row_count;
        let image_len =
            IMAGE_HEADER_LEN + entries_len + token_ends_len + inline_rows_len + token_bytes_len;
        if image_len > BUFFER_SIZE {
            return Err(Error::corrupt(
                "dictionary block does not fit one ring slot",
            ));
        }

        let mut slab = fill.get_contiguous_bytes(image_len);
        let (header, image) = slab.as_mut_slice().split_at_mut(IMAGE_HEADER_LEN);
        let (entries, image) = image.split_at_mut(entries_len);
        let (token_ends, image) = image.split_at_mut(token_ends_len);
        let (inline_rows, token_bytes) = image.split_at_mut(inline_rows_len);
        header[0..4].copy_from_slice(&(token_count as u32).to_le_bytes());
        header[4..8].copy_from_slice(&(inline_row_count as u32).to_le_bytes());
        header[8..12].copy_from_slice(&(token_bytes_len as u32).to_le_bytes());
        header[12..16].fill(0);

        let (mut inline_rows_written, mut token_start, mut token_end) = (0, 0, 0);
        for index in 0..token_count {
            let token = EncodedToken::read(&mut rest)?;
            let previous_token_start = token_start;
            token_start = token_end;
            token_bytes.copy_within(
                previous_token_start..previous_token_start + token.shared_len,
                token_start,
            );
            token_end = token_start + token.shared_len + token.suffix_len;
            token_bytes[token_start + token.shared_len..token_end]
                .copy_from_slice(&token.suffix[..token.suffix_len]);
            token_ends[4 * index..4 * index + 4].copy_from_slice(&(token_end as u32).to_le_bytes());

            let (location, len) = match token.rows {
                EncodedRows::List { offset, len } => (offset, len),
                EncodedRows::Inline(rows) => {
                    let first_inline_row = inline_rows_written;
                    for row in &rows[..token.row_count as usize] {
                        inline_rows[4 * inline_rows_written..4 * inline_rows_written + 4]
                            .copy_from_slice(&row.to_le_bytes());
                        inline_rows_written += 1;
                    }
                    (first_inline_row as u64, 0)
                }
            };
            let entry = &mut entries[IMAGE_ENTRY_LEN * index..IMAGE_ENTRY_LEN * (index + 1)];
            entry[0..8].copy_from_slice(&location.to_le_bytes());
            entry[8..12].copy_from_slice(&len.to_le_bytes());
            entry[12..16].copy_from_slice(&token.row_count.to_le_bytes());
        }
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

/// One token of a block's file form.
struct EncodedToken {
    /// How many bytes of the token before it this token starts with.
    shared_len: usize,
    /// The rest of the token, in the first `suffix_len` places.
    suffix: [u8; MAX_INDEXED_TOKEN_LEN],
    suffix_len: usize,
    row_count: u32,
    rows: EncodedRows,
}

enum EncodedRows {
    /// The token's rows, in the first `row_count` places.
    Inline([u32; MAX_INLINE_ROWS]),
    /// Where the token's posting list is in the index file.
    List { offset: u64, len: u32 },
}

impl EncodedToken {
    /// Read the next token `rest` holds.
    fn read(rest: &mut FragmentReader<'_>) -> Result<Self, Error> {
        let shared_len = rest.read_varint()? as usize;
        let suffix_len = rest.read_varint()? as usize;
        if shared_len > MAX_INDEXED_TOKEN_LEN || suffix_len > MAX_INDEXED_TOKEN_LEN - shared_len {
            return Err(Error::corrupt("token is longer than an index stores"));
        }
        let mut suffix = [0u8; MAX_INDEXED_TOKEN_LEN];
        rest.read_exact(&mut suffix[..suffix_len])?;
        let row_count = rest.read_varint()?;
        if row_count == 0 || row_count > u32::MAX as u64 {
            return Err(Error::corrupt("token row count is invalid"));
        }
        let rows = if row_count as usize > MAX_INLINE_ROWS {
            let offset = rest.read_varint()?;
            let len = rest.read_varint()?;
            if len > u32::MAX as u64 {
                return Err(Error::corrupt("posting list length is invalid"));
            }
            EncodedRows::List {
                offset,
                len: len as u32,
            }
        } else {
            let mut rows = [0u32; MAX_INLINE_ROWS];
            let mut row = 0u64;
            for inline_row in &mut rows[..row_count as usize] {
                row += rest.read_varint()?;
                if row > u32::MAX as u64 {
                    return Err(Error::corrupt("inline row is out of range"));
                }
                *inline_row = row as u32;
            }
            EncodedRows::Inline(rows)
        };
        Ok(Self {
            shared_len,
            suffix,
            suffix_len,
            row_count: row_count as u32,
            rows,
        })
    }
}
