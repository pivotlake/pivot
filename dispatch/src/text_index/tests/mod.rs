//! The three pieces of a text index: gathered in working memory and in their
//! caches, encoded and decoded. Every test installs its own single-worker
//! memory context.

mod dict_index;
mod index_file;
mod postings;
mod sparse_index;
mod token_run;

use crate::io::{FileRange, LocalFile, OpenFile};
use crate::memory::{CacheId, CacheKey, FileBytes, Slab, SlabAllocator, memory_ctx};
use crate::text_index::{DictIndex, IndexFileFooter, Postings, SparseIndex, TokenRows};

/// The `index`-th of a run of ascending tokens.
fn token(index: usize) -> Vec<u8> {
    format!("token{index:08}").into_bytes()
}

/// A range that tells which token it was stored for.
fn range(index: usize) -> FileRange {
    FileRange::new(index * 100, 100)
}

/// A key for the piece at `offset` of some index file.
fn key(offset: usize) -> CacheKey {
    let index_file = LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
    CacheKey::new(OpenFile::Local(index_file), offset, 1)
}

fn encoded_len(encoded: &[Slab]) -> usize {
    encoded.iter().map(Slab::size).sum()
}

/// `bytes` as encoded input.
fn encoded_from(bytes: &[u8]) -> Vec<Slab> {
    let mut slab = SlabAllocator::new(false).get_slab_of_size(bytes.len(), false);
    slab.as_mut_slice().copy_from_slice(bytes);
    vec![slab]
}

/// How many bytes of the slot `cache` is filling on this worker are taken.
fn count_bytes_cached(cache: CacheId) -> usize {
    memory_ctx().fill_cursor(cache).next_byte()
}

fn bytes_of(file: &FileBytes) -> Vec<u8> {
    file.runs().flatten().copied().collect()
}

/// The bytes of `file` at `range`, as a read of that range would return them.
fn read_range(file: &[u8], range: FileRange) -> Vec<Slab> {
    encoded_from(&file[range.offset..range.offset + range.len])
}

fn read_footer(file: &[u8]) -> IndexFileFooter {
    let footer_bytes = file[file.len() - IndexFileFooter::ENCODED_LEN..]
        .try_into()
        .unwrap();
    IndexFileFooter::decode(footer_bytes, file.len()).unwrap()
}

fn read_sparse_index(file: &[u8]) -> SparseIndex {
    let mut allocator = SlabAllocator::new(false);
    let encoded = read_range(file, read_footer(file).sparse_index);
    SparseIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish()
}

/// The rows of `token` as a lookup finds them in `file`: through the footer,
/// the sparse index, one dict index and, for a token in many rows, its
/// postings.
fn find_rows(file: &[u8], token: &[u8]) -> Option<Vec<u32>> {
    let mut allocator = SlabAllocator::new(false);
    let dict_index_range = read_sparse_index(file).find_dict_index(token)?;
    let dict_index = DictIndex::decode(&read_range(file, dict_index_range), &mut allocator)
        .unwrap()
        .finish();
    match dict_index.find_rows(token)? {
        TokenRows::KeptInDict(few_rows) => Some(few_rows.as_slice().to_vec()),
        TokenRows::InPostings { postings, .. } => {
            let postings = Postings::decode(&read_range(file, postings), &mut allocator)
                .unwrap()
                .finish(&mut allocator);
            let mut rows = Vec::new();
            postings.for_each_row(|row| rows.push(row));
            Some(rows)
        }
    }
}
