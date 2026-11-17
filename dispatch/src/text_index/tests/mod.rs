//! The three pieces of a text index: gathered in working memory and in their
//! caches, encoded and decoded. Every test installs its own single-worker
//! memory context.

mod dict_index;
mod index_file;
mod postings;
mod sparse_index;

use crate::io::{FileRange, LocalFile, OpenFile};
use crate::memory::{CacheId, CacheKey, Slab, SlabAllocator, memory_ctx};

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
