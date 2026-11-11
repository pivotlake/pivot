//! What the compressed cache holds for a local file is found again through
//! every later open of that file: the cache knows a file by which file it
//! is, not by the handle it was read through.

use std::path::{Path, PathBuf};

use dispatch::io::{LocalFile, OpenFile};
use dispatch::memory::CacheLookup;
use dispatch::{Dispatch, memory_ctx};

/// The size of the blocks the cache holds a file in.
const BLOCK_BYTES: usize = 4096;

/// A deterministic byte for the file offset `offset`, so what the cache
/// holds is told apart from what the file holds.
fn pattern_at(offset: usize) -> u8 {
    offset.wrapping_mul(7).wrapping_add(1) as u8
}

/// Open `path` anew for reading.
fn open_local(path: &Path) -> OpenFile {
    OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap())
}

/// Look the first block of the file at `path` up through a new open of it.
fn look_up_first_block(path: &Path) -> Vec<CacheLookup> {
    memory_ctx()
        .page_caches()
        .compressed
        .get(&open_local(path), 0, BLOCK_BYTES)
}

/// Cache the pattern as the first block of the file at `path`, on a worker of
/// `dispatch`, the way a read through one open of the file leaves it.
fn cache_first_block(dispatch: &Dispatch, path: &Path) {
    let path: PathBuf = path.to_path_buf();
    dispatch
        .dispatcher()
        .run_on_worker(move || {
            for lookup in look_up_first_block(&path) {
                let block = lookup
                    .missing_extent()
                    .expect("nothing of the file is cached yet");
                for position in 0..block.len() {
                    let byte = pattern_at(block.file_offset() + position);
                    unsafe { *block.dest().add(position) = byte };
                }
                block.commit();
            }
        })
        .unwrap();
}

/// What a worker of `dispatch` finds cached of the first block of the file at
/// `path` through a new open of it: `None` when any of the block is missing.
fn read_cached_first_block(dispatch: &Dispatch, path: &Path) -> Option<Vec<u8>> {
    let path: PathBuf = path.to_path_buf();
    dispatch
        .dispatcher()
        .run_on_worker(move || {
            let lookups = look_up_first_block(&path);
            if lookups
                .iter()
                .any(|lookup| lookup.missing_extent().is_some())
            {
                return None;
            }
            let mut cached = Vec::new();
            for lookup in lookups {
                cached.extend_from_slice(&lookup.into_data());
            }
            Some(cached)
        })
        .unwrap()
}

/// The pattern over the first block of a file.
fn first_block_pattern() -> Vec<u8> {
    (0..BLOCK_BYTES).map(pattern_at).collect()
}

#[test]
fn a_second_open_of_an_unchanged_file_hits_what_the_first_cached() {
    let dispatch = Dispatch::spin_up(1, 16, None);
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), vec![7u8; BLOCK_BYTES]).unwrap();
    cache_first_block(&dispatch, file.path());

    let reopened = read_cached_first_block(&dispatch, file.path());

    assert_eq!(reopened, Some(first_block_pattern()));
    dispatch.exit();
}

#[test]
fn a_file_overwritten_in_place_is_served_what_was_cached_before() {
    let dispatch = Dispatch::spin_up(1, 16, None);
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), vec![7u8; BLOCK_BYTES]).unwrap();
    cache_first_block(&dispatch, file.path());

    std::fs::write(file.path(), vec![9u8; 2 * BLOCK_BYTES]).unwrap();
    let overwritten = read_cached_first_block(&dispatch, file.path());

    assert_eq!(overwritten, Some(first_block_pattern()));
    dispatch.exit();
}
