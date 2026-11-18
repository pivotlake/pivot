//! A file put in the compressed cache as it is written is found there by a
//! later open of it, and is written out from the cache's copy.

use std::path::{Path, PathBuf};

use dispatch::io::{LocalFile, OpenFile};
use dispatch::memory::{CachedFileWriter, FileBytes, SlabAllocator};
use dispatch::{Dispatch, memory_ctx};

/// A deterministic byte for the file offset `offset`.
fn pattern_at(offset: usize) -> u8 {
    offset.wrapping_mul(7).wrapping_add(1) as u8
}

/// The pattern over the first `len` bytes of a file.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(pattern_at).collect()
}

/// Open `path` anew for reading.
fn open_local(path: &Path) -> OpenFile {
    OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap())
}

/// Put the pattern in the cache as the file at `path`, appended in pieces of
/// `piece_lens` bytes, on a worker of `dispatch`. Returns the bytes the
/// writer handed back for the file's write.
fn append_pattern(dispatch: &Dispatch, path: &Path, piece_lens: &[usize]) -> Vec<u8> {
    let path: PathBuf = path.to_path_buf();
    let piece_lens = piece_lens.to_vec();
    dispatch
        .dispatcher()
        .run_on_worker(move || {
            let mut allocator = SlabAllocator::new(false);
            let mut writer =
                CachedFileWriter::new(&memory_ctx().page_caches().compressed, open_local(&path));
            let mut returned = Vec::new();
            let mut offset = 0;
            for piece_len in piece_lens {
                let mut slab = allocator.get_slab_of_size(piece_len, false);
                for (position, byte) in slab.as_mut_slice().iter_mut().enumerate() {
                    *byte = pattern_at(offset + position);
                }
                let mut piece = FileBytes::new();
                piece.push_slab(slab);
                offset += piece_len;
                returned.extend(writer.append(piece).runs().flatten());
            }
            writer.finish();
            returned
        })
        .unwrap()
}

/// What a worker of `dispatch` finds cached of the first `len` bytes of the
/// file at `path` through a new open of it: `None` when any of them is
/// missing.
fn read_cached(dispatch: &Dispatch, path: &Path, len: usize) -> Option<Vec<u8>> {
    let path: PathBuf = path.to_path_buf();
    dispatch
        .dispatcher()
        .run_on_worker(move || {
            let lookups = memory_ctx()
                .page_caches()
                .compressed
                .get(&open_local(&path), 0, len);
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

#[test]
fn a_file_appended_in_pieces_that_end_inside_blocks_is_found_whole() {
    let dispatch = Dispatch::spin_up(1, 16, None);
    let file = tempfile::NamedTempFile::new().unwrap();
    let piece_lens = [5_000, 100, 9_000, 3];
    let file_len: usize = piece_lens.iter().sum();

    append_pattern(&dispatch, file.path(), &piece_lens);

    assert_eq!(
        read_cached(&dispatch, file.path(), file_len),
        Some(pattern(file_len))
    );
    dispatch.exit();
}

#[test]
fn a_file_larger_than_a_ring_buffer_is_found_whole() {
    let dispatch = Dispatch::spin_up(1, 16, None);
    let file = tempfile::NamedTempFile::new().unwrap();
    let piece_lens = [1_500_000, 1_500_000, 1_500_000];
    let file_len: usize = piece_lens.iter().sum();

    append_pattern(&dispatch, file.path(), &piece_lens);

    assert_eq!(
        read_cached(&dispatch, file.path(), file_len),
        Some(pattern(file_len))
    );
    dispatch.exit();
}

#[test]
fn the_bytes_handed_back_for_the_write_are_the_bytes_appended() {
    let dispatch = Dispatch::spin_up(1, 16, None);
    let file = tempfile::NamedTempFile::new().unwrap();
    let piece_lens = [5_000, 100, 9_000, 3];
    let file_len: usize = piece_lens.iter().sum();

    let returned = append_pattern(&dispatch, file.path(), &piece_lens);

    assert_eq!(returned, pattern(file_len));
    dispatch.exit();
}
