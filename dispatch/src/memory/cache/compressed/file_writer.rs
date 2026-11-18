//! [`CachedFileWriter`]: puts a file in the compressed cache as it is
//! written, and has the file written out from the cache's copy.
//!
//! ```text
//!   encoded pages ──copy──▶ cache extents ──▶ returned FileBytes ──▶ the write
//!   (given up as copied)    (committed as      (views of the extents,
//!                            they fill)         pinning their slots)
//! ```
//!
//! A page's place in its file is known only once the file is laid out, and an
//! extent holds file blocks at their place, so the bytes are copied in once.
//! From then on the extents are the only copy: the write goes out from views
//! of them, which keep their slots from being evicted until it is done, and a
//! later read of the file finds them.
//!
//! A file arrives a piece at a time and a piece seldom ends on a block
//! boundary, so the extent holding the file's last block stays reserved
//! between pieces and the next piece goes on from the middle of that block.

use crate::io::OpenFile;
use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::compressed::extent::BLOCK_SIZE;
use crate::memory::cache::compressed::lookup::MissingExtent;
use crate::memory::cache::id::CacheId;
use crate::memory::cache::ring_cache::RingCache;
use crate::memory::context::memory_ctx;
use crate::memory::file_bytes::FileBytes;
use crate::memory::ring::BUFFER_SIZE;
use std::collections::VecDeque;

/// Puts one file's bytes in a compressed cache in the order they are
/// appended. Lives on the worker that writes the file.
pub struct CachedFileWriter {
    cache: CacheId,
    /// The file as its readers will open it.
    open_file: OpenFile,
    /// How many bytes of the file are in the cache.
    len: usize,
    /// The extents reserved for the file's next bytes, in file order. The
    /// first is the one being filled. Between appends at most one is left:
    /// the one holding the file's last block, while that block has room.
    reserved: VecDeque<MissingExtent>,
}

impl CachedFileWriter {
    /// A writer of the file readers will open as `open_file`, none of which
    /// is in `cache` yet.
    pub fn new(cache: &CompressedCache, open_file: OpenFile) -> Self {
        Self {
            cache: cache.id(),
            open_file,
            len: 0,
            reserved: VecDeque::new(),
        }
    }

    /// Copy `bytes`, the file's next ones, into the cache, and return them as
    /// views of the cache for the file's write. Each run of `bytes` is given
    /// up as soon as it is copied.
    pub fn append(&mut self, bytes: FileBytes) -> FileBytes {
        let mut cached = FileBytes::new();
        let mut bytes_left = bytes.len();
        // Where the bytes copied into the extent being filled, and not yet
        // returned as a view, start in the file.
        let mut view_start = self.len;
        for run in bytes.into_runs() {
            let mut source = run.as_slice();
            while !source.is_empty() {
                if self.reserved.is_empty() {
                    self.reserve_extents(bytes_left.min(BUFFER_SIZE));
                }
                let extent = self
                    .reserved
                    .front()
                    .expect("extents were reserved for the bytes left");
                let extent_end = extent.file_offset() + extent.len();
                let copied_len = source.len().min(extent_end - self.len);
                // SAFETY: the destination is inside the extent, whose slot
                // the extent pins. This writer reserved the extent and no
                // reader knows the file yet, so nothing else reads or writes
                // these bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        extent.dest().add(self.len - extent.file_offset()),
                        copied_len,
                    );
                }
                self.len += copied_len;
                bytes_left -= copied_len;
                source = &source[copied_len..];
                if self.len == extent_end {
                    cached.push_cached(extent.view_bytes(view_start, self.len));
                    view_start = self.len;
                    extent.commit();
                    self.reserved.pop_front();
                }
            }
        }
        if let Some(extent) = self.reserved.front()
            && view_start < self.len
        {
            cached.push_cached(extent.view_bytes(view_start, self.len));
        }
        cached
    }

    /// The file is complete: make its last block, which may be short, one a
    /// reader finds.
    pub fn finish(mut self) {
        if let Some(extent) = self.reserved.pop_front() {
            extent.commit_prefix(self.len - extent.file_offset());
        }
        debug_assert!(
            self.reserved.is_empty(),
            "only the extent of the file's last block is left between appends"
        );
    }

    /// Reserve extents for the file's next `len` bytes, which start on a
    /// block boundary.
    fn reserve_extents(&mut self, len: usize) {
        debug_assert_eq!(self.len % BLOCK_SIZE, 0);
        let lookups =
            memory_ctx()
                .caches()
                .compressed(self.cache)
                .get(&self.open_file, self.len, len);
        for lookup in lookups {
            let extent = lookup
                .missing_extent()
                .filter(|extent| extent.is_fill_owner())
                .expect("nothing else puts a file that is being written in the cache");
            self.reserved.push_back(extent.clone());
        }
    }
}
