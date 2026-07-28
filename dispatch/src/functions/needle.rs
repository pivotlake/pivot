//! [`NeedleSearcher`] — one literal needle searched once per underlying
//! `StringViewArray` data buffer, with every match offset kept for the rows
//! that point into that buffer.
//!
//! A `StringViewArray` holds a vector of views (u128) that are either inlined
//! strings or pointers into data buffers. Those buffers are reused across
//! record batches, the same string may be pointed at from many views (a
//! dictionary, for example), and they are usually very large: often the
//! original allocation of a decompressed source page. So the searcher runs the
//! `memchr::memmem::Finder` *once* per underlying physical buffer, which both
//! avoids repeating work and lets the vectorized search run over long inputs.
//! Each row then costs a binary search over the recorded offsets.

use arrow_array::StringViewArray;
use arrow_buffer::Bytes;
use memchr::memmem::Finder;
use std::sync::{Arc, Weak};

/// Cached needle-match offsets for a single underlying data buffer.
///
/// Stores a weak reference to the buffer so we can detect when it has been
/// freed (and therefore its pointer may be reused by a new allocation).
#[derive(Debug, Default)]
struct BufferFindOffsets {
    /// Address of the buffer's backing allocation, used as an identity key.
    base_ptr: usize,
    /// Weak ref to the buffer — when all strong refs are dropped the buffer
    /// is gone and this entry must be evicted to avoid pointer-reuse collisions.
    buffer: Weak<Bytes>,
    /// Sorted byte offsets within the buffer where the needle starts.
    offsets: Vec<usize>,
}

/// Stand-in for a buffer that was never scanned, so resolution needs no
/// `Option` in the per-row path.
static EMPTY_BUFFER: BufferFindOffsets = BufferFindOffsets {
    base_ptr: 0,
    buffer: Weak::new(),
    offsets: Vec::new(),
};

impl BufferFindOffsets {
    /// Returns `true` if the underlying buffer is still alive.
    fn is_valid(&self) -> bool {
        self.buffer.strong_count() > 0
    }
}

/// One data buffer of an array, resolved to the needle offsets already found
/// inside it. Both `base_offset` and the offsets address the whole allocation
/// the finder scanned, not the array's slice of it.
///
/// It holds the cache entry by reference rather than copying the offsets slice
/// out, which keeps it two words wide. Every row indexes this by its view's
/// buffer index, and at two words that is a shift; a three-word struct costs a
/// multiply and a third load on each of the hundred million of them.
#[derive(Clone, Copy)]
pub(crate) struct ScannedBuffer<'a> {
    /// Offset of the array's buffer within the allocation, to be added to a
    /// view's own offset.
    pub(crate) base_offset: usize,
    /// Where the needle was found in that allocation.
    found: &'a BufferFindOffsets,
}

impl ScannedBuffer<'_> {
    /// Whether the needle occurs entirely within `[from, until)`. Callers that
    /// only need the answer take this rather than
    /// [`find_at_or_after`](Self::find_at_or_after), so the per-row loop never
    /// carries a position it is going to discard.
    #[inline(always)]
    pub(crate) fn contains_in_range(&self, from: usize, until: usize, needle_len: usize) -> bool {
        let offsets = &self.found.offsets;
        let idx = offsets.partition_point(|&offset| offset < from);
        offsets
            .get(idx)
            .is_some_and(|&offset| offset + needle_len <= until)
    }

    /// The first needle occurrence starting at or after `from` and ending at or
    /// before `until`, as an offset into the allocation. For an ordered walk,
    /// where the match position is where the next segment starts looking.
    #[inline(always)]
    pub(crate) fn find_at_or_after(
        &self,
        from: usize,
        until: usize,
        needle_len: usize,
    ) -> Option<usize> {
        let offsets = &self.found.offsets;
        let idx = offsets.partition_point(|&offset| offset < from);
        offsets
            .get(idx)
            .copied()
            .filter(|&offset| offset + needle_len <= until)
    }
}

/// Searches one needle across the data buffers of successive
/// [`StringViewArray`]s, caching what it finds per buffer. See the module
/// docs for why the search is per buffer rather than per row.
pub(crate) struct NeedleSearcher {
    buffers_with_offsets: Vec<BufferFindOffsets>,
    finder: Finder<'static>,
}

impl NeedleSearcher {
    pub(crate) fn new<B: ?Sized + AsRef<[u8]>>(needle: &B) -> Self {
        Self {
            buffers_with_offsets: vec![],
            finder: Finder::new(needle).into_owned(),
        }
    }

    pub(crate) fn needle_len(&self) -> usize {
        self.finder.needle().len()
    }

    /// The finder itself, for values too short to live in a data buffer.
    pub(crate) fn finder(&self) -> &Finder<'static> {
        &self.finder
    }

    /// Run the finder over every data buffer of `col` not searched already.
    #[inline]
    pub(crate) fn scan_buffers(&mut self, col: &StringViewArray) {
        // Remove any buffers that aren't valid - it's critical to this to ensure we don't take
        // a new buffer and accidentally think it's one we already ran on (we compare pointers to
        // see if buffers have already been run on)
        self.buffers_with_offsets.retain_mut(|f| f.is_valid());
        for buffer in col.data_buffers() {
            // Get the underlying allocation (not the slice view)
            let base_ptr = buffer.bytes().as_ptr() as usize;

            if self
                .buffers_with_offsets
                .iter_mut()
                .any(|b| b.base_ptr == base_ptr)
            {
                // We've already ran on this buffer,
                continue;
            }

            self.buffers_with_offsets.push(BufferFindOffsets {
                base_ptr,
                buffer: Arc::downgrade(buffer.bytes()),
                offsets: self.finder.find_iter(&buffer.bytes()[..]).collect(),
            })
        }
    }

    /// Resolve each of `col`'s data buffers to the offsets found in it,
    /// positionally, so a view's `buffer_index` indexes the result.
    /// [`scan_buffers`](Self::scan_buffers) must have run for this array.
    #[inline]
    pub(crate) fn scanned_buffers(&self, col: &StringViewArray) -> Vec<ScannedBuffer<'_>> {
        col.data_buffers()
            .iter()
            .map(|buffer| {
                let base_ptr = buffer.bytes().as_ptr() as usize;
                let found = self
                    .buffers_with_offsets
                    .iter()
                    .find(|f| f.base_ptr == base_ptr)
                    .unwrap_or(&EMPTY_BUFFER);
                ScannedBuffer {
                    base_offset: buffer.ptr_offset(),
                    found,
                }
            })
            .collect()
    }
}
