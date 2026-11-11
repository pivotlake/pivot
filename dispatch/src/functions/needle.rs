//! [`NeedleSearcher`] — one literal needle searched once over the bytes of
//! each underlying `StringViewArray` data buffer that rows point into, with
//! every match offset kept for the rows that point there.
//!
//! A `StringViewArray` holds a vector of views (u128) that are either inlined
//! strings or pointers into data buffers. Those buffers are reused across
//! record batches, the same string may be pointed at from many views (a
//! dictionary, for example), and they are usually very large: often the
//! original allocation of a decompressed source page. So the searcher runs the
//! `memchr::memmem::Finder` over long stretches of the underlying physical
//! buffer rather than row by row, which both avoids repeating work and lets
//! the vectorized search run over long inputs. Each row then costs a binary
//! search over the recorded offsets.
//!
//! Only the stretch of a buffer that an array's views reference is searched,
//! and the searched stretch grows as later arrays reference more of it. An
//! array usually covers a small part of its buffers: a batch decoded from a
//! page holds a few thousand of the page's rows, and carries its column's
//! dictionary page whether or not a row points into it. The bytes a batch
//! does reference were just read by whoever built it and are still in cache;
//! the rest of the buffer is not, and its rows may reach another worker's
//! searcher rather than this one.

use crate::env::MAX_INLINE_STRING_VIEW;
use arrow_array::StringViewArray;
use arrow_buffer::Bytes;
use memchr::memmem::Finder;
use std::ops::Range;
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
    /// The part of the buffer searched so far.
    searched_range: Range<usize>,
    /// Sorted byte offsets within the buffer where the needle starts, for
    /// every occurrence that lies inside `searched_range`.
    offsets: Vec<usize>,
}

/// Stand-in for a buffer that was never scanned, so resolution needs no
/// `Option` in the per-row path.
static EMPTY_BUFFER: BufferFindOffsets = BufferFindOffsets {
    base_ptr: 0,
    buffer: Weak::new(),
    searched_range: 0..0,
    offsets: Vec::new(),
};

impl BufferFindOffsets {
    /// Returns `true` if the underlying buffer is still alive.
    fn is_valid(&self) -> bool {
        self.buffer.strong_count() > 0
    }

    /// Extends the searched part of `allocation` to include `span`.
    ///
    /// A span reaching past the end of the searched part is the common case
    /// (a page read front to back), and only the new bytes are searched: the
    /// search restarts one needle short of the old end, so an occurrence
    /// lying across it is found. A span reaching back before the searched
    /// part searches the two joined together again from the span's start.
    fn search_span(&mut self, allocation: &[u8], span: Range<usize>, finder: &Finder<'_>) {
        if self.searched_range.start <= span.start && span.end <= self.searched_range.end {
            return;
        }
        if self.searched_range.is_empty() || span.start < self.searched_range.start {
            let joined_span = if self.searched_range.is_empty() {
                span
            } else {
                span.start..span.end.max(self.searched_range.end)
            };
            self.offsets.clear();
            self.offsets.extend(
                finder
                    .find_iter(&allocation[joined_span.clone()])
                    .map(|at| at + joined_span.start),
            );
            self.searched_range = joined_span;
            return;
        }
        let needle_overlap = finder.needle().len().saturating_sub(1);
        let search_start = self
            .searched_range
            .end
            .saturating_sub(needle_overlap)
            .max(self.searched_range.start);
        // Every occurrence already recorded starts before the ones found
        // here, except an empty needle's at the old end itself.
        let first_new_offset = self.offsets.last().map_or(0, |&last| last + 1);
        self.offsets.extend(
            finder
                .find_iter(&allocation[search_start..span.end])
                .map(|at| at + search_start)
                .filter(|&at| at >= first_new_offset),
        );
        self.searched_range.end = span.end;
    }
}

/// For each data buffer of `col`, the stretch of its allocation that the
/// array's out-of-line views reference, from the first referenced byte to the
/// last; `None` for a buffer no view points into.
fn find_referenced_spans(col: &StringViewArray) -> Vec<Option<Range<usize>>> {
    const NO_BUFFER: usize = usize::MAX;
    let mut bounds = vec![(usize::MAX, 0usize); col.data_buffers().len()];
    // Consecutive rows mostly point into the same buffer, so that buffer's
    // bounds are kept out of the vector until a row points elsewhere: updated
    // in place they would chain every row's store into the next row's load.
    let mut current_buffer = NO_BUFFER;
    let (mut span_start, mut span_end) = (usize::MAX, 0usize);
    for &view in col.views().iter() {
        let len = view as u32 as usize;
        if len <= MAX_INLINE_STRING_VIEW {
            continue;
        }
        let buffer_index = (view >> 64) as u32 as usize;
        let start = (view >> 96) as u32 as usize;
        if buffer_index != current_buffer {
            if current_buffer != NO_BUFFER {
                bounds[current_buffer] = (span_start, span_end);
            }
            current_buffer = buffer_index;
            (span_start, span_end) = bounds[buffer_index];
        }
        span_start = span_start.min(start);
        span_end = span_end.max(start + len);
    }
    if current_buffer != NO_BUFFER {
        bounds[current_buffer] = (span_start, span_end);
    }
    bounds
        .into_iter()
        .zip(col.data_buffers())
        .map(|((span_start, span_end), buffer)| {
            let base_offset = buffer.ptr_offset();
            (span_start < span_end).then(|| base_offset + span_start..base_offset + span_end)
        })
        .collect()
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

    /// Run the finder over the bytes of `col`'s data buffers that its views
    /// reference and that were not searched already.
    #[inline]
    pub(crate) fn scan_buffers(&mut self, col: &StringViewArray) {
        // Remove any buffers that aren't valid - it's critical to this to ensure we don't take
        // a new buffer and accidentally think it's one we already ran on (we compare pointers to
        // see if buffers have already been run on)
        self.buffers_with_offsets.retain_mut(|f| f.is_valid());
        let spans = find_referenced_spans(col);
        for (buffer, span) in col.data_buffers().iter().zip(spans) {
            let Some(span) = span else { continue };
            // Get the underlying allocation (not the slice view)
            let base_ptr = buffer.bytes().as_ptr() as usize;

            let known_buffer = self
                .buffers_with_offsets
                .iter()
                .position(|b| b.base_ptr == base_ptr);
            let buffer_offsets = match known_buffer {
                Some(index) => &mut self.buffers_with_offsets[index],
                None => {
                    self.buffers_with_offsets.push(BufferFindOffsets {
                        base_ptr,
                        buffer: Arc::downgrade(buffer.bytes()),
                        searched_range: 0..0,
                        offsets: Vec::new(),
                    });
                    self.buffers_with_offsets.last_mut().unwrap()
                }
            };
            buffer_offsets.search_span(&buffer.bytes()[..], span, &self.finder);
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
