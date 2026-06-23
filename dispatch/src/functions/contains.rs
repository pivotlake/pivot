use crate::env::MAX_INLINE_STRING_VIEW;
use arrow::array::ByteView;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::{BooleanBufferBuilder, Bytes};
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

impl BufferFindOffsets {
    /// Returns `true` if the underlying buffer is still alive.
    fn is_valid(&self) -> bool {
        self.buffer.strong_count() > 0
    }

    /// Returns whether the needle appears fully within the byte range
    /// `[start, start + len)`. Uses binary search over the sorted offsets.
    #[inline(always)]
    fn contains_in_range(&self, start: usize, len: usize, needle_len: usize) -> bool {
        let idx = self.offsets.partition_point(|&o| o < start);
        self.offsets
            .get(idx)
            .is_some_and(|&o| o + needle_len <= start + len)
    }
}

/// The `Contains` struct contains a specialized implementation for running contains with a needle
/// (e.g., WHERE LIKE '%google%') on `StringViewArray`s. The idea in general is to only do one pass
/// finding occurrences of the needle per underlying buffer.
///
/// In contrast to StringArray, StringViewArray has a vector of views (u128) which are either
/// pointers to underlying buffers or inlined strings. These underlying buffers can be re-used
/// across RecordBatches, and the same string may be pointed to many types from different places
/// (for example if it's in a Dict). They are also usually very large - they may be the original
/// allocations of the decompressed source pages.
/// The implementation here aims to only run the  memchr::memchr::Finder *once* per entire
/// underlying physical buffer.
///
/// This is advantageous for two reasons:
/// 1. We're never running twice on one buffer
/// 2. We can take advantage of vectorized capabilities by running on longer buffers
pub struct Contains {
    buffers_with_offsets: Vec<BufferFindOffsets>,
    finder: Finder<'static>,
}

impl Contains {
    /// Create a new `Contains` searcher for the given needle.
    pub fn new<B: ?Sized + AsRef<[u8]>>(needle: &B) -> Self {
        Self {
            buffers_with_offsets: vec![],
            finder: Finder::new(needle).into_owned(),
        }
    }

    /// Test every string in `col` for the needle, returning a [`BooleanArray`]
    /// mask. `true` at position *i* means the string at *i* contains the needle.
    ///
    /// The searcher is stateful: buffer scan results are cached across calls so
    /// that repeated invocations on arrays sharing the same backing buffers
    /// (common with dictionary-encoded or sliced data) avoid redundant work.
    pub fn run(&mut self, col: &StringViewArray) -> BooleanArray {
        self.run_find_on_underlying_buffers(col);
        self.create_bitmask(col, None)
    }

    /// Like [`run`](Self::run), but only inspects rows where `selection` is set
    /// (i.e. `true` and non-null); every other row is left `false`. The result
    /// is therefore `selection AND contains`, used by the filter operator to run
    /// the substring scan only over the rows that survived the cheaper
    /// predicates.
    pub fn run_selected(
        &mut self,
        col: &StringViewArray,
        selection: &BooleanArray,
    ) -> BooleanArray {
        self.run_find_on_underlying_buffers(col);
        self.create_bitmask(col, Some(selection))
    }

    /// Run find on underlying buffers *if this is the first time we've seen them*.
    fn run_find_on_underlying_buffers(&mut self, col: &StringViewArray) {
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

    /// Build a boolean mask indicating which strings in `array` contain the needle.
    ///
    /// Assumes [`run_find_on_underlying_buffers`](Self::run_find_on_underlying_buffers)
    /// has already been called for this array.
    ///
    /// Each view is either *inline* (≤ 12 bytes, stored in the view itself) or
    /// *buffer-backed* (a pointer into a data buffer). Inline strings are checked
    /// directly with the finder; buffer-backed strings are checked via a binary
    /// search over the pre-computed needle offsets for their buffer.
    fn create_bitmask(
        &self,
        array: &StringViewArray,
        selection: Option<&BooleanArray>,
    ) -> BooleanArray {
        let needle_len = self.finder.needle().len();
        let empty = BufferFindOffsets::default();

        // Resolve each data buffer to its (ptr_offset, cached find offsets).
        // ptr_offset is needed because the view's offset is relative to the
        // buffer slice, not the underlying allocation we scanned.
        let buffers: Vec<_> = array
            .data_buffers()
            .iter()
            .map(|b| {
                let ptr = b.bytes().as_ptr() as usize;
                let find_offsets = self
                    .buffers_with_offsets
                    .iter()
                    .find(|f| f.base_ptr == ptr)
                    .unwrap_or(&empty);
                (b.ptr_offset(), find_offsets)
            })
            .collect();

        let row_count = array.len();
        let mut bitmap = BooleanBufferBuilder::new(row_count);
        bitmap.append_n(row_count, false);

        for (i, &view) in array.views().iter().enumerate() {
            // Skip rows the caller already excluded; they stay `false`. A null
            // selection bit (an excluded row) is treated the same as `false`.
            if selection.is_some_and(|s| !(s.is_valid(i) && s.value(i))) {
                continue;
            }

            let len = view as u32;

            let found = if len as usize > MAX_INLINE_STRING_VIEW {
                let bv = ByteView::from(view);
                let (base_offset, find_offsets) = buffers[bv.buffer_index as usize];
                let start = bv.offset as usize + base_offset;
                find_offsets.contains_in_range(start, len as usize, needle_len)
            } else if len as usize >= needle_len {
                let bytes = view.to_le_bytes();
                self.finder.find(&bytes[4..4 + len as usize]).is_some()
            } else {
                false
            };

            bitmap.set_bit(i, found);
        }

        bitmap.finish().into()
    }
}

#[cfg(test)]
mod tests {
    use super::Contains;
    use arrow_array::{BooleanArray, StringViewArray};

    fn run(needle: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let result = Contains::new(needle).run(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    fn run_selected(needle: &str, values: &[&str], selection: &[Option<bool>]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let selection = BooleanArray::from(selection.to_vec());
        let result = Contains::new(needle).run_selected(&array, &selection);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    #[test]
    fn selected_skips_unselected_rows() {
        let values = ["world a", "world b", "world c"];

        // Middle row matches but is not selected; it must come back false.
        let result = run_selected("world", &values, &[Some(true), Some(false), Some(true)]);

        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn selected_treats_null_selection_as_excluded() {
        let values = ["world a", "world b"];

        let result = run_selected("world", &values, &[Some(true), None]);

        assert_eq!(result, vec![true, false]);
    }

    #[test]
    fn matches_substring() {
        let values = ["hello world", "foo", "world peace"];

        let result = run("world", &values);

        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn no_matches() {
        let values = ["abc", "def", "ghi"];

        let result = run("xyz", &values);

        assert_eq!(result, vec![false, false, false]);
    }

    #[test]
    fn all_match() {
        let values = ["aaa", "baa", "caa"];

        let result = run("aa", &values);

        assert_eq!(result, vec![true, true, true]);
    }

    #[test]
    fn empty_needle_matches_all() {
        let values = ["hello", "", "x"];

        let result = run("", &values);

        assert_eq!(result, vec![true, true, true]);
    }

    #[test]
    fn empty_array() {
        let result = run("needle", &[]);

        assert_eq!(result, Vec::<bool>::new());
    }

    #[test]
    fn needle_longer_than_value() {
        let values = ["hi", "a", ""];

        let result = run("longer_needle", &values);

        assert_eq!(result, vec![false, false, false]);
    }

    #[test]
    fn exact_match() {
        let values = ["needle", "not", "needle"];

        let result = run("needle", &values);

        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn short_inline_strings() {
        let values = ["ab", "abc", "bc", "a"];

        let result = run("bc", &values);

        assert_eq!(result, vec![false, true, true, false]);
    }

    #[test]
    fn long_strings_use_buffer_path() {
        let values = [
            "this is a long string with google in it",
            "this is a long string without the word",
            "another long string containing google here",
        ];

        let result = run("google", &values);

        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn reuses_buffer_offsets_across_calls() {
        let array = StringViewArray::from_iter_values([
            "a]long string containing needle inside",
            "a long string without the target word",
        ]);
        let mut contains = Contains::new("needle");

        let r1 = contains.run(&array);
        let r2 = contains.run(&array);

        assert!(r1.value(0));
        assert!(!r1.value(1));
        assert!(r2.value(0));
        assert!(!r2.value(1));
    }
}
