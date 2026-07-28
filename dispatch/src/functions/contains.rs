use crate::env::MAX_INLINE_STRING_VIEW;
use crate::functions::needle::NeedleSearcher;
use arrow::array::ByteView;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBufferBuilder;

/// The `Contains` struct contains a specialized implementation for running contains with a needle
/// (e.g., WHERE LIKE '%google%') on `StringViewArray`s. The search itself is
/// [`NeedleSearcher`]'s one pass per underlying buffer; this adds the per-row
/// mask on top of it.
pub struct Contains {
    searcher: NeedleSearcher,
}

impl Contains {
    /// Create a new `Contains` searcher for the given needle.
    pub fn new<B: ?Sized + AsRef<[u8]>>(needle: &B) -> Self {
        Self {
            searcher: NeedleSearcher::new(needle),
        }
    }

    /// Test every string in `col` for the needle, returning a [`BooleanArray`]
    /// mask. `true` at position *i* means the string at *i* contains the needle.
    ///
    /// The searcher is stateful: buffer scan results are cached across calls so
    /// that repeated invocations on arrays sharing the same backing buffers
    /// (common with dictionary-encoded or sliced data) avoid redundant work.
    pub fn run(&mut self, col: &StringViewArray) -> BooleanArray {
        self.searcher.scan_buffers(col);
        self.create_bitmask(col)
    }

    /// Build a boolean mask indicating which strings in `array` contain the needle.
    ///
    /// Each view is either *inline* (≤ 12 bytes, stored in the view itself) or
    /// *buffer-backed* (a pointer into a data buffer). Inline strings are checked
    /// directly with the finder; buffer-backed strings are checked via a binary
    /// search over the pre-computed needle offsets for their buffer.
    fn create_bitmask(&self, array: &StringViewArray) -> BooleanArray {
        let needle_len = self.searcher.needle_len();
        let buffers = self.searcher.scanned_buffers(array);

        let row_count = array.len();
        let mut bitmap = BooleanBufferBuilder::new(row_count);

        for &view in array.views().iter() {
            let len = view as u32;

            let found = if len as usize > MAX_INLINE_STRING_VIEW {
                let bv = ByteView::from(view);
                let buffer = buffers[bv.buffer_index as usize];
                let start = bv.offset as usize + buffer.base_offset;
                buffer.contains_in_range(start, start + len as usize, needle_len)
            } else if len as usize >= needle_len {
                let bytes = view.to_le_bytes();
                self.searcher
                    .finder()
                    .find(&bytes[4..4 + len as usize])
                    .is_some()
            } else {
                false
            };

            bitmap.append(found);
        }

        bitmap.finish().into()
    }
}

#[cfg(test)]
mod tests {
    use super::Contains;
    use arrow_array::StringViewArray;

    fn run(needle: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let result = Contains::new(needle).run(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
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
