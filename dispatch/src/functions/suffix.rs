use crate::env::MAX_INLINE_STRING_VIEW;
use arrow::array::ByteView;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBufferBuilder;

/// The `Suffix` struct runs ends-with matching (e.g., WHERE p_type LIKE '%BRASS')
/// on `StringViewArray`s.
///
/// Unlike [`Prefix`](super::Prefix), a view's inline bytes hold the string's
/// *first* four bytes, so they cannot pre-reject a suffix probe. A string short
/// enough to inline still compares entirely within the view; only longer
/// strings read their tail from the data buffers.
pub struct Suffix {
    suffix: Vec<u8>,
}

impl Suffix {
    /// Create a new `Suffix` matcher for the given suffix.
    pub fn new<B: ?Sized + AsRef<[u8]>>(suffix: &B) -> Self {
        Self {
            suffix: suffix.as_ref().to_vec(),
        }
    }

    /// Test every string in `col` for the suffix, returning a [`BooleanArray`]
    /// mask. `true` at position *i* means the string at *i* ends with the suffix.
    pub fn run(&self, col: &StringViewArray) -> BooleanArray {
        let suffix = self.suffix.as_slice();
        let suffix_len = suffix.len();
        let data_buffers = col.data_buffers();

        let row_count = col.len();
        let mut bitmap = BooleanBufferBuilder::new(row_count);
        bitmap.append_n(row_count, false);

        for (i, &view) in col.views().iter().enumerate() {
            let len = (view as u32) as usize;
            if len < suffix_len {
                continue;
            }
            let matched = if len <= MAX_INLINE_STRING_VIEW {
                let view_bytes = view.to_le_bytes();
                view_bytes[4 + len - suffix_len..4 + len] == suffix[..]
            } else {
                let byte_view = ByteView::from(view);
                let start = byte_view.offset as usize;
                let buffer = &data_buffers[byte_view.buffer_index as usize];
                buffer[start + len - suffix_len..start + len] == suffix[..]
            };
            bitmap.set_bit(i, matched);
        }

        // A NULL input row must stay NULL in the mask (not `true`, which an
        // empty suffix would otherwise produce), so the filter drops it.
        BooleanArray::new(bitmap.finish(), col.nulls().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::Suffix;
    use arrow_array::{Array, StringViewArray};

    fn run(suffix: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let result = Suffix::new(suffix).run(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    #[test]
    fn matches_short_inline_strings() {
        let result = run("bc", &["abc", "bc", "cb", "c"]);

        assert_eq!(result, vec![true, true, false, false]);
    }

    #[test]
    fn matches_long_buffer_backed_strings() {
        let values = [
            "a long path ending in ECONOMY BRASS",
            "a long path ending in ECONOMY STEEL",
            "another long path ending in BRASSY",
        ];

        let result = run("BRASS", &values);

        assert_eq!(result, vec![true, false, false]);
    }

    #[test]
    fn suffix_spanning_the_whole_inline_view() {
        let result = run("exactly12chr", &["exactly12chr", "exactly12chx"]);

        assert_eq!(result, vec![true, false]);
    }

    #[test]
    fn substring_elsewhere_does_not_match() {
        let result = run("world", &["world news", "hello world"]);

        assert_eq!(result, vec![false, true]);
    }

    #[test]
    fn empty_suffix_matches_all() {
        let result = run("", &["a", "", "some long string beyond inline"]);

        assert_eq!(result, vec![true, true, true]);
    }

    #[test]
    fn suffix_longer_than_value() {
        let result = run("longer than the value", &["short", ""]);

        assert_eq!(result, vec![false, false]);
    }

    #[test]
    fn null_rows_stay_null_even_for_empty_suffix() {
        let array = StringViewArray::from(vec![Some("abc"), None]);

        let result = Suffix::new("").run(&array);

        assert!(result.value(0));
        assert!(result.is_null(1));
    }

    #[test]
    fn exact_match_is_a_suffix() {
        let result = run("needle", &["needle", "eedle"]);

        assert_eq!(result, vec![true, false]);
    }
}
