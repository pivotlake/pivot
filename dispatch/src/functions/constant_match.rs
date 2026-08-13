use crate::env::MAX_INLINE_STRING_VIEW;
use arrow::array::ByteView;
use arrow_array::builder::make_view;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBuffer;

/// The `ConstantMatch` struct tests every string in a `StringViewArray` against
/// one constant (e.g. WHERE SearchPhrase <> '').
///
/// StringViewArray's layout makes this cheap: a view's low 32 bits hold the
/// string's length and the next 32 its first four bytes. A constant short
/// enough to inline therefore reduces the predicate to one 128-bit integer
/// compare per row, and an empty constant to a single "is the length zero"
/// test, neither of which dereferences a data buffer. Both compile to a
/// branch-free loop over the views. Only a constant too long to inline reads
/// payload bytes, and then only for rows whose length and first four bytes
/// already match.
pub struct ConstantMatch {
    needle: Vec<u8>,
    /// The needle laid out as a view. Meaningful whole only when the needle
    /// inlines; for a longer one just its low 64 bits (length and prefix) are.
    needle_view: u128,
}

impl ConstantMatch {
    /// Create a new `ConstantMatch` matcher for the given constant.
    pub fn new<B: ?Sized + AsRef<[u8]>>(needle: &B) -> Self {
        let needle = needle.as_ref().to_vec();
        Self {
            needle_view: make_view(&needle, 0, 0),
            needle,
        }
    }

    /// Mask of the rows equal to the constant. A NULL input row stays NULL.
    pub fn run_equal(&self, col: &StringViewArray) -> BooleanArray {
        self.run(col, false)
    }

    /// Mask of the rows differing from the constant. A NULL input row stays
    /// NULL, so the filter drops it rather than counting it as different.
    pub fn run_not_equal(&self, col: &StringViewArray) -> BooleanArray {
        self.run(col, true)
    }

    fn run(&self, col: &StringViewArray, negate: bool) -> BooleanArray {
        let views = col.views();
        let matches = if self.needle.is_empty() {
            // The length is the view's low 32 bits, so an empty constant never
            // needs the rest of the view.
            collect_match_bits(views, negate, |view| (view as u32) == 0)
        } else if self.needle.len() <= MAX_INLINE_STRING_VIEW {
            let needle_view = self.needle_view;
            collect_match_bits(views, negate, |view| view == needle_view)
        } else {
            // Length and prefix together sit in the view's low 64 bits, so one
            // integer compare rejects every row that cannot match before any
            // payload is read. A value this long is always buffer-backed.
            let needle = self.needle.as_slice();
            let needle_head = self.needle_view as u64;
            let data_buffers = col.data_buffers();
            collect_match_bits(views, negate, |view| {
                if view as u64 != needle_head {
                    return false;
                }
                let byte_view = ByteView::from(view);
                let start = byte_view.offset as usize;
                let buffer = &data_buffers[byte_view.buffer_index as usize];
                buffer[start..start + needle.len()] == needle[..]
            })
        };

        BooleanArray::new(matches, col.nulls().cloned())
    }
}

/// Test every view with `is_equal` and pack the results a word at a time,
/// negating whole words for a `<>` comparison rather than per row. Bits past
/// the last row are left as the negation produces them; the buffer's length
/// keeps them out of every reader's view.
fn collect_match_bits(
    views: &[u128],
    negate: bool,
    is_equal: impl Fn(u128) -> bool,
) -> BooleanBuffer {
    let mut words: Vec<u64> = Vec::with_capacity(views.len().div_ceil(64));
    for chunk in views.chunks(64) {
        let mut packed = 0u64;
        for (bit, &view) in chunk.iter().enumerate() {
            packed |= (is_equal(view) as u64) << bit;
        }
        words.push(if negate { !packed } else { packed });
    }
    BooleanBuffer::new(words.into(), 0, views.len())
}

#[cfg(test)]
mod tests {
    use super::ConstantMatch;
    use arrow_array::{Array, StringViewArray};

    fn run_not_equal(needle: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let result = ConstantMatch::new(needle).run_not_equal(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    fn run_equal(needle: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let result = ConstantMatch::new(needle).run_equal(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    #[test]
    fn separates_empty_strings_from_the_rest() {
        let result = run_not_equal("", &["", "a", "", "a much longer value than inlines"]);

        assert_eq!(result, vec![false, true, false, true]);
    }

    #[test]
    fn matches_an_inline_constant() {
        let result = run_equal("abc", &["abc", "abd", "ab", "abcd", ""]);

        assert_eq!(result, vec![true, false, false, false, false]);
    }

    #[test]
    fn matches_a_constant_of_exactly_the_inline_limit() {
        let result = run_equal(
            "012345678901",
            &["012345678901", "01234567890", "0123456789012"],
        );

        assert_eq!(result, vec![true, false, false]);
    }

    #[test]
    fn matches_a_buffer_backed_constant() {
        let values = [
            "http://example.com/some/long/path",
            "http://example.com/some/long/pat",
            "http://example.org/some/long/path",
            "",
        ];

        let result = run_equal("http://example.com/some/long/path", &values);

        assert_eq!(result, vec![true, false, false, false]);
    }

    #[test]
    fn mismatch_beyond_the_prefix_of_a_long_constant() {
        let values = ["abcdefghijklmnop-one", "abcdefghijklmnop-two"];

        let result = run_not_equal("abcdefghijklmnop-one", &values);

        assert_eq!(result, vec![false, true]);
    }

    #[test]
    fn packs_more_than_one_word_of_rows() {
        let values: Vec<String> = (0..100)
            .map(|i| {
                if i % 3 == 0 {
                    String::new()
                } else {
                    format!("v{i}")
                }
            })
            .collect();
        let array = StringViewArray::from_iter_values(values.iter().map(|s| s.as_str()));

        let result = ConstantMatch::new("").run_not_equal(&array);

        assert_eq!(result.len(), 100);
        assert!((0..100).all(|i| result.value(i) == (i % 3 != 0)));
    }

    #[test]
    fn null_rows_stay_null() {
        let array = StringViewArray::from(vec![Some("abc"), None, Some("")]);

        let result = ConstantMatch::new("").run_not_equal(&array);

        assert!(result.value(0));
        assert!(result.is_null(1));
        assert!(!result.value(2));
    }

    #[test]
    fn handles_an_empty_column() {
        let array = StringViewArray::from(Vec::<Option<&str>>::new());

        let result = ConstantMatch::new("").run_not_equal(&array);

        assert_eq!(result.len(), 0);
    }
}
