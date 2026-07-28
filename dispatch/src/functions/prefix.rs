use crate::env::MAX_INLINE_STRING_VIEW;
use arrow::array::ByteView;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBufferBuilder;

/// The `Prefix` struct runs starts-with matching (e.g., WHERE URL LIKE 'http://x%')
/// on `StringViewArray`s.
///
/// StringViewArray's layout makes this cheap: every view (u128) inlines the string's
/// first four bytes, so most non-matching rows are rejected by comparing those inline
/// bytes without ever dereferencing into the data buffers. Only rows whose first four
/// bytes match the prefix read the full string.
pub struct Prefix {
    prefix: Vec<u8>,
}

impl Prefix {
    /// Create a new `Prefix` matcher for the given prefix.
    pub fn new<B: ?Sized + AsRef<[u8]>>(prefix: &B) -> Self {
        Self {
            prefix: prefix.as_ref().to_vec(),
        }
    }

    /// Test every string in `col` for the prefix, returning a [`BooleanArray`]
    /// mask. `true` at position *i* means the string at *i* starts with the prefix.
    pub fn run(&self, col: &StringViewArray) -> BooleanArray {
        let prefix = self.prefix.as_slice();
        let prefix_len = prefix.len();
        // Both inline and buffer-backed views store the string's first four
        // bytes at view bytes 4..8; compare those before touching any buffer.
        let inline_check_len = prefix_len.min(4);
        let data_buffers = col.data_buffers();

        let row_count = col.len();
        let mut bitmap = BooleanBufferBuilder::new(row_count);
        bitmap.append_n(row_count, false);

        for (i, &view) in col.views().iter().enumerate() {
            let len = (view as u32) as usize;
            if len < prefix_len {
                continue;
            }
            let view_bytes = view.to_le_bytes();
            if view_bytes[4..4 + inline_check_len] != prefix[..inline_check_len] {
                continue;
            }
            let matched = if prefix_len <= inline_check_len {
                true
            } else if len <= MAX_INLINE_STRING_VIEW {
                view_bytes[4..4 + prefix_len] == prefix[..]
            } else {
                let byte_view = ByteView::from(view);
                let start = byte_view.offset as usize;
                let buffer = &data_buffers[byte_view.buffer_index as usize];
                buffer[start..start + prefix_len] == prefix[..]
            };
            bitmap.set_bit(i, matched);
        }

        // A NULL input row must stay NULL in the mask (not `true`, which an
        // empty prefix would otherwise produce), so the filter drops it.
        BooleanArray::new(bitmap.finish(), col.nulls().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::Prefix;
    use arrow_array::{Array, StringViewArray};

    fn run(prefix: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let result = Prefix::new(prefix).run(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    #[test]
    fn matches_short_inline_strings() {
        let result = run("ab", &["abc", "ab", "ba", "a"]);

        assert_eq!(result, vec![true, true, false, false]);
    }

    #[test]
    fn matches_long_buffer_backed_strings() {
        let values = [
            "http://example.com/some/long/path",
            "https://example.com/some/long/path",
            "http://example.org/another/long/path",
        ];

        let result = run("http://example.com", &values);

        assert_eq!(result, vec![true, false, false]);
    }

    #[test]
    fn mismatch_beyond_the_inline_view_bytes() {
        let values = [
            "abcd-long-enough-to-not-inline",
            "abce-long-enough-to-not-inline",
        ];

        let result = run("abcd", &values);

        assert_eq!(result, vec![true, false]);
    }

    #[test]
    fn substring_elsewhere_does_not_match() {
        let result = run("world", &["hello world", "worldwide news coverage"]);

        assert_eq!(result, vec![false, true]);
    }

    #[test]
    fn empty_prefix_matches_all() {
        let result = run("", &["a", "", "some long string beyond inline"]);

        assert_eq!(result, vec![true, true, true]);
    }

    #[test]
    fn prefix_longer_than_value() {
        let result = run("longer than the value", &["short", ""]);

        assert_eq!(result, vec![false, false]);
    }

    #[test]
    fn null_rows_stay_null_even_for_empty_prefix() {
        let array = StringViewArray::from(vec![Some("abc"), None]);

        let result = Prefix::new("").run(&array);

        assert!(result.value(0));
        assert!(result.is_null(1));
    }

    #[test]
    fn exact_match_is_a_prefix() {
        let result = run("needle", &["needle", "needl"]);

        assert_eq!(result, vec![true, false]);
    }
}
