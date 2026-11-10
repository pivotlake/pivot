use crate::text_index::tokenizer::is_token_byte;
use crate::text_index::{TextQuery, TokenMatch};
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBufferBuilder;
use memchr::memmem::Finder;

/// The `HasTokens` struct tests each string of a `StringViewArray` for whole
/// tokens: `has_token(message, 'timeout')`, and the all-of and any-of forms
/// over several tokens.
///
/// A token is found in a string where its bytes occur with no token byte on
/// either side (see [`tokenizer`](crate::text_index::tokenizer)):
///
/// ```text
///   token "time"      "time out"   found, a separator follows
///                     "timeout"    not found, the occurrence runs on into "out"
/// ```
///
/// This is the same answer a text index gives for the tokens it stores, read
/// from the text itself, so it holds for files with and without an index.
pub struct HasTokens {
    tokens: Vec<Finder<'static>>,
    token_match: TokenMatch,
}

impl HasTokens {
    pub fn new(query: &TextQuery) -> Self {
        Self {
            tokens: query
                .tokens()
                .map(|token| Finder::new(token).into_owned())
                .collect(),
            token_match: query.token_match(),
        }
    }

    /// Test every string in `col`, returning a [`BooleanArray`] mask. `true`
    /// at position *i* means the string at *i* holds the tokens asked for.
    pub fn run(&self, col: &StringViewArray) -> BooleanArray {
        let row_count = col.len();
        let mut bitmap = BooleanBufferBuilder::new(row_count);
        for row in 0..row_count {
            let text = col.value(row).as_bytes();
            let mut tokens = self.tokens.iter();
            let matches = match self.token_match {
                TokenMatch::All => tokens.all(|token| holds_token(text, token)),
                TokenMatch::Any => tokens.any(|token| holds_token(text, token)),
            };
            bitmap.append(matches);
        }

        // A NULL input row must stay NULL in the mask (not `true`, which an
        // all-of over no tokens would otherwise produce), so the filter drops it.
        BooleanArray::new(bitmap.finish(), col.nulls().cloned())
    }
}

/// Whether `token` occurs in `text` as a whole token.
fn holds_token(text: &[u8], token: &Finder<'_>) -> bool {
    let token_len = token.needle().len();
    let mut search_from = 0;
    while let Some(found) = token.find(&text[search_from..]) {
        let start = search_from + found;
        let end = start + token_len;
        let starts_a_token = start == 0 || !is_token_byte(text[start - 1]);
        let ends_a_token = end == text.len() || !is_token_byte(text[end]);
        if starts_a_token && ends_a_token {
            return true;
        }
        search_from = start + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::HasTokens;
    use crate::text_index::{TextQuery, TokenMatch};
    use arrow_array::{Array, StringViewArray};

    fn run(tokens: &str, token_match: TokenMatch, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let query = TextQuery::from_text(tokens.as_bytes(), token_match);
        let result = HasTokens::new(&query).run(&array);
        (0..result.len()).map(|i| result.value(i)).collect()
    }

    #[test]
    fn matches_a_whole_token_only() {
        let values = [
            "request timeout",
            "timeouts everywhere",
            "a timeout.",
            "time",
        ];

        let result = run("timeout", TokenMatch::All, &values);

        assert_eq!(result, vec![true, false, true, false]);
    }

    #[test]
    fn matches_a_token_after_a_longer_one_holding_it() {
        let values = ["retimeout timeout", "retimeout"];

        let result = run("timeout", TokenMatch::All, &values);

        assert_eq!(result, vec![true, false]);
    }

    #[test]
    fn matches_in_strings_too_long_to_be_inline() {
        let values = [
            "GET /api/v2/users?id=42 returned timeout after 30s",
            "GET /api/v2/users?id=42 returned ok after 30 seconds",
        ];

        let result = run("timeout", TokenMatch::All, &values);

        assert_eq!(result, vec![true, false]);
    }

    #[test]
    fn all_needs_every_token() {
        let values = ["GET users timeout", "GET users", "timeout"];

        let result = run("timeout GET", TokenMatch::All, &values);

        assert_eq!(result, vec![true, false, false]);
    }

    #[test]
    fn any_needs_one_token() {
        let values = ["GET users timeout", "POST users", "timeout"];

        let result = run("timeout GET", TokenMatch::Any, &values);

        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn matching_is_case_sensitive() {
        let values = ["Timeout", "timeout"];

        let result = run("timeout", TokenMatch::All, &values);

        assert_eq!(result, vec![false, true]);
    }

    #[test]
    fn a_null_row_stays_null() {
        let array = StringViewArray::from(vec![Some("timeout"), None]);
        let query = TextQuery::from_text(b"", TokenMatch::All);

        let result = HasTokens::new(&query).run(&array);

        assert!(result.value(0));
        assert!(result.is_null(1));
    }
}
