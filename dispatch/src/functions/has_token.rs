use crate::text_index::split_tokens;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBufferBuilder;

/// `has_token(text, token)`: whether `token` is one of the tokens of a text,
/// as a text index splits it (see [`split_tokens`]). The token is matched
/// byte for byte, so `Disk` is not found in a text that says `disk`.
///
/// This is what a search by token answers when it reads the text itself, and
/// so what an index of the column has to agree with.
pub struct HasToken {
    token: Vec<u8>,
}

impl HasToken {
    /// A matcher for `token`, which must be one token: not empty, and with
    /// nothing in it that separates tokens. Text has no token that is not.
    pub fn new(token: &[u8]) -> Self {
        assert!(
            Self::is_one_token(token),
            "has_token looks for one token, and {token:?} is not one"
        );
        Self {
            token: token.to_vec(),
        }
    }

    /// Whether `text` is exactly one token.
    pub fn is_one_token(text: &[u8]) -> bool {
        !text.is_empty() && split_tokens(text).eq([text])
    }

    /// Test every text of `texts` for the token. A row without a text stays
    /// NULL in the answer.
    pub fn run(&self, texts: &StringViewArray) -> BooleanArray {
        let mut has_token = BooleanBufferBuilder::new(texts.len());
        for row in 0..texts.len() {
            let text = texts.value(row).as_bytes();
            has_token.append(split_tokens(text).any(|token| token == self.token));
        }
        BooleanArray::new(has_token.finish(), texts.nulls().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::HasToken;
    use arrow_array::{Array, StringViewArray};

    fn run(token: &str, texts: &[&str]) -> Vec<bool> {
        let texts = StringViewArray::from_iter_values(texts.iter().copied());
        let result = HasToken::new(token.as_bytes()).run(&texts);
        (0..result.len()).map(|row| result.value(row)).collect()
    }

    #[test]
    fn a_text_has_the_tokens_it_is_cut_into() {
        let texts = ["disk /dev/sda1 is full", "net down", "sda1"];

        let result = run("sda1", &texts);

        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn a_part_of_a_token_is_not_a_token() {
        let texts = ["diskette", "a disk", "disk1"];

        let result = run("disk", &texts);

        assert_eq!(result, vec![false, true, false]);
    }

    #[test]
    fn letter_case_tells_tokens_apart() {
        let texts = ["Disk full", "disk full"];

        let result = run("disk", &texts);

        assert_eq!(result, vec![false, true]);
    }

    #[test]
    fn a_row_without_text_stays_null() {
        let texts = StringViewArray::from_iter([Some("disk"), None]);

        let result = HasToken::new(b"disk").run(&texts);

        assert!(result.value(0));
        assert!(result.is_null(1));
    }

    #[test]
    fn one_token_has_no_separator_and_is_not_empty() {
        assert!(HasToken::is_one_token(b"sda1"));
        assert!(!HasToken::is_one_token(b"dev sda1"));
        assert!(!HasToken::is_one_token(b"sda1."));
        assert!(!HasToken::is_one_token(b""));
    }
}
