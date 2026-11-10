//! Splitting text into tokens.
//!
//! ```text
//!   "GET /api/v2/users?id=42 timeout"
//!    └┬┘  └┬┘ └┤ └─┬─┘ └┤ └┤ └──┬──┘
//!    GET  api v2 users id 42 timeout
//! ```
//!
//! A token is a maximal run of token bytes: ASCII letters and digits, and
//! every byte of a multi-byte UTF-8 character. Everything else separates.
//! Tokens are compared byte for byte, so matching is case-sensitive.
//!
//! The index and the scalar functions share this one definition: a row
//! matches `has_token(text, 'users')` exactly when `users` is one of the
//! tokens this module yields for the row.

/// The longest token the index stores. Longer runs (encoded blobs, hashes
/// glued together) are left out of the index, so a search for one cannot be
/// answered from it and has to read the text.
pub const MAX_INDEXED_TOKEN_LEN: usize = 255;

const TOKEN_BYTE: [bool; 256] = {
    let mut table = [false; 256];
    let mut byte = 0usize;
    while byte < 256 {
        table[byte] = (byte as u8).is_ascii_alphanumeric() || byte >= 0x80;
        byte += 1;
    }
    table
};

/// Whether `byte` is part of a token rather than a separator.
#[inline]
pub fn is_token_byte(byte: u8) -> bool {
    TOKEN_BYTE[byte as usize]
}

/// The tokens of `text`, in order of appearance. A token appearing twice is
/// yielded twice.
pub fn split_tokens(text: &[u8]) -> Tokens<'_> {
    Tokens { rest: text }
}

/// Whether `text` is exactly one token: non-empty, with no separator in it.
pub fn is_single_token(text: &[u8]) -> bool {
    !text.is_empty() && text.iter().all(|&byte| is_token_byte(byte))
}

/// Iterator over the tokens of a text. See [`split_tokens`].
pub struct Tokens<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for Tokens<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let start = self.rest.iter().position(|&byte| is_token_byte(byte))?;
        let after_start = &self.rest[start..];
        let len = after_start
            .iter()
            .position(|&byte| !is_token_byte(byte))
            .unwrap_or(after_start.len());
        let (token, rest) = after_start.split_at(len);
        self.rest = rest;
        Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_of(text: &str) -> Vec<&str> {
        split_tokens(text.as_bytes())
            .map(|token| std::str::from_utf8(token).unwrap())
            .collect()
    }

    #[test]
    fn separators_split_and_are_dropped() {
        let tokens = tokens_of("GET /api/v2/users?id=42 timeout");

        assert_eq!(
            tokens,
            vec!["GET", "api", "v2", "users", "id", "42", "timeout"]
        );
    }

    #[test]
    fn text_without_token_bytes_has_no_tokens() {
        let tokens = tokens_of(" -- / ");

        assert!(tokens.is_empty());
    }

    #[test]
    fn multi_byte_characters_stay_inside_their_token() {
        let tokens = tokens_of("naïve café-crème");

        assert_eq!(tokens, vec!["naïve", "café", "crème"]);
    }

    #[test]
    fn a_repeated_token_is_yielded_each_time() {
        let tokens = tokens_of("retry retry");

        assert_eq!(tokens, vec!["retry", "retry"]);
    }

    #[test]
    fn a_single_token_has_no_separator_and_is_not_empty() {
        assert!(is_single_token(b"timeout"));
        assert!(!is_single_token(b"time out"));
        assert!(!is_single_token(b""));
    }
}
