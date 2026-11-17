//! What a token is.
//!
//! Text is cut at every ASCII byte that is neither a letter nor a digit, and
//! the pieces left between the cuts are the tokens:
//!
//! ```text
//!   "disk /dev/sda1 is 91% full"   ──▶   disk  dev  sda1  is  91  full
//! ```
//!
//! Bytes outside ASCII never cut, so a character encoded in several bytes
//! stays whole inside its token. A token is its bytes as they appear in the
//! text: `Disk` and `disk` are two tokens.
//!
//! An index is built from the tokens of its column and searched by token, so
//! both sides go through this one function.

/// The tokens of `text`, in the order they appear. A token that appears
/// several times is yielded each time.
pub fn split_tokens(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    text.split(|byte| byte.is_ascii() && !byte.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
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
    fn text_is_cut_at_punctuation_and_spaces() {
        let tokens = tokens_of("disk /dev/sda1 is 91% full");

        assert_eq!(tokens, ["disk", "dev", "sda1", "is", "91", "full"]);
    }

    #[test]
    fn text_of_separators_alone_has_no_tokens() {
        let tokens = tokens_of("  ...  --");

        assert!(tokens.is_empty());
    }

    #[test]
    fn empty_text_has_no_tokens() {
        let tokens = tokens_of("");

        assert!(tokens.is_empty());
    }

    #[test]
    fn a_character_of_several_bytes_does_not_cut_its_token() {
        let tokens = tokens_of("über-größe 東京:駅");

        assert_eq!(tokens, ["über", "größe", "東京", "駅"]);
    }

    #[test]
    fn letter_case_tells_tokens_apart() {
        let tokens = tokens_of("Disk disk");

        assert_eq!(tokens, ["Disk", "disk"]);
    }

    #[test]
    fn a_token_appearing_twice_is_yielded_twice() {
        let tokens = tokens_of("ping ping");

        assert_eq!(tokens, ["ping", "ping"]);
    }
}
