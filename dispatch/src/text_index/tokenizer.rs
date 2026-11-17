//! What a token is.
//!
//! Text is cut at every ASCII byte that is neither a letter nor a digit, and
//! the pieces left between the cuts are the tokens:
//!
//! ```text
//!   "Disk /dev/sda1 is 91% full"   ──▶   Disk  dev  sda1  is  91  full
//! ```
//!
//! Bytes outside ASCII never cut, so a character encoded in several bytes
//! stays whole inside its token.
//!
//! Whether `Disk` and `disk` are one token is the index's [`LetterCase`]. An
//! index is built from the tokens of its column and searched by token, so
//! both sides go through one [`Tokenizer`] with the index's letter case.

/// Whether tokens differing only in the case of their letters are one token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LetterCase {
    /// A token is its bytes as they appear in the text: `Disk` and `disk`
    /// are two tokens.
    Kept,
    /// The ASCII letters of a token are lowered: `Disk` and `disk` are the
    /// one token `disk`. Letters outside ASCII are left as they are.
    Lowered,
}

/// The tokens of `text` as they appear in it, in order. A token that appears
/// several times is yielded each time.
pub fn split_tokens(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    text.split(|byte| byte.is_ascii() && !byte.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
}

/// Splits text into the tokens of an index with a given [`LetterCase`].
pub struct Tokenizer {
    letter_case: LetterCase,
    /// The token being visited with its letters lowered, when they are.
    lowered_token: Vec<u8>,
}

impl Tokenizer {
    pub fn new(letter_case: LetterCase) -> Self {
        Self {
            letter_case,
            lowered_token: Vec::new(),
        }
    }

    /// Visit the tokens of `text`, in the order they appear. A token that
    /// appears several times is visited each time.
    pub fn for_each_token(&mut self, text: &[u8], mut visit: impl FnMut(&[u8])) {
        for token in split_tokens(text) {
            match self.letter_case {
                LetterCase::Kept => visit(token),
                LetterCase::Lowered => {
                    self.lowered_token.clear();
                    self.lowered_token
                        .extend(token.iter().map(u8::to_ascii_lowercase));
                    visit(&self.lowered_token);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_of(text: &str, letter_case: LetterCase) -> Vec<String> {
        let mut tokens = Vec::new();
        Tokenizer::new(letter_case).for_each_token(text.as_bytes(), |token| {
            tokens.push(String::from_utf8(token.to_vec()).unwrap())
        });
        tokens
    }

    #[test]
    fn text_is_cut_at_punctuation_and_spaces() {
        let tokens = tokens_of("disk /dev/sda1 is 91% full", LetterCase::Kept);

        assert_eq!(tokens, ["disk", "dev", "sda1", "is", "91", "full"]);
    }

    #[test]
    fn text_of_separators_alone_has_no_tokens() {
        let tokens = tokens_of("  ...  --", LetterCase::Kept);

        assert!(tokens.is_empty());
    }

    #[test]
    fn empty_text_has_no_tokens() {
        let tokens = tokens_of("", LetterCase::Kept);

        assert!(tokens.is_empty());
    }

    #[test]
    fn a_character_of_several_bytes_does_not_cut_its_token() {
        let tokens = tokens_of("über-größe 東京:駅", LetterCase::Kept);

        assert_eq!(tokens, ["über", "größe", "東京", "駅"]);
    }

    #[test]
    fn a_token_appearing_twice_is_visited_twice() {
        let tokens = tokens_of("ping ping", LetterCase::Kept);

        assert_eq!(tokens, ["ping", "ping"]);
    }

    #[test]
    fn kept_letter_case_tells_tokens_apart() {
        let tokens = tokens_of("Disk disk", LetterCase::Kept);

        assert_eq!(tokens, ["Disk", "disk"]);
    }

    #[test]
    fn lowered_letter_case_makes_them_one_token() {
        let tokens = tokens_of("Disk DISK disk", LetterCase::Lowered);

        assert_eq!(tokens, ["disk", "disk", "disk"]);
    }

    #[test]
    fn lowering_leaves_letters_outside_ascii_as_they_are() {
        let tokens = tokens_of("ÜBER Größe", LetterCase::Lowered);

        assert_eq!(tokens, ["Über", "größe"]);
    }
}
