//! Statement framing for the interactive SQL buffer.

/// Complete statements found in an input buffer and the unfinished tail.
#[derive(Debug, PartialEq, Eq)]
pub struct SplitStatements {
    pub complete: Vec<String>,
    pub remainder: String,
}

#[derive(Debug)]
enum State {
    Normal,
    SingleQuote,
    DoubleQuote,
    DollarQuote(String),
    LineComment,
    BlockComment(usize),
}

/// Split on semicolons that are outside SQL strings, quoted identifiers, and
/// comments. PostgreSQL dollar-quoted strings and nested block comments are
/// recognized so pasted function bodies and complex comments frame correctly.
pub fn split_complete(input: &str) -> SplitStatements {
    let bytes = input.as_bytes();
    let mut state = State::Normal;
    let mut statements = Vec::new();
    let mut statement_start = 0;
    let mut index = 0;

    while index < bytes.len() {
        match &mut state {
            State::Normal => match bytes[index] {
                b'\'' => {
                    state = State::SingleQuote;
                    index += 1;
                }
                b'"' => {
                    state = State::DoubleQuote;
                    index += 1;
                }
                b'-' if bytes.get(index + 1) == Some(&b'-') => {
                    state = State::LineComment;
                    index += 2;
                }
                b'/' if bytes.get(index + 1) == Some(&b'*') => {
                    state = State::BlockComment(1);
                    index += 2;
                }
                b'$' => {
                    if let Some(delimiter) = dollar_delimiter(input, index) {
                        index += delimiter.len();
                        state = State::DollarQuote(delimiter);
                    } else {
                        index += 1;
                    }
                }
                b';' => {
                    let statement = input[statement_start..=index].trim();
                    if contains_sql(statement) {
                        statements.push(statement.to_string());
                    }
                    index += 1;
                    statement_start = index;
                }
                _ => index += char_width(input, index),
            },
            State::SingleQuote => {
                if bytes[index] == b'\'' {
                    if bytes.get(index + 1) == Some(&b'\'') {
                        index += 2;
                    } else {
                        state = State::Normal;
                        index += 1;
                    }
                } else if bytes[index] == b'\\' && bytes.get(index + 1).is_some() {
                    index += 1 + char_width(input, index + 1);
                } else {
                    index += char_width(input, index);
                }
            }
            State::DoubleQuote => {
                if bytes[index] == b'"' {
                    if bytes.get(index + 1) == Some(&b'"') {
                        index += 2;
                    } else {
                        state = State::Normal;
                        index += 1;
                    }
                } else {
                    index += char_width(input, index);
                }
            }
            State::DollarQuote(delimiter) => {
                if input[index..].starts_with(delimiter.as_str()) {
                    index += delimiter.len();
                    state = State::Normal;
                } else {
                    index += char_width(input, index);
                }
            }
            State::LineComment => {
                if bytes[index] == b'\n' {
                    state = State::Normal;
                }
                index += char_width(input, index);
            }
            State::BlockComment(depth) => {
                if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
                    *depth += 1;
                    index += 2;
                } else if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') {
                    *depth -= 1;
                    index += 2;
                    if *depth == 0 {
                        state = State::Normal;
                    }
                } else {
                    index += char_width(input, index);
                }
            }
        }
    }

    let remainder = input[statement_start..].trim_start();
    let has_unfinished_block_comment = matches!(state, State::BlockComment(_));
    SplitStatements {
        complete: statements,
        remainder: if contains_sql(remainder) || has_unfinished_block_comment {
            remainder.to_string()
        } else {
            String::new()
        },
    }
}

fn dollar_delimiter(input: &str, start: usize) -> Option<String> {
    let suffix = &input[start + 1..];
    let close = suffix.find('$')?;
    let tag = &suffix[..close];
    let valid = tag.is_empty()
        || tag.chars().enumerate().all(|(index, character)| {
            character == '_'
                || character.is_ascii_alphabetic()
                || !character.is_ascii()
                || index > 0 && character.is_ascii_digit()
        });
    valid.then(|| input[start..start + close + 2].to_string())
}

fn char_width(input: &str, index: usize) -> usize {
    input[index..].chars().next().unwrap().len_utf8()
}

/// Whether a fragment contains anything except whitespace and comments.
fn contains_sql(fragment: &str) -> bool {
    let bytes = fragment.as_bytes();
    let mut index = 0;
    let mut block_depth = 0;
    while index < bytes.len() {
        if block_depth > 0 {
            if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
                block_depth += 1;
                index += 2;
            } else if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') {
                block_depth -= 1;
                index += 2;
            } else {
                index += char_width(fragment, index);
            }
        } else if bytes[index].is_ascii_whitespace() || bytes[index] == b';' {
            index += 1;
        } else if bytes[index] == b'-' && bytes.get(index + 1) == Some(&b'-') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += char_width(fragment, index);
            }
        } else if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            block_depth = 1;
            index += 2;
        } else {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{SplitStatements, split_complete};

    fn assert_split(input: &str, complete: &[&str], remainder: &str) {
        assert_eq!(
            split_complete(input),
            SplitStatements {
                complete: complete
                    .iter()
                    .map(|statement| (*statement).to_string())
                    .collect(),
                remainder: remainder.to_string(),
            }
        );
    }

    #[test]
    fn splits_several_statements_and_keeps_the_tail() {
        let input = "SELECT 1; SELECT ';' AS value; SELECT";

        assert_split(input, &["SELECT 1;", "SELECT ';' AS value;"], "SELECT");
    }

    #[test]
    fn leaves_a_trailing_meta_command_for_the_shell() {
        assert_split("SELECT 1; \\timing", &["SELECT 1;"], "\\timing");
    }

    #[test]
    fn quoted_strings_and_identifiers_do_not_end_at_their_semicolons() {
        let input =
            r#"SELECT ';', 'it''s; fine', "semi;""colon"; SELECT 'escaped \'; still quoted';"#;

        assert_split(
            input,
            &[
                r#"SELECT ';', 'it''s; fine', "semi;""colon";"#,
                r#"SELECT 'escaped \'; still quoted';"#,
            ],
            "",
        );
    }

    #[test]
    fn dollar_quotes_hide_semicolons_but_positional_parameters_do_not() {
        let input = "SELECT $$a;b$$; SELECT $body_2$c;d$body_2$; SELECT $雪$c;d$雪$; SELECT $1;";

        assert_split(
            input,
            &[
                "SELECT $$a;b$$;",
                "SELECT $body_2$c;d$body_2$;",
                "SELECT $雪$c;d$雪$;",
                "SELECT $1;",
            ],
            "",
        );
    }

    #[test]
    fn line_and_nested_block_comments_hide_semicolons() {
        let input = "/* outer; /* nested; */ */ SELECT 1; -- between;\nSELECT 2;";

        assert_split(
            input,
            &[
                "/* outer; /* nested; */ */ SELECT 1;",
                "-- between;\nSELECT 2;",
            ],
            "",
        );
    }

    #[test]
    fn empty_and_comment_only_statements_are_ignored() {
        let input = " ; -- only;\n; /* closed; */ ; \n";

        assert_split(input, &[], "");
    }

    #[test]
    fn trailing_complete_comments_are_not_kept_as_a_remainder() {
        let input = "SELECT 1; -- trailing;\n/* closed; */";

        assert_split(input, &["SELECT 1;"], "");
    }

    #[test]
    fn every_unterminated_quoted_form_stays_in_the_buffer() {
        assert_split("SELECT 'hello;\nworld", &[], "SELECT 'hello;\nworld");
        assert_split(
            "SELECT \"unfinished;\nidentifier",
            &[],
            "SELECT \"unfinished;\nidentifier",
        );
        assert_split(
            "SELECT $body$unfinished;\nbody",
            &[],
            "SELECT $body$unfinished;\nbody",
        );
    }

    #[test]
    fn an_unterminated_block_comment_stays_in_the_buffer() {
        assert_split("/* unfinished;", &[], "/* unfinished;");
        assert_split(
            "SELECT 1; /* outer; /* nested */",
            &["SELECT 1;"],
            "/* outer; /* nested */",
        );
    }

    #[test]
    fn closing_a_multiline_construct_allows_its_statement_to_complete() {
        let input = "SELECT 1 /* open;\nstill open */;";

        assert_split(input, &[input], "");
    }

    #[test]
    fn unicode_around_semicolons_preserves_byte_boundaries() {
        let input = r#"SELECT '雪;☃', "列;名"; SELECT 'escaped \☃;'; SELECT 'é"#;

        assert_split(
            input,
            &[r#"SELECT '雪;☃', "列;名";"#, r#"SELECT 'escaped \☃;';"#],
            "SELECT 'é",
        );
    }
}
