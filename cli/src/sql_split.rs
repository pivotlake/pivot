//! Splitting console input into complete SQL statements.
//!
//! A statement ends at a semicolon that sits outside every quoted string and
//! comment, the same boundary psql uses. Input may hold several statements,
//! and its tail may be an unfinished statement still being typed; the caller
//! keeps that tail and feeds it back in with the next line appended.

/// Split `input` into the complete statements it holds (in order, trimmed,
/// without their terminating semicolon) and the unterminated remainder.
/// Statements that are empty after trimming are dropped, so a stray `;`
/// produces nothing.
pub fn split_statements(input: &str) -> (Vec<String>, String) {
    #[derive(PartialEq)]
    enum State {
        Normal,
        SingleQuote,
        DoubleQuote,
        LineComment,
        BlockComment(usize),
    }

    let mut statements = Vec::new();
    let mut current = String::new();
    let mut state = State::Normal;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match state {
            State::Normal => match c {
                ';' => {
                    let statement = current.trim();
                    if !statement.is_empty() {
                        statements.push(statement.to_string());
                    }
                    current.clear();
                    continue;
                }
                '\'' => state = State::SingleQuote,
                '"' => state = State::DoubleQuote,
                '-' if chars.peek() == Some(&'-') => state = State::LineComment,
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    current.push_str("/*");
                    state = State::BlockComment(1);
                    continue;
                }
                _ => {}
            },
            // A doubled quote (`''` / `""`) reads as leave-then-reenter, which
            // needs no lookahead and still never ends a statement inside it.
            State::SingleQuote if c == '\'' => state = State::Normal,
            State::DoubleQuote if c == '"' => state = State::Normal,
            State::LineComment if c == '\n' => state = State::Normal,
            State::BlockComment(depth) => match c {
                '*' if chars.peek() == Some(&'/') => {
                    chars.next();
                    current.push_str("*/");
                    state = if depth == 1 {
                        State::Normal
                    } else {
                        State::BlockComment(depth - 1)
                    };
                    continue;
                }
                // Postgres block comments nest.
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    current.push_str("/*");
                    state = State::BlockComment(depth + 1);
                    continue;
                }
                _ => {}
            },
            _ => {}
        }
        current.push(c);
    }
    (statements, current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_semicolon_inside_a_string_does_not_end_the_statement() {
        let (statements, rest) = split_statements("SELECT 'a;b' AS x;");

        assert_eq!(statements, vec!["SELECT 'a;b' AS x"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn two_statements_on_one_line_split_in_order() {
        let (statements, rest) = split_statements("SELECT 1; SELECT 2;");

        assert_eq!(statements, vec!["SELECT 1", "SELECT 2"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn an_unterminated_statement_is_returned_as_the_remainder() {
        let (statements, rest) = split_statements("SELECT 1;\nSELECT 2,");

        assert_eq!(statements, vec!["SELECT 1"]);
        assert_eq!(rest, "\nSELECT 2,");
    }

    #[test]
    fn a_semicolon_inside_comments_does_not_split() {
        let (statements, rest) = split_statements("SELECT 1 -- one; two\n+ 2; /* a;b */ SELECT 3;");

        assert_eq!(
            statements,
            vec!["SELECT 1 -- one; two\n+ 2", "/* a;b */ SELECT 3"]
        );
        assert_eq!(rest, "");
    }

    #[test]
    fn a_doubled_quote_stays_inside_the_string() {
        let (statements, rest) = split_statements("SELECT 'it''s;fine';");

        assert_eq!(statements, vec!["SELECT 'it''s;fine'"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn empty_statements_are_dropped() {
        let (statements, rest) = split_statements(" ; ;; SELECT 1; ");

        assert_eq!(statements, vec!["SELECT 1"]);
        assert_eq!(rest, " ");
    }
}
