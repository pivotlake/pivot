//! Prepared statements for the extended query protocol.
//!
//! A client that prepares a statement sends `Parse` with the SQL, `Bind` with
//! one value per placeholder, then `Execute`. Pivot's planner has no notion of
//! a parameter, so a prepared statement is kept here as the literal SQL
//! *between* its placeholders: binding joins those fragments back together with
//! one rendered literal in each gap, and the result is planned and run exactly
//! like a statement that arrived on the simple protocol.
//!
//! Both placeholder spellings are recognised: `?`, numbered left to right, and
//! `$n`, which names the value it wants.
//!
//! Parameters we are asked to infer are described to the client as `TEXT`,
//! since inferring a real type would need planner support. A value therefore
//! arrives as its text form and is substituted as a quoted literal, which the
//! planner casts to whatever the statement needs (a `BIGINT` column accepts
//! `'4'`). A client that declares parameter types itself at `Parse` is taken at
//! its word, and those values are decoded as the declared type.
//!
//! A bound SQL NULL becomes a `NULL` literal, which the planner cannot yet fold
//! into a constant: it errors on a numeric column and stores the text `NULL` on
//! a string one. The limitation belongs to the planner's scalar conversion, and
//! a `NULL` written by hand into a simple query hits it just the same.

use std::str;

use async_trait::async_trait;
use pgwire::api::portal::{Format, Portal};
use pgwire::api::results::FieldInfo;
use pgwire::api::stmt::QueryParser;
use pgwire::api::{ClientInfo, Type};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use thiserror::Error;

/// SQL split around the parameter placeholders it contains.
#[derive(Clone, Debug)]
pub struct ParameterizedSql {
    /// The literal SQL around the placeholders: `fragments[i]` precedes the
    /// value of `parameters[i]`, and the final fragment is the tail after the
    /// last placeholder. Always one longer than `parameters`.
    fragments: Vec<String>,
    /// The bound value each placeholder reads, zero-based, in the order the
    /// placeholders appear.
    parameters: Vec<usize>,
}

impl ParameterizedSql {
    /// Split `sql` around its placeholders. Placeholders inside string
    /// literals, quoted identifiers and comments are text like any other, so
    /// those regions are skipped.
    pub(crate) fn parse(sql: &str) -> Self {
        let bytes = sql.as_bytes();
        let mut fragments = Vec::new();
        let mut parameters = Vec::new();
        // Start of the fragment being accumulated: just past the last
        // placeholder, or the start of the statement.
        let mut fragment_start = 0;
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'\'' | b'"' => i = skip_quoted(bytes, i),
                b'-' if bytes.get(i + 1) == Some(&b'-') => i = skip_line_comment(bytes, i),
                b'/' if bytes.get(i + 1) == Some(&b'*') => i = skip_block_comment(bytes, i),
                b'?' => {
                    fragments.push(sql[fragment_start..i].to_string());
                    parameters.push(parameters.len());
                    i += 1;
                    fragment_start = i;
                }
                b'$' => {
                    let (number, end) = read_number(bytes, i + 1);
                    // `$0` names no parameter and `$` alone is not one either:
                    // leave both to the planner to complain about.
                    if number == 0 {
                        i = end.max(i + 1);
                        continue;
                    }
                    fragments.push(sql[fragment_start..i].to_string());
                    parameters.push(number - 1);
                    i = end;
                    fragment_start = i;
                }
                _ => i += 1,
            }
        }
        fragments.push(sql[fragment_start..].to_string());
        Self {
            fragments,
            parameters,
        }
    }

    /// How many values a `Bind` has to carry, which is the highest-numbered
    /// parameter the statement reads.
    pub(crate) fn parameter_count(&self) -> usize {
        self.parameters
            .iter()
            .max()
            .map_or(0, |highest| highest + 1)
    }

    /// Join the fragments back together with one rendered literal per
    /// placeholder.
    fn render(&self, values: &[String]) -> Result<String, Error> {
        let mut sql = String::new();
        for (fragment, &parameter) in self.fragments.iter().zip(&self.parameters) {
            let value = values.get(parameter).ok_or(Error::MissingParameter {
                wanted: parameter + 1,
                bound: values.len(),
            })?;
            sql.push_str(fragment);
            sql.push_str(value);
        }
        // One more fragment than placeholders: the tail after the last one.
        sql.push_str(self.fragments.last().expect("at least one fragment"));
        Ok(sql)
    }
}

/// Index just past the string literal or quoted identifier that starts at
/// `start`. A doubled quote inside one is an escaped quote and does not end it.
/// An unterminated quote runs to the end of the statement, which the planner
/// then rejects.
fn skip_quoted(bytes: &[u8], start: usize) -> usize {
    let quote = bytes[start];
    let mut i = start + 1;
    while i < bytes.len() {
        if bytes[i] == quote {
            if bytes.get(i + 1) == Some(&quote) {
                i += 2;
                continue;
            }
            return i + 1;
        }
        i += 1;
    }
    i
}

/// Index just past the `--` comment that starts at `start`, which the next
/// newline ends.
fn skip_line_comment(bytes: &[u8], start: usize) -> usize {
    match bytes[start..].iter().position(|&b| b == b'\n') {
        Some(offset) => start + offset + 1,
        None => bytes.len(),
    }
}

/// Index just past the `/* … */` comment that starts at `start`.
fn skip_block_comment(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 2;
    while i + 1 < bytes.len() {
        if bytes[i] == b'*' && bytes[i + 1] == b'/' {
            return i + 2;
        }
        i += 1;
    }
    bytes.len()
}

/// The decimal number at `start` and the index just past it. Yields zero when
/// there is no digit there, which no placeholder can be.
fn read_number(bytes: &[u8], start: usize) -> (usize, usize) {
    let mut end = start;
    let mut number = 0usize;
    while let Some(digit) = bytes.get(end).and_then(|b| (*b as char).to_digit(10)) {
        number = number.saturating_mul(10).saturating_add(digit as usize);
        end += 1;
    }
    (number, end)
}

/// Everything that can go wrong turning a bound portal into a statement to run.
#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("statement reads parameter ${wanted} but only {bound} were bound")]
    MissingParameter { wanted: usize, bound: usize },
    #[error("parameter ${index} is not valid UTF-8")]
    NotUtf8 { index: usize },
    #[error("parameter ${index} was sent as binary {type_name}, which is not supported")]
    UnsupportedBinary { index: usize, type_name: String },
    #[error(
        "the extended query protocol serves only statements that return no rows; \
         run this one as a simple query"
    )]
    RowsNotSupported,
}

impl Error {
    /// Convert into a `PgWireError::UserError` so it serialises as a normal
    /// error response on the wire.
    pub(crate) fn into_pgwire(self) -> PgWireError {
        const FEATURE_NOT_SUPPORTED: &str = "0A000";
        const PROTOCOL_VIOLATION: &str = "08P01";

        let sqlstate = match self {
            Error::MissingParameter { .. } | Error::NotUtf8 { .. } => PROTOCOL_VIOLATION,
            Error::UnsupportedBinary { .. } | Error::RowsNotSupported => FEATURE_NOT_SUPPORTED,
        };
        PgWireError::UserError(Box::new(ErrorInfo::new(
            "ERROR".to_string(),
            sqlstate.to_string(),
            self.to_string(),
        )))
    }
}

/// Splits each `Parse`d statement around its placeholders. The statement is not
/// planned here: a placeholder is not SQL the planner understands, so nothing
/// can be resolved until `Bind` supplies the values.
#[derive(Debug)]
pub struct PlaceholderParser;

#[async_trait]
impl QueryParser for PlaceholderParser {
    type Statement = ParameterizedSql;

    async fn parse_sql<C>(
        &self,
        _client: &C,
        sql: &str,
        _types: &[Option<Type>],
    ) -> PgWireResult<Self::Statement>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        Ok(ParameterizedSql::parse(sql))
    }

    /// Every parameter the client left to us is described as `TEXT` (see the
    /// module docs). A type the client declared itself wins over this, which
    /// pgwire's `do_describe_statement` applies.
    fn get_parameter_types(&self, statement: &Self::Statement) -> PgWireResult<Vec<Type>> {
        Ok(vec![Type::TEXT; statement.parameter_count()])
    }

    /// No statement served here returns rows, so there is never a row
    /// description to give.
    fn get_result_schema(
        &self,
        _statement: &Self::Statement,
        _column_format: Option<&Format>,
    ) -> PgWireResult<Vec<FieldInfo>> {
        Ok(Vec::new())
    }
}

/// The statement a bound portal stands for: its SQL with every placeholder
/// replaced by the literal its bound value renders as.
pub(crate) fn bound_sql(portal: &Portal<ParameterizedSql>) -> Result<String, Error> {
    let values = (0..portal.parameter_len())
        .map(|index| render_parameter(portal, index))
        .collect::<Result<Vec<_>, _>>()?;
    portal.statement.statement.render(&values)
}

/// Render one bound value as the SQL literal that replaces its placeholder.
fn render_parameter(portal: &Portal<ParameterizedSql>, index: usize) -> Result<String, Error> {
    // Parameters are `$1`-based everywhere a client can see them.
    let number = index + 1;
    // A NULL parameter carries no bytes at all.
    let Some(bytes) = &portal.parameters[index] else {
        return Ok("NULL".to_string());
    };
    if portal.parameter_format.is_text(index) {
        // The client sent the value's text form, which is the literal once
        // quoted.
        let text = str::from_utf8(bytes).map_err(|_| Error::NotUtf8 { index: number })?;
        return Ok(quote_literal(text));
    }

    // Binary. Anything we described ourselves is TEXT, whose binary form is its
    // UTF-8 text, so a value only lands on a typed arm when the client declared
    // the type at `Parse`.
    let declared = portal
        .statement
        .parameter_types
        .get(index)
        .and_then(Option::as_ref)
        .unwrap_or(&Type::TEXT);
    // Decode as the Rust type that carries the declared Postgres type;
    // `to_string` renders it as the numeric or boolean literal it stands for.
    macro_rules! literal_as {
        ($rust:ty) => {
            portal
                .parameter::<$rust>(index, declared)
                .map_err(|_| Error::UnsupportedBinary {
                    index: number,
                    type_name: declared.name().to_string(),
                })?
                .map_or_else(|| "NULL".to_string(), |value| value.to_string())
        };
    }
    if *declared == Type::BOOL {
        Ok(literal_as!(bool))
    } else if *declared == Type::INT2 {
        Ok(literal_as!(i16))
    } else if *declared == Type::INT4 {
        Ok(literal_as!(i32))
    } else if *declared == Type::INT8 {
        Ok(literal_as!(i64))
    } else if *declared == Type::FLOAT4 {
        Ok(literal_as!(f32))
    } else if *declared == Type::FLOAT8 {
        Ok(literal_as!(f64))
    } else if is_textual(declared) {
        let text = str::from_utf8(bytes).map_err(|_| Error::NotUtf8 { index: number })?;
        Ok(quote_literal(text))
    } else {
        Err(Error::UnsupportedBinary {
            index: number,
            type_name: declared.name().to_string(),
        })
    }
}

/// Whether a declared type's binary form is plain UTF-8 text.
fn is_textual(declared: &Type) -> bool {
    [
        Type::TEXT,
        Type::VARCHAR,
        Type::BPCHAR,
        Type::NAME,
        Type::UNKNOWN,
    ]
    .contains(declared)
}

/// Quote `value` as a SQL string literal, doubling the quotes inside it.
fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(sql: &str, values: &[&str]) -> String {
        let values: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        ParameterizedSql::parse(sql).render(&values).unwrap()
    }

    #[test]
    fn question_marks_are_numbered_left_to_right() {
        let statement = ParameterizedSql::parse("INSERT INTO t VALUES (?, ?), (?, ?)");

        assert_eq!(statement.parameter_count(), 4);
        assert_eq!(
            statement
                .render(&["1", "'a'", "2", "'b'"].map(String::from))
                .unwrap(),
            "INSERT INTO t VALUES (1, 'a'), (2, 'b')"
        );
    }

    #[test]
    fn dollar_placeholders_read_the_value_they_name() {
        let statement = ParameterizedSql::parse("INSERT INTO t VALUES ($2, $1, $2)");

        assert_eq!(statement.parameter_count(), 2);
        assert_eq!(
            statement.render(&["'a'", "'b'"].map(String::from)).unwrap(),
            "INSERT INTO t VALUES ('b', 'a', 'b')"
        );
    }

    #[test]
    fn a_statement_without_placeholders_is_unchanged() {
        let statement = ParameterizedSql::parse("INSERT INTO t VALUES (1)");

        assert_eq!(statement.parameter_count(), 0);
        assert_eq!(statement.render(&[]).unwrap(), "INSERT INTO t VALUES (1)");
    }

    #[test]
    fn quotes_and_comments_hide_their_placeholders() {
        assert_eq!(
            render("INSERT INTO t VALUES ('why?', ?)", &["1"]),
            "INSERT INTO t VALUES ('why?', 1)"
        );
        assert_eq!(
            render(r#"INSERT INTO "why?" VALUES (?)"#, &["1"]),
            r#"INSERT INTO "why?" VALUES (1)"#
        );
        assert_eq!(
            render("INSERT INTO t VALUES (?) -- or ? later\n", &["1"]),
            "INSERT INTO t VALUES (1) -- or ? later\n"
        );
        assert_eq!(
            render("INSERT /* ? */ INTO t VALUES (?)", &["1"]),
            "INSERT /* ? */ INTO t VALUES (1)"
        );
    }

    #[test]
    fn an_escaped_quote_does_not_end_a_literal() {
        assert_eq!(
            render("INSERT INTO t VALUES ('it''s ?', ?)", &["1"]),
            "INSERT INTO t VALUES ('it''s ?', 1)"
        );
    }

    #[test]
    fn a_statement_reading_an_unbound_parameter_is_rejected() {
        let statement = ParameterizedSql::parse("INSERT INTO t VALUES ($1, $2)");

        let error = statement.render(&["1".to_string()]).unwrap_err();

        assert!(matches!(
            error,
            Error::MissingParameter {
                wanted: 2,
                bound: 1
            }
        ));
    }

    #[test]
    fn a_literal_quotes_the_quotes_inside_it() {
        assert_eq!(quote_literal("it's"), "'it''s'");
    }
}
