//! [`HasTokens`] — the token-matching functions over text:
//!
//! ```text
//!   has_token(message, 'timeout')            the row holds the token
//!   has_all_tokens(message, 'GET timeout')   the row holds every token of the list
//!   has_any_token(message, 'retry timeout')  the row holds at least one of them
//! ```
//!
//! A token is a run of letters and digits; the token list is a string split
//! the same way the text is, so `'GET timeout'`, `'GET,timeout'` and
//! `'GET /timeout'` name the same two tokens. Matching is case-sensitive and
//! whole-token: `has_token(message, 'time')` does not match `timeout`.
//!
//! The functions are evaluated from the text itself. A table with a text
//! index on the column additionally uses [`HasTokens::query`] to skip rows
//! the index rules out before they are read.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::{ArrayRef, RecordBatch};
use dispatch::HasTokens as DispatchHasTokens;
use dispatch::text_index::{TextQuery, TokenMatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// Which of the token-matching functions a call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenFunction {
    HasToken,
    HasAllTokens,
    HasAnyToken,
}

impl TokenFunction {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "has_token" => Some(Self::HasToken),
            "has_all_tokens" => Some(Self::HasAllTokens),
            "has_any_token" => Some(Self::HasAnyToken),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::HasToken => "has_token",
            Self::HasAllTokens => "has_all_tokens",
            Self::HasAnyToken => "has_any_token",
        }
    }

    fn token_match(self) -> TokenMatch {
        match self {
            // A single token: all-of and any-of agree.
            Self::HasToken | Self::HasAllTokens => TokenMatch::All,
            Self::HasAnyToken => TokenMatch::Any,
        }
    }
}

/// SQL `has_token(input, token)`, `has_all_tokens(input, tokens)` or
/// `has_any_token(input, tokens)`.
#[derive(Debug, Clone)]
pub struct HasTokens {
    pub input: Box<Expression>,
    pub function: TokenFunction,
    /// The constant second argument: one token for `has_token`, otherwise text
    /// that is split into the tokens to look for.
    pub tokens: String,
}

impl Display for HasTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}, '{}')",
            self.function.name(),
            self.input,
            self.tokens
        )
    }
}

impl HasTokens {
    /// The tokens this call looks for and how they combine.
    pub fn query(&self) -> TextQuery {
        TextQuery::from_text(self.tokens.as_bytes(), self.function.token_match())
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self.input.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(compile::Error::UnsupportedExpressionForTokenMatchInput(
                    expr.clone(),
                ));
            }
        }

        let input_builder = self.input.compile()?;
        let query = self.query();
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let has_tokens = DispatchHasTokens::new(&query);
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let col = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringViewArray>()
                    .unwrap();
                ExprResult::Array(Arc::new(has_tokens.run(col)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn has_token_keeps_the_rows_holding_the_token(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE has_token(name, 'alice')",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }

    #[rstest]
    fn has_token_does_not_match_part_of_a_token(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE has_token(name, 'ali')",
        );

        assert!(rows.is_empty());
    }

    #[rstest]
    fn has_any_token_keeps_the_rows_holding_one_of_the_tokens(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE has_any_token(name, 'alice bob')",
        );

        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter()
                .all(|r| r["name"] == "alice" || r["name"] == "bob")
        );
    }

    #[rstest]
    fn has_all_tokens_needs_every_token_in_the_row(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE has_all_tokens(name, 'alice bob')",
        );

        assert!(rows.is_empty());
    }

    #[rstest]
    fn has_token_rejects_an_argument_that_is_not_one_token(mut testing_planner: TestingPlanner) {
        let error = testing_planner
            .plan("SELECT name FROM example_table WHERE has_token(name, 'alice bob')")
            .unwrap_err();

        assert!(error.to_string().contains("exactly one token"));
    }
}
