//! [`HasToken`] — SQL `has_token(text, token)`: whether a text has a token.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::{ArrayRef, RecordBatch};
use dispatch::HasToken as DispatchHasToken;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `has_token(text, token)`: whether `token` is one of the tokens `text`
/// is cut into, the way a text index cuts it: at every ASCII byte that is
/// neither a letter nor a digit. Letter case counts.
///
/// The text is a column and the token a constant that is itself one token,
/// which is what lets a text index of the column answer the same question.
#[derive(Debug, Clone)]
pub struct HasToken {
    pub text: Box<Expression>,
    pub token: String,
}

impl Display for HasToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "has_token({}, '{}')", self.text, self.token)
    }
}

impl HasToken {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self.text.as_ref() {
            Expression::Ref(reference) if reference.return_type == Type::Utf8 => {}
            expression => {
                return Err(compile::Error::UnsupportedExpressionForHasTokenText(
                    expression.clone(),
                ));
            }
        }
        let text_builder = self.text.compile()?;
        let token = self.token.clone();
        Ok(Box::new(move || {
            let mut text_expression = text_builder();
            let has_token = DispatchHasToken::new(token.as_bytes());
            Box::new(move |batch: &RecordBatch| {
                let text = text_expression(batch);
                let (texts, _) = text.as_datum().get();
                let texts = texts
                    .as_any()
                    .downcast_ref::<arrow_array::StringViewArray>()
                    .unwrap();
                ExprResult::Array(Arc::new(has_token.run(texts)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn filters_rows_whose_text_has_the_token(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE has_token(name, 'alice')",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row["name"] == "alice"));
    }

    #[rstest]
    fn a_part_of_a_token_matches_no_row(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE has_token(name, 'ali')",
        );

        assert!(rows.is_empty());
    }

    #[rstest]
    fn a_search_for_more_than_one_token_is_refused(mut testing_planner: TestingPlanner) {
        let result = testing_planner
            .plan("SELECT name FROM example_table WHERE has_token(name, 'alice bob')");

        let error = result.unwrap_err().to_string();
        assert!(error.contains("one token"), "{error}");
    }
}
