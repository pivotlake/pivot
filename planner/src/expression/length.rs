//! [`Length`] — SQL `length(string)`, the Unicode character count.

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `length(string)` — the number of Unicode *characters* (not bytes) in
/// the string, as `BIGINT`. Arrives as a `BOUND_FUNCTION` named `length`.
#[derive(Debug, Clone)]
pub struct Length {
    pub input: Box<Expression>,
}

impl TryFrom<duckdb_expression::Function> for Length {
    type Error = Error;
    fn try_from(mut f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        if f.params.len() != 1 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 1,
                actual,
            });
        }
        let input = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(Length { input })
    }
}

impl Display for Length {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "length({})", self.input)
    }
}

impl Length {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                // `length()` counts Unicode characters, not bytes (arrow's
                // own length kernel returns *byte* lengths for Utf8View, so it
                // can't be used here).
                //
                // In UTF-8 each character is encoded as exactly one leading
                // byte followed by zero or more continuation bytes, and a
                // continuation byte is the only kind that starts with the bit
                // pattern `10xxxxxx`. So the number of characters equals the
                // number of *non*-continuation bytes. `b & 0xC0` masks off the
                // low 6 bits, leaving the top two; `!= 0x80` (i.e. `!= 10xxxxxx`)
                // is true for every leading byte. Counting those is a single
                // branch-free pass that vectorizes and avoids the per-codepoint
                // decoding `chars().count()` would do.
                let lengths: Int64Array = strings
                    .iter()
                    .map(|v| v.map(|s| s.bytes().filter(|&b| (b & 0xC0) != 0x80).count() as i64))
                    .collect();
                ExprResult::Array(Arc::new(lengths) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn counts_unicode_chars(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "strs",
            &[(
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "hello",
                    "héllo",
                    "",
                    "日本語abc",
                ])) as ArrayRef,
            )],
        );

        let mut rows = run(&mut testing_planner, "SELECT length(s) FROM strs");

        rows.sort_by_key(|r| r["col0"].as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| r["col0"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 5, 5, 6]
        );
    }
}
