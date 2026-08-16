//! [`Substring`] — SQL `substring(string, start[, length])` with constant
//! positions, counted in characters as DuckDB counts them.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `substring(input, start[, length])`, DuckDB's `BOUND_FUNCTION` named
/// `substring` (the `substring(x FROM a FOR b)` syntax binds to the same
/// three-argument call).
///
/// The supported surface is a constant `start >= 0` and a constant `length >=
/// 0` (or no length at all, which reads to the end of the string). DuckDB's
/// zero start denotes one position before the string, consuming one unit of an
/// explicit length; the builder canonicalizes that to an equivalent one-based
/// range so execution needs no special case. Negative positions, which count
/// back from the end, fail loudly at plan time. Positions count characters,
/// not bytes, matching DuckDB on multi-byte input.
#[derive(Debug, Clone)]
pub struct Substring {
    pub input: Box<Expression>,
    /// One-based character position of the first kept character.
    pub start: u64,
    /// Characters kept from `start` on; `None` reads to the end.
    pub length: Option<u64>,
}

impl Display for Substring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.length {
            Some(length) => write!(f, "substring({}, {}, {})", self.input, self.start, length),
            None => write!(f, "substring({}, {})", self.input, self.start),
        }
    }
}

/// The byte range of the character range `skip..skip + take`, against which
/// the string is sliced. A range past the end clamps to it: SQL's substring
/// of a too-short string is the empty string, not an error.
fn char_range_bytes(value: &str, skip: usize, take: Option<u64>) -> (usize, usize) {
    let begin = value
        .char_indices()
        .nth(skip)
        .map_or(value.len(), |(offset, _)| offset);
    let end = match take {
        Some(take) => value[begin..]
            .char_indices()
            .nth(take as usize)
            .map_or(value.len(), |(offset, _)| begin + offset),
        None => value.len(),
    };
    (begin, end)
}

impl Substring {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        let skip = (self.start - 1) as usize;
        let take = self.length;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                // Arrow's substring kernels are byte-positioned or lack a
                // string-view path, so the slicing is done here: one pass per
                // value, sliced on char boundaries found by scanning only the
                // value's prefix.
                let out: StringViewArray = strings
                    .iter()
                    .map(|value| {
                        value.map(|value| {
                            let (begin, end) = char_range_bytes(value, skip, take);
                            &value[begin..end]
                        })
                    })
                    .collect();
                ExprResult::Array(Arc::new(out) as ArrayRef)
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

    fn phones_table(testing_planner: &mut TestingPlanner) {
        testing_planner.add_table(
            "phones",
            &[(
                "number",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    Some("13-715-945-6730"),
                    Some("31-133-332-6649"),
                    Some("ab"),
                    None,
                ])) as ArrayRef,
            )],
        );
    }

    #[rstest]
    fn takes_a_constant_prefix_range(mut testing_planner: TestingPlanner) {
        phones_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT substring(number FROM 1 FOR 2) AS code FROM phones WHERE number IS NOT NULL",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"code": "13"}, {"code": "31"}, {"code": "ab"}])
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[rstest]
    fn a_range_past_the_end_yields_the_empty_string(mut testing_planner: TestingPlanner) {
        phones_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT substring(number, 4, 3) AS mid FROM phones WHERE number = 'ab'",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"mid": ""}]).as_array().unwrap().clone()
        );
    }

    #[rstest]
    fn without_a_length_reads_to_the_end(mut testing_planner: TestingPlanner) {
        phones_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT substring(number, 4) AS rest FROM phones WHERE number LIKE '13%'",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"rest": "715-945-6730"}])
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[rstest]
    fn positions_count_characters_not_bytes(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "notes",
            &[(
                "text",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec!["наём truth"])) as ArrayRef,
            )],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT substring(text, 3, 2) AS piece FROM notes",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"piece": "ём"}])
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[rstest]
    fn null_input_stays_null(mut testing_planner: TestingPlanner) {
        phones_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT count(substring(number, 1, 2)) AS with_code FROM phones",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"with_code": 3}])
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[rstest]
    fn a_zero_start_matches_duckdb_without_a_runtime_special_case(
        mut testing_planner: TestingPlanner,
    ) {
        phones_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT substring(number, 0, 4) AS four, \
                    substring(number, 0, 1) AS one, \
                    substring(number, 0, 0) AS zero, \
                    substring(number, 0) AS rest \
             FROM phones WHERE number LIKE '13%'",
        );

        assert_eq!(
            rows,
            serde_json::json!([{
                "four": "13-",
                "one": "",
                "zero": "",
                "rest": "13-715-945-6730"
            }])
            .as_array()
            .unwrap()
            .clone()
        );
    }

    #[rstest]
    fn a_negative_start_is_rejected(mut testing_planner: TestingPlanner) {
        phones_table(&mut testing_planner);

        let error = testing_planner
            .plan("SELECT substring(number, -1, 2) FROM phones")
            .unwrap_err()
            .to_string();

        assert!(error.contains("substring"), "unexpected error: {error}");
    }
}
