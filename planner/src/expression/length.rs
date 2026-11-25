//! [`Length`] — SQL `length(string)`, the number of characters in the string.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `length(string)`, also spelled `len`: the number of Unicode code points
/// in the string, as `BIGINT`. This is how DuckDB counts when it folds a
/// constant argument, so a column and a constant agree.
#[derive(Debug, Clone)]
pub struct Length {
    pub input: Box<Expression>,
    pub return_type: Type,
}

impl Display for Length {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "length({})", self.input)
    }
}

/// The number of characters in `string`. Every character of an ASCII string
/// is one byte, and checking for that is cheaper than counting.
fn count_characters(string: &str) -> i64 {
    let character_count = if string.is_ascii() {
        string.len()
    } else {
        string.chars().count()
    };
    character_count as i64
}

impl Length {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let lengths: Int64Array = arr
                    .as_string_view()
                    .iter()
                    .map(|string| string.map(count_characters))
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

    fn strs_table(testing_planner: &mut TestingPlanner) {
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
    }

    #[rstest]
    #[case::length("length")]
    #[case::len("len")]
    fn counts_characters(mut testing_planner: TestingPlanner, #[case] function: &str) {
        strs_table(&mut testing_planner);

        let mut rows = run(
            &mut testing_planner,
            &format!("SELECT {function}(s) FROM strs"),
        );

        rows.sort_by_key(|r| only_column(r).as_i64().unwrap());
        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 5, 5, 6]
        );
    }

    #[rstest]
    fn a_column_and_a_constant_have_the_same_length(mut testing_planner: TestingPlanner) {
        strs_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT s FROM strs WHERE length(s) = length('日本語abc')",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(only_column(&rows[0]), "日本語abc");
    }

    #[rstest]
    fn a_null_string_has_a_null_length(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "strs",
            &[(
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    Some("a string longer than twelve bytes"),
                    None,
                    Some("ab"),
                ])) as ArrayRef,
            )],
        );

        let rows = run(&mut testing_planner, "SELECT length(s) FROM strs");

        // A row whose only column is NULL comes back with no column at all.
        let mut lengths: Vec<Option<i64>> = rows
            .iter()
            .map(|row| {
                let columns = row.as_object().unwrap();
                columns.values().next().and_then(|length| length.as_i64())
            })
            .collect();
        lengths.sort();
        assert_eq!(lengths, vec![None, Some(2), Some(33)]);
    }
}
