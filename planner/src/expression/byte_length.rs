//! [`ByteLength`] — SQL `strlen(string)`, the number of bytes in the string.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `strlen(string)`: the number of bytes in the string's UTF-8 encoding,
/// as `BIGINT`.
#[derive(Debug, Clone)]
pub struct ByteLength {
    pub input: Box<Expression>,
    pub return_type: Type,
}

impl Display for ByteLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "strlen({})", self.input)
    }
}

impl ByteLength {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                // Each view keeps its string's length in its low 32 bits, so
                // this reads the views alone and never the string payload
                // buffers. A null row keeps the input's null bit; whatever its
                // view holds is never looked at.
                let byte_counts: Vec<i64> = strings
                    .views()
                    .iter()
                    .map(|view| i64::from(*view as u32))
                    .collect();
                let byte_counts = Int64Array::new(byte_counts.into(), strings.nulls().cloned());
                ExprResult::Array(Arc::new(byte_counts) as ArrayRef)
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
    fn counts_bytes(mut testing_planner: TestingPlanner) {
        strs_table(&mut testing_planner);

        let mut rows = run(&mut testing_planner, "SELECT strlen(s) FROM strs");

        rows.sort_by_key(|r| only_column(r).as_i64().unwrap());
        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 5, 6, 12]
        );
    }

    #[rstest]
    fn a_column_and_a_constant_have_the_same_byte_length(mut testing_planner: TestingPlanner) {
        strs_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT s FROM strs WHERE strlen(s) = strlen('日本語abc')",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(only_column(&rows[0]), "日本語abc");
    }

    #[rstest]
    fn a_null_string_has_a_null_byte_length(mut testing_planner: TestingPlanner) {
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

        let rows = run(&mut testing_planner, "SELECT strlen(s) FROM strs");

        // A row whose only column is NULL comes back with no column at all.
        let mut byte_lengths: Vec<Option<i64>> = rows
            .iter()
            .map(|row| {
                let columns = row.as_object().unwrap();
                columns
                    .values()
                    .next()
                    .and_then(|byte_length| byte_length.as_i64())
            })
            .collect();
        byte_lengths.sort();
        assert_eq!(byte_lengths, vec![None, Some(2), Some(33)]);
    }
}
