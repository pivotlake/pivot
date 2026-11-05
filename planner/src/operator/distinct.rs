//! [`Distinct`] — the distinct values of a tuple of input columns.
//!
//! Emits each distinct key tuple once, as the key columns alone, in `keys`
//! order. Lowered from a `SELECT DISTINCT` over all of its child's columns, and
//! synthesized by the delim join translation as the duplicate-elimination stage
//! feeding the subquery side's delim scans.

use crate::compile::Error;
use crate::operator::aggregate::grouped::build_dedup_operator;
use crate::operator::aggregate::restore_key_columns;
use crate::types::Type;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;

/// A keys-only dedup: `GROUP BY keys` emitting the keys themselves.
#[derive(Debug)]
pub struct Distinct {
    /// The deduplicated columns: each an index into the input's output plus
    /// its type, in output order.
    pub keys: Vec<(usize, Type)>,
}

impl fmt::Display for Distinct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns: Vec<String> = self.keys.iter().map(|(idx, _)| idx.to_string()).collect();
        write!(f, "Distinct(keys: [{}])", columns.join(", "))
    }
}

impl Distinct {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        input_nullability: Vec<bool>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // The dedup groups a date as its day count; restore the declared key
        // types on its output.
        let deduped = build_dedup_operator(input, &self.keys, &input_nullability)?;
        let key_types: Vec<Type> = self.keys.iter().map(|(_, ty)| ty.clone()).collect();
        Ok(restore_key_columns(deduped, &key_types))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Date32Array};
    use arrow_schema::DataType;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn distinct_date_keys_stay_dates(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "days",
            &[(
                "d",
                Type::Date,
                Arc::new(Date32Array::from(vec![10, 20, 10])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT DISTINCT d FROM days ORDER BY d",
        );

        assert_eq!(batches[0].schema().field(0).data_type(), &DataType::Date32);
        assert_eq!(
            batches_to_json(&batches),
            serde_json::json!([{"d": "1970-01-11"}, {"d": "1970-01-21"}])
                .as_array()
                .unwrap()
                .clone()
        );
    }
}
