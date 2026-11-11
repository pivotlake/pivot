//! Global `SELECT COUNT(DISTINCT x)` (no GROUP BY): a dedicated single-level fast
//! path. The grouped forms (alone or mixed with other aggregates) are a two-level
//! GROUP BY handled in [`super::grouped_distinct`].

use super::grouped::{build_dedup_operator, column_nullable};
use super::{Aggregate, is_groupable_key_type};
use crate::compile::Error;
use crate::expression::NumericAggregate;
use crate::types::Type;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use dispatch::{
    AggregationKind, AggregationSlot, HashOnlyIntKeyExtractor, RecordBatchOperatorSpec,
    StringKeyExtractor,
};

impl Aggregate {
    /// Global `COUNT(DISTINCT x)`: dedup `x` in a keys-only group that emits each
    /// hash partition's distinct-key count (not the keys), then SUM those
    /// per-partition counts. Skips materialising the whole distinct-value column
    /// just to count it. Integer columns use the keys-only
    /// [`HashOnlyIntKeyExtractor`] (8-byte entries; a bijective hash makes
    /// hash-equality exactly key-equality), half the per-entry bytes, the
    /// dominant cost for this latency-bound build. Strings keep the arena-backed
    /// [`StringKeyExtractor`] (no exact 64-bit bijection for them). Every other
    /// groupable column, and any nullable one, goes through [`count_deduped`].
    pub(super) fn compile_global_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        distinct: &NumericAggregate,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Reject a type no extractor can group (a float), naming the column,
        // as `compile_resolved` does for a GROUP BY key.
        let column = distinct.column();
        if !is_groupable_key_type(&column.return_type) {
            return Err(Error::DataTypeNotSupportedForGroupBy {
                column: column.name.clone(),
                data_type: column.return_type.clone(),
            });
        }

        // The hash-only dedup counts every distinct key, but SQL does not count
        // NULL as a distinct value, so a nullable column takes the general dedup.
        if column_nullable(nullability, column.column_idx) {
            return count_deduped(input, distinct, nullability);
        }

        let distinct_cols = vec![column.column_idx];
        let counts =
            match &column.return_type {
                Type::Int8 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int8Type>>(distinct_cols),
                Type::Int16 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int16Type>>(distinct_cols),
                Type::Int32 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int32Type>>(distinct_cols),
                Type::Int64 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int64Type>>(distinct_cols),
                Type::Utf8 => input.group_by_distinct_count::<StringKeyExtractor>(distinct_cols),
                // Every other groupable type (temporal, decimal, unsigned,
                // boolean) has no hash-only extractor: dedup it the general way.
                _ => return count_deduped(input, distinct, nullability),
            };
        // Per-partition distinct counts are i64; their total can't exceed the
        // row count, so the narrow accumulator suffices. The result is the
        // `COUNT(DISTINCT)` type DuckDB declares (`BIGINT`).
        Ok(counts.aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::Sum,
            0,
            crate::types::physical_arrow_type(&distinct.return_type),
        )]))
    }
}

/// `COUNT(DISTINCT x)` through the general keys-only dedup, for any groupable
/// type: dedup `x` (the NULLs collapse into one key row) and count only the
/// non-NULL survivors.
fn count_deduped(
    input: RecordBatchOperatorSpec,
    distinct: &NumericAggregate,
    nullability: &[bool],
) -> Result<RecordBatchOperatorSpec, Error> {
    let keys = [(
        distinct.column().column_idx,
        distinct.column().return_type.clone(),
    )];
    let deduped = build_dedup_operator(input, &keys, nullability)?;
    Ok(deduped.aggregate::<i64>(vec![AggregationSlot::new(
        AggregationKind::Count,
        0,
        crate::types::physical_arrow_type(&distinct.return_type),
    )]))
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float64Array,
        TimestampMicrosecondArray, UInt32Array, UInt64Array,
    };
    use rstest::rstest;
    use std::sync::Arc;

    /// Five rows with no NULLs, so every column is non-nullable: three distinct
    /// values in each column except the boolean `flag`, which has two.
    /// `wide_price` is a `Decimal128` carrier and `big` reaches `u64::MAX`.
    fn add_events_table(testing_planner: &TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[
                (
                    "d",
                    Type::Date,
                    Arc::new(Date32Array::from(vec![10, 20, 10, 30, 20])) as ArrayRef,
                ),
                (
                    "price",
                    Type::Decimal {
                        precision: 10,
                        scale: 2,
                    },
                    Arc::new(
                        Decimal64Array::from(vec![1050, 250, 1050, 400, 250])
                            .with_precision_and_scale(10, 2)
                            .unwrap(),
                    ) as ArrayRef,
                ),
                (
                    "wide_price",
                    Type::Decimal {
                        precision: 30,
                        scale: 2,
                    },
                    Arc::new(
                        Decimal128Array::from(vec![1050, 250, 1050, 400, 250])
                            .with_precision_and_scale(30, 2)
                            .unwrap(),
                    ) as ArrayRef,
                ),
                (
                    "ts",
                    Type::Timestamp,
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        1_000_000, 2_000_000, 1_000_000, 3_000_000, 2_000_000,
                    ])) as ArrayRef,
                ),
                (
                    "u",
                    Type::UInt32,
                    Arc::new(UInt32Array::from(vec![7, 8, 9, 7, 8])) as ArrayRef,
                ),
                (
                    "big",
                    Type::UInt64,
                    Arc::new(UInt64Array::from(vec![u64::MAX, 1, u64::MAX, 2, 1])) as ArrayRef,
                ),
                (
                    "flag",
                    Type::Boolean,
                    Arc::new(BooleanArray::from(vec![true, false, true, true, false])) as ArrayRef,
                ),
                (
                    "f",
                    Type::Float64,
                    Arc::new(Float64Array::from(vec![1.5, 2.5, 1.5, 4.0, 2.5])) as ArrayRef,
                ),
            ],
        );
    }

    #[rstest]
    #[case::date("COUNT(DISTINCT d)", 3)]
    #[case::decimal("COUNT(DISTINCT price)", 3)]
    #[case::wide_decimal("COUNT(DISTINCT wide_price)", 3)]
    #[case::timestamp("COUNT(DISTINCT ts)", 3)]
    #[case::unsigned("COUNT(DISTINCT u)", 3)]
    #[case::unsigned_max("COUNT(DISTINCT big)", 3)]
    #[case::boolean("COUNT(DISTINCT flag)", 2)]
    #[case::computed_boolean("COUNT(DISTINCT u > 7)", 2)]
    fn global_count_distinct_over_non_nullable_column(
        mut testing_planner: TestingPlanner,
        #[case] aggregate: &str,
        #[case] expected: i64,
    ) {
        add_events_table(&testing_planner);

        let rows = run(
            &mut testing_planner,
            &format!("SELECT {aggregate} AS n FROM events"),
        );

        assert_eq!(rows[0]["n"].as_i64(), Some(expected));
    }

    #[rstest]
    fn global_count_distinct_over_float_is_rejected(mut testing_planner: TestingPlanner) {
        add_events_table(&testing_planner);

        let error = compile_expecting_error(
            &mut testing_planner,
            "SELECT COUNT(DISTINCT f) AS n FROM events",
        );

        assert!(
            error.contains("\"f\"") && error.contains("Float64"),
            "{error}"
        );
    }

    #[rstest]
    fn global_count_distinct_string_excludes_nulls(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT COUNT(DISTINCT s) AS d FROM nullable_table",
        );

        assert_eq!(rows[0]["d"].as_i64(), Some(3)); // x, y, z
    }

    #[rstest]
    fn global_count_distinct_int_excludes_nulls(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT COUNT(DISTINCT b) AS d FROM nullable_table",
        );

        assert_eq!(rows[0]["d"].as_i64(), Some(3)); // 10, 30, 50
    }
}
