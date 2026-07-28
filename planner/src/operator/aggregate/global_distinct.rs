//! Global `SELECT COUNT(DISTINCT x)` (no GROUP BY): a dedicated single-level fast
//! path. The grouped forms (alone or mixed with other aggregates) are a two-level
//! GROUP BY handled in [`super::grouped_distinct`].

use super::Aggregate;
use super::grouped::{build_dedup_operator, column_nullable};
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
    /// [`StringKeyExtractor`] (no exact 64-bit bijection for them).
    pub(super) fn compile_global_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        distinct: &NumericAggregate,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // The hash-only dedup counts every distinct key, but SQL does not count
        // NULL as a distinct value. Over a nullable column, dedup the keys (the
        // NULLs collapse into one key row) and count only the non-NULL
        // survivors.
        if column_nullable(nullability, distinct.column().column_idx) {
            let keys = [(
                distinct.column().column_idx,
                distinct.column().return_type.clone(),
            )];
            let deduped = build_dedup_operator(input, &keys, nullability)?;
            return Ok(deduped.aggregate::<i64>(vec![AggregationSlot::new(
                AggregationKind::Count,
                0,
                crate::types::physical_arrow_type(&distinct.return_type),
            )]));
        }

        let distinct_cols = vec![distinct.column().column_idx];
        let counts =
            match &distinct.column().return_type {
                Type::Int8 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int8Type>>(distinct_cols),
                Type::Int16 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int16Type>>(distinct_cols),
                Type::Int32 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int32Type>>(distinct_cols),
                Type::Int64 => input
                    .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int64Type>>(distinct_cols),
                Type::Utf8 => input.group_by_distinct_count::<StringKeyExtractor>(distinct_cols),
                dt => {
                    return Err(Error::DataTypeNotSupportedForGroupBy {
                        column: distinct.column().name.clone(),
                        data_type: dt.clone(),
                    });
                }
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

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

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
