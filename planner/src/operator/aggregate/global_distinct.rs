//! Global `SELECT COUNT(DISTINCT x)` (no GROUP BY): a dedicated single-level fast
//! path. The grouped forms (alone or mixed with other aggregates) are a two-level
//! GROUP BY handled in [`super::grouped_distinct`].

use super::Aggregate;
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
    ) -> Result<RecordBatchOperatorSpec, Error> {
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
                dt => return Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
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
