//! `COUNT(DISTINCT x)` as the node's sole aggregate, lowered to a two-level
//! GROUP BY — mirroring how DuckDB pre-aggregates a distinct argument before
//! counting it.
//!
//! * **Global** (`SELECT COUNT(DISTINCT x)`): dedup `x` in a keys-only group
//!   that emits each hash partition's distinct count, then `SUM` them.
//! * **Grouped** (`SELECT g, COUNT(DISTINCT x) … GROUP BY g`): dedup the
//!   `(g, x)` pairs with a two-int-key keys-only GROUP BY, then a GROUP BY on
//!   column 0 (`g`) counts the distinct rows per group.
//!
//! Only a single integer group key is supported in the grouped form. NULLs in
//! `x` are read through the value bits, so a nullable `x` with NULLs present can
//! differ from DuckDB by one.

use super::Aggregate;
use crate::compile::Error;
use crate::expression::{Expression, NumericAggregate};
use crate::types::Type;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use dispatch::{
    AggregationKind, AggregationSlot, HashOnlyIntKeyExtractor, IntKeyExtractor,
    IntPairKeyExtractor, RecordBatchOperatorSpec, StringKeyExtractor,
};

impl Aggregate {
    pub(super) fn compile_count_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        distinct: &NumericAggregate,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let x_col = distinct.column.column_idx;

        if self.groups.is_empty() {
            return self.compile_global_distinct(input, distinct);
        }

        if self.groups.len() != 1 {
            return Err(Error::UnsupportedAggregateGroupAmount(self.groups.len()));
        }
        let g = match &self.groups[0] {
            Expression::Ref(r) => r,
            e => return Err(Error::UnexpectedAggExpression(e.clone())),
        };
        let g_col = g.column_idx;

        // Dedup (g, x) pairs with a keys-only group (the inner GROUP BY emits
        // just [g, x] — no accumulator), then count rows per g. The outer
        // GROUP BY on column 0 (g) is the actual per-group distinct count.
        macro_rules! two_level {
            ($g:ty, $x:ty) => {{
                let deduped =
                    input.group_by_distinct::<IntPairKeyExtractor<$g, $x>>(vec![g_col, x_col]);
                Ok(deduped.group_by_count::<IntKeyExtractor<$g>>(0))
            }};
        }
        macro_rules! by_x {
            ($g:ty) => {
                match &distinct.column.return_type {
                    Type::Int8 => two_level!($g, Int8Type),
                    Type::Int16 => two_level!($g, Int16Type),
                    Type::Int32 => two_level!($g, Int32Type),
                    Type::Int64 => two_level!($g, Int64Type),
                    dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
                }
            };
        }
        match &g.return_type {
            Type::Int8 => by_x!(Int8Type),
            Type::Int16 => by_x!(Int16Type),
            Type::Int32 => by_x!(Int32Type),
            Type::Int64 => by_x!(Int64Type),
            dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
        }
    }

    /// Global `COUNT(DISTINCT x)`: dedup `x` in a keys-only group that emits each
    /// hash partition's distinct-key count (not the keys), then SUM those
    /// per-partition counts. Skips materialising the whole distinct-value column
    /// just to count it. Integer columns use the keys-only
    /// [`HashOnlyIntKeyExtractor`] (8-byte entries; a bijective hash makes
    /// hash-equality exactly key-equality) — half the per-entry bytes, the
    /// dominant cost for this latency-bound build. Strings keep the arena-backed
    /// [`StringKeyExtractor`] (no exact 64-bit bijection for them).
    fn compile_global_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        distinct: &NumericAggregate,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let x = vec![distinct.column.column_idx];
        let counts = match &distinct.column.return_type {
            Type::Int8 => input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int8Type>>(x),
            Type::Int16 => input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int16Type>>(x),
            Type::Int32 => input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int32Type>>(x),
            Type::Int64 => input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int64Type>>(x),
            Type::Utf8 => input.group_by_distinct_count::<StringKeyExtractor>(x),
            dt => return Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
        };
        // Per-partition distinct counts are i64; their total can't exceed the
        // row count, so the narrow accumulator suffices.
        Ok(counts.aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::Sum, 0)]))
    }
}
