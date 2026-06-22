//! `COUNT(DISTINCT x)` as the node's sole aggregate, lowered to a two-level
//! GROUP BY, mirroring how DuckDB pre-aggregates a distinct argument before
//! counting it.
//!
//! * **Global** (`SELECT COUNT(DISTINCT x)`): dedup `x` in a keys-only group
//!   that emits each hash partition's distinct count, then `SUM` them.
//! * **Grouped** (`SELECT g…, COUNT(DISTINCT x) … GROUP BY g…`): dedup the
//!   `(g…, x)` tuples with a keys-only GROUP BY, then a GROUP BY on the leading
//!   group columns counts the distinct rows per group.
//!
//! A single integer group key with an integer distinct argument takes the packed
//! [`IntPairKeyExtractor`] fast path; a string group key, several group keys, or
//! a string distinct argument byte-encode the tuple with [`RowKeyExtractor`]. A
//! computed group key (such as `date_trunc(...)`) is first materialised into a
//! leading column, so it flows through either path like a plain column. NULLs in
//! `x` are read through the value bits, so a nullable `x` with NULLs present can
//! differ from DuckDB by one.

use super::{Aggregate, row_key_schema};
use crate::compile::Error;
use crate::expression::NumericAggregate;
use crate::types::Type;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use dispatch::{
    AggregationKind, AggregationSlot, Compiled, CountSlot, Distinct, HashOnlyIntKeyExtractor,
    IntKeyExtractor, IntPairKeyExtractor, RecordBatchOperatorSpec, RowKeyExtractor,
    StringKeyExtractor,
};

impl Aggregate {
    pub(super) fn compile_count_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        distinct: &NumericAggregate,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if self.groups.is_empty() {
            return self.compile_global_distinct(input, distinct);
        }

        // Materialise the group keys over a (possibly re-projected) input: any
        // computed key becomes a leading column, shifting the distinct argument's
        // column right by the number of materialised keys. After this every group
        // key is a column, so both lowering paths below treat computed and plain
        // keys identically.
        let (input, groups, key_shift) = self.materialize_group_keys(input)?;
        let distinct_col = distinct.column.column_idx + key_shift;
        let distinct_type = &distinct.column.return_type;

        // Fast path: a single integer group key + integer distinct argument pack
        // into the specialised u128 pair extractor for the inner (group, distinct)
        // dedup (half the entry bytes, radix-partitioned) and the dedicated int
        // extractor for the outer count, measurably cheaper than byte-encoding a
        // two-column tuple. The inner emits `[group, distinct]`, so the outer
        // counts column 0 (the group). Non-integer shapes fall through to the row
        // path below; the moving arms all `return`, so `input` stays owned
        // otherwise.
        if let [(group_col, group_type)] = groups.as_slice() {
            let group_col = *group_col;
            macro_rules! two_level {
                ($group:ty, $distinct:ty) => {{
                    let deduped = input
                        .group_by_aggregate::<IntPairKeyExtractor<$group, $distinct>, Distinct>(
                            vec![group_col, distinct_col],
                            Vec::new(),
                            None,
                        );
                    return Ok(deduped
                        .group_by_aggregate::<IntKeyExtractor<$group>, Compiled<(CountSlot,)>>(
                            vec![0],
                            vec![AggregationSlot::new(AggregationKind::CountStar, 0)],
                            None,
                        ));
                }};
            }
            macro_rules! by_distinct_type {
                ($group:ty) => {
                    match distinct_type {
                        Type::Int8 => two_level!($group, Int8Type),
                        Type::Int16 => two_level!($group, Int16Type),
                        Type::Int32 => two_level!($group, Int32Type),
                        Type::Int64 => two_level!($group, Int64Type),
                        _ => {}
                    }
                };
            }
            match group_type {
                Type::Int8 => by_distinct_type!(Int8Type),
                Type::Int16 => by_distinct_type!(Int16Type),
                Type::Int32 => by_distinct_type!(Int32Type),
                Type::Int64 => by_distinct_type!(Int64Type),
                _ => {}
            }
        }

        // General path: byte-encode the (group…, distinct) tuple for the inner
        // dedup, then count the deduped rows per group tuple. The inner emits the
        // key columns in key order (the group columns, then the distinct column),
        // so the outer groups by the leading `0..groups.len()` columns. Handles a
        // string group key, several group keys, or a string distinct argument.
        let mut inner_cols: Vec<usize> = groups.iter().map(|(col, _)| *col).collect();
        inner_cols.push(distinct_col);
        let inner_schema = row_key_schema(
            groups
                .iter()
                .map(|(_, ty)| ty)
                .chain(std::iter::once(distinct_type)),
        )
        .ok_or_else(|| Error::DataTypeNotSupportedForGroupBy(distinct_type.clone()))?;
        let outer_schema = row_key_schema(groups.iter().map(|(_, ty)| ty))
            .ok_or_else(|| Error::DataTypeNotSupportedForGroupBy(groups[0].1.clone()))?;
        let outer_cols: Vec<usize> = (0..groups.len()).collect();

        let deduped = input.group_by_aggregate_config::<RowKeyExtractor, Distinct>(
            inner_cols,
            Vec::new(),
            None,
            inner_schema,
        );
        Ok(deduped.group_by_aggregate_config::<RowKeyExtractor, Compiled<(CountSlot,)>>(
            outer_cols,
            vec![AggregationSlot::new(AggregationKind::CountStar, 0)],
            None,
            outer_schema,
        ))
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
        let distinct_cols = vec![distinct.column.column_idx];
        let counts = match &distinct.column.return_type {
            Type::Int8 => {
                input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int8Type>>(distinct_cols)
            }
            Type::Int16 => {
                input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int16Type>>(distinct_cols)
            }
            Type::Int32 => {
                input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int32Type>>(distinct_cols)
            }
            Type::Int64 => {
                input.group_by_distinct_count::<HashOnlyIntKeyExtractor<Int64Type>>(distinct_cols)
            }
            Type::Utf8 => input.group_by_distinct_count::<StringKeyExtractor>(distinct_cols),
            dt => return Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
        };
        // Per-partition distinct counts are i64; their total can't exceed the
        // row count, so the narrow accumulator suffices.
        Ok(counts.aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::Sum, 0)]))
    }
}
