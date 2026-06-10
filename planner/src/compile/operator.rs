//! Per-operator compile impls.
//!
//! Each [`Operator`](crate::operator::Operator) variant has a `compile`
//! method here that translates it into a [`RecordBatchOperatorSpec`] call.

use crate::catalog::{Catalog, DynamicScanPredicate};
use crate::compile::dummy_scan::DummyScanNullaryFactory;
use crate::compile::{DynamicFilterSlots, Error, ExprEvalFn, ExprFn, ExprResult};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::Expression;
use crate::operator::{
    Aggregate, CreateTable, DummyScan, Filter, Input, Limit, Materialize, OrderBy,
    OrderByDirection, Projection, TopN,
};
use crate::types::Type;
use arrow::compute::kernels::boolean::and;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::{Field, Schema};
use dispatch::{
    DataFlowDispatcher, DynamicFilterSlot, IntKeyExtractor, OrderBy as DispatchOrderBy,
    Projection as DispatchProjection, RecordBatchOperatorSpec, StringKeyExtractor,
};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

impl Projection {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Fast path: every projection is a plain column reference, so we can
        // select columns zero-copy and preserve the input schema's fields.
        if let Some(idxs) = self
            .projections
            .iter()
            .map(|e| match e {
                Expression::Ref(n) => Some(n.column_idx),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            let idxs = Arc::new(idxs);
            return Ok(input.project(move || {
                let idxs = idxs.clone();
                move |batch: RecordBatch| {
                    // A late-materialized narrow scan appends row-group metadata
                    // columns the downstream materializer reads positionally from
                    // the end. A plain column projection would drop them, so carry
                    // any such trailing pair through untouched.
                    let meta = dispatch::trailing_metadata_columns(batch.schema_ref());
                    if meta == 0 {
                        batch.project(&idxs).unwrap()
                    } else {
                        let n = batch.num_columns();
                        let mut cols: Vec<usize> = idxs.as_ref().clone();
                        cols.extend((n - meta)..n);
                        batch.project(&cols).unwrap()
                    }
                }
            }));
        }

        // General path: at least one projection is a computed expression
        // (e.g. the `sum / count` divide of an AVG). Evaluate each projection
        // per batch and assemble a new RecordBatch, deriving the output schema
        // from the produced arrays.
        let builders: Arc<Vec<ExprFn>> = Arc::new(
            self.projections
                .iter()
                .map(|e| e.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );

        Ok(input.project(move || {
            let mut evals: Vec<ExprEvalFn> = builders.iter().map(|b| b()).collect();
            move |batch: RecordBatch| {
                let n = batch.num_rows();
                let columns: Vec<ArrayRef> = evals
                    .iter_mut()
                    .map(|eval| match eval(&batch) {
                        ExprResult::Array(a) => a,
                        // A constant column (e.g. `SELECT 1, …`): broadcast to
                        // the batch's row count so the columns line up.
                        ExprResult::Scalar(s) => {
                            let arr = s.into_inner();
                            let zeros = arrow_array::UInt32Array::from(vec![0u32; n]);
                            arrow::compute::take(&arr, &zeros, None).unwrap()
                        }
                    })
                    .collect();
                let fields: Vec<Field> = columns
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Field::new(format!("col{i}"), c.data_type().clone(), true))
                    .collect();
                RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
            }
        }))
    }
}

/// Pick the aggregate accumulator width by column type — the single rule shared
/// by the global and grouped paths: `i128` only when a `SUM` reads a 64-bit
/// column (whose total can overflow `i64`), else `i64`.
fn sum_reads_wide_column(exprs: &[Expression]) -> bool {
    use crate::expression::AggregateFunc;
    exprs.iter().any(|e| {
        matches!(
            e,
            Expression::AggregateFunc(AggregateFunc::Sum(a)) if a.column.return_type == Type::Int64
        )
    })
}

/// Arrow type a group-key column carries through the row-encoded multi-key
/// extractor. Mirrors how each planner type executes: integers and strings map
/// 1-1; timestamps run as their stored Int64 epoch seconds; dates encode as
/// Int32 — wide enough for any stored day-count width (the scan hands dates
/// over in their parquet-physical type, which the extractor's reader casts up
/// losslessly once per batch).
fn row_key_data_type(t: &Type) -> Result<arrow_schema::DataType, Error> {
    use arrow_schema::DataType;
    Ok(match t {
        Type::Int8 => DataType::Int8,
        Type::Int16 => DataType::Int16,
        Type::Int32 => DataType::Int32,
        Type::Int64 => DataType::Int64,
        Type::Utf8 => DataType::Utf8View,
        Type::Date => DataType::Int32,
        Type::Timestamp => DataType::Int64,
        t => return Err(Error::DataTypeNotSupportedForGroupBy(t.clone())),
    })
}

/// Keys-only dedup of `cols` via the row-encoded multi-key extractor — the
/// inner stage of a grouped `COUNT(DISTINCT)` whose tuple shape has no packed
/// extractor.
fn row_key_distinct(
    input: RecordBatchOperatorSpec,
    cols: Vec<usize>,
    types: &[Type],
) -> Result<RecordBatchOperatorSpec, Error> {
    use dispatch::{RowKeyExtractor, RowKeySchema};
    let schema = RowKeySchema::new(
        types
            .iter()
            .map(row_key_data_type)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(input.group_by_distinct_config::<RowKeyExtractor>(cols, schema))
}

/// `COUNT(*)` per group tuple over the leading columns of `input` (a deduped
/// inner stage's output) — the outer stage of a grouped `COUNT(DISTINCT)`.
fn count_per_group(
    input: RecordBatchOperatorSpec,
    group_types: &[Type],
    top_k: Option<(usize, usize)>,
) -> Result<RecordBatchOperatorSpec, Error> {
    use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
    use dispatch::{
        AggregationKind, AggregationSlot, Compiled, Count, RowKeyExtractor, RowKeySchema,
    };

    let key_cols: Vec<usize> = (0..group_types.len()).collect();
    let slots = vec![AggregationSlot::new(AggregationKind::CountStar, 0)];
    macro_rules! count_with {
        ($K:ty) => {
            Ok(input.group_by_aggregate::<$K, Compiled<(Count,)>>(key_cols, slots, top_k))
        };
    }
    match group_types {
        [Type::Int8] => count_with!(IntKeyExtractor<Int8Type>),
        [Type::Int16] => count_with!(IntKeyExtractor<Int16Type>),
        [Type::Int32] => count_with!(IntKeyExtractor<Int32Type>),
        [Type::Int64] => count_with!(IntKeyExtractor<Int64Type>),
        [Type::Utf8] => count_with!(StringKeyExtractor),
        _ => {
            let schema = RowKeySchema::new(
                group_types
                    .iter()
                    .map(row_key_data_type)
                    .collect::<Result<Vec<_>, _>>()?,
            );
            Ok(
                input.group_by_aggregate_config::<RowKeyExtractor, Compiled<(Count,)>>(
                    key_cols, slots, top_k, schema,
                ),
            )
        }
    }
}

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use dispatch::{AggregationKind, AggregationSlot};

        // Derived group keys — computed purely from other (plain-column) keys,
        // or constants — can't split or merge groups, so they are dropped
        // before grouping and recomputed from the surviving keys afterwards.
        // `GROUP BY ip, ip - 1` then groups on `ip` alone (narrower hash-table
        // entries, and a single int key keeps its specialised extractor).
        if let Some((reduced, post)) = self.split_derived_keys() {
            let grouped = reduced.compile(input)?;
            return Projection { projections: post }.compile(grouped);
        }

        // `COUNT(DISTINCT x)` (the sole aggregate) lowers to a two-level GROUP BY
        // rather than a single-pass accumulator — see [`compile_count_distinct`].
        // Both the global and the single-group-key forms route here.
        if let [Expression::AggregateFunc(AggregateFunc::CountDistinct(a))] =
            self.expressions.as_slice()
        {
            return self.compile_count_distinct(input, a);
        }

        // Grouped aggregate mixing one `COUNT(DISTINCT x)` with non-distinct
        // aggregates (SUM/COUNT/COUNT(*)) — e.g. ClickBench Q9. Also a two-level
        // GROUP BY: the inner `(group, x)` group computes the non-distinct
        // partials, the outer re-sums them and counts rows for the distinct.
        let n_distinct = self
            .expressions
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Expression::AggregateFunc(AggregateFunc::CountDistinct(_))
                )
            })
            .count();
        if !self.groups.is_empty() && n_distinct == 1 {
            return self.compile_grouped_mixed_distinct(input);
        }

        // Global aggregates (no GROUP BY). A bare `COUNT(*)` keeps the
        // dedicated row-counter; anything else (SUM/COUNT, possibly several)
        // compiles to the multi-aggregate operator, one output column per
        // expression. `AVG` never appears here: DuckDB lowers it to a
        // `sum`+`count` pair with a downstream divide (the same lowering the
        // grouped path relies on), so only count/sum slots reach this point.
        if self.groups.is_empty() {
            if matches!(
                self.expressions.as_slice(),
                [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
            ) {
                return Ok(input.count());
            }

            let slots =
                self.expressions
                    .iter()
                    .map(|e| match e {
                        Expression::AggregateFunc(AggregateFunc::Sum(a)) => Ok(
                            AggregationSlot::new(AggregationKind::Sum, a.column.column_idx),
                        ),
                        Expression::AggregateFunc(AggregateFunc::Count(a)) => Ok(
                            AggregationSlot::new(AggregationKind::Count, a.column.column_idx),
                        ),
                        // COUNT(*) ignores its column; the index is a placeholder.
                        Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {
                            Ok(AggregationSlot::new(AggregationKind::CountStar, 0))
                        }
                        // Global string extremes are not supported (no query
                        // shape needs them yet); integer/date ones are.
                        Expression::AggregateFunc(AggregateFunc::Min(a))
                            if a.column.return_type != Type::Utf8 =>
                        {
                            Ok(AggregationSlot::new(
                                AggregationKind::Min,
                                a.column.column_idx,
                            ))
                        }
                        Expression::AggregateFunc(AggregateFunc::Max(a))
                            if a.column.return_type != Type::Utf8 =>
                        {
                            Ok(AggregationSlot::new(
                                AggregationKind::Max,
                                a.column.column_idx,
                            ))
                        }
                        expr => Err(Error::UnsupportedAggregateExpression(expr.clone())),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            // Pick the accumulator width by column type, the same rule the
            // grouped path uses: i128 only when a SUM reads a 64-bit column
            // (whose total can overflow i64), else i64.
            return Ok(if sum_reads_wide_column(&self.expressions) {
                input.aggregate::<i128>(slots)
            } else {
                input.aggregate::<i64>(slots)
            });
        }

        // Grouped. A single plain-column key with a lone COUNT(*) uses the
        // dedicated count path (which also handles string keys); everything
        // else — multi-key grouping, computed keys, sum/count/avg aggregates —
        // goes through the general grouped compiler.
        let simple_count = matches!(self.groups.as_slice(), [Expression::Ref(_)])
            && matches!(
                self.expressions.as_slice(),
                [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
            );
        if !simple_count {
            return self.compile_grouped(input);
        }

        let Expression::Ref(group) = &self.groups[0] else {
            unreachable!("checked by simple_count")
        };
        let col = group.column_idx;
        match &group.return_type {
            Type::Int8 => {
                Ok(input.group_by_count::<IntKeyExtractor<arrow_array::types::Int8Type>>(col))
            }
            Type::Int16 => {
                Ok(input.group_by_count::<IntKeyExtractor<arrow_array::types::Int16Type>>(col))
            }
            Type::Int32 => {
                Ok(input.group_by_count::<IntKeyExtractor<arrow_array::types::Int32Type>>(col))
            }
            Type::Int64 => {
                Ok(input.group_by_count::<IntKeyExtractor<arrow_array::types::Int64Type>>(col))
            }
            Type::Utf8 => Ok(input.group_by_count::<StringKeyExtractor>(col)),
            dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
        }
    }

    /// Partition the group keys into *base* keys (plain columns, plus computed
    /// expressions that read non-key columns) and *derived* keys — expressions
    /// whose every input is itself a base key column, including constants and
    /// duplicate columns. A derived key is a pure function of the base keys, so
    /// grouping with or without it yields the same groups.
    ///
    /// Returns `None` when nothing is derived (or everything would be — a
    /// degenerate constant-only GROUP BY). Otherwise returns the reduced
    /// aggregate (base keys only) plus the projection that rebuilds the
    /// original output layout `[keys…, values…]` from the reduced output,
    /// recomputing each derived key from the base keys' emitted values.
    fn split_derived_keys(&self) -> Option<(Aggregate, Vec<Expression>)> {
        use std::collections::HashMap;

        // Pass 1: every first occurrence of a plain column key is a base key.
        let mut base_pos: HashMap<usize, usize> = HashMap::new();
        let mut base: Vec<Expression> = Vec::new();
        for g in &self.groups {
            if let Expression::Ref(r) = g
                && !base_pos.contains_key(&r.column_idx)
            {
                base_pos.insert(r.column_idx, base.len());
                base.push(g.clone());
            }
        }

        // Pass 2: a computed key whose inputs are all base columns is derived;
        // anything else is a (computed) base key. A repeated plain column is
        // derived too (it re-reads its first occurrence).
        enum Kind {
            Base(usize),
            Derived,
        }
        let mut kinds: Vec<Kind> = Vec::with_capacity(self.groups.len());
        let mut any_derived = false;
        for g in &self.groups {
            match g {
                Expression::Ref(r) => {
                    let pos = base_pos[&r.column_idx];
                    // First occurrence is the base; a repeated column re-reads it.
                    if kinds
                        .iter()
                        .any(|k| matches!(k, Kind::Base(p) if *p == pos))
                    {
                        kinds.push(Kind::Derived);
                        any_derived = true;
                    } else {
                        kinds.push(Kind::Base(pos));
                    }
                }
                computed => {
                    let mut cols = Vec::new();
                    computed.referenced_columns(&mut cols);
                    if cols.iter().all(|c| base_pos.contains_key(c)) {
                        kinds.push(Kind::Derived);
                        any_derived = true;
                    } else {
                        kinds.push(Kind::Base(base.len()));
                        base.push(computed.clone());
                    }
                }
            }
        }
        if !any_derived || base.is_empty() {
            return None;
        }

        // The reduced output is [base keys…, values…]; rebuild the original
        // layout, remapping derived keys' column refs onto the emitted base
        // key positions.
        let mut projections: Vec<Expression> =
            Vec::with_capacity(self.groups.len() + self.expressions.len());
        for (g, kind) in self.groups.iter().zip(&kinds) {
            match kind {
                Kind::Base(pos) => projections.push(Expression::Ref(crate::expression::Ref {
                    column_idx: *pos,
                    return_type: g.result_type().unwrap_or(Type::Int64),
                })),
                Kind::Derived => projections.push(g.remap_refs(&base_pos)),
            }
        }
        for v in 0..self.expressions.len() {
            projections.push(Expression::Ref(crate::expression::Ref {
                column_idx: base.len() + v,
                return_type: Type::Int64,
            }));
        }

        let reduced = Aggregate {
            groups: base,
            expressions: self.expressions.clone(),
            top_k: self.top_k,
            output_limit: self.output_limit,
        };
        Some((reduced, projections))
    }

    /// Answer an unfiltered global `MIN`/`MAX`-only aggregate straight from
    /// table metadata (e.g. parquet row-group statistics), skipping the scan
    /// entirely. Applies when the aggregate sits directly on a scan with no
    /// dynamic predicates, every expression is a `MIN`/`MAX` over a plain
    /// column, and the table can prove both bounds for each column
    /// ([`Table::column_min_max`](crate::catalog::Table::column_min_max));
    /// any miss returns `None` and the ordinary scan-based path runs.
    pub(crate) fn try_compile_from_stats(
        &self,
        scan: &Input,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Option<RecordBatchOperatorSpec>, Error> {
        use crate::expression::AggregateFunc;

        if !self.groups.is_empty() || self.expressions.is_empty() {
            return Ok(None);
        }
        if !scan.dynamic_filters.is_empty() {
            return Ok(None);
        }

        let mut fields = Vec::with_capacity(self.expressions.len());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.expressions.len());
        for e in &self.expressions {
            let (agg, is_min) = match e {
                Expression::AggregateFunc(AggregateFunc::Min(a)) => (a, true),
                Expression::AggregateFunc(AggregateFunc::Max(a)) => (a, false),
                _ => return Ok(None),
            };
            // The aggregate's ref indexes the scan's output columns; map it
            // back to the table column the stats are kept under.
            let table_col = match scan.columns.get(agg.column.column_idx) {
                Some(Expression::Ref(r)) => r.column_idx,
                _ => return Ok(None),
            };
            let Some((min, max)) = scan.table.column_min_max(table_col) else {
                return Ok(None);
            };
            let scalar = if is_min { min } else { max };
            // Emit Int64, the same output type as the scan-based global
            // MIN/MAX (stats carry the column's physical storage type).
            let arr = scalar.into_inner();
            let casted = arrow::compute::cast(&arr, &arrow_schema::DataType::Int64)
                .map_err(|_| Error::UnsupportedAggregateExpression(e.clone()))?;
            fields.push(Field::new(
                if is_min { "min" } else { "max" },
                arrow_schema::DataType::Int64,
                false,
            ));
            columns.push(casted);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("stat scalars are single-row arrays");
        Ok(Some(
            dispatch::values_input(dispatcher, [batch]).record_batches(),
        ))
    }

    /// Compile `COUNT(DISTINCT x)` — the sole aggregate of the node — into a
    /// two-level GROUP BY, mirroring how DuckDB pre-aggregates a distinct
    /// argument before counting it.
    ///
    /// * **Global** (`SELECT COUNT(DISTINCT x)`): dedup `x` in a keys-only group
    ///   that emits each hash partition's distinct count, then `SUM` them.
    /// * **Grouped** (`SELECT g, COUNT(DISTINCT x) … GROUP BY g`): dedup the
    ///   `(g, x)` pairs with a two-int-key keys-only GROUP BY, then a GROUP BY on
    ///   column 0 (`g`) counts the distinct rows per group.
    ///
    /// Only a single integer group key is supported in the grouped form (the
    /// two-int-key extractor). NULLs in `x` are read through the value bits, so a
    /// nullable `x` with NULLs present can differ from DuckDB by one.
    fn compile_count_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        distinct: &crate::expression::NumericAggregate,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
        use dispatch::{
            AggregationKind, AggregationSlot, HashOnlyIntKeyExtractor, IntPairKeyExtractor,
        };

        let x = &distinct.column;
        let x_col = x.column_idx;

        // Global COUNT(DISTINCT x): dedup x in a keys-only group that emits each
        // hash partition's distinct-key count (not the keys), then SUM those
        // per-partition counts. Skips materialising the whole distinct-value
        // column just to count it. For integer columns we use the keys-only
        // HashOnlyIntKeyExtractor (8-byte entries; a bijective hash makes
        // hash-equality exactly key-equality) — half the per-entry bytes, which
        // is the dominant cost for this latency-bound build. Strings keep the
        // arena-backed StringKeyExtractor (no exact 64-bit bijection for them).
        if self.groups.is_empty() {
            let counts =
                match &x.return_type {
                    Type::Int8 => input
                        .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int8Type>>(vec![x_col]),
                    Type::Int16 => input
                        .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int16Type>>(vec![x_col]),
                    Type::Int32 => input
                        .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int32Type>>(vec![x_col]),
                    Type::Int64 => input
                        .group_by_distinct_count::<HashOnlyIntKeyExtractor<Int64Type>>(vec![x_col]),
                    Type::Utf8 => input.group_by_distinct_count::<StringKeyExtractor>(vec![x_col]),
                    dt => return Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
                };
            // Per-partition distinct counts are i64; their total can't exceed the
            // row count, so the narrow accumulator suffices.
            return Ok(counts.aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::Sum, 0)]));
        }

        // Grouped: dedup the (groups…, x) tuples with a keys-only group (the
        // inner GROUP BY emits just the key columns — no accumulator), then
        // count rows per group tuple. Group keys must be plain columns.
        let key_refs: Vec<&crate::expression::Ref> = self
            .groups
            .iter()
            .map(|g| match g {
                Expression::Ref(r) => Ok(r),
                e => Err(Error::UnexpectedAggExpression(e.clone())),
            })
            .collect::<Result<_, _>>()?;

        // Inner dedup over [groups…, x].
        let mut inner_cols: Vec<usize> = key_refs.iter().map(|r| r.column_idx).collect();
        inner_cols.push(x_col);
        let mut inner_types: Vec<Type> = key_refs.iter().map(|r| r.return_type.clone()).collect();
        inner_types.push(x.return_type.clone());

        // A single int group key with an int x keeps the packed pair extractor;
        // everything else (string keys, several keys) row-encodes the tuple.
        let deduped = if let [gt, xt] = inner_types.as_slice() {
            macro_rules! pair {
                ($g:ty, $x:ty) => {
                    input.group_by_distinct::<IntPairKeyExtractor<$g, $x>>(inner_cols.clone())
                };
            }
            macro_rules! by_x {
                ($g:ty) => {
                    match xt {
                        Type::Int8 => pair!($g, Int8Type),
                        Type::Int16 => pair!($g, Int16Type),
                        Type::Int32 => pair!($g, Int32Type),
                        Type::Int64 => pair!($g, Int64Type),
                        _ => row_key_distinct(input, inner_cols, &inner_types)?,
                    }
                };
            }
            match gt {
                Type::Int8 => by_x!(Int8Type),
                Type::Int16 => by_x!(Int16Type),
                Type::Int32 => by_x!(Int32Type),
                Type::Int64 => by_x!(Int64Type),
                _ => row_key_distinct(input, inner_cols, &inner_types)?,
            }
        } else {
            row_key_distinct(input, inner_cols, &inner_types)?
        };

        // Outer: count the deduped rows per group tuple (columns 0..k of the
        // inner output). The distinct count is exactly that row count, so a
        // top-k on it can be applied per partition.
        let group_types: Vec<Type> = key_refs.iter().map(|r| r.return_type.clone()).collect();
        count_per_group(deduped, &group_types, self.top_k)
    }

    /// Compile a grouped aggregate mixing one `COUNT(DISTINCT x)` with
    /// non-distinct aggregates (e.g. `g, SUM(a), COUNT(*), AVG(b),
    /// COUNT(DISTINCT x) … GROUP BY g`).
    ///
    /// Lowered to a two-level GROUP BY by exploiting decomposability: the
    /// non-distinct aggregates are folds of per-subgroup partials (sums add,
    /// extremes take the extreme of extremes), and `COUNT(DISTINCT x)` is the
    /// number of distinct `(groups…, x)` subgroups. So:
    /// * **Inner** GROUP BY `(groups…, x)` computes each non-distinct
    ///   aggregate's partial → `[groups…, x, p0, p1, …]`.
    /// * **Outer** GROUP BY `groups…` re-folds each partial (SUM over a sum or
    ///   count partial, MIN/MAX over an extreme partial) and uses `COUNT(*)`
    ///   of the inner rows for the distinct count.
    ///
    /// The outer slots are emitted in the original expression order so the
    /// output column layout matches the plan's aggregate output and the
    /// downstream projection (e.g. the AVG divide) lines up. AVG is already
    /// split by DuckDB into `sum`+`count` exprs, handled generically here.
    /// Group keys must be plain columns; any int/string mix is supported (a
    /// single int key with an int `x` keeps the packed pair extractor, other
    /// shapes row-encode the tuple).
    fn compile_grouped_mixed_distinct(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
        use dispatch::{
            AggregationKind, AggregationRowValueExtractor, AggregationSlot, IntPairKeyExtractor,
            MixedRowValueExtractor, RowKeyExtractor, RowKeySchema,
        };

        // The mixed-slot stages accumulate sums in i64; a SUM over a 64-bit
        // column alongside MIN/MAX would risk silent overflow — refuse.
        let any_extremes = self.expressions.iter().any(|e| {
            matches!(
                e,
                Expression::AggregateFunc(AggregateFunc::Min(_) | AggregateFunc::Max(_))
            )
        });
        if any_extremes && sum_reads_wide_column(&self.expressions) {
            return Err(Error::UnsupportedWideSumWithExtremes);
        }

        let key_refs: Vec<&crate::expression::Ref> = self
            .groups
            .iter()
            .map(|g| match g {
                Expression::Ref(r) => Ok(r),
                e => Err(Error::UnexpectedAggExpression(e.clone())),
            })
            .collect::<Result<_, _>>()?;
        let k = key_refs.len();

        // Walk the expressions once, building the inner (non-distinct partials
        // over `(groups…, x)`) and outer (re-fold partials + COUNT(*) for the
        // distinct) slot lists. The outer slots stay in expression order so the
        // output columns match the plan's aggregate layout. The inner emits
        // `[groups…, x, partial0, …]`, so partial `j` is at column `k + 1 + j`.
        let mut x: Option<&crate::expression::Ref> = None;
        let mut inner_slots: Vec<AggregationSlot> = Vec::new();
        let mut outer_slots: Vec<AggregationSlot> = Vec::new();
        // Coalesce inner partials that compute the same thing, so each is
        // scattered/merged once: all COUNT/COUNT(*) slots are identical
        // (pivot's Count contributes +1 per row regardless of column, ==
        // COUNT(*)), and same-kind aggregates of the same column are
        // identical. Fewer inner slots ⇒ a narrower hash-table entry, the
        // dominant cost of the high-cardinality inner. `partial_key` is
        // (kind-class, column): None column = a count.
        let mut partials: Vec<((AggregationKind, Option<usize>), usize)> = Vec::new();
        let partial_base = k + 1;
        let mut intern = |inner_slots: &mut Vec<AggregationSlot>,
                          key: (AggregationKind, Option<usize>),
                          slot: AggregationSlot|
         -> usize {
            if let Some(&(_, col)) = partials.iter().find(|(p, _)| *p == key) {
                return col;
            }
            let col = partial_base + inner_slots.len();
            inner_slots.push(slot);
            partials.push((key, col));
            col
        };
        for e in &self.expressions {
            match e {
                Expression::AggregateFunc(AggregateFunc::CountDistinct(a)) => {
                    if x.is_some() {
                        return Err(Error::UnsupportedAggregateExpression(e.clone()));
                    }
                    x = Some(&a.column);
                    // distinct count = number of inner (distinct-tuple) rows.
                    outer_slots.push(AggregationSlot::new(AggregationKind::CountStar, 0));
                }
                Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {
                    let col = intern(
                        &mut inner_slots,
                        (AggregationKind::CountStar, None),
                        AggregationSlot::new(AggregationKind::CountStar, 0),
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                Expression::AggregateFunc(AggregateFunc::Count(a)) => {
                    // Same value as COUNT(*) in pivot's Count semantics → coalesce.
                    let col = intern(
                        &mut inner_slots,
                        (AggregationKind::CountStar, None),
                        AggregationSlot::new(AggregationKind::Count, a.column.column_idx),
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
                    let col = intern(
                        &mut inner_slots,
                        (AggregationKind::Sum, Some(a.column.column_idx)),
                        AggregationSlot::new(AggregationKind::Sum, a.column.column_idx),
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                Expression::AggregateFunc(AggregateFunc::Min(a) | AggregateFunc::Max(a)) => {
                    let is_min = matches!(e, Expression::AggregateFunc(AggregateFunc::Min(_)));
                    let kind = match (is_min, &a.column.return_type) {
                        (true, Type::Utf8) => AggregationKind::MinStr,
                        (false, Type::Utf8) => AggregationKind::MaxStr,
                        (true, _) => AggregationKind::Min,
                        (false, _) => AggregationKind::Max,
                    };
                    // An extreme of per-subgroup extremes is the extreme.
                    let col = intern(
                        &mut inner_slots,
                        (kind, Some(a.column.column_idx)),
                        AggregationSlot::new(kind, a.column.column_idx),
                    );
                    outer_slots.push(AggregationSlot::new(kind, col));
                }
                expr => return Err(Error::UnsupportedAggregateExpression(expr.clone())),
            }
        }
        // Guaranteed by the caller (exactly one CountDistinct), but stay total.
        let x = x.ok_or_else(|| Error::UnsupportedAggregateExpressionAmount(0))?;

        // Inner stage: GROUP BY (groups…, x), computing the interned partials.
        let mut inner_cols: Vec<usize> = key_refs.iter().map(|r| r.column_idx).collect();
        inner_cols.push(x.column_idx);
        let mut inner_types: Vec<Type> = key_refs.iter().map(|r| r.return_type.clone()).collect();
        inner_types.push(x.return_type.clone());

        let inner_has_extremes = inner_slots.iter().any(|s| {
            matches!(
                s.kind,
                AggregationKind::Min
                    | AggregationKind::Max
                    | AggregationKind::MinStr
                    | AggregationKind::MaxStr
            )
        });

        // Monomorphise an aggregate stage over key type + slot arity, with the
        // mixed extractor when extremes are present.
        macro_rules! staged {
            ($spec:expr, $K:ty, $cols:expr, $slots:expr, $config:expr, $extremes:expr, $top_k:expr) => {{
                let slots = $slots;
                macro_rules! call {
                    ($V:ty) => {
                        $spec.group_by_aggregate_config::<$K, $V>($cols, slots, $top_k, $config)
                    };
                }
                if $extremes {
                    match slots.len() {
                        1 => call!(MixedRowValueExtractor<1>),
                        2 => call!(MixedRowValueExtractor<2>),
                        3 => call!(MixedRowValueExtractor<3>),
                        4 => call!(MixedRowValueExtractor<4>),
                        5 => call!(MixedRowValueExtractor<5>),
                        6 => call!(MixedRowValueExtractor<6>),
                        n => return Err(Error::UnsupportedAggregateExpressionAmount(n)),
                    }
                } else {
                    match slots.len() {
                        1 => call!(AggregationRowValueExtractor<1>),
                        2 => call!(AggregationRowValueExtractor<2>),
                        3 => call!(AggregationRowValueExtractor<3>),
                        4 => call!(AggregationRowValueExtractor<4>),
                        5 => call!(AggregationRowValueExtractor<5>),
                        6 => call!(AggregationRowValueExtractor<6>),
                        n => return Err(Error::UnsupportedAggregateExpressionAmount(n)),
                    }
                }
            }};
        }

        // Inner: a single int group key with an int x keeps the packed pair
        // extractor; other tuple shapes row-encode.
        let inner = if let [
            Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64,
            Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64,
        ] = inner_types.as_slice()
        {
            macro_rules! pair_inner {
                ($g:ty, $x:ty) => {
                    staged!(
                        input,
                        IntPairKeyExtractor<$g, $x>,
                        inner_cols.clone(),
                        inner_slots.clone(),
                        (),
                        inner_has_extremes,
                        None
                    )
                };
            }
            macro_rules! by_x {
                ($g:ty) => {
                    match &inner_types[1] {
                        Type::Int8 => pair_inner!($g, Int8Type),
                        Type::Int16 => pair_inner!($g, Int16Type),
                        Type::Int32 => pair_inner!($g, Int32Type),
                        Type::Int64 => pair_inner!($g, Int64Type),
                        _ => unreachable!("matched as int above"),
                    }
                };
            }
            match &inner_types[0] {
                Type::Int8 => by_x!(Int8Type),
                Type::Int16 => by_x!(Int16Type),
                Type::Int32 => by_x!(Int32Type),
                Type::Int64 => by_x!(Int64Type),
                _ => unreachable!("matched as int above"),
            }
        } else {
            let schema = RowKeySchema::new(
                inner_types
                    .iter()
                    .map(row_key_data_type)
                    .collect::<Result<Vec<_>, _>>()?,
            );
            staged!(
                input,
                RowKeyExtractor,
                inner_cols.clone(),
                inner_slots.clone(),
                schema,
                inner_has_extremes,
                None
            )
        };

        // Outer: GROUP BY the leading group columns of the inner output,
        // re-folding partials in expression order.
        let group_types: Vec<Type> = key_refs.iter().map(|r| r.return_type.clone()).collect();
        let outer_cols: Vec<usize> = (0..k).collect();
        let outer_has_extremes = outer_slots.iter().any(|s| {
            matches!(
                s.kind,
                AggregationKind::Min
                    | AggregationKind::Max
                    | AggregationKind::MinStr
                    | AggregationKind::MaxStr
            )
        });
        let outer = match group_types.as_slice() {
            [Type::Int8] => staged!(
                inner,
                IntKeyExtractor<Int8Type>,
                outer_cols,
                outer_slots,
                (),
                outer_has_extremes,
                self.top_k
            ),
            [Type::Int16] => staged!(
                inner,
                IntKeyExtractor<Int16Type>,
                outer_cols,
                outer_slots,
                (),
                outer_has_extremes,
                self.top_k
            ),
            [Type::Int32] => staged!(
                inner,
                IntKeyExtractor<Int32Type>,
                outer_cols,
                outer_slots,
                (),
                outer_has_extremes,
                self.top_k
            ),
            [Type::Int64] => staged!(
                inner,
                IntKeyExtractor<Int64Type>,
                outer_cols,
                outer_slots,
                (),
                outer_has_extremes,
                self.top_k
            ),
            [Type::Utf8] => staged!(
                inner,
                StringKeyExtractor,
                outer_cols,
                outer_slots,
                (),
                outer_has_extremes,
                self.top_k
            ),
            _ => {
                let schema =
                    RowKeySchema::new(group_types.iter().map(row_key_data_type).collect::<Result<
                        Vec<_>,
                        _,
                    >>(
                    )?);
                staged!(
                    inner,
                    RowKeyExtractor,
                    outer_cols,
                    outer_slots,
                    schema,
                    outer_has_extremes,
                    self.top_k
                )
            }
        };
        Ok(outer)
    }

    /// Compile a grouped multi-aggregate (`GROUP BY k1, k2 …` with one or more
    /// of COUNT(*)/SUM/COUNT) into the multi-slot group operator. DuckDB lowers
    /// grouped `AVG(c)` to `sum(c)`+`count(c)` with a downstream divide
    /// projection, so the aggregate node here only ever holds count/sum slots.
    ///
    /// Keys are normalised first: computed key expressions (`GROUP BY
    /// extract(minute FROM ts)`, `GROUP BY CASE …`, a constant) are
    /// materialised into leading columns by a projection that also carries the
    /// aggregate input columns through. The extractor is then chosen by the
    /// normalised key shape — a single int/string key and the packed
    /// two-int-key pairs keep their specialised extractors; any other mix
    /// (three or more keys, strings alongside ints, dates) uses the
    /// row-encoded multi-key extractor.
    fn compile_grouped(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
        use dispatch::{
            AggregationKind, AggregationRowValueExtractor, AggregationSlot, Compiled, Count,
            IntPairKeyExtractor, Sum,
        };

        // The columns the aggregate slots read (deduplicated, in first-use
        // order) — these ride along when a key-materialising projection is
        // inserted.
        let mut agg_cols: Vec<usize> = Vec::new();
        for e in &self.expressions {
            match e {
                Expression::AggregateFunc(
                    AggregateFunc::Sum(a)
                    | AggregateFunc::Count(a)
                    | AggregateFunc::Min(a)
                    | AggregateFunc::Max(a),
                ) => {
                    if !agg_cols.contains(&a.column.column_idx) {
                        agg_cols.push(a.column.column_idx);
                    }
                }
                Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {}
                expr => return Err(Error::UnsupportedAggregateExpression(expr.clone())),
            }
        }

        // Normalise the keys: with computed key expressions, project
        // `[key0 … keyN-1, agg input cols …]` and group on the leading
        // columns; with plain column refs, group on them directly.
        let computed_keys = self.groups.iter().any(|g| !matches!(g, Expression::Ref(_)));
        let key_types: Vec<Type> = self
            .groups
            .iter()
            .map(|g| {
                g.result_type()
                    .ok_or_else(|| Error::UnexpectedAggExpression(g.clone()))
            })
            .collect::<Result<_, _>>()?;
        let (input, key_cols, agg_col_pos): (_, Vec<usize>, _) = if !computed_keys {
            let cols = self
                .groups
                .iter()
                .map(|g| match g {
                    Expression::Ref(r) => r.column_idx,
                    _ => unreachable!("checked by computed_keys"),
                })
                .collect();
            // Slots read their original input columns.
            let pos: Vec<usize> = agg_cols.clone();
            (input, cols, pos)
        } else {
            let key_fns: Arc<Vec<ExprFn>> = Arc::new(
                self.groups
                    .iter()
                    .map(|g| g.compile())
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let passthrough = agg_cols.clone();
            let projected = input.project(move || {
                let mut evals: Vec<ExprEvalFn> = key_fns.iter().map(|b| b()).collect();
                let passthrough = passthrough.clone();
                move |batch: RecordBatch| {
                    let n = batch.num_rows();
                    let mut columns: Vec<ArrayRef> = evals
                        .iter_mut()
                        .map(|eval| match eval(&batch) {
                            ExprResult::Array(a) => a,
                            // A constant key (e.g. `GROUP BY 1, URL`):
                            // broadcast to the batch's row count so the key
                            // column lines up with the others.
                            ExprResult::Scalar(s) => {
                                let arr = s.into_inner();
                                let zeros = arrow_array::UInt32Array::from(vec![0u32; n]);
                                arrow::compute::take(&arr, &zeros, None).unwrap()
                            }
                        })
                        .collect();
                    columns.extend(passthrough.iter().map(|&c| batch.column(c).clone()));
                    let fields: Vec<Field> = columns
                        .iter()
                        .enumerate()
                        .map(|(i, c)| Field::new(format!("c{i}"), c.data_type().clone(), true))
                        .collect();
                    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
                }
            });
            let k = self.groups.len();
            // Slot input column `agg_cols[j]` now lives at projected column `k + j`.
            let pos: Vec<usize> = (k..k + agg_cols.len()).collect();
            (projected, (0..k).collect(), pos)
        };
        // Where a slot's original input column lives after normalisation.
        let col_of = |orig: usize| -> usize {
            let j = agg_cols.iter().position(|&c| c == orig).unwrap();
            agg_col_pos[j]
        };

        let slots: Vec<AggregationSlot> = self
            .expressions
            .iter()
            .map(|e| match e {
                Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {
                    Ok(AggregationSlot::new(AggregationKind::CountStar, 0))
                }
                Expression::AggregateFunc(AggregateFunc::Sum(a)) => Ok(AggregationSlot::new(
                    AggregationKind::Sum,
                    col_of(a.column.column_idx),
                )),
                Expression::AggregateFunc(AggregateFunc::Count(a)) => Ok(AggregationSlot::new(
                    AggregationKind::Count,
                    col_of(a.column.column_idx),
                )),
                Expression::AggregateFunc(AggregateFunc::Min(a)) => {
                    let kind = if a.column.return_type == Type::Utf8 {
                        AggregationKind::MinStr
                    } else {
                        AggregationKind::Min
                    };
                    Ok(AggregationSlot::new(kind, col_of(a.column.column_idx)))
                }
                Expression::AggregateFunc(AggregateFunc::Max(a)) => {
                    let kind = if a.column.return_type == Type::Utf8 {
                        AggregationKind::MaxStr
                    } else {
                        AggregationKind::Max
                    };
                    Ok(AggregationSlot::new(kind, col_of(a.column.column_idx)))
                }
                expr => Err(Error::UnsupportedAggregateExpression(expr.clone())),
            })
            .collect::<Result<_, _>>()?;

        // A lone COUNT(*) compiles to the dedicated count value (straight-line,
        // no per-row slot dispatch) regardless of key shape.
        let lone_count = matches!(
            self.expressions.as_slice(),
            [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
        );

        // MIN/MAX slots need the mixed-slot extractor (heterogeneous, fold- and
        // arena-aware); the additive extractors can't represent them.
        let has_extremes = slots.iter().any(|s| {
            matches!(
                s.kind,
                AggregationKind::Min
                    | AggregationKind::Max
                    | AggregationKind::MinStr
                    | AggregationKind::MaxStr
            )
        });
        // The mixed extractor accumulates sums in i64; a SUM over a 64-bit
        // column needs the i128 width it doesn't have. Refuse loudly rather
        // than risk silent overflow.
        if has_extremes && sum_reads_wide_column(&self.expressions) {
            return Err(Error::UnsupportedWideSumWithExtremes);
        }

        // Per-slot signature (kind + the `SUM` column's type), used to pick a
        // compiled, monomorphised value extractor when the signature matches one
        // we've specialised; any other signature falls back to the generic enum
        // extractor (`AggregationRowValueExtractor<N>`).
        enum Sig {
            Count,
            Sum(Type),
        }
        let sig: Vec<Sig> = self
            .expressions
            .iter()
            .map(|e| match e {
                Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
                    Sig::Sum(a.column.return_type.clone())
                }
                // CountStar / Count — validated above when building `slots`.
                _ => Sig::Count,
            })
            .collect();

        let top_k = self.top_k;
        let output_limit = self.output_limit;

        // Accumulator width, by the same column-type rule as the global path:
        // i128 only when a SUM reads a 64-bit column, else i64 (narrow entries).
        // Grouped sums are almost always over narrow columns, so this is i64
        // in practice; the i128 arm keeps a wide grouped sum correct rather
        // than silently overflowing the slot.
        let wide = sum_reads_wide_column(&self.expressions);

        // The generic enum value extractor, monomorphised by slot arity and
        // accumulator width, for a given key extractor.
        macro_rules! by_arity {
            ($K:ty, $acc:ty) => {
                match slots.len() {
                    1 => Ok(input
                        .group_by_aggregate_limited::<$K, AggregationRowValueExtractor<1, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    2 => Ok(input
                        .group_by_aggregate_limited::<$K, AggregationRowValueExtractor<2, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    3 => Ok(input
                        .group_by_aggregate_limited::<$K, AggregationRowValueExtractor<3, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    4 => Ok(input
                        .group_by_aggregate_limited::<$K, AggregationRowValueExtractor<4, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    5 => Ok(input
                        .group_by_aggregate_limited::<$K, AggregationRowValueExtractor<5, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    6 => Ok(input
                        .group_by_aggregate_limited::<$K, AggregationRowValueExtractor<6, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            };
        }
        // The mixed-slot extractor, monomorphised by arity, for a typed key.
        macro_rules! mixed_by_arity {
            ($K:ty) => {
                match slots.len() {
                    1 => Ok(input
                        .group_by_aggregate_limited::<$K, dispatch::MixedRowValueExtractor<1>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    2 => Ok(input
                        .group_by_aggregate_limited::<$K, dispatch::MixedRowValueExtractor<2>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    3 => Ok(input
                        .group_by_aggregate_limited::<$K, dispatch::MixedRowValueExtractor<3>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    4 => Ok(input
                        .group_by_aggregate_limited::<$K, dispatch::MixedRowValueExtractor<4>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    5 => Ok(input
                        .group_by_aggregate_limited::<$K, dispatch::MixedRowValueExtractor<5>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    6 => Ok(input
                        .group_by_aggregate_limited::<$K, dispatch::MixedRowValueExtractor<6>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            Default::default(),
                        )),
                    n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            };
        }
        // Full value-extractor selection for a typed key extractor.
        macro_rules! with_key {
            ($K:ty) => {{
                if has_extremes {
                    mixed_by_arity!($K)
                } else if lone_count {
                    Ok(input.group_by_aggregate_limited::<$K, Compiled<(Count,)>>(
                        key_cols,
                        slots,
                        top_k,
                        output_limit,
                        Default::default(),
                    ))
                } else if wide {
                    by_arity!($K, i128)
                } else {
                    by_arity!($K, i64)
                }
            }};
        }

        // Single-key shapes keep their specialised extractors.
        if let [t] = key_types.as_slice() {
            return match t {
                Type::Int8 => with_key!(IntKeyExtractor<Int8Type>),
                Type::Int16 => with_key!(IntKeyExtractor<Int16Type>),
                Type::Int32 => with_key!(IntKeyExtractor<Int32Type>),
                Type::Int64 => with_key!(IntKeyExtractor<Int64Type>),
                Type::Utf8 => with_key!(StringKeyExtractor),
                // Timestamps execute as Int64 epoch seconds.
                Type::Timestamp => with_key!(IntKeyExtractor<Int64Type>),
                _ => self.compile_grouped_row_key(input, key_cols, &key_types, slots, lone_count),
            };
        }

        // Two integer keys: the packed-u128 pair extractor, including the
        // compiled straight-line shape for COUNT(*)+SUM(i16)+SUM(i16)+COUNT.
        if let [a, b] = key_types.as_slice() {
            macro_rules! pair {
                ($a:ty, $b:ty) => {{
                    type Key = IntPairKeyExtractor<$a, $b>;
                    return match sig.as_slice() {
                        // COUNT(*), SUM(i16), SUM(i16), COUNT — compiled to
                        // straight-line code with no per-row enum dispatch
                        // (count + sum + avg over two int keys).
                        [
                            Sig::Count,
                            Sig::Sum(Type::Int16),
                            Sig::Sum(Type::Int16),
                            Sig::Count,
                        ] => {
                            type V = Compiled<(Count, Sum<Int16Type>, Sum<Int16Type>, Count)>;
                            Ok(input.group_by_aggregate_limited::<Key, V>(
                                key_cols,
                                slots,
                                top_k,
                                output_limit,
                                Default::default(),
                            ))
                        }
                        _ => with_key!(Key),
                    };
                }};
            }
            match (a, b) {
                (Type::Int64, Type::Int32) => pair!(Int64Type, Int32Type),
                (Type::Int32, Type::Int32) => pair!(Int32Type, Int32Type),
                (Type::Int16, Type::Int32) => pair!(Int16Type, Int32Type),
                (Type::Int16, Type::Int16) => pair!(Int16Type, Int16Type),
                (Type::Int64, Type::Int64) => pair!(Int64Type, Int64Type),
                (Type::Int32, Type::Int64) => pair!(Int32Type, Int64Type),
                _ => {}
            }
        }

        // Everything else: the row-encoded multi-key extractor.
        self.compile_grouped_row_key(input, key_cols, &key_types, slots, lone_count)
    }

    /// Compile a grouped aggregate over the row-encoded multi-key extractor —
    /// the general path for key shapes without a specialised extractor.
    fn compile_grouped_row_key(
        &self,
        input: RecordBatchOperatorSpec,
        key_cols: Vec<usize>,
        key_types: &[Type],
        slots: Vec<dispatch::AggregationSlot>,
        lone_count: bool,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use dispatch::{
            AggregationKind, AggregationRowValueExtractor, Compiled, Count, MixedRowValueExtractor,
            RowKeyExtractor, RowKeySchema,
        };

        let schema = RowKeySchema::new(
            key_types
                .iter()
                .map(row_key_data_type)
                .collect::<Result<Vec<_>, _>>()?,
        );
        let top_k = self.top_k;
        let output_limit = self.output_limit;
        let wide = sum_reads_wide_column(&self.expressions);

        let has_extremes = slots.iter().any(|s| {
            matches!(
                s.kind,
                AggregationKind::Min
                    | AggregationKind::Max
                    | AggregationKind::MinStr
                    | AggregationKind::MaxStr
            )
        });
        if has_extremes && wide {
            return Err(Error::UnsupportedWideSumWithExtremes);
        }
        if has_extremes {
            macro_rules! mixed {
                ($n:literal) => {
                    Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, MixedRowValueExtractor<$n>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        ))
                };
            }
            return match slots.len() {
                1 => mixed!(1),
                2 => mixed!(2),
                3 => mixed!(3),
                4 => mixed!(4),
                5 => mixed!(5),
                6 => mixed!(6),
                n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
            };
        }

        if lone_count {
            return Ok(
                input.group_by_aggregate_limited::<RowKeyExtractor, Compiled<(Count,)>>(
                    key_cols,
                    slots,
                    top_k,
                    output_limit,
                    schema,
                ),
            );
        }
        macro_rules! by_arity {
            ($acc:ty) => {
                match slots.len() {
                    1 => Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, AggregationRowValueExtractor<1, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        )),
                    2 => Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, AggregationRowValueExtractor<2, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        )),
                    3 => Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, AggregationRowValueExtractor<3, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        )),
                    4 => Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, AggregationRowValueExtractor<4, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        )),
                    5 => Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, AggregationRowValueExtractor<5, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        )),
                    6 => Ok(input
                        .group_by_aggregate_limited::<RowKeyExtractor, AggregationRowValueExtractor<6, $acc>>(
                            key_cols,
                            slots,
                            top_k,
                            output_limit,
                            schema,
                        )),
                    n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            };
        }
        if wide {
            by_arity!(i128)
        } else {
            by_arity!(i64)
        }
    }
}

impl Input {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        slots: &mut DynamicFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let column_indices: Vec<usize> = self
            .columns
            .iter()
            .filter_map(|e| match e {
                // DuckDB emits column_idx == usize::MAX as a sentinel for
                // "no column needed" (e.g. COUNT(*) scans). Skip these.
                Expression::Ref(r) if r.column_idx == usize::MAX => None,
                Expression::Ref(r) => Some(Ok(r.column_idx)),
                _ => Some(Err(Error::UnexpectedInputExpression(e.clone()))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projection = DispatchProjection::columns(column_indices);
        let dynamic_filters = build_dynamic_scan_predicates(&self.dynamic_filters, slots);
        self.table
            .compile(
                dispatcher,
                projection,
                dynamic_filters,
                self.emit_row_group_metadata,
            )
            .map_err(Error::TableScan)
    }
}

impl Materialize {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let projection = DispatchProjection::columns(self.columns.iter().copied());
        Ok(self.table.materialize(input, projection))
    }
}

/// Get-or-create the shared [`DynamicFilterSlot`] for `slot_id` within this
/// compile. A producer (`TopN`) and the consumer scans referencing the same id
/// resolve to one `Arc`; a later compile of the same (cached) plan mints fresh,
/// empty slots — so no stale boundary or pooled scan memory is reused.
fn slot_for(slots: &mut DynamicFilterSlots, slot_id: usize) -> Arc<DynamicFilterSlot> {
    Arc::clone(
        slots
            .entry(slot_id)
            .or_insert_with(|| Arc::new(RwLock::new(None))),
    )
}

/// Lower the Top-N's dynamic filters into logical [`DynamicScanPredicate`]s the
/// table can use for pruning. Each carries the column, comparison, and the
/// shared slot the Top-N fills with its live boundary; turning that into actual
/// (e.g. row-group) elimination is the storage backend's job.
fn build_dynamic_scan_predicates(
    filters: &[DynamicFilter],
    slots: &mut DynamicFilterSlots,
) -> Vec<DynamicScanPredicate> {
    filters
        .iter()
        .map(|df| DynamicScanPredicate {
            column_idx: df.column_idx,
            compare_type: df.compare_type,
            slot: slot_for(slots, df.slot_id),
        })
        .collect()
}

impl Filter {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let filters = Arc::new(
            self.conditions
                .iter()
                .map(|e| e.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        assert!(!filters.is_empty());

        Ok(input.filter(|| {
            let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
            move |batch: &RecordBatch| {
                eval_fns
                    .iter_mut()
                    .map(|f| {
                        let result = f(batch);
                        let (arr, _) = result.as_datum().get();
                        arr.as_any().downcast_ref::<BooleanArray>().unwrap().clone()
                    })
                    .reduce(|left, right| and(&left, &right).unwrap())
                    .unwrap()
            }
        }))
    }
}

impl OrderBy {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let orders = self
            .order_bys
            .iter()
            .map(|node| {
                let col = match &node.expression {
                    Expression::Ref(r) => Ok(r.column_idx),
                    expr => Err(Error::UnsupportedOrderByExpression(expr.clone())),
                }?;
                let descending = matches!(node.direction, OrderByDirection::Desc);
                Ok(DispatchOrderBy::new(col, descending, false))
            })
            .collect::<Result<Vec<_>, _>>()?;
        //Tmp -- until we have a order by without limit
        Ok(input.order_by_limit(orders, 1_000_000_000))
    }
}

impl TopN {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        slots: &mut DynamicFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let orders = self
            .order_bys
            .iter()
            .map(|node| {
                let col = match &node.expression {
                    Expression::Ref(r) => Ok(r.column_idx),
                    expr => Err(Error::UnsupportedTopKExpression(expr.clone())),
                }?;
                let descending = matches!(node.direction, OrderByDirection::Desc);
                Ok(DispatchOrderBy::new(col, descending, false))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // If DuckDB's Top-N optimizer marked this node as a dynamic-filter
        // producer, hand it the shared slot (minted fresh per compile, shared
        // with the consumer scan by `slot_id`) so it publishes its running
        // boundary into it, tightening row-group pruning at sibling scans.
        let dynamic_filter = self
            .produces_dynamic_filter
            .as_ref()
            .map(|df| slot_for(slots, df.slot_id));
        Ok(input.order_by_limit_offset(orders, self.limit, self.offset, dynamic_filter))
    }
}

impl Limit {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Only the unordered shape reaches here — `Limit → OrderBy` was fused
        // into a TopN during planning (except an unbounded `OFFSET`-only
        // modifier, where the dispatch operator's `usize::MAX` convention
        // applies and the sort below it emits a single batch whose order the
        // limit stage preserves).
        Ok(input.limit(self.limit.unwrap_or(usize::MAX), self.offset))
    }
}

impl CreateTable {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<dyn Catalog>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if self.or_replace {
            return Err(Error::UnsupportedCreateTableOrReplace);
        }
        if self.temporary {
            return Err(Error::UnsupportedTemporaryCreateTable);
        }
        if self.has_query {
            return Err(Error::UnsupportedCreateTableAs);
        }
        if self.constraint_count != 0 {
            return Err(Error::UnsupportedCreateTableConstraints(
                self.constraint_count,
            ));
        }

        // The catalog does the up-front work — fetching every data file's footer
        // in parallel over the worker pool, here on the coordinator — and returns
        // the plan that writes the materialized table into the catalog. (Running
        // that fetch dataflow from a per-worker nullary would nest a dataflow
        // inside a worker and deadlock the pool.)
        catalog
            .create_table(self.request.clone(), dispatcher)
            .map_err(Error::CreateTable)
    }
}

impl DummyScan {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // One nullary per worker sharing a flag, so exactly one emits the single
        // dummy row the parent projection runs over (mirrors `CreateTable`).
        let emitted = Arc::new(AtomicBool::new(false));
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| DummyScanNullaryFactory::new(emitted.clone())),
        ))
    }
}
