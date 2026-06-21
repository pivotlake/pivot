//! Per-operator compile impls.
//!
//! Each [`Operator`](crate::operator::Operator) variant has a `compile`
//! method here that translates it into a [`RecordBatchOperatorSpec`] call.

use crate::catalog::{Catalog, DynamicScanPredicate, QueryContext};
use crate::compile::dummy_scan::DummyScanNullaryFactory;
use crate::compile::{DynamicFilterSlots, Error, ExprEvalFn, ExprFn, ExprResult};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::Expression;
use crate::operator::{
    Aggregate, CreateTable, DummyScan, Filter, Input, Materialize, OrderBy, OrderByDirection,
    Projection, TopN,
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
                let columns: Vec<ArrayRef> = evals
                    .iter_mut()
                    .map(|eval| match eval(&batch) {
                        ExprResult::Array(a) => a,
                        ExprResult::Scalar(s) => s.into_inner(),
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

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use dispatch::{AggregationKind, AggregationSlot};

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

        // Grouped. A single key with a lone COUNT(*) uses the dedicated count
        // path (which also handles string keys); multi-key grouping or
        // sum/count/avg aggregates use the multi-aggregate path.
        let simple_count = self.groups.len() == 1
            && matches!(
                self.expressions.as_slice(),
                [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
            );
        if !simple_count {
            return self.compile_grouped(input);
        }

        if self.expressions.len() != 1 {
            return Err(Error::UnsupportedAggregateExpressionAmount(
                self.expressions.len(),
            ));
        }

        match &self.expressions[0] {
            Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {}
            expr => return Err(Error::UnsupportedAggregateExpression(expr.clone())),
        }

        match self.groups.len() {
            0 => Ok(input.count()),
            1 => match &self.groups[0] {
                Expression::Ref(group) => {
                    let col = group.column_idx;
                    match &group.return_type {
                        Type::Int8 => Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int8Type>>(col)),
                        Type::Int16 => Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int16Type>>(col)),
                        Type::Int32 => Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int32Type>>(col)),
                        Type::Int64 => Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int64Type>>(col)),
                        Type::Utf8 => Ok(input.group_by_count::<StringKeyExtractor>(col)),
                        dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
                    }
                }
                // Computed group key (e.g. `GROUP BY date_trunc('minute',
                // EventTime)` or `GROUP BY CASE …`). Materialise the key into a
                // single column with a projection, then group on it, picking the
                // key extractor from the expression's static result type. Numeric
                // keys (date_trunc/minute yield Int64) use the Int64 extractor;
                // a string-valued CASE uses the string extractor.
                computed => {
                    let key_fn = computed.compile()?;
                    let keyed = input.project(move || {
                        let mut eval = key_fn();
                        move |batch: RecordBatch| {
                            let col: ArrayRef = match eval(&batch) {
                                ExprResult::Array(a) => a,
                                ExprResult::Scalar(s) => s.into_inner(),
                            };
                            let field = Field::new("k", col.data_type().clone(), true);
                            RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![col])
                                .unwrap()
                        }
                    });
                    match computed.result_type() {
                        Some(Type::Utf8) => Ok(keyed.group_by_count::<StringKeyExtractor>(0)),
                        _ => Ok(keyed
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int64Type>>(0)),
                    }
                }
            },
            n => Err(Error::UnsupportedAggregateGroupAmount(n)),
        }
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
                match &x.return_type {
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

    /// Compile a grouped aggregate mixing one `COUNT(DISTINCT x)` with
    /// non-distinct aggregates (e.g. ClickBench Q9:
    /// `RegionID, SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth),
    /// COUNT(DISTINCT UserID) GROUP BY RegionID`).
    ///
    /// Lowered to a two-level GROUP BY by exploiting decomposability: the
    /// non-distinct aggregates (SUM/COUNT/COUNT(*)) are sums of per-subgroup
    /// partials, and `COUNT(DISTINCT x)` is the number of distinct `(group, x)`
    /// subgroups. So:
    /// * **Inner** GROUP BY `(group, x)` computes each non-distinct aggregate's
    ///   partial → `[group, x, p0, p1, …]`.
    /// * **Outer** GROUP BY `group` re-sums each partial (SUM over the inner
    ///   column) and uses `COUNT(*)` of the inner rows for the distinct count.
    ///
    /// The outer slots are emitted in the original expression order — distinct
    /// expr → `COUNT(*)`, each non-distinct expr → `SUM` of its inner partial —
    /// so the output column layout matches DuckDB's aggregate output and the
    /// downstream projection (e.g. the AVG divide) lines up. AVG is already split
    /// by DuckDB into `sum`+`count` exprs, handled generically here. Single
    /// integer group key only (the inner uses the two-int-key extractor).
    fn compile_grouped_mixed_distinct(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
        use dispatch::{
            AggregationKind, AggregationRowValueExtractor, AggregationSlot, IntPairKeyExtractor,
        };

        if self.groups.len() != 1 {
            return Err(Error::UnsupportedAggregateGroupAmount(self.groups.len()));
        }
        let g = match &self.groups[0] {
            Expression::Ref(r) => r,
            e => return Err(Error::UnexpectedAggExpression(e.clone())),
        };
        let g_col = g.column_idx;

        // Walk the expressions once, building the inner (non-distinct partials
        // over `(g, x)`) and outer (re-sum partials + COUNT(*) for the distinct)
        // slot lists. The outer slots stay in expression order so the output
        // columns match DuckDB's aggregate layout. The inner emits
        // `[g, x, partial0, partial1, …]`, so partial `k` is at column `2 + k`.
        let mut x: Option<&crate::expression::Ref> = None;
        let mut inner_slots: Vec<AggregationSlot> = Vec::new();
        let mut outer_slots: Vec<AggregationSlot> = Vec::new();
        // Coalesce inner partials that compute the *same* column, so each is
        // scattered/merged once: all COUNT/COUNT(*) slots are identical (pivot's
        // Count contributes +1 per row regardless of column/null, == COUNT(*)),
        // and SUMs of the same column are identical. Fewer inner slots ⇒ a
        // narrower scattered hash-table entry, which is the dominant cost of the
        // high-cardinality inner. `partial_key`: None = a count, Some(col) = a sum.
        // Maps that key to the inner output column (`2 + k`).
        let mut partials: Vec<(Option<usize>, usize)> = Vec::new();
        let mut intern = |inner_slots: &mut Vec<AggregationSlot>,
                          key: Option<usize>,
                          slot: AggregationSlot|
         -> usize {
            if let Some(&(_, col)) = partials.iter().find(|(k, _)| *k == key) {
                return col;
            }
            let col = 2 + inner_slots.len();
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
                    // distinct count = number of inner (distinct-pair) rows for g.
                    outer_slots.push(AggregationSlot::new(AggregationKind::CountStar, 0));
                }
                Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {
                    let col = intern(
                        &mut inner_slots,
                        None,
                        AggregationSlot::new(AggregationKind::CountStar, 0),
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                Expression::AggregateFunc(AggregateFunc::Count(a)) => {
                    // Same value as COUNT(*) in pivot's Count semantics → coalesce.
                    let col = intern(
                        &mut inner_slots,
                        None,
                        AggregationSlot::new(AggregationKind::Count, a.column.column_idx),
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
                    let col = intern(
                        &mut inner_slots,
                        Some(a.column.column_idx),
                        AggregationSlot::new(AggregationKind::Sum, a.column.column_idx),
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                expr => return Err(Error::UnsupportedAggregateExpression(expr.clone())),
            }
        }
        // Guaranteed by the caller (exactly one CountDistinct), but stay total.
        let x = x.ok_or_else(|| Error::UnsupportedAggregateExpressionAmount(0))?;
        let x_col = x.column_idx;

        // group_by_aggregate monomorphised over key type + slot arity.
        macro_rules! grouped {
            ($spec:expr, $K:ty, $cols:expr, $slots:expr) => {{
                let slots = $slots;
                match slots.len() {
                    1 => $spec.group_by_aggregate::<$K, AggregationRowValueExtractor<1>>(
                        $cols, slots, None,
                    ),
                    2 => $spec.group_by_aggregate::<$K, AggregationRowValueExtractor<2>>(
                        $cols, slots, None,
                    ),
                    3 => $spec.group_by_aggregate::<$K, AggregationRowValueExtractor<3>>(
                        $cols, slots, None,
                    ),
                    4 => $spec.group_by_aggregate::<$K, AggregationRowValueExtractor<4>>(
                        $cols, slots, None,
                    ),
                    5 => $spec.group_by_aggregate::<$K, AggregationRowValueExtractor<5>>(
                        $cols, slots, None,
                    ),
                    6 => $spec.group_by_aggregate::<$K, AggregationRowValueExtractor<6>>(
                        $cols, slots, None,
                    ),
                    n => return Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            }};
        }
        macro_rules! build {
            ($G:ty, $X:ty) => {{
                let inner = grouped!(
                    input,
                    IntPairKeyExtractor<$G, $X>,
                    vec![g_col, x_col],
                    inner_slots
                );
                Ok(grouped!(inner, IntKeyExtractor<$G>, vec![0], outer_slots))
            }};
        }
        macro_rules! by_x {
            ($G:ty) => {
                match &x.return_type {
                    Type::Int8 => build!($G, Int8Type),
                    Type::Int16 => build!($G, Int16Type),
                    Type::Int32 => build!($G, Int32Type),
                    Type::Int64 => build!($G, Int64Type),
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

    /// Compile a grouped multi-aggregate (`GROUP BY k1, k2 …` with one or more
    /// of COUNT(*)/SUM/COUNT) into the multi-slot group operator. DuckDB lowers
    /// grouped `AVG(c)` to `sum(c)`+`count(c)` with a downstream divide
    /// projection, so the aggregate node here only ever holds count/sum slots.
    fn compile_grouped(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use arrow_array::types::{Int16Type, Int32Type, Int64Type};
        use dispatch::{
            AggregationKind, AggregationRowValueExtractor, AggregationSlot, Compiled, Count,
            IntPairKeyExtractor, RowKeyExtractor, Sum,
        };

        let key_refs: Vec<&crate::expression::Ref> = self
            .groups
            .iter()
            .map(|g| match g {
                Expression::Ref(r) => Ok(r),
                e => Err(Error::UnexpectedAggExpression(e.clone())),
            })
            .collect::<Result<_, _>>()?;

        let key_cols: Vec<usize> = key_refs.iter().map(|r| r.column_idx).collect();

        let slots: Vec<AggregationSlot> = self
            .expressions
            .iter()
            .map(|e| match e {
                Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {
                    Ok(AggregationSlot::new(AggregationKind::CountStar, 0))
                }
                Expression::AggregateFunc(AggregateFunc::Sum(a)) => Ok(AggregationSlot::new(
                    AggregationKind::Sum,
                    a.column.column_idx,
                )),
                Expression::AggregateFunc(AggregateFunc::Count(a)) => Ok(AggregationSlot::new(
                    AggregationKind::Count,
                    a.column.column_idx,
                )),
                expr => Err(Error::UnsupportedAggregateExpression(expr.clone())),
            })
            .collect::<Result<_, _>>()?;

        // Per-slot signature (kind + the `SUM` column's type), used to pick a
        // compiled, monomorphised value extractor when the signature matches one
        // we've specialised; any other signature falls back to the generic enum
        // extractor (`AggregationRowValueExtractor<N>`) below.
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

        // Accumulator width, by the same column-type rule as the global path:
        // i128 only when a SUM reads a 64-bit column, else i64 (narrow entries).
        // Grouped sums are almost always over narrow columns, so this is i64
        // in practice; the i128 arm keeps a wide grouped sum correct rather
        // than silently overflowing the slot.
        let wide = sum_reads_wide_column(&self.expressions);

        // Monomorphise over the two key types, the slot arity (N), and the
        // accumulator width ($acc). The compiled q32 shape is i16-only (never
        // wide), so only the generic fallback needs the width parameter.
        macro_rules! by_arity {
            ($a:ty, $b:ty, $acc:ty) => {{
                type Key = IntPairKeyExtractor<$a, $b>;
                match sig.as_slice() {
                    // COUNT(*), SUM(i16), SUM(i16), COUNT — compiled to straight-line
                    // code with no per-row enum dispatch. (q32: count + sum + avg.)
                    [
                        Sig::Count,
                        Sig::Sum(Type::Int16),
                        Sig::Sum(Type::Int16),
                        Sig::Count,
                    ] => {
                        type V = Compiled<(Count, Sum<Int16Type>, Sum<Int16Type>, Count)>;
                        Ok(input.group_by_aggregate::<Key, V>(key_cols, slots, top_k))
                    }
                    // Fallback: the generic enum extractor, monomorphised by arity.
                    _ => match slots.len() {
                        1 => Ok(input
                            .group_by_aggregate::<Key, AggregationRowValueExtractor<1, $acc>>(
                                key_cols, slots, top_k,
                            )),
                        2 => Ok(input
                            .group_by_aggregate::<Key, AggregationRowValueExtractor<2, $acc>>(
                                key_cols, slots, top_k,
                            )),
                        3 => Ok(input
                            .group_by_aggregate::<Key, AggregationRowValueExtractor<3, $acc>>(
                                key_cols, slots, top_k,
                            )),
                        4 => Ok(input
                            .group_by_aggregate::<Key, AggregationRowValueExtractor<4, $acc>>(
                                key_cols, slots, top_k,
                            )),
                        5 => Ok(input
                            .group_by_aggregate::<Key, AggregationRowValueExtractor<5, $acc>>(
                                key_cols, slots, top_k,
                            )),
                        6 => Ok(input
                            .group_by_aggregate::<Key, AggregationRowValueExtractor<6, $acc>>(
                                key_cols, slots, top_k,
                            )),
                        n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                    },
                }
            }};
        }

        // Pick the accumulator width once, then dispatch on the key types.
        macro_rules! by_keys {
            ($a:ty, $b:ty) => {
                if wide {
                    by_arity!($a, $b, i128)
                } else {
                    by_arity!($a, $b, i64)
                }
            };
        }

        // The general fallback: encode the whole key tuple into one byte blob.
        // Handles any shape the specialised extractors don't — a string key, 3+
        // keys, or an integer pair we haven't monomorphised.
        macro_rules! row_fallback {
            ($acc:ty) => {{
                let Some(schema) = row_key_schema(&key_refs) else {
                    return Err(Error::DataTypeNotSupportedForGroupBy(
                        key_refs[0].return_type.clone(),
                    ));
                };
                macro_rules! by_n {
                    ($n:literal) => {{
                        type V = AggregationRowValueExtractor<$n, $acc>;
                        Ok(input.group_by_aggregate_config::<RowKeyExtractor, V>(
                            key_cols, slots, top_k, schema,
                        ))
                    }};
                }
                match slots.len() {
                    1 => by_n!(1),
                    2 => by_n!(2),
                    3 => by_n!(3),
                    4 => by_n!(4),
                    5 => by_n!(5),
                    6 => by_n!(6),
                    n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            }};
        }

        // Two integer keys pack into the specialised u128 pair extractor; every
        // other shape falls back to the row-encoded extractor. `input` is
        // consumed exactly once, in whichever branch runs.
        let int_pair = key_cols.len() == 2
            && matches!(
                (&key_refs[0].return_type, &key_refs[1].return_type),
                (Type::Int64, Type::Int32)
                    | (Type::Int32, Type::Int32)
                    | (Type::Int16, Type::Int32)
                    | (Type::Int16, Type::Int16)
                    | (Type::Int64, Type::Int64)
                    | (Type::Int32, Type::Int64)
            );
        if int_pair {
            match (&key_refs[0].return_type, &key_refs[1].return_type) {
                (Type::Int64, Type::Int32) => by_keys!(Int64Type, Int32Type),
                (Type::Int32, Type::Int32) => by_keys!(Int32Type, Int32Type),
                (Type::Int16, Type::Int32) => by_keys!(Int16Type, Int32Type),
                (Type::Int16, Type::Int16) => by_keys!(Int16Type, Int16Type),
                (Type::Int64, Type::Int64) => by_keys!(Int64Type, Int64Type),
                (Type::Int32, Type::Int64) => by_keys!(Int32Type, Int64Type),
                _ => unreachable!("int_pair guard restricts to these arms"),
            }
        } else if wide {
            row_fallback!(i128)
        } else {
            row_fallback!(i64)
        }
    }
}

/// Map the GROUP BY key columns' planner types to the arrow types the
/// [`dispatch::RowKeyExtractor`] encodes, in key order. Returns `None` if any
/// key column has a type the row encoding doesn't support, so the caller can
/// report it unsupported rather than panicking in `RowKeySchema::new`.
fn row_key_schema(key_refs: &[&crate::expression::Ref]) -> Option<dispatch::RowKeySchema> {
    use arrow_schema::DataType;
    let types = key_refs
        .iter()
        .map(|r| match r.return_type {
            Type::Int8 => Some(DataType::Int8),
            Type::Int16 => Some(DataType::Int16),
            Type::Int32 => Some(DataType::Int32),
            Type::Int64 => Some(DataType::Int64),
            Type::Utf8 => Some(DataType::Utf8View),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(dispatch::RowKeySchema::new(types))
}

impl Input {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        ctx: &dyn QueryContext,
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
                ctx,
            )
            .map_err(Error::TableScan)
    }
}

impl Materialize {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        ctx: &dyn QueryContext,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let projection = DispatchProjection::columns(self.columns.iter().copied());
        Ok(self.table.materialize(input, projection, ctx))
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
