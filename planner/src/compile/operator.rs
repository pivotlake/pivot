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
    Aggregate, CreateTable, DummyScan, Filter, Input, OrderBy, OrderByDirection, Projection, TopN,
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
                move |batch: RecordBatch| batch.project(&idxs).unwrap()
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

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        use crate::expression::AggregateFunc;
        use dispatch::{AggKind, AggSpec};

        // Global aggregates (no GROUP BY). A bare `COUNT(*)` keeps the
        // dedicated row-counter; anything else (SUM/AVG/COUNT, possibly
        // several) compiles to the multi-aggregate operator, one output
        // column per expression.
        if self.groups.is_empty() {
            if matches!(
                self.expressions.as_slice(),
                [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
            ) {
                return Ok(input.count());
            }

            let specs = self
                .expressions
                .iter()
                .map(|e| match e {
                    Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
                        Ok(AggSpec::new(AggKind::Sum, a.column.column_idx))
                    }
                    Expression::AggregateFunc(AggregateFunc::Count(a)) => {
                        Ok(AggSpec::new(AggKind::Count, a.column.column_idx))
                    }
                    Expression::AggregateFunc(AggregateFunc::Avg(a)) => {
                        Ok(AggSpec::new(AggKind::Avg, a.column.column_idx))
                    }
                    // COUNT(*) ignores its column; the index is a placeholder.
                    Expression::AggregateFunc(AggregateFunc::CountStar(_)) => {
                        Ok(AggSpec::new(AggKind::CountStar, 0))
                    }
                    expr => Err(Error::UnsupportedAggregateExpression(expr.clone())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(input.aggregate(specs));
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
                // EventTime)`). Materialise the key into a single Int64 column
                // with a projection, then group on that column. date_trunc and
                // the other supported scalar key expressions all yield Int64.
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
                    Ok(keyed.group_by_count::<IntKeyExtractor<arrow_array::types::Int64Type>>(0))
                }
            },
            n => Err(Error::UnsupportedAggregateGroupAmount(n)),
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
            IntPairKeyExtractor, Sum,
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

        if key_cols.len() != 2 {
            return Err(Error::UnsupportedAggregateGroupAmount(key_cols.len()));
        }

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

        // Monomorphise over the two key types and the slot arity (N).
        macro_rules! by_arity {
            ($a:ty, $b:ty) => {{
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
                    _ => {
                        match slots.len() {
                            1 => Ok(input
                                .group_by_aggregate::<Key, AggregationRowValueExtractor<1>>(
                                    key_cols, slots, top_k,
                                )),
                            2 => Ok(input
                                .group_by_aggregate::<Key, AggregationRowValueExtractor<2>>(
                                    key_cols, slots, top_k,
                                )),
                            3 => Ok(input
                                .group_by_aggregate::<Key, AggregationRowValueExtractor<3>>(
                                    key_cols, slots, top_k,
                                )),
                            4 => Ok(input
                                .group_by_aggregate::<Key, AggregationRowValueExtractor<4>>(
                                    key_cols, slots, top_k,
                                )),
                            5 => Ok(input
                                .group_by_aggregate::<Key, AggregationRowValueExtractor<5>>(
                                    key_cols, slots, top_k,
                                )),
                            6 => Ok(input
                                .group_by_aggregate::<Key, AggregationRowValueExtractor<6>>(
                                    key_cols, slots, top_k,
                                )),
                            n => Err(Error::UnsupportedAggregateExpressionAmount(n)),
                        }
                    }
                }
            }};
        }

        match (&key_refs[0].return_type, &key_refs[1].return_type) {
            (Type::Int64, Type::Int32) => by_arity!(Int64Type, Int32Type),
            (Type::Int32, Type::Int32) => by_arity!(Int32Type, Int32Type),
            (Type::Int16, Type::Int32) => by_arity!(Int16Type, Int32Type),
            (Type::Int16, Type::Int16) => by_arity!(Int16Type, Int16Type),
            (Type::Int64, Type::Int64) => by_arity!(Int64Type, Int64Type),
            (Type::Int32, Type::Int64) => by_arity!(Int32Type, Int64Type),
            (a, _) => Err(Error::DataTypeNotSupportedForGroupBy(a.clone())),
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
            .compile(dispatcher, projection, dynamic_filters)
            .map_err(Error::TableScan)
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
