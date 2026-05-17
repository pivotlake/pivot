//! Per-operator compile impls.
//!
//! Each [`Operator`](crate::operator::Operator) variant has a `compile`
//! method here that translates it into a [`RecordBatchOperatorSpec`] call.

use crate::compile::create_table::CreateTableNullaryFactory;
use crate::compile::{Error, ExprEvalFn};
use crate::expression::Expression;
use crate::operator::{
    Aggregate, CreateTable, Filter, Input, OrderBy, OrderByDirection, Projection, TopN,
};
use crate::types::Type;
use arrow::compute::kernels::boolean::and;
use arrow_array::{BooleanArray, RecordBatch};
use dispatch::{DataFlowDispatcher, IntKeyExtractor, OrderBy as DispatchOrderBy, Projection as DispatchProjection, RecordBatchOperatorSpec, StringKeyExtractor};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use crate::catalog::Catalog;

impl Projection {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let idxs = Arc::new(
            self.projections
                .iter()
                .map(|e| match e {
                    Expression::Ref(n) => Ok(n.column_idx),
                    _ => Err(Error::UnsupportedProjectionExpression(e.clone())),
                })
                .collect::<Result<Vec<_>, _>>()?,
        );

        Ok(input.project(|| {
            let idxs = idxs.clone();
            move |batch: RecordBatch| batch.project(&idxs).unwrap()
        }))
    }
}

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if self.expressions.len() != 1 {
            return Err(Error::UnsupportedAggregateExpressionAmount(
                self.expressions.len(),
            ));
        }
        match &self.expressions[0] {
            Expression::AggregateFunc(crate::expression::AggregateFunc::CountStar(_)) => {}
            expr => return Err(Error::UnsupportedAggregateExpression(expr.clone())),
        }

        match self.groups.len() {
            0 => Ok(input.count()),
            1 => {
                let group = match &self.groups[0] {
                    Expression::Ref(r) => r,
                    expr => return Err(Error::UnexpectedAggExpression(expr.clone())),
                };
                let col = group.column_idx;
                match &group.return_type {
                    Type::Int8 => {
                        Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int8Type>>(col))
                    }
                    Type::Int16 => {
                        Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int16Type>>(col))
                    }
                    Type::Int32 => {
                        Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int32Type>>(col))
                    }
                    Type::Int64 => {
                        Ok(input
                            .group_by_count::<IntKeyExtractor<arrow_array::types::Int64Type>>(col))
                    }
                    Type::Utf8 => Ok(input.group_by_count::<StringKeyExtractor>(col)),
                    dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
                }
            }
            n => Err(Error::UnsupportedAggregateGroupAmount(n)),
        }
    }
}

impl Input {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
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
        Ok(self.table.compile(dispatcher, projection))
    }
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
        Ok(input.order_by_limit(orders, self.limit))
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

        let already_created = Arc::new(AtomicBool::new(false));

        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| {
                CreateTableNullaryFactory::new(
                    catalog.clone(),
                    self.request.clone(),
                    already_created.clone(),
                )
            }),
        ))
    }
}
