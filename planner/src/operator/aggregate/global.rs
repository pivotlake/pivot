//! Global aggregates (no GROUP BY).

use super::{Aggregate, aggregation_slots, needs_wide_accumulator};
use crate::catalog::CatalogTransaction;
use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression};
use crate::types::Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::sync::Arc;

impl Aggregate {
    /// Every global aggregate (`COUNT(*)`/SUM/COUNT/MIN/MAX, possibly several,
    /// including a bare `COUNT(*)`) compiles to the multi-aggregate operator,
    /// one output column per expression. `AVG` never appears here: DuckDB lowers
    /// it to a `sum`+`count` pair with a downstream divide, so only
    /// count/sum/min/max slots reach this point.
    pub(super) fn compile_global(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let slots = aggregation_slots(&self.expressions)?;
        // Pick the accumulator width by column type (see `needs_wide_accumulator`).
        Ok(if needs_wide_accumulator(&self.expressions) {
            input.aggregate::<i128>(slots)
        } else {
            input.aggregate::<i64>(slots)
        })
    }

    /// Answer an unfiltered global `MIN`/`MAX`-only aggregate straight from
    /// table metadata (e.g. parquet row-group statistics), skipping the scan
    /// entirely. Applies when the aggregate sits directly on a scan with no
    /// dynamic predicates, every expression is a `MIN`/`MAX` over a plain
    /// column, and the table can prove both bounds for each column
    /// ([`Table::column_min_max`](crate::catalog::Table::column_min_max)); any
    /// miss returns `None` and the ordinary scan-based path runs.
    pub(crate) fn try_compile_from_stats(
        &self,
        inputs: &[crate::PlanNode],
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<Option<RecordBatchOperatorSpec>, Error> {
        // Only fires on an aggregate sitting directly on a bare scan (a single
        // Input child with no children of its own), so the table's metadata
        // describes exactly the rows the aggregate would see.
        let [child] = inputs else {
            return Ok(None);
        };
        let crate::Operator::Input(scan) = &child.operator else {
            return Ok(None);
        };
        if !child.inputs.is_empty() {
            return Ok(None);
        }

        if !self.groups.is_empty() || self.expressions.is_empty() {
            return Ok(None);
        }
        if !scan.dynamic_filters.is_empty() {
            return Ok(None);
        }

        // A lone unfiltered COUNT(*) is the sum of every row group's row count.
        // Emit the same Int64 "count" column the scan-based aggregate path does.
        if self.is_lone_count_star() {
            let Some(count) = scan.table.row_count(transaction) else {
                return Ok(None);
            };
            let array = Int64Array::from(vec![count]);
            let schema = Arc::new(Schema::new(vec![Field::new(
                "count",
                DataType::Int64,
                false,
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(array)])
                .expect("single-row count batch");
            return Ok(Some(
                dispatch::values_input(dispatcher, [batch]).record_batches(),
            ));
        }

        let mut fields = Vec::with_capacity(self.expressions.len());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.expressions.len());
        for e in &self.expressions {
            let (agg, is_min) = match e {
                Expression::AggregateFunc(AggregateFunc::Min(a)) => (a, true),
                Expression::AggregateFunc(AggregateFunc::Max(a)) => (a, false),
                _ => return Ok(None),
            };
            // Stats answer only a `MIN`/`MAX` over a plain column. A computed
            // argument (`MIN(a * b)`) has no column stats to read, so decline and
            // let the materialised scan-based path handle it.
            let Expression::Ref(column) = agg.argument.as_ref() else {
                return Ok(None);
            };
            // Only the integer/temporal columns the scan-based global path emits
            // as Int64 are sound here: their stats cast losslessly to Int64.
            // Float/decimal/boolean/Int128 would silently truncate, overflow, or
            // rescale under that cast, so decline and let the ordinary scan-based
            // path handle, or reject, them.
            match column.return_type {
                Type::Int8
                | Type::Int16
                | Type::Int32
                | Type::Int64
                | Type::Date
                | Type::Timestamp => {}
                _ => return Ok(None),
            }
            // The aggregate's ref indexes the scan's output columns; map it back
            // to the table column the stats are kept under.
            let table_col = match scan.columns.get(column.column_idx) {
                Some(Expression::Ref(r)) => r.column_idx,
                _ => return Ok(None),
            };
            let Some((min, max)) = scan.table.column_min_max(table_col, transaction) else {
                return Ok(None);
            };
            let scalar = if is_min { min } else { max };
            // Match the scan-based global MIN/MAX output type: a temporal column
            // surfaces as its real Date32/Timestamp, every other column as Int64.
            // Stats carry the column's physical storage int, which casts
            // losslessly to the target. A cast failure is just another "can't
            // answer from stats": fall back to the scan rather than failing.
            let target =
                super::temporal_output_type(&agg.column().return_type).unwrap_or(DataType::Int64);
            let arr = scalar.into_inner();
            let Ok(casted) = arrow::compute::cast(&arr, &target) else {
                return Ok(None);
            };
            fields.push(Field::new(if is_min { "min" } else { "max" }, target, true));
            columns.push(casted);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("stat scalars are single-row arrays");
        Ok(Some(
            dispatch::values_input(dispatcher, [batch]).record_batches(),
        ))
    }
}
