//! Global aggregates (no GROUP BY).

use super::{Aggregate, aggregation_slots, sum_reads_wide_column};
use crate::catalog::QueryContext;
use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression};
use crate::operator::Input;
use crate::types::Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::sync::Arc;

impl Aggregate {
    /// A bare `COUNT(*)` keeps the dedicated row-counter; anything else
    /// (SUM/COUNT/MIN/MAX, possibly several) compiles to the multi-aggregate
    /// operator, one output column per expression. `AVG` never appears here:
    /// DuckDB lowers it to a `sum`+`count` pair with a downstream divide, so
    /// only count/sum/min/max slots reach this point.
    pub(super) fn compile_global(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if self.is_lone_count_star() {
            return Ok(input.count());
        }

        let slots = aggregation_slots(&self.expressions)?;
        // Pick the accumulator width by column type (see `sum_reads_wide_column`).
        Ok(if sum_reads_wide_column(&self.expressions) {
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
        scan: &Input,
        dispatcher: &DataFlowDispatcher,
        ctx: &dyn QueryContext,
    ) -> Result<Option<RecordBatchOperatorSpec>, Error> {
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
            // Only the integer/temporal columns the scan-based global path emits
            // as Int64 are sound here: their stats cast losslessly to Int64.
            // Float/decimal/boolean/Int128 would silently truncate or overflow
            // (and the scan path doesn't support them either), so decline and
            // let the ordinary path handle, or reject, them.
            match agg.column.return_type {
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
            let table_col = match scan.columns.get(agg.column.column_idx) {
                Some(Expression::Ref(r)) => r.column_idx,
                _ => return Ok(None),
            };
            let Some((min, max)) = scan.table.column_min_max(table_col, ctx) else {
                return Ok(None);
            };
            let scalar = if is_min { min } else { max };
            // Emit Int64, the same output type as the scan-based global MIN/MAX
            // (stats carry the column's physical storage type). A cast failure
            // is just another "can't answer from stats": fall back to the scan
            // rather than failing the whole query.
            let arr = scalar.into_inner();
            let Ok(casted) = arrow::compute::cast(&arr, &DataType::Int64) else {
                return Ok(None);
            };
            fields.push(Field::new(
                if is_min { "min" } else { "max" },
                DataType::Int64,
                true,
            ));
            columns.push(casted);
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .expect("stat scalars are single-row arrays");
        Ok(Some(
            dispatch::values_input(dispatcher, [batch]).record_batches(),
        ))
    }
}
