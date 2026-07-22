//! [`Input`] — scans a [`Table`] from the catalog.

use super::slot_for;
use crate::catalog::{CatalogTransaction, DynamicScanPredicate, Table};
use crate::compile::{DynamicFilterSlots, Error};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::Expression;
use dispatch::{DataFlowDispatcher, Projection as DispatchProjection, RecordBatchOperatorSpec};
use std::fmt;

/// Scans a [`Table`] from the catalog. `columns` lists the requested output
/// columns (each as a [`Ref`](crate::expression::Ref) into the table schema)
/// and `filters` are predicates pushed down into the scan.
#[derive(Debug, Clone)]
pub struct Input {
    pub table: Box<dyn Table>,
    pub columns: Vec<Expression>,
    /// Runtime-populated predicates the scan reads from shared slots — installed
    /// by a Top-N (or, in future, a hash join) elsewhere in the plan. Each
    /// prunes row groups against the producer's live boundary value.
    pub dynamic_filters: Vec<DynamicFilter>,
    /// When `true`, the scan tags each emitted row with its global row-group ID
    /// and per-row index (extra metadata columns appended after the data
    /// columns). Set on the narrow scan of a late-materialized query so a
    /// downstream [`Materialize`](crate::operator::Materialize) can fetch the
    /// remaining columns for only the surviving rows. Always `false` for
    /// ordinary scans.
    pub emit_row_group_metadata: bool,
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Resolve each scanned column's positional index back to its real name
        // from the table schema, so a plan dump reads like the source query
        // (e.g. `Input([URL:Utf8])`) rather than `Input([#2:Utf8])`. Falls back
        // to the bare reference for the COUNT(*) sentinel index or any non-`Ref`.
        let schema = self.table.columns();
        let cols = self
            .columns
            .iter()
            .map(|c| match c {
                Expression::Ref(r) => match schema.get(r.column_idx) {
                    Some(column) => format!("{}:{}", column.name, r.return_type),
                    None => r.to_string(),
                },
                _ => c.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Input([{cols}])")
    }
}

impl Input {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
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
            .compile_scan(
                dispatcher,
                projection,
                dynamic_filters,
                self.emit_row_group_metadata,
                transaction,
            )
            .map_err(Error::TableScan)
    }
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

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::DataType;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn date_column_scans_as_a_real_date(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "d",
                Type::Date,
                Arc::new(Int32Array::from(vec![0, 7])) as ArrayRef,
            )],
        );

        let batches = run_batches(&mut testing_planner, "SELECT d FROM events");

        // The DATE column is stored as Int32 days but must surface as a real
        // Date32 (reinterpreted zero-copy at the scan), so it renders as a date.
        assert_eq!(batches[0].schema().field(0).data_type(), &DataType::Date32);
    }
}
