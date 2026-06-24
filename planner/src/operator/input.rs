//! [`Input`] — scans a [`Table`] from the catalog.

use super::slot_for;
use crate::catalog::{DuckDBTableAdapter, DynamicScanPredicate, QueryContext, Table};
use crate::compile::{DynamicFilterSlots, Error};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::Expression;
use dispatch::{DataFlowDispatcher, Projection as DispatchProjection, RecordBatchOperatorSpec};
use duckdb_planner::operator as duckdb_operator;
use std::any::Any;
use std::fmt;

/// Scans a [`Table`] from the catalog. `columns` lists the requested output
/// columns (each as a [`Ref`](crate::expression::Ref) into the table schema)
/// and `filters` are predicates pushed down into the scan.
#[derive(Debug)]
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

impl TryFrom<duckdb_operator::Input> for Input {
    type Error = super::Error;
    fn try_from(s: duckdb_operator::Input) -> Result<Self, Self::Error> {
        let any: Box<dyn Any> = s.table;
        let wrapper: Box<DuckDBTableAdapter> = any
            .downcast::<DuckDBTableAdapter>()
            .expect("Input.table should be a DuckDBTableAdapter");
        Ok(Input {
            table: wrapper.table,
            columns: s
                .columns
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            dynamic_filters: s
                .dynamic_filters
                .into_iter()
                .map(DynamicFilter::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            emit_row_group_metadata: s.emit_row_group_metadata,
        })
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = self
            .columns
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Input([{cols}])")
    }
}

impl Input {
    /// The scan's real data column indices (DuckDB's `usize::MAX` "no column"
    /// sentinel, used by `COUNT(*)` scans, dropped).
    fn data_column_indices(&self) -> Result<Vec<usize>, Error> {
        self.columns
            .iter()
            .filter_map(|e| match e {
                Expression::Ref(r) if r.column_idx == usize::MAX => None,
                Expression::Ref(r) => Some(Ok(r.column_idx)),
                _ => Some(Err(Error::UnexpectedInputExpression(e.clone()))),
            })
            .collect()
    }

    /// Whether this scan reads at least one real data column. The condition-cache
    /// populate path needs one: it strips the two metadata columns back off, which
    /// a zero-data-column (e.g. `COUNT(*)`) scan can't survive.
    pub(crate) fn has_data_columns(&self) -> bool {
        self.data_column_indices().is_ok_and(|c| !c.is_empty())
    }

    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        ctx: &dyn QueryContext,
        slots: &mut DynamicFilterSlots,
        // Emit row-group metadata even if the plan didn't ask for it. Set when a
        // `Filter` above this scan populates the condition cache and needs the
        // `(global_row_group, row_idx)` columns to attribute survivors.
        force_row_group_metadata: bool,
        // When `Some(id)`, replay the cached survivors for filter `id` (a previous
        // run of the same query recorded them) so only those rows are decoded.
        replay_condition_filter: Option<u64>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let projection = DispatchProjection::columns(self.data_column_indices()?);
        let dynamic_filters = build_dynamic_scan_predicates(&self.dynamic_filters, slots);
        self.table
            .compile(
                dispatcher,
                projection,
                dynamic_filters,
                self.emit_row_group_metadata || force_row_group_metadata,
                replay_condition_filter,
                ctx,
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
