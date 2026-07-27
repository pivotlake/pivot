//! [`Materialize`] — late-materialization fetch of extra columns.

use crate::catalog::BoundTable;
use crate::compile::Error;
use dispatch::{Projection as DispatchProjection, RecordBatchOperatorSpec};
use std::fmt;

/// Late-materialization fetch: re-reads `columns` from the table for the rows
/// that survived the narrow pipeline below it.
///
/// The bridge synthesizes this by collapsing DuckDB's `late_materialization`
/// row-id SEMI join (see the duckdb-planner `Materialize` operator). Its single
/// input is the narrow pipeline whose scan is tagged with row-group metadata
/// (see [`Input::emit_row_group_metadata`](crate::operator::Input::emit_row_group_metadata));
/// this node fetches the requested columns for only the surviving rows, emitting
/// them in `columns` order — which matches the order DuckDB's full-column Get
/// produced, so the projection kept above it lines up positionally without
/// remapping.
#[derive(Debug)]
pub struct Materialize {
    pub table: Box<dyn BoundTable>,
    /// BoundTable-schema (storage) column indices to fetch, in output order.
    pub columns: Vec<usize>,
}

impl fmt::Display for Materialize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = self
            .columns
            .iter()
            .map(|c| format!("#{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Materialize([{cols}])")
    }
}

impl Materialize {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let projection = DispatchProjection::columns(self.columns.iter().copied());
        self.table
            .materialize(input, projection)
            .map_err(Error::TableScan)
    }
}
