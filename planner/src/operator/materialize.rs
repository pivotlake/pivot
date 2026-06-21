//! [`Materialize`] — late-materialization fetch of extra columns.

use crate::catalog::{DuckDBTableAdapter, QueryContext, Table};
use crate::compile::Error;
use dispatch::{Projection as DispatchProjection, RecordBatchOperatorSpec};
use duckdb_planner::operator as duckdb_operator;
use std::any::Any;
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
    pub table: Box<dyn Table>,
    /// Table-schema (storage) column indices to fetch, in output order.
    pub columns: Vec<usize>,
}

impl TryFrom<duckdb_operator::Materialize> for Materialize {
    type Error = super::Error;
    fn try_from(m: duckdb_operator::Materialize) -> Result<Self, Self::Error> {
        // Mirror `Input`: the resolved DuckDB table is a `DuckDBTableAdapter`
        // wrapping the pivot `Table`; downcast and take it.
        let any: Box<dyn Any> = m.table;
        let wrapper: Box<DuckDBTableAdapter> = any
            .downcast::<DuckDBTableAdapter>()
            .expect("Materialize.table should be a DuckDBTableAdapter");
        Ok(Materialize {
            table: wrapper.table,
            columns: m.columns,
        })
    }
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
        ctx: &dyn QueryContext,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let projection = DispatchProjection::columns(self.columns.iter().copied());
        self.table
            .materialize(input, projection, ctx)
            .map_err(Error::TableScan)
    }
}
