//! [`Materialize`] — late-materialization fetch of extra columns.

use super::input::{describe_scan_columns, plan_scan_projection};
use crate::catalog::BoundTable;
use crate::compile::Error;
use crate::expression::Expression;
use dispatch::RecordBatchOperatorSpec;
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
    /// Output columns to fetch, in output order: each a [`Ref`](crate::expression::Ref)
    /// into the table schema or a pushed variant field extract (a
    /// [`VariantGet`](crate::expression::VariantGet) over one), exactly the
    /// shape an [`Input`](super::Input)'s columns take.
    pub columns: Vec<Expression>,
}

impl fmt::Display for Materialize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = describe_scan_columns(self.table.as_ref(), &self.columns);
        write!(f, "Materialize([{cols}])")
    }
}

impl Materialize {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let (projection, extract_projection) =
            plan_scan_projection(&self.columns, self.table.as_ref())?;
        let fetched = self
            .table
            .materialize(input, projection)
            .map_err(Error::TableScan)?;
        match extract_projection {
            Some(extract_projection) => extract_projection.compile(fetched),
            None => Ok(fetched),
        }
    }
}
