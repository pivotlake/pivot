//! [`Insert`] - `INSERT INTO <table>` fed by a source plan (a `VALUES` list or
//! any `SELECT`).

use crate::catalog::{Catalog, QueryContext};
use crate::compile::Error;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

/// `INSERT INTO <table>`, fed by its single child (the source plan).
///
/// Compiles like any other operator: the target table (resolved from the
/// catalog) wraps the compiled source in the dataflow that writes it - see
/// [`Table::insert`](crate::catalog::Table::insert). That dataflow durably
/// commits every row before finishing and emits a single one-row batch
/// carrying the written-row count, which the server reports as `INSERT 0 n`
/// instead of streaming rows to the client.
#[derive(Debug)]
pub struct Insert {
    /// The table being inserted into.
    pub table: String,
    /// For each table column (schema order), the source-output column that
    /// provides it. Empty when the statement listed no columns, meaning the
    /// source columns are already in table order.
    pub column_map: Vec<usize>,
}

impl fmt::Display for Insert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.column_map.is_empty() {
            write!(f, "Insert({})", self.table)
        } else {
            write!(f, "Insert({}, columns: {:?})", self.table, self.column_map)
        }
    }
}

impl Insert {
    /// Compile the insert: reorder `source`'s columns into table-schema order
    /// and hand it to the target table's writer, which returns the dataflow
    /// that writes, commits, and emits the row count.
    pub(crate) fn compile(
        &self,
        source: RecordBatchOperatorSpec,
        catalog: &Arc<dyn Catalog>,
        ctx: &dyn QueryContext,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let source = if self.column_map.is_empty() {
            source
        } else {
            let map = Arc::new(self.column_map.clone());
            source.project(move || {
                let map = map.clone();
                // The map's indices were bound by the planner against the
                // source's output, so they are always in range.
                move |batch| batch.project(&map).unwrap()
            })
        };
        let table = catalog
            .table(&self.table)
            .ok_or_else(|| Error::InsertTableMissing(self.table.clone()))?;
        table.insert(source, ctx).map_err(Error::Insert)
    }
}
