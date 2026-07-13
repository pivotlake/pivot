//! INSERT INTO a table from one source plan.

use crate::catalog::CatalogTransaction;
use crate::compile::Error;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct Insert {
    pub table: String,
    pub column_map: Vec<usize>,
}

impl fmt::Display for Insert {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.column_map.is_empty() {
            write!(formatter, "Insert({})", self.table)
        } else {
            write!(
                formatter,
                "Insert({}, columns: {:?})",
                self.table, self.column_map
            )
        }
    }
}

impl Insert {
    /// Reorder an explicit INSERT column list into table order, then delegate
    /// the physical write plan to the transaction-bound catalog table.
    pub(crate) fn compile(
        &self,
        source: RecordBatchOperatorSpec,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let source = if self.column_map.is_empty() {
            source
        } else {
            let column_map = Arc::new(self.column_map.clone());
            source.project(move || {
                let column_map = column_map.clone();
                move |batch| batch.project(&column_map).unwrap()
            })
        };
        let table = transaction
            .table(&self.table)
            .ok_or_else(|| Error::InsertTableMissing(self.table.clone()))?;
        table.insert(source, transaction).map_err(Error::Insert)
    }
}
