//! [`CreateTable`]: `CREATE TABLE` with an explicit column list.

use crate::catalog::{CatalogTransaction, CreateTableRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// CREATE TABLE with an explicit column list.
#[derive(Debug)]
pub struct CreateTable {
    pub request: CreateTableRequest,
    pub or_replace: bool,
    pub temporary: bool,
    pub has_query: bool,
    pub constraint_count: usize,
}

impl fmt::Display for CreateTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns = self
            .request
            .columns
            .iter()
            .map(|c| format!("{}:{}", c.name, c.col_type))
            .collect::<Vec<_>>()
            .join(", ");
        // Sort by key so the rendered output is deterministic; `HashMap`
        // iteration order is randomized per process and would otherwise flake
        // any snapshot/equality test that includes options.
        let mut options: Vec<(&String, &String)> = self.request.options.iter().collect();
        options.sort_by(|a, b| a.0.cmp(b.0));
        let options_str = options
            .iter()
            .map(|(k, v)| format!("{k:?}: {v:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "CreateTable({}, [{columns}], options: {{{options_str}}})",
            self.request.name
        )
    }
}

impl CreateTable {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
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

        // Resolve the create against the datastore the statement named (routing,
        // validation, locating the table's files), then compile the resolved
        // creation into the dataflow that fetches every data file's footer in
        // parallel over the pool and, at its terminal, stages the materialized
        // table for transaction commit. (Running that fetch dataflow from a
        // per-worker nullary would nest a dataflow inside a worker and deadlock the
        // pool, so it is built here on the coordinator.)
        let creation = transaction
            .bind_create_table(self.request.clone())
            .map_err(Error::CreateTable)?;
        creation.compile(dispatcher).map_err(Error::CreateTable)
    }
}
