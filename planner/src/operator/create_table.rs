//! [`CreateTable`] — `CREATE TABLE` with an explicit column list.

use crate::catalog::{Catalog, Column, CreateTableRequest};
use crate::compile::Error;
use crate::types::type_from_logical;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::Arc;

/// CREATE TABLE with an explicit column list.
#[derive(Debug)]
pub struct CreateTable {
    pub request: CreateTableRequest,
    pub or_replace: bool,
    pub temporary: bool,
    pub has_query: bool,
    pub constraint_count: usize,
}

impl TryFrom<duckdb_operator::CreateTable> for CreateTable {
    type Error = super::Error;

    fn try_from(create_table: duckdb_operator::CreateTable) -> Result<Self, Self::Error> {
        Ok(Self {
            request: CreateTableRequest {
                name: create_table.name,
                columns: create_table
                    .columns
                    .into_iter()
                    .map(|column| {
                        Ok(Column {
                            name: column.name,
                            col_type: type_from_logical(column.col_type)?,
                        })
                    })
                    .collect::<Result<Vec<_>, super::Error>>()?,
                options: create_table.options,
                if_not_exists: create_table.if_not_exists,
            },
            or_replace: create_table.or_replace,
            temporary: create_table.temporary,
            has_query: create_table.has_query,
            constraint_count: create_table.constraint_count,
        })
    }
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
        // Sort by key so the rendered output is deterministic — `HashMap`
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
        catalog: &Arc<dyn Catalog>,
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

        // The catalog does the up-front work — fetching every data file's footer
        // in parallel over the worker pool, here on the coordinator — and returns
        // the plan that writes the materialized table into the catalog. (Running
        // that fetch dataflow from a per-worker nullary would nest a dataflow
        // inside a worker and deadlock the pool.)
        catalog
            .create_table(self.request.clone(), dispatcher)
            .map_err(Error::CreateTable)
    }
}
