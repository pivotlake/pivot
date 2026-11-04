//! A database attached with a default schema resolves the names that name no
//! schema in that schema, both as the current database and when a reference
//! qualifies a table with the database alone.

use std::sync::Arc;

use duckdb_planner::{
    DatabaseAttachment, DuckDBColumn, DuckDBTable, DuckDBTransaction, Error, LogicalTypeId,
    PlannerContext,
};

struct IdTable;

impl DuckDBTable for IdTable {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(IdTable)
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![DuckDBColumn {
            name: "id".to_string(),
            duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            decimal_width: 0,
            decimal_scale: 0,
        }]
    }
}

/// Defines the schemas `main` and `public`, with the table `t` in `public` alone.
struct PublicTransaction;

impl DuckDBTransaction for PublicTransaction {
    fn does_schema_exist(
        &self,
        _datastore: &str,
        schema: &str,
    ) -> duckdb_planner::catalog_provider::Result<bool> {
        Ok(schema == "main" || schema == "public")
    }

    fn bind_table(
        &self,
        _datastore: &str,
        schema: &str,
        table_name: &str,
    ) -> duckdb_planner::catalog_provider::Result<Option<Box<dyn DuckDBTable>>> {
        let table: Box<dyn DuckDBTable> = Box::new(IdTable);
        Ok((schema == "public" && table_name == "t").then_some(table))
    }
}

fn context_with_default_schema(default_schema: &str) -> PlannerContext {
    let database = DatabaseAttachment {
        name: "db".to_string(),
        default_schema: default_schema.to_string(),
    };
    PlannerContext::new(Vec::new(), vec![database], "db".to_string()).unwrap()
}

#[test]
fn an_unqualified_name_resolves_in_the_default_schema() {
    let mut planner = context_with_default_schema("public");

    let plan = planner.plan("SELECT id FROM t", Arc::new(PublicTransaction));

    plan.unwrap();
}

#[test]
fn a_database_qualified_name_resolves_in_its_default_schema() {
    let mut planner = context_with_default_schema("public");

    let plan = planner.plan("SELECT id FROM db.t", Arc::new(PublicTransaction));

    plan.unwrap();
}

#[test]
fn an_unqualified_name_is_not_looked_up_outside_the_default_schema() {
    let mut planner = context_with_default_schema("main");

    let result = planner.plan("SELECT id FROM t", Arc::new(PublicTransaction));

    assert!(matches!(result, Err(Error::DuckDBPlanning(_))));
}
