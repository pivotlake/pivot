use std::sync::{Arc, Mutex};

use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use insta::assert_snapshot;
use planner::Planner;
use planner::catalog::{Catalog, Column, CreateTableRequest, Table};
use planner::expression::TableFilter;
use planner::types::Type;

#[allow(unused_imports)]
use crate::common::*;

/// Stand-in for a real table that records every `pushdown_filter` call so the
/// test can assert what DuckDB tried to push, and answers `accept_pushdown`
/// to drive the "rejected" vs "accepted" plan shapes.
#[derive(Clone, Debug)]
struct RecordingTable {
    columns: Vec<Column>,
    accept_pushdown: bool,
    /// Shared between the original (held by the test) and the per-binding
    /// clone (held by the catalog), so a test can inspect filters that were
    /// pushed into the clone.
    received: Arc<Mutex<Vec<TableFilter>>>,
}

impl RecordingTable {
    fn new(columns: Vec<Column>, accept_pushdown: bool) -> Self {
        Self {
            columns,
            accept_pushdown,
            received: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Table for RecordingTable {
    fn compile(
        &self,
        _dispatcher: &DataFlowDispatcher,
        _projection: Projection,
        _dynamic_filters: Vec<planner::catalog::DynamicScanPredicate>,
    ) -> planner::catalog::Result<RecordBatchOperatorSpec> {
        unreachable!("plan-only test should not reach compile")
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> planner::catalog::Result<bool> {
        self.received.lock().unwrap().push(filter);
        Ok(self.accept_pushdown)
    }
}

/// Minimal `Catalog` that resolves a single, known table name. Used instead
/// of the shared `TestCatalog` because we need a hand-rolled `Table` impl.
#[derive(Debug)]
struct SingleTableCatalog {
    name: String,
    table: RecordingTable,
}

impl Catalog for SingleTableCatalog {
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        (name == self.name).then(|| Box::new(self.table.clone()) as Box<dyn Table>)
    }

    fn create_table(
        &self,
        _request: CreateTableRequest,
        _dispatcher: &dispatch::DataFlowDispatcher,
    ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
        unreachable!("test catalog does not support CREATE TABLE")
    }
}

fn two_int_cols() -> Vec<Column> {
    vec![
        Column {
            name: "a".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "b".to_string(),
            col_type: Type::Int32,
        },
    ]
}

fn build_planner(table: RecordingTable) -> Planner {
    let catalog = Arc::new(SingleTableCatalog {
        name: "t".to_string(),
        table,
    });
    Planner::new(catalog)
}

/// `Catalog::table` is consulted by name; the looked-up `Table::columns` is
/// what the planner shows in `Input([...])`.
#[test]
fn catalog_resolves_named_table() {
    let table = RecordingTable::new(two_int_cols(), false);
    let mut planner = build_planner(table.clone());

    let plan = planner.plan("SELECT a, b FROM t").unwrap();

    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32, #1:Int32)
      Input([#0:Int32, #1:Int32])
    ");
}

/// Unknown table names surface as a planning error rather than panicking.
#[test]
fn catalog_returns_error_for_unknown_table() {
    let table = RecordingTable::new(two_int_cols(), false);
    let mut planner = build_planner(table.clone());

    let err = planner.plan("SELECT a FROM nonexistent").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("nonexistent") || msg.to_lowercase().contains("table"),
        "expected error to mention the missing table, got: {msg}"
    );
}

fn pushdown_snapshot(received: &[TableFilter]) -> String {
    received
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// When the table refuses pushdown, the planner keeps the explicit `Filter`
/// operator above the scan — and `pushdown_filter` is still consulted with
/// the predicate DuckDB tried to push.
#[test]
fn pushdown_rejected_keeps_filter_operator() {
    let table = RecordingTable::new(two_int_cols(), false);
    let mut planner = build_planner(table.clone());

    let plan = planner.plan("SELECT a FROM t WHERE a <> 0").unwrap();

    let received = table.received.lock().unwrap();
    assert_snapshot!(pushdown_snapshot(&received), @"#0:Int32 <> 0:Int32 -> Boolean");
    drop(received);

    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Filter(#0:Int32 <> 0:Int32 -> Boolean)
        Input([#0:Int32])
    ");
}

/// When the table accepts pushdown, the synthesized `Filter` is dropped from
/// the plan — only the `Input` remains, since the predicate is now the
/// table's responsibility.
#[test]
fn pushdown_accepted_drops_filter_operator() {
    let table = RecordingTable::new(two_int_cols(), true);
    let mut planner = build_planner(table.clone());

    let plan = planner.plan("SELECT a FROM t WHERE a <> 0").unwrap();

    let received = table.received.lock().unwrap();
    assert_snapshot!(pushdown_snapshot(&received), @"#0:Int32 <> 0:Int32 -> Boolean");
    drop(received);

    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Input([#0:Int32])
    ");
}

/// Queries without a `WHERE` clause never invoke `pushdown_filter`.
#[test]
fn no_where_clause_skips_pushdown() {
    let table = RecordingTable::new(two_int_cols(), true);
    let mut planner = build_planner(table.clone());

    let _plan = planner.plan("SELECT a FROM t").unwrap();

    assert!(table.received.lock().unwrap().is_empty());
}
