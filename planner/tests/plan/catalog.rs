use std::sync::{Arc, Mutex};

use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use insta::assert_snapshot;
use planner::catalog::{BoundTable, CatalogTransaction, Column, TableReference, TableRevision};
use planner::expression::{Expression, TableFilter};
use planner::types::Type;
use planner::{DEFAULT_DATASTORE_NAME, DEFAULT_SCHEMA_NAME, Planner};

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

impl BoundTable for RecordingTable {
    fn table_reference(&self) -> TableReference {
        TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: DEFAULT_SCHEMA_NAME.to_string(),
            table: "t".to_string(),
        }
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: format!("{DEFAULT_DATASTORE_NAME}:t"),
            version: "0".to_string(),
        }
    }

    fn compile_scan(
        &self,
        _dispatcher: &DataFlowDispatcher,
        _projection: Projection,
        _dynamic_filters: Vec<planner::catalog::DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
    ) -> planner::catalog::Result<RecordBatchOperatorSpec> {
        unreachable!("plan-only test should not reach compile")
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> planner::catalog::Result<bool> {
        self.received.lock().unwrap().push(filter);
        Ok(self.accept_pushdown)
    }
}

/// Minimal `Catalog` that resolves a single, known table name. Used instead
/// of the shared `TestCatalog` because we need a hand-rolled `BoundTable` impl.
/// Its transaction snapshot is the catalog itself: the table never changes.
#[derive(Debug)]
struct SingleTableCatalog {
    name: String,
    table: RecordingTable,
}

impl CatalogTransaction for SingleTableCatalog {
    // The one table lives in the default schema, so that is the only schema
    // this catalog defines.
    fn does_schema_exist(&self, _datastore: &str, schema: &str) -> planner::catalog::Result<bool> {
        Ok(schema == DEFAULT_SCHEMA_NAME)
    }

    fn bind_table(
        &self,
        reference: &TableReference,
    ) -> planner::catalog::Result<Option<Box<dyn BoundTable>>> {
        Ok((reference.table == self.name)
            .then(|| Box::new(self.table.clone()) as Box<dyn BoundTable>))
    }

    fn table_revision(
        &self,
        reference: &TableReference,
    ) -> planner::catalog::Result<Option<TableRevision>> {
        Ok((reference.table == self.name).then(|| TableRevision {
            identity: format!("{}:{}", reference.datastore, reference.table),
            version: "0".to_string(),
        }))
    }
}

impl SingleTableCatalog {
    fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        Arc::new(SingleTableCatalog {
            name: self.name.clone(),
            table: self.table.clone(),
        })
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

fn timestamp_col(col_type: Type) -> Vec<Column> {
    vec![Column {
        name: "ts".to_string(),
        col_type,
    }]
}

fn build_planner(table: RecordingTable) -> (Planner, Arc<SingleTableCatalog>) {
    let catalog = Arc::new(SingleTableCatalog {
        name: "t".to_string(),
        table,
    });
    (
        Planner::from_datastore_names(
            vec![DEFAULT_DATASTORE_NAME.to_string()],
            DEFAULT_DATASTORE_NAME.to_string(),
        )
        .expect("planner context"),
        catalog,
    )
}

fn plan_sql(
    planner: &mut Planner,
    catalog: &Arc<SingleTableCatalog>,
    sql: &str,
) -> Result<planner::Plan, planner::Error> {
    planner.plan(sql, catalog.begin_transaction())
}

/// `Catalog::table` is consulted by name; the looked-up `BoundTable::columns` is
/// what the planner shows in `Input([...])`.
#[test]
fn catalog_resolves_named_table() {
    let table = RecordingTable::new(two_int_cols(), false);
    let (mut planner, catalog) = build_planner(table.clone());

    let plan = plan_sql(&mut planner, &catalog, "SELECT a, b FROM t").unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, b:Int32)
      Input([a:Int32, b:Int32])
    ");
}

/// Unknown table names surface as a planning error rather than panicking.
#[test]
fn catalog_returns_error_for_unknown_table() {
    let table = RecordingTable::new(two_int_cols(), false);
    let (mut planner, catalog) = build_planner(table.clone());

    let err = plan_sql(&mut planner, &catalog, "SELECT a FROM nonexistent").unwrap_err();
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
    let (mut planner, catalog) = build_planner(table.clone());

    let plan = plan_sql(&mut planner, &catalog, "SELECT a FROM t WHERE a <> 0").unwrap();

    let received = table.received.lock().unwrap();
    assert_snapshot!(pushdown_snapshot(&received), @"a:Int32 <> 0:Int32 -> Boolean");
    drop(received);

    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Filter(a:Int32 <> 0:Int32 -> Boolean)
        Input([a:Int32])
    ");
}

/// When the table accepts pushdown, the synthesized `Filter` is dropped from
/// the plan — only the `Input` remains, since the predicate is now the
/// table's responsibility.
#[test]
fn pushdown_accepted_drops_filter_operator() {
    let table = RecordingTable::new(two_int_cols(), true);
    let (mut planner, catalog) = build_planner(table.clone());

    let plan = plan_sql(&mut planner, &catalog, "SELECT a FROM t WHERE a <> 0").unwrap();

    let received = table.received.lock().unwrap();
    assert_snapshot!(pushdown_snapshot(&received), @"a:Int32 <> 0:Int32 -> Boolean");
    drop(received);

    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Input([a:Int32])
    ");
}

/// DuckDB's query-stable `now()` is folded to one TIMESTAMPTZ constant, so
/// the resulting bare-column comparison reaches the table's pushdown hook.
/// The same DuckDB property that permits that fold also prevents plan reuse.
#[test]
fn now_filter_is_pushed_down_and_plan_is_not_cacheable() {
    let table = RecordingTable::new(timestamp_col(Type::TimestampTz), false);
    let (mut planner, catalog) = build_planner(table.clone());

    let plan = plan_sql(
        &mut planner,
        &catalog,
        "SELECT ts FROM t WHERE ts > now() - interval '4 minutes'",
    )
    .unwrap();

    let received = table.received.lock().unwrap();
    assert_eq!(received.len(), 1);
    let TableFilter::Expression(filter) = &received[0] else {
        panic!("expected an expression filter, got {}", received[0]);
    };
    let Expression::Compare(compare) = filter.as_ref() else {
        panic!("expected a comparison, got {filter}");
    };
    assert!(
        matches!(compare.left.as_ref(), Expression::Ref(_)),
        "{filter}"
    );
    assert!(
        matches!(compare.right.as_ref(), Expression::Constant(_)),
        "{filter}"
    );
    drop(received);

    assert!(!plan.is_cacheable());
    let rendered = plan.to_string();
    assert!(rendered.contains("Filter(ts:TimestampTz >"), "{rendered}");
    assert!(rendered.contains("Input([ts:TimestampTz])"), "{rendered}");
}

/// DuckDB pushes the casted comparison for a zone-less column too. The Delta
/// binding can peel this exact cast because the planner session is fixed to
/// UTC, where TIMESTAMP-to-TIMESTAMPTZ preserves the epoch-microsecond value.
#[test]
fn now_filter_on_timestamp_without_time_zone_is_pushed_down() {
    let table = RecordingTable::new(timestamp_col(Type::Timestamp), false);
    let (mut planner, catalog) = build_planner(table.clone());

    let plan = plan_sql(
        &mut planner,
        &catalog,
        "SELECT ts FROM t WHERE ts > now() - interval '4 minutes'",
    )
    .unwrap();

    let received = table.received.lock().unwrap();
    assert_eq!(received.len(), 1);
    let TableFilter::Expression(filter) = &received[0] else {
        panic!("expected an expression filter, got {}", received[0]);
    };
    let Expression::Compare(compare) = filter.as_ref() else {
        panic!("expected a comparison, got {filter}");
    };
    let Expression::Cast(cast) = compare.left.as_ref() else {
        panic!("expected DuckDB's timestamp cast, got {filter}");
    };
    assert_eq!(cast.target, Type::TimestampTz);
    assert!(matches!(cast.source(), Expression::Ref(_)), "{filter}");
    assert!(
        matches!(compare.right.as_ref(), Expression::Constant(_)),
        "{filter}"
    );
    drop(received);

    assert!(!plan.is_cacheable());
    let rendered = plan.to_string();
    assert!(
        rendered.contains("Filter(cast(ts:Timestamp as TimestampTz) >"),
        "{rendered}"
    );
    assert!(rendered.contains("Input([ts:Timestamp])"), "{rendered}");
}

/// Queries without a `WHERE` clause never invoke `pushdown_filter`.
#[test]
fn no_where_clause_skips_pushdown() {
    let table = RecordingTable::new(two_int_cols(), true);
    let (mut planner, catalog) = build_planner(table.clone());

    let _plan = plan_sql(&mut planner, &catalog, "SELECT a FROM t").unwrap();

    assert!(table.received.lock().unwrap().is_empty());
}
