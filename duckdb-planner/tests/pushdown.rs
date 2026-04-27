//! Tests that exercise the [`DuckDBTable::pushdown_filter`] hook.
//!
//! When the catalog's table opts into pushdown (`pushdown_filter` returns
//! `true`), DuckDB consumes the filter at the scan and the planner emits no
//! `Filter` operator above the `Input`. When the table refuses
//! (`pushdown_filter` returns `false`), the filter survives as a `Filter`
//! operator wrapping the `Input`.

use duckdb_planner::expression::TableFilter;
use duckdb_planner::{DuckDBBind, DuckDBColumn, DuckDBTable, LogicalTypeId, PlannerContext};
use insta::assert_snapshot;
use rstest::{fixture, rstest};
use std::collections::HashMap;
use std::sync::Arc;

/// A `users` table whose `pushdown_filter` answer is configurable per-test.
struct UsersTable {
    accept_pushdown: bool,
}

impl DuckDBTable for UsersTable {
    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![
            DuckDBColumn {
                name: "id".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            },
            DuckDBColumn {
                name: "name".to_string(),
                duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
            },
            DuckDBColumn {
                name: "score".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            },
        ]
    }

    fn pushdown_filter(&self, _filter: TableFilter) -> bool {
        self.accept_pushdown
    }
}

struct TestCatalog {
    tables: HashMap<String, Arc<dyn DuckDBTable>>,
}

impl DuckDBBind for TestCatalog {
    fn try_bind(&self, name: &str) -> Option<Arc<dyn DuckDBTable>> {
        self.tables.get(name).cloned()
    }
}

fn planner_with(accept_pushdown: bool) -> PlannerContext {
    let mut tables: HashMap<String, Arc<dyn DuckDBTable>> = HashMap::new();
    tables.insert(
        "users".to_string(),
        Arc::new(UsersTable { accept_pushdown }),
    );
    PlannerContext::new(Arc::new(TestCatalog { tables }))
}

#[fixture]
fn accepting_planner() -> PlannerContext {
    planner_with(true)
}

#[fixture]
fn rejecting_planner() -> PlannerContext {
    planner_with(false)
}

/// When the table accepts pushdown, the comparison against `score` is consumed
/// at the scan, so no `Filter` operator appears in the plan.
#[rstest]
fn pushdown_accepted_removes_filter(mut accepting_planner: PlannerContext) {
    let plan = accepting_planner
        .plan("SELECT * FROM users WHERE score <> 42")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#1:INTEGER, #2:VARCHAR, #0:INTEGER)
      Input([#2:INTEGER, #0:INTEGER, #1:VARCHAR])
    ");
}

/// When the table refuses pushdown, the same predicate stays as a `Filter`
/// operator above the `Input`.
#[rstest]
fn pushdown_rejected_keeps_filter(mut rejecting_planner: PlannerContext) {
    let plan = rejecting_planner
        .plan("SELECT * FROM users WHERE score <> 42")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#1:INTEGER, #2:VARCHAR, #0:INTEGER)
      Filter(#0:INTEGER <> 42:INTEGER -> BOOLEAN)
        Input([#2:INTEGER, #0:INTEGER, #1:VARCHAR])
    ");
}
