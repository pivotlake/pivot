use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb_planner::LogicalOp;
use duckdb_planner::duckdb_bridge::duckdb_types::LogicalOperatorType;
use duckdb_planner::{
    DuckDBBind, DuckDBColumn, DuckDBTable, DuckDBTransaction, LogicalTypeId, PlannerContext,
};

struct CountingTable {
    rows: u64,
    exact: bool,
    asked: Arc<AtomicUsize>,
}

impl DuckDBTable for CountingTable {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(CountingTable {
            rows: self.rows,
            exact: self.exact,
            asked: self.asked.clone(),
        })
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![DuckDBColumn::plain("id", LogicalTypeId::INTEGER)]
    }

    fn estimate_row_count(&self) -> Option<u64> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        Some(self.rows)
    }

    fn exact_row_count(&self) -> Option<u64> {
        if !self.exact {
            return None;
        }
        self.asked.fetch_add(1, Ordering::Relaxed);
        Some(self.rows)
    }
}

struct StatsCatalog;

impl DuckDBBind for StatsCatalog {}

struct StatsTransaction {
    asked: Arc<AtomicUsize>,
}

impl DuckDBTransaction for StatsTransaction {
    fn does_schema_exist(&self, _datastore: &str, schema: &str) -> bool {
        schema == "main"
    }

    fn bind_table(
        &self,
        _datastore: &str,
        _schema: &str,
        table_name: &str,
    ) -> Option<Box<dyn DuckDBTable>> {
        let (rows, exact) = match table_name {
            "empty" => (0, true),
            "estimated_empty" => (0, false),
            "small" => (10, false),
            "big" => (1_000_000, false),
            _ => return None,
        };
        Some(Box::new(CountingTable {
            rows,
            exact,
            asked: self.asked.clone(),
        }))
    }
}

fn contains_operator(op: LogicalOp<'_>, expected: LogicalOperatorType) -> bool {
    op.op_type().unwrap() == expected
        || op
            .children()
            .unwrap()
            .into_iter()
            .any(|child| contains_operator(child, expected.clone()))
}

fn planner() -> PlannerContext {
    PlannerContext::new(
        Arc::new(StatsCatalog),
        vec!["db".to_string()],
        "db".to_string(),
    )
    .unwrap()
}

#[test]
fn join_planning_consults_table_row_counts() {
    let asked = Arc::new(AtomicUsize::new(0));
    let mut planner = planner();

    planner
        .plan(
            "SELECT small.id FROM small JOIN big ON small.id = big.id",
            Arc::new(StatsTransaction {
                asked: asked.clone(),
            }),
        )
        .unwrap();

    assert!(asked.load(Ordering::Relaxed) > 0);
}

#[test]
fn exact_zero_row_count_produces_an_empty_result() {
    let asked = Arc::new(AtomicUsize::new(0));
    let mut planner = planner();

    let plan = planner
        .plan(
            "SELECT id FROM empty UNION ALL SELECT id FROM empty",
            Arc::new(StatsTransaction { asked }),
        )
        .unwrap();

    assert_eq!(
        plan.root().unwrap().op_type().unwrap(),
        LogicalOperatorType::LOGICAL_EMPTY_RESULT
    );
}

#[test]
fn estimated_zero_row_count_keeps_the_scan() {
    let asked = Arc::new(AtomicUsize::new(0));
    let mut planner = planner();

    let plan = planner
        .plan(
            "SELECT id FROM estimated_empty",
            Arc::new(StatsTransaction { asked }),
        )
        .unwrap();

    assert!(contains_operator(
        plan.root().unwrap(),
        LogicalOperatorType::LOGICAL_GET
    ));
}

#[test]
fn empty_correlated_subquery_drops_its_delim_dependency() {
    let asked = Arc::new(AtomicUsize::new(0));
    let mut planner = planner();

    let plan = planner
        .plan(
            "SELECT big.id, (SELECT empty.id FROM empty WHERE empty.id = big.id) FROM big",
            Arc::new(StatsTransaction { asked }),
        )
        .unwrap();

    assert!(!contains_operator(
        plan.root().unwrap(),
        LogicalOperatorType::LOGICAL_DELIM_JOIN
    ));
}
