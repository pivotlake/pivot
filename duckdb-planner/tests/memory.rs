use duckdb_planner::{
    DuckDBBind, DuckDBColumn, DuckDBTable, LogicalTypeId, Operator, PlannerContext,
};
use std::sync::Arc;

struct UsersTable;

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
                name: "age".to_string(),
                duckdb_logical_type_id: LogicalTypeId::SMALLINT as u8,
            },
        ]
    }
}

struct TestCatalog;

impl DuckDBBind for TestCatalog {
    fn try_bind(&self, table_name: &str) -> Option<Arc<dyn DuckDBTable>> {
        match table_name {
            "users" => Some(Arc::new(UsersTable)),
            _ => None,
        }
    }
}

fn get_rss_bytes() -> usize {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

#[test]
#[ignore] // long-running stress test — run with: cargo test --test memory -- --ignored --nocapture
fn no_memory_leak_across_repeated_plans() {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let total_iterations = 10_000_000_000u64;
    let per_thread = total_iterations / num_threads as u64;

    println!("Spawning {num_threads} threads, {per_thread} plans each");

    let rss_before = get_rss_bytes();

    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            s.spawn(move || {
                let mut ctx = PlannerContext::new(Arc::new(TestCatalog));

                for i in 0..per_thread {
                    let plan = ctx
                        .plan("SELECT id, name FROM users WHERE age <> 0")
                        .unwrap();
                    assert!(matches!(&plan.operator, Operator::Projection(_)));
                    if i > 0 && i % 10_000 == 0 {
                        let rss = get_rss_bytes();
                        let growth = rss.saturating_sub(rss_before);
                        println!(
                            "[thread {thread_id}] {}M plans — RSS: {} MB (+{} MB)",
                            i / 1_000_000,
                            rss / (1024 * 1024),
                            growth / (1024 * 1024),
                        );
                    }
                }
            });
        }
    });

    let rss_after = get_rss_bytes();
    let growth = rss_after.saturating_sub(rss_before);

    assert!(
        growth < 10 * 1024 * 1024,
        "RSS grew by {growth} bytes after 10B plans — possible memory leak"
    );
}
