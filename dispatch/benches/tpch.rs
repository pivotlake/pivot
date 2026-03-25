//! TPC-H performance harness.
//!
//! Runs queries from the TPC-H suite against a local parquet dataset,
//! printing per-iteration wall-clock times.
//!
//! # Environment variables
//!
//! - `SOURCE_DIRECTORY` (required) — path to directory containing TPC-H table directories
//!   (e.g., `lineitem/`, `orders/`)
//! - `WORKER_COUNT` — number of worker threads (default: number of CPU cores)
//! - `QUERY` — which query/queries to run, comma-separated (default: all)
//! - `QUERY_TEST_COUNT` — number of iterations (default: 1)
//! - `SLEEP` — seconds to sleep between iterations (optional)
//!
//! # Usage
//!
//! ```sh
//! SOURCE_DIRECTORY=/path/to/tpch QUERY=12 QUERY_TEST_COUNT=5 cargo bench --bench tpch
//! ```

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::thread::sleep;
use std::time::{Duration, Instant};

use arrow::compute::concat_batches as arrow_concat_batches;
use arrow::util::display::ArrayFormatter;
use arrow_array::RecordBatch;
use tracing_subscriber::{EnvFilter, fmt};

use dispatch::{ParquetTable, Projection, table_input};

static SOURCE_DIRECTORY: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from(std::env::var("SOURCE_DIRECTORY").unwrap()));

fn get_env_var_with_default<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn concat_batches(batches: &[RecordBatch]) -> RecordBatch {
    assert!(!batches.is_empty(), "Cannot concatenate empty batch list");
    let schema = batches[0].schema();
    arrow_concat_batches(&schema, batches).expect("Failed to concatenate batches")
}

fn batch_to_tsv(batch: &RecordBatch) -> String {
    let mut result = String::new();
    let formatters: Vec<_> = batch
        .columns()
        .iter()
        .map(|col| ArrayFormatter::try_new(col.as_ref(), &Default::default()).unwrap())
        .collect();

    for row in 0..batch.num_rows() {
        for (col_idx, formatter) in formatters.iter().enumerate() {
            if col_idx > 0 {
                result.push('\t');
            }
            result.push_str(&formatter.value(row).to_string());
        }
        result.push('\n');
    }
    result
}

fn assert_result(expected: &str, batches: &[RecordBatch]) {
    let batch = concat_batches(batches);
    let actual = batch_to_tsv(&batch);
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Result mismatch.\n\nActual:\n{}\n\nExpected:\n{}",
        actual,
        expected
    );
}

// ── Tables ─────────────────────────────────────────────────────────────────

struct TpchTables {
    lineitem: Arc<ParquetTable>,
    orders: Arc<ParquetTable>,
}

impl TpchTables {
    fn load(base_dir: &PathBuf) -> Self {
        Self {
            lineitem: Arc::new(
                ParquetTable::from_directory(&base_dir.join("lineitem"))
                    .expect("Could not load lineitem table"),
            ),
            orders: Arc::new(
                ParquetTable::from_directory(&base_dir.join("orders"))
                    .expect("Could not load orders table"),
            ),
        }
    }
}

// ── Count sanity check ─────────────────────────────────────────────────────
//
// SELECT
//     (SELECT COUNT(*) FROM lineitem) AS lineitem_count,
//     (SELECT COUNT(*) FROM orders) AS orders_count

fn run_query_join_orders(tables: &TpchTables) {
    let orders = table_input(&tables.orders, Projection::from_field_names(tables.orders.schema(), ["o_orderkey"]), false);
    let joined = table_input(
        &tables.lineitem,
        Projection::from_field_names(tables.lineitem.schema(), ["l_orderkey"]),
        false,
    ).join(orders, 0, 0);
    joined.count().collect();
}

fn run_query_count(tables: &TpchTables) {
    const EXPECTED: &str = "600037902\n150000000\n";

    let lineitem_count = table_input(
        &tables.lineitem,
        Projection::from_field_names(tables.lineitem.schema(), ["l_orderkey", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipmode"]),
        false,
    )
    .count();


    let orders_count = table_input(
        &tables.orders,
        Projection::from_field_names(tables.orders.schema(), ["o_orderkey", "o_orderpriority"]),
        false,
    )
    .count();

    let results = lineitem_count.concat(orders_count).collect();
    let batch = concat_batches(&results);

    println!("Actual {:?}", batch_to_tsv(&batch));

    assert_result(EXPECTED, &results);
}

// ── Query 12 ───────────────────────────────────────────────────────────────
//
// SELECT
//     l_shipmode,
//     SUM(CASE
//         WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
//         THEN 1 ELSE 0
//     END) AS high_line_count,
//     SUM(CASE
//         WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
//         THEN 1 ELSE 0
//     END) AS low_line_count
// FROM
//     orders,
//     lineitem
// WHERE
//     o_orderkey = l_orderkey
//     AND l_shipmode IN ('MAIL', 'SHIP')
//     AND l_commitdate < l_receiptdate
//     AND l_shipdate < l_commitdate
//     AND l_receiptdate >= DATE '1994-01-01'
//     AND l_receiptdate < DATE '1994-01-01' + INTERVAL '1' YEAR
// GROUP BY
//     l_shipmode
// ORDER BY
//     l_shipmode;

fn run_query_12(tables: &TpchTables) {
//     const EXPECTED: &str = r#"MAIL	623115	934713
// SHIP	622979	934534"#;
//
//     let _lineitem = table_input(
//         &tables.lineitem,
//         Projection::from_field_names(
//             tables.lineitem.schema(),
//             ["l_orderkey", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipmode"],
//         ),
//         false,
//     );
//
//     let _orders = table_input(
//         &tables.orders,
//         Projection::from_field_names(tables.orders.schema(), ["o_orderkey", "o_orderpriority"]),
//         false,
//     );
//
//     // TODO: join lineitem and orders on l_orderkey = o_orderkey,
//     // filter on l_shipmode, date predicates,
//     // group by l_shipmode with conditional aggregation,
//     // order by l_shipmode.
//     todo!("join + group-by aggregation not yet implemented");
//
//     #[allow(unreachable_code)]
//     assert_result(EXPECTED, &todo!());
}

// ── Query registry ────────────────────────────────────────────────────────

const QUERIES: &[(u32, fn(&TpchTables))] = &[
    (0, run_query_count),
    (1, run_query_join_orders),
    (12, run_query_12),
];

fn get_query_fn(id: u32) -> fn(&TpchTables) {
    QUERIES
        .iter()
        .find(|(qid, _)| *qid == id)
        .unwrap_or_else(|| panic!("Unknown query: {id}"))
        .1
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stdout)
        .init();

    let num_workers: usize = std::env::var("WORKER_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| core_affinity::get_core_ids().unwrap().len());
    dispatch::init(num_workers);

    let tables = TpchTables::load(&SOURCE_DIRECTORY);

    let queries: Vec<(u32, fn(&TpchTables))> = match std::env::var("QUERY") {
        Ok(val) => val
            .split(',')
            .map(|s| {
                let id: u32 = s
                    .trim()
                    .parse()
                    .expect("QUERY must be comma-separated numbers");
                (id, get_query_fn(id))
            })
            .collect(),
        Err(_) => QUERIES.to_vec(),
    };
    let iterations = get_env_var_with_default("QUERY_TEST_COUNT", 1);

    for (id, run) in &queries {
        println!("=== TPC-H Query {} ===", id);
        for i in 0..iterations {
            let start = Instant::now();
            run(&tables);
            println!(
                "[{}/{}] Query {} — {}ms",
                i + 1,
                iterations,
                id,
                start.elapsed().as_millis()
            );
            if let Ok(a) = std::env::var("SLEEP") {
                sleep(Duration::from_secs(a.parse().unwrap()));
            }
        }
    }
}
