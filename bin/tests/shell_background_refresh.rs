//! The shell keeps its tables current the way the server does: a background
//! sweep reloads them from the store, so a commit another process lands in a
//! shared datastore shows up in later queries without reopening the shell.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arrow_array::Int64Array;
use bin::execution::{ExecuteOptions, Executor, StatementOutput};
use bin::shell::ShellInstance;

#[test]
fn a_commit_from_outside_the_shell_becomes_visible_to_later_queries() {
    let directory = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let instance = runtime
        .block_on(async {
            ShellInstance::open_with_resources(
                directory.path().to_str().unwrap(),
                1,
                32,
                Duration::from_millis(100),
            )
        })
        .unwrap();
    runtime.block_on(async {
        execute(instance.executor(), "CREATE TABLE people (id BIGINT)").await;
        execute(instance.executor(), "INSERT INTO people VALUES (1), (2)").await;
    });
    let before = runtime.block_on(count_ids(instance.executor()));

    commit_a_copy_of_the_data_file_from_outside(&table_dir(directory.path(), "people"));
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut after = before;
    while after == before && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        after = runtime.block_on(count_ids(instance.executor()));
    }

    assert_eq!(before, 2);
    assert_eq!(
        after, 4,
        "the background refresh never picked up the commit"
    );
}

/// Stand in for another process writing to the same datastore: duplicate the
/// table's one data file under a new name and commit it as the next Delta
/// version, straight onto disk, so the shell's in-memory table set knows
/// nothing of it until its refresh sweep runs.
fn commit_a_copy_of_the_data_file_from_outside(table: &Path) {
    let log = table.join("_delta_log");
    let mut commits: Vec<PathBuf> = std::fs::read_dir(&log)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    commits.sort();
    let latest = commits.last().unwrap();
    let added: Vec<String> = std::fs::read_to_string(latest)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .filter_map(|action| action["add"]["path"].as_str().map(str::to_string))
        .collect();
    assert_eq!(added.len(), 1, "the INSERT should have committed one file");

    let copy_name = "external-copy.parquet";
    std::fs::copy(table.join(&added[0]), table.join(copy_name)).unwrap();
    let size = std::fs::metadata(table.join(copy_name)).unwrap().len();
    let action = serde_json::json!({
        "add": {
            "path": copy_name,
            "partitionValues": {},
            "size": size,
            "modificationTime": 0,
            "dataChange": true
        }
    });
    let next_version = commits.len();
    std::fs::write(
        log.join(format!("{next_version:020}.json")),
        format!("{action}\n"),
    )
    .unwrap();
}

/// Where the datastore rooted at `root` keeps `table`, read from its manifest.
fn table_dir(root: &Path, table: &str) -> PathBuf {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("_pivot_manifest.json")).unwrap()).unwrap();
    let id = manifest["schemas"]
        .as_array()
        .unwrap()
        .iter()
        .find(|schema| schema["name"].as_str() == Some("main"))
        .unwrap()["table_ids"][table]
        .as_str()
        .unwrap();
    root.join(manifest["table_locations"][id].as_str().unwrap())
}

async fn count_ids(executor: &Executor) -> i64 {
    let StatementOutput::Rows { batches, .. } =
        execute(executor, "SELECT COUNT(id) FROM people").await
    else {
        panic!("COUNT did not return rows");
    };
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

async fn execute(executor: &Executor, sql: &str) -> StatementOutput<arrow_array::RecordBatch> {
    executor
        .execute(sql.to_string(), ExecuteOptions::default())
        .await
        .unwrap()
        .output
}
