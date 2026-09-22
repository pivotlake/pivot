use std::fs::{self, File, FileTimes};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use arrow_array::Int64Array;
use bin::execution::{Command, ExecuteOptions, StatementOutput};
use bin::shell::{ShellInstance, ShellTarget};
use datastore_pivot::DEFAULT_REFRESH_INTERVAL;

fn find_table_dir(root: &Path, table: &str) -> PathBuf {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("_pivot_manifest.json")).unwrap()).unwrap();
    root.join(manifest["schemas"][0]["table_ids"][table].as_str().unwrap())
}

fn age_file(path: &Path) {
    File::open(path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(5 * 3600)))
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn vacuum_in_shell_reclaims_expired_files_and_dropped_tables() {
    let directory = tempfile::tempdir().unwrap();
    let instance = ShellInstance::open_with_resources(
        &ShellTarget::pivot(directory.path().to_str().unwrap()),
        1,
        32,
        DEFAULT_REFRESH_INTERVAL,
    )
    .unwrap();
    let executor = instance.executor();
    for sql in [
        "CREATE TABLE events (id BIGINT)",
        "INSERT INTO events VALUES (1)",
        "CREATE TABLE discarded (id BIGINT)",
        "DROP TABLE discarded",
    ] {
        executor
            .execute(sql.into(), ExecuteOptions::default())
            .await
            .unwrap();
    }
    let table_dir = find_table_dir(directory.path(), "events");
    for entry in fs::read_dir(&table_dir).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|extension| extension == "parquet")
        {
            age_file(&path);
        }
    }
    let old_orphan = table_dir.join("old.parquet");
    let young_orphan = table_dir.join("young.parquet");
    fs::write(&old_orphan, b"orphan").unwrap();
    fs::write(&young_orphan, b"orphan").unwrap();
    age_file(&old_orphan);
    let manifest_path = directory.path().join("_pivot_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let dropped_dir = directory
        .path()
        .join(manifest["dropped_tables"][0]["location"].as_str().unwrap());

    assert!(executor.describe("VACUUM;").await.unwrap().is_none());
    assert!(old_orphan.exists(), "describing VACUUM must not run it");
    let result = executor
        .execute("/* cleanup */ vAcUuM;".into(), ExecuteOptions::default())
        .await
        .unwrap();

    assert!(matches!(
        result.output,
        StatementOutput::Command(Command::Vacuum)
    ));
    assert!(!old_orphan.exists());
    assert!(young_orphan.exists());
    assert!(dropped_dir.exists(), "a recent drop stays within retention");
    let rows = executor
        .execute("SELECT id FROM events".into(), ExecuteOptions::default())
        .await
        .unwrap();
    let StatementOutput::Rows { batches, .. } = rows.output else {
        panic!("expected rows")
    };
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );

    age_file(&young_orphan);
    manifest["dropped_tables"][0]["dropped_at_ms"] = 0.into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    executor
        .execute("VACUUM".into(), ExecuteOptions::default())
        .await
        .unwrap();

    assert!(!young_orphan.exists(), "a later command runs another sweep");
    assert!(!dropped_dir.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn vacuum_reports_unsupported_forms_and_cleanup_errors() {
    let directory = tempfile::tempdir().unwrap();
    let instance = ShellInstance::open_with_resources(
        &ShellTarget::pivot(directory.path().to_str().unwrap()),
        1,
        32,
        DEFAULT_REFRESH_INTERVAL,
    )
    .unwrap();
    let executor = instance.executor();
    executor
        .execute(
            "CREATE TABLE events (id BIGINT)".into(),
            ExecuteOptions::default(),
        )
        .await
        .unwrap();
    let table_dir = find_table_dir(directory.path(), "events");
    let orphan = table_dir.join("old.parquet");
    fs::write(&orphan, b"orphan").unwrap();
    age_file(&orphan);

    for sql in ["ANALYZE", "VACUUM ANALYZE", "VACUUM events", "VACUUM FULL"] {
        assert!(
            executor
                .execute(sql.into(), ExecuteOptions::default())
                .await
                .is_err(),
            "{sql}"
        );
        assert!(orphan.exists(), "unsupported SQL must not vacuum");
    }
    fs::write(
        table_dir.join("_delta_log/00000000000000000001.json"),
        b"invalid json",
    )
    .unwrap();
    let result = executor
        .execute("VACUUM".into(), ExecuteOptions::default())
        .await;

    assert!(result.is_err(), "failed cleanup must not report success");
    assert!(orphan.exists(), "failed refresh must not delete data");
}
