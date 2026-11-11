//! A routine round remembers of a partition's layout only the pairs it may
//! merge, so a pair below its bar is left standing, and a `COMPACT ... FINAL`
//! that follows measures the pair again and merges it, whichever format
//! keeps the table.

mod iceberg_support;

use std::time::Duration;

use datastore_pivotlake::{CompactionConfig, FileRef, TableFormat};
use iceberg_support::{Database, collect_i64s};
use planner::catalog::SchemaQualifiedTableName;

/// Every file counts as large and two files are a full clique, so a round
/// runs layout optimization alone.
fn layout_only() -> CompactionConfig {
    CompactionConfig {
        target_bytes: 2,
        merge_target_bytes: u64::MAX,
        min_files: usize::MAX,
        max_concurrent_merges: 1,
        layout_clique_size: 2,
        poll_interval: Duration::from_secs(3600),
    }
}

/// The live files of the table `t` of `db`.
fn list_files(db: &Database) -> Vec<FileRef> {
    db.datastore
        .table_files(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap()
        .unwrap()
}

/// Two sorted files whose ranges cross without either lying inside the
/// other's: a pair below the routine bar, which only a final sweep merges.
fn check_a_final_sweep_merges_the_pair_a_routine_round_left(format: TableFormat) {
    let mut db = Database::in_format_with_compaction(format, layout_only());
    db.execute("CREATE TABLE t (v BIGINT) SORTED BY (v)");
    db.execute("INSERT INTO t VALUES (0), (100)");
    db.execute("INSERT INTO t VALUES (50), (150)");
    let inserted = list_files(&db);

    db.compact("t", false);
    let left_by_the_routine_round = list_files(&db);
    db.compact("t", true);
    let left_by_the_final_sweep = list_files(&db);

    assert_eq!(left_by_the_routine_round, inserted);
    assert_eq!(left_by_the_final_sweep.len(), 1);
    assert!(!inserted.contains(&left_by_the_final_sweep[0]));
    let rows = db.execute("SELECT v FROM t ORDER BY v");
    assert_eq!(collect_i64s(&rows, 0), [0, 50, 100, 150]);
}

#[test]
fn a_final_sweep_merges_the_pair_a_routine_round_left_in_a_delta_table() {
    check_a_final_sweep_merges_the_pair_a_routine_round_left(TableFormat::Delta);
}

#[test]
fn a_final_sweep_merges_the_pair_a_routine_round_left_in_an_iceberg_table() {
    check_a_final_sweep_merges_the_pair_a_routine_round_left(TableFormat::Iceberg);
}
