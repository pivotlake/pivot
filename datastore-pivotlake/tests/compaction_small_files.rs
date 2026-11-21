//! Which small files a compaction round merges under the default sizes,
//! whichever format keeps the table: a routine round leaves a few small
//! files as they are, and a `COMPACT ... FINAL` merges them, as many at a
//! time as one merge takes.

mod iceberg_support;

use std::time::Duration;

use datastore_pivotlake::{
    CompactionConfig, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_PARALLELISM,
    DEFAULT_LAYOUT_CLIQUE_SIZE, DEFAULT_MIN_FILES_TO_MERGE, TableFormat,
    default_max_merge_input_bytes,
};
use iceberg_support::{Database, collect_i64s};
use planner::catalog::SchemaQualifiedTableName;

const FORMATS: [TableFormat; 2] = [TableFormat::Delta, TableFormat::Iceberg];

/// The default sizes, for a compacter only a `COMPACT` drives.
fn build_default_sizes_config() -> CompactionConfig {
    CompactionConfig {
        target_bytes: DEFAULT_COMPACT_BYTES,
        max_merge_input_bytes: default_max_merge_input_bytes(DEFAULT_COMPACT_BYTES),
        min_files: DEFAULT_MIN_FILES_TO_MERGE,
        max_concurrent_merges: DEFAULT_COMPACT_PARALLELISM,
        layout_clique_size: DEFAULT_LAYOUT_CLIQUE_SIZE,
        poll_interval: Duration::from_secs(3600),
    }
}

/// A database kept in `format` whose table `t` holds one file of one row
/// for each of `values`.
fn build_database_with_files(
    format: TableFormat,
    values: std::ops::RangeInclusive<i64>,
) -> Database {
    let mut database = Database::in_format_with_compaction(format, build_default_sizes_config());
    database.execute("CREATE TABLE t (value BIGINT)");
    for value in values {
        database.execute(&format!("INSERT INTO t VALUES ({value})"));
    }
    database
}

/// How many files the table `t` holds.
fn count_files(database: &Database) -> usize {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    database
        .datastore
        .table_files(&name)
        .unwrap()
        .unwrap()
        .len()
}

#[test]
fn a_routine_round_leaves_the_two_small_files_that_a_final_sweep_merges() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=2);

        database.compact("t", false);
        let files_after_the_routine_round = count_files(&database);
        database.compact("t", true);

        assert_eq!(files_after_the_routine_round, 2, "{format}");
        assert_eq!(count_files(&database), 1, "{format}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(collect_i64s(&rows, 0), [1, 2], "{format}");
    }
}

#[test]
fn a_final_sweep_merges_no_more_than_ten_small_files_at_a_time() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=12);

        let passes = database.compact("t", true);

        assert_eq!(
            passes, 3,
            "{format}: a merge of ten files, one of its output with the two left, then a pass that finds none"
        );
        assert_eq!(count_files(&database), 1, "{format}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(
            collect_i64s(&rows, 0),
            (1..=12).collect::<Vec<i64>>(),
            "{format}"
        );
    }
}
