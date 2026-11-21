//! The small-file merges of one partition a routine round swaps into the
//! table together, whichever format keeps the table: several merges are
//! written at the same time and one commit holds every one that was
//! written, or none of them. A merge that fails stays out of the commit,
//! and the round reports it.

mod iceberg_support;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use datastore_pivotlake::test_support::commit_external_replace;
use datastore_pivotlake::{CompacterHandle, Error, FileRef, TableFormat};
use iceberg_support::{Database, collect_i64s, int_parquet_bytes};
use planner::catalog::SchemaQualifiedTableName;
use uuid::Uuid;

const FORMATS: [TableFormat; 2] = [TableFormat::Delta, TableFormat::Iceberg];

/// The files of the table `t`.
fn list_files(database: &Database) -> Vec<FileRef> {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    database.datastore.table_files(&name).unwrap().unwrap()
}

/// The version the table `t` is at: one above for every commit.
fn read_version(database: &Database) -> u64 {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    database.datastore.table_handle(&name).unwrap().version()
}

/// The names of the Parquet files under `directory`, at any depth.
fn list_parquet_names(directory: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            names.extend(list_parquet_names(&path));
        } else if path
            .extension()
            .is_some_and(|extension| extension == "parquet")
        {
            names.insert(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    names
}

/// A database kept in `format` whose table `t` holds one file of one row
/// for each of `values`, in the order of the values.
fn build_database_with_files(
    format: TableFormat,
    values: std::ops::RangeInclusive<i64>,
) -> Database {
    let mut database = Database::in_format(format);
    database.execute("CREATE TABLE t (value BIGINT)");
    insert_each(&mut database, values);
    database
}

/// Insert each of `values` into the table `t`: one file of one row for each.
fn insert_each(database: &mut Database, values: std::ops::RangeInclusive<i64>) {
    for value in values {
        database.execute(&format!("INSERT INTO t VALUES ({value})"));
    }
}

/// A compacter over the table `t` of a database, which runs a routine round
/// when a test asks for one.
struct Compacter {
    runtime: tokio::runtime::Runtime,
    handle: CompacterHandle,
    id: Uuid,
}

impl Compacter {
    /// A compacter of `database` under which one merge takes no more than
    /// `files_per_merge` files of the size the table `t` holds now, a file
    /// larger than the largest of them is small no longer when
    /// `are_outputs_small` is not set, and the merges of
    /// `max_concurrent_merges` commits are in flight at once.
    fn start(
        database: &Database,
        files_per_merge: u64,
        are_outputs_small: bool,
        max_concurrent_merges: usize,
    ) -> Self {
        let largest_file = list_files(database).iter().map(|file| file.size).max();
        let largest_file = largest_file.unwrap();
        let target_bytes = if are_outputs_small {
            datastore_pivotlake::DEFAULT_COMPACT_BYTES
        } else {
            2 * largest_file + 1
        };
        let (handle, actor) = CompacterHandle::new(
            target_bytes,
            files_per_merge * largest_file + largest_file / 2,
            datastore_pivotlake::DEFAULT_MIN_FILES_TO_MERGE,
            max_concurrent_merges,
            datastore_pivotlake::DEFAULT_LAYOUT_CLIQUE_SIZE,
            None,
            Arc::clone(&database.datastore),
        );
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.spawn(actor.run());
        let name = SchemaQualifiedTableName::in_default_schema("t");
        let id = database.datastore.table_handle(&name).unwrap().id();
        Self {
            runtime,
            handle,
            id,
        }
    }

    /// Run one routine round over the table, and tell how many passes it
    /// took.
    fn run_round(&self) -> Result<u64, Error> {
        let name = SchemaQualifiedTableName::in_default_schema("t");
        let round = self.handle.compact(name, self.id, false);
        self.runtime.block_on(round)
    }
}

#[test]
fn the_merges_of_a_partition_are_swapped_in_by_one_commit() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=12);
        let compacter = Compacter::start(&database, 6, false, 1);
        let version_before = read_version(&database);

        let passes = compacter.run_round().unwrap();

        assert_eq!(passes, 2, "{format}: two merges in one commit, then none");
        assert_eq!(read_version(&database), version_before + 1, "{format}");
        assert_eq!(list_files(&database).len(), 2, "{format}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(
            collect_i64s(&rows, 0),
            (1..=12).collect::<Vec<i64>>(),
            "{format}"
        );
    }
}

#[test]
fn one_commit_swaps_in_no_more_than_five_merges() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=36);
        let compacter = Compacter::start(&database, 6, false, 1);
        let version_before = read_version(&database);

        let passes = compacter.run_round().unwrap();

        assert_eq!(
            passes, 3,
            "{format}: five merges in one commit, the sixth in another, then none"
        );
        assert_eq!(read_version(&database), version_before + 2, "{format}");
        assert_eq!(list_files(&database).len(), 6, "{format}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(
            collect_i64s(&rows, 0),
            (1..=36).collect::<Vec<i64>>(),
            "{format}"
        );
    }
}

#[test]
fn a_round_after_a_commit_of_several_merges_merges_the_files_inserted_since() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=12);
        let compacter = Compacter::start(&database, 6, false, 1);
        compacter.run_round().unwrap();
        let merged = list_files(&database);
        let version_before = read_version(&database);
        insert_each(&mut database, 13..=18);
        let version_inserted = read_version(&database);

        let passes = compacter.run_round().unwrap();

        let files = list_files(&database);
        let kept = files.iter().filter(|file| merged.contains(file)).count();
        assert_eq!(version_inserted, version_before + 6, "{format}");
        assert_eq!(passes, 2, "{format}: one merge, then none");
        assert_eq!(read_version(&database), version_inserted + 1, "{format}");
        assert_eq!(files.len(), 3, "{format}: {files:?}");
        assert_eq!(kept, 2, "{format}: the outputs of the round before stay");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(
            collect_i64s(&rows, 0),
            (1..=18).collect::<Vec<i64>>(),
            "{format}"
        );
    }
}

#[test]
fn a_round_merges_the_outputs_of_the_merges_it_committed_together() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=41);
        let compacter = Compacter::start(&database, 8, true, 1);
        let version_before = read_version(&database);

        let passes = compacter.run_round().unwrap();

        assert_eq!(
            passes, 3,
            "{format}: five merges in one commit, one of their outputs with the file left, then none"
        );
        assert_eq!(read_version(&database), version_before + 2, "{format}");
        assert_eq!(list_files(&database).len(), 1, "{format}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(
            collect_i64s(&rows, 0),
            (1..=41).collect::<Vec<i64>>(),
            "{format}"
        );
    }
}

#[test]
fn merges_in_flight_together_share_no_small_file_of_a_partition() {
    for format in FORMATS {
        let mut database = build_database_with_files(format, 1..=36);
        let compacter = Compacter::start(&database, 6, false, 3);

        let round = compacter.run_round();

        assert!(round.is_ok(), "{format}: {:?}", round.err());
        assert_eq!(list_files(&database).len(), 6, "{format}");
        let rows = database.execute("SELECT value FROM t ORDER BY value");
        assert_eq!(
            collect_i64s(&rows, 0),
            (1..=36).collect::<Vec<i64>>(),
            "{format}"
        );
    }
}

#[test]
fn a_failed_merge_keeps_its_inputs_while_the_merge_beside_it_is_committed() {
    for format in FORMATS {
        let database = build_database_with_files(format, 1..=12);
        let compacter = Compacter::start(&database, 6, false, 1);
        let table_path = database.table_path("t");
        let emptied = empty_first_inserted_file(&table_path);
        let names_before = list_parquet_names(&table_path);
        let files_before = list_files(&database);
        let version_before = read_version(&database);

        let round = compacter.run_round();

        assert!(round.is_err(), "{format}");
        assert_eq!(read_version(&database), version_before + 1, "{format}");
        let files = list_files(&database);
        let kept: Vec<&FileRef> = files
            .iter()
            .filter(|file| files_before.contains(file))
            .collect();
        assert_eq!(kept.len(), 6, "{format}: the inputs of the failed merge");
        assert!(
            kept.iter().any(|file| file.path.name() == emptied),
            "{format}: {kept:?}"
        );
        let live: BTreeSet<String> = files
            .iter()
            .map(|file| file.path.name().to_string())
            .collect();
        let names = list_parquet_names(&table_path);
        let written: Vec<&String> = names.difference(&names_before).collect();
        assert_eq!(written.len(), 1, "{format}: the output of the other merge");
        assert!(live.contains(written[0]), "{format}: {written:?}");
    }
}

#[test]
fn a_group_whose_only_merge_failed_commits_nothing_and_leaves_no_output() {
    for format in FORMATS {
        let database = build_database_with_files(format, 1..=6);
        let compacter = Compacter::start(&database, 6, false, 1);
        let table_path = database.table_path("t");
        empty_first_inserted_file(&table_path);
        let names_before = list_parquet_names(&table_path);
        let files_before = list_files(&database);
        let version_before = read_version(&database);

        let round = compacter.run_round();

        assert!(round.is_err(), "{format}");
        assert_eq!(read_version(&database), version_before, "{format}");
        assert_eq!(list_files(&database), files_before, "{format}");
        assert_eq!(list_parquet_names(&table_path), names_before, "{format}");
    }
}

#[test]
fn a_commit_another_writer_conflicts_with_leaves_none_of_its_outputs() {
    let mut database = build_database_with_files(TableFormat::Iceberg, 1..=12);
    let compacter = Compacter::start(&database, 6, false, 1);
    let table_path = database.table_path("t");
    let retired = list_files(&database).remove(0).path;
    commit_external_replace(
        &database.root_uri(),
        &database.table_directory("t"),
        &[retired.as_str()],
        "outside.parquet",
        &int_parquet_bytes(&[100]),
        1,
    )
    .unwrap();
    let names_before = list_parquet_names(&table_path);

    let round = compacter.run_round();

    assert!(round.is_ok(), "{:?}", round.err());
    let live: BTreeSet<String> = list_files(&database)
        .iter()
        .map(|file| file.path.name().to_string())
        .collect();
    let written: BTreeSet<String> = list_parquet_names(&table_path)
        .difference(&names_before)
        .cloned()
        .collect();
    assert!(written.is_subset(&live), "{written:?} of {live:?}");
    let rows = database.execute("SELECT count(*) FROM t");
    assert_eq!(collect_i64s(&rows, 0), [12]);
}

/// Leave the file that holds the value 1, the first inserted, without a
/// byte, and tell its name: the merge that takes it fails.
fn empty_first_inserted_file(table_path: &Path) -> String {
    let names = list_parquet_names(table_path);
    let is_first_inserted = |name: &String| int_rows_of(table_path, name) == [1];
    let first_inserted = names.into_iter().find(is_first_inserted).unwrap();
    empty_parquet_file(table_path, &first_inserted);
    first_inserted
}

/// The values of the one BIGINT column of the Parquet file `name` under
/// `directory`, at any depth.
fn int_rows_of(directory: &Path, name: &str) -> Vec<i64> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    let path = find_parquet_file(directory, name);
    let file = std::fs::File::open(path).unwrap();
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    reader
        .flat_map(|batch| {
            let batch = batch.unwrap();
            let values = batch.column(0).as_primitive::<Int64Type>();
            values.values().to_vec()
        })
        .collect()
}

/// Where the Parquet file `name` is under `directory`, at any depth.
fn find_parquet_file(directory: &Path, name: &str) -> std::path::PathBuf {
    let mut directories = vec![directory.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                directories.push(path);
            } else if path.file_name().is_some_and(|file_name| file_name == name) {
                return path;
            }
        }
    }
    panic!("no file {name} under {}", directory.display());
}

/// Leave the Parquet file `name` under `directory` without a byte, behind
/// the table's back: whoever reads its rows fails.
fn empty_parquet_file(directory: &Path, name: &str) {
    std::fs::File::create(find_parquet_file(directory, name)).unwrap();
}
