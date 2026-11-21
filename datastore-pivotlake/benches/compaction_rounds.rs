//! What compaction rounds over a table of many files cost in time, and what
//! the compacter holds of the table between them.
//!
//! The table is built here, in the format asked for: ten `Int64` columns,
//! partitioned by `p` when it has more than one partition and sorted by `s`.
//! Its files are hard links to a few real Parquet files, so every one has a
//! footer to read while the table costs no disk: a link to a file of one
//! row is a small file, a link to a file past half the target a large one.
//! Every large file is committed with a stretch of the sort key of its own,
//! so no layout merge ever qualifies, and the commits are sized so an
//! Iceberg table's manifests come close to the size manifests are merged up
//! to, each holding files of every partition.
//!
//! A run measures either the rounds' times (`--measures rounds`) or what the
//! compacter holds (`--measures held`). The first goes through these, in
//! order, and prints what each took:
//!
//! - `first round`: a routine round over the table as nothing has examined
//!   it yet. Nothing qualifies for a merge.
//! - `unchanged round`: a routine round over the table as the round before
//!   left it.
//! - `round after a rewrite`: a routine round after another writer replaced
//!   one large file with another, which rewrites one of the table's large
//!   manifests. Nothing qualifies for a merge. Measured twice: as the round
//!   finds the table (`cold`), and after a scan that selects no file has read
//!   what the commit wrote (`after a scan`).
//! - `merging round`: a routine round after a commit added small files beside
//!   a manifest's worth of large ones, a few to each partition. The round
//!   merges them some ten at a time, and commits up to five merges of a
//!   partition together, each commit rewriting the manifest that holds
//!   their files. The time is given for the round and for each commit: what
//!   selecting its merges, running them and committing them take together.
//!
//! The second commits every large file, runs the first round and a few
//! unchanged ones, and prints `held`: the growth of what the process has
//! allocated and not freed across them, as its allocator counts it, with
//! nothing left in the datastore's table metadata cache before or after:
//! what the compacter holds of the table between rounds, in all and for each
//! file.
//!
//! # Running
//!
//! ```sh
//! cargo bench -p datastore-pivotlake --bench compaction_rounds -- \
//!     --directory <where the database is written> --format iceberg
//! ```
//!
//! Arguments: `--directory` (required), `--format` (`iceberg` by default, or
//! `delta`), `--files` (default 100000), `--partitions` (default 1),
//! `--small-files` (default 200), `--workers` (default 8), `--repeats`
//! (default 5; the unchanged round runs four times as often), `--measures`
//! (`rounds` by default, or `held`).

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
use arrow_schema::{DataType, Field, Schema};
use catalog::datastore::DatastoreTransaction;
use datastore_pivotlake::{
    CatalogTable, CompacterHandle, FileEntry, FileRef, PivotlakeDatastore, TableFormat,
};
use dispatch::Dispatch;
use iceberg_tables::table_metadata_cache::{
    DEFAULT_TABLE_METADATA_CACHE_BYTES, TableMetadataCache,
};
use object_storage::{LocalStore, ObjectPath, ObjectStore};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use tikv_jemalloc_sys::mallctl;
use uuid::Uuid;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;

/// The compaction target: a file at least half of it is a large one.
const TARGET_BYTES: u64 = 64 * KIB;

/// The rows of a large file, which make it more than half the target.
const LARGE_FILE_ROWS: usize = 1000;

/// How many small files' bytes a small-file merge takes.
const SMALL_FILES_PER_MERGE: u64 = 10;

/// How large a manifest a commit of an Iceberg table aims to write: close
/// below the size manifests are merged up to.
const MANIFEST_TARGET_BYTES: u64 = 7 * MIB + MIB / 2;

/// How many files the first commit adds, whose manifest tells what one file
/// takes in a manifest.
const FIRST_COMMIT_FILES: usize = 256;

/// How many files a commit of a table without manifests adds.
const FILES_PER_COMMIT_WITHOUT_MANIFESTS: usize = 20_000;

/// The columns of the table: the partition column, the sort column, and
/// eight more.
const COLUMN_COUNT: usize = 10;

/// The stretch of the sort column a large file is committed with.
const SORT_VALUES_PER_FILE: i64 = 10_000;

/// The value of the sort column in every row of a small file, below every
/// large file's.
const SMALL_FILE_SORT_VALUE: i64 = -2_000_000;

/// What the process's allocator answers when asked for `name`, a value it
/// keeps as a `Value`.
fn read_allocator_value<Value: Copy>(name: &CStr, mut value: Value) -> Value {
    let mut length = size_of::<Value>();
    let status = unsafe {
        mallctl(
            name.as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &raw mut length,
            null_mut(),
            0,
        )
    };
    assert_eq!(status, 0, "the allocator keeps {name:?}");
    value
}

/// The bytes the process has allocated and not freed, as its allocator
/// counts them now, once what this thread freed and its allocator still
/// keeps at hand for it is handed back.
fn read_live_bytes() -> usize {
    let flushed = unsafe {
        mallctl(
            c"thread.tcache.flush".as_ptr(),
            null_mut(),
            null_mut(),
            null_mut(),
            0,
        )
    };
    assert_eq!(flushed, 0, "the allocator hands back what the thread freed");
    let mut epoch: u64 = 1;
    let refreshed = unsafe {
        mallctl(
            c"epoch".as_ptr(),
            null_mut(),
            null_mut(),
            (&raw mut epoch).cast::<c_void>(),
            size_of::<u64>(),
        )
    };
    assert_eq!(refreshed, 0, "the allocator refreshes what it counts");
    read_allocator_value::<usize>(c"stats.allocated", 0)
}

struct Arguments {
    directory: PathBuf,
    format: TableFormat,
    files: usize,
    partitions: usize,
    small_files: usize,
    workers: usize,
    repeats: usize,
    /// Whether the run measures what the compacter holds, and not what the
    /// rounds take.
    measures_held: bool,
}

impl Arguments {
    fn parse() -> Self {
        let mut arguments = Self {
            directory: PathBuf::new(),
            format: TableFormat::Iceberg,
            files: 100_000,
            partitions: 1,
            small_files: 200,
            workers: 8,
            repeats: 5,
            measures_held: false,
        };
        let mut given = std::env::args().skip(1);
        while let Some(name) = given.next() {
            // Cargo passes `--bench` to every bench binary.
            if name == "--bench" {
                continue;
            }
            let value = given
                .next()
                .unwrap_or_else(|| panic!("{name} takes a value"));
            let read_count = || -> usize {
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("{name} takes a whole number, not `{value}`"))
            };
            match name.as_str() {
                "--directory" => arguments.directory = PathBuf::from(&value),
                "--format" => arguments.format = value.parse().unwrap(),
                "--files" => arguments.files = read_count(),
                "--partitions" => arguments.partitions = read_count(),
                "--small-files" => arguments.small_files = read_count(),
                "--workers" => arguments.workers = read_count(),
                "--repeats" => arguments.repeats = read_count(),
                "--measures" => {
                    arguments.measures_held = match value.as_str() {
                        "rounds" => false,
                        "held" => true,
                        _ => panic!("--measures is `rounds` or `held`, not `{value}`"),
                    }
                }
                _ => panic!("unknown argument {name}"),
            }
        }
        assert!(
            !arguments.directory.as_os_str().is_empty(),
            "--directory names where the database is written"
        );
        assert!(arguments.partitions >= 1, "--partitions is at least 1");
        arguments
    }

    fn is_partitioned(&self) -> bool {
        self.partitions > 1
    }
}

/// The table's columns: the partition column `p`, the sort column `s`, then
/// `c2`, `c3` and so on, each named for its position.
fn build_columns() -> Vec<Column> {
    (0..COLUMN_COUNT)
        .map(|position| Column {
            name: match position {
                0 => "p".to_string(),
                1 => "s".to_string(),
                position => format!("c{position}"),
            },
            col_type: Type::Int64,
        })
        .collect()
}

fn int_array(value: i64) -> ArrayRef {
    Arc::new(Int64Array::from(vec![value]))
}

/// Write the Parquet file at `path`: `rows` rows of the table's columns,
/// every one holding `partition` in `p` and `SMALL_FILE_SORT_VALUE` in `s`,
/// and values that do not compress in the others.
fn write_parquet_file(path: &Path, partition: i64, rows: usize) {
    let fields: Vec<Field> = build_columns()
        .iter()
        .map(|column| Field::new(&column.name, DataType::Int64, true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = (0..COLUMN_COUNT)
        .map(|position| {
            let values: Vec<i64> = match position {
                0 => vec![partition; rows],
                1 => vec![SMALL_FILE_SORT_VALUE; rows],
                position => (0..rows)
                    .map(|row| {
                        let mixed = (row as u64 + 1)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            .wrapping_add(position as u64);
                        (mixed ^ (mixed >> 29)) as i64
                    })
                    .collect(),
            };
            Arc::new(Int64Array::from(values)) as ArrayRef
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// The real files the table's files are links to: a small one and a large
/// one for each partition.
struct LinkedFiles {
    directory: PathBuf,
    small_bytes: u64,
    large_bytes: u64,
}

impl LinkedFiles {
    fn write(directory: &Path, partitions: usize) -> Self {
        std::fs::create_dir_all(directory).unwrap();
        for partition in 0..partitions {
            write_parquet_file(
                &directory.join(format!("small-{partition}")),
                partition as i64,
                1,
            );
            write_parquet_file(
                &directory.join(format!("large-{partition}")),
                partition as i64,
                LARGE_FILE_ROWS,
            );
        }
        let read_size = |name: &str| std::fs::metadata(directory.join(name)).unwrap().len();
        Self {
            directory: directory.to_path_buf(),
            small_bytes: read_size("small-0"),
            large_bytes: read_size("large-0"),
        }
    }
}

/// What tells the files of one table apart, and where they stand.
struct TableFiles<'a> {
    table_directory: PathBuf,
    linked: &'a LinkedFiles,
    arguments: &'a Arguments,
}

impl TableFiles<'_> {
    /// Link the real file `linked_name` into the table as `name`, a file of
    /// `partition` holding `rows` rows of `size` bytes whose sort column
    /// spans `sort_range`, and hand back the entry a commit records for it.
    fn link(
        &self,
        linked_name: &str,
        name: &str,
        partition: i64,
        rows: i64,
        size: u64,
        sort_range: (i64, i64),
    ) -> FileEntry {
        std::fs::hard_link(
            self.linked.directory.join(linked_name),
            self.table_directory.join(name),
        )
        .unwrap();
        let mut min_values = HashMap::with_capacity(COLUMN_COUNT);
        let mut max_values = HashMap::with_capacity(COLUMN_COUNT);
        let mut null_counts = HashMap::with_capacity(COLUMN_COUNT);
        for (position, column) in build_columns().into_iter().enumerate() {
            let (low, high) = match position {
                0 => (partition, partition),
                1 => sort_range,
                position => (position as i64, i64::MAX - position as i64),
            };
            min_values.insert(column.name.clone(), int_array(low));
            max_values.insert(column.name.clone(), int_array(high));
            null_counts.insert(column.name, 0);
        }
        let partition_values = self
            .arguments
            .is_partitioned()
            .then(|| PartitionValues::from([("p".to_string(), Scalar::new(int_array(partition)))]));
        FileEntry {
            file: FileRef {
                path: ObjectPath::new(name),
                size,
            },
            partition: partition_values,
            stats: Some(Arc::new(FileStats {
                num_records: Some(rows),
                min_values,
                max_values,
                null_counts,
            })),
            sorted_by: Arc::default(),
        }
    }

    /// The entry of the large file numbered `number`, linked into the
    /// table: in the partition its number falls in, on a stretch of the
    /// sort column of its own.
    fn link_large(&self, number: usize) -> FileEntry {
        self.link_large_into(number, number % self.arguments.partitions)
    }

    /// The entry of the large file numbered `number`, linked into the
    /// table in `partition`, on a stretch of the sort column of its own.
    fn link_large_into(&self, number: usize, partition: usize) -> FileEntry {
        let lowest = number as i64 * SORT_VALUES_PER_FILE;
        self.link(
            &format!("large-{partition}"),
            &format!("large-{number:08}.parquet"),
            partition as i64,
            LARGE_FILE_ROWS as i64,
            self.linked.large_bytes,
            (lowest, lowest + SORT_VALUES_PER_FILE - 1),
        )
    }

    /// The entry of the small file numbered `number`, linked into the
    /// table: in the partition its number falls in.
    fn link_small(&self, number: usize) -> FileEntry {
        let partition = number % self.arguments.partitions;
        let sort_value = SMALL_FILE_SORT_VALUE + 1 + number as i64;
        self.link(
            &format!("small-{partition}"),
            &format!("small-{number:08}.parquet"),
            partition as i64,
            1,
            self.linked.small_bytes,
            (sort_value, sort_value),
        )
    }
}

/// The length of the largest manifest in the Iceberg table directory
/// `table_directory`; `None` for a table that keeps no manifest.
fn find_largest_manifest_bytes(table_directory: &Path) -> Option<u64> {
    let metadata = std::fs::read_dir(table_directory.join("metadata")).ok()?;
    metadata
        .map(|entry| entry.unwrap())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.ends_with(".avro") && !name.starts_with("snap-")
        })
        .map(|entry| entry.metadata().unwrap().len())
        .max()
}

/// A database with the one table the rounds run over, and the compacter
/// that runs them.
struct Harness {
    runtime: tokio::runtime::Runtime,
    datastore: Arc<PivotlakeDatastore>,
    compacter: CompacterHandle,
    name: SchemaQualifiedTableName,
    id: Uuid,
}

impl Harness {
    fn find_table(&self) -> CatalogTable {
        self.datastore.table_handle(&self.name).unwrap()
    }

    /// Commit `added` in place of the files at `removed`, as another writer
    /// would, and publish the version that wrote.
    fn commit(&self, removed: &[ObjectPath], added: &[FileEntry]) {
        let mut table = self.find_table();
        table.replace_data_files(removed, added).unwrap();
        self.datastore.publish_table(table);
    }

    /// Run one routine round over the table; hands back how long it took
    /// and how many passes it made.
    fn run_round(&self) -> (Duration, u64) {
        let started = Instant::now();
        let passes = self
            .runtime
            .block_on(self.compacter.compact(self.name.clone(), self.id, false))
            .unwrap();
        (started.elapsed(), passes)
    }

    /// Plan a scan of the table that selects no file.
    fn scan_past_every_file(&self, arguments: &Arguments) {
        let past_every_file = arguments.files as i64 * SORT_VALUES_PER_FILE * 2;
        let filter = Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx: 1,
                return_type: Type::Int64,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(int_array(
                past_every_file,
            )))),
            compare_type: CompareType::GreaterEqual,
            return_type: Type::Boolean,
        });
        let selected = self.find_table().select_scan(&[filter]).unwrap();
        assert!(selected.row_groups().is_empty(), "the scan selects no file");
    }
}

/// Create the table `name` in `datastore`.
fn create_table(
    runtime: &tokio::runtime::Runtime,
    datastore: &Arc<PivotlakeDatastore>,
    dispatch: &Dispatch,
    name: &str,
    arguments: &Arguments,
) {
    let partitioned_by = if arguments.is_partitioned() {
        vec!["p".to_string()]
    } else {
        Vec::new()
    };
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns: build_columns(),
        partitioned_by,
        sorted_by: vec!["s".to_string()],
        options: HashMap::new(),
        if_not_exists: false,
    };
    let transaction = Arc::clone(datastore).begin_transaction();
    transaction
        .bind_create_table(request)
        .unwrap()
        .compile(dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    runtime.block_on(transaction.commit()).unwrap();
}

/// The median of `values` with the least and the greatest.
fn summarize(mut values: Vec<Duration>) -> String {
    values.sort_unstable();
    let format = |duration: Duration| format!("{:.3} ms", duration.as_secs_f64() * 1000.0);
    format!(
        "{} ({} to {})",
        format(values[values.len() / 2]),
        format(values[0]),
        format(values[values.len() - 1])
    )
}

fn main() {
    let arguments = Arguments::parse();
    std::fs::create_dir_all(&arguments.directory).unwrap();
    let directory = tempfile::tempdir_in(&arguments.directory).unwrap();
    let linked = LinkedFiles::write(&directory.path().join("linked"), arguments.partitions);
    assert!(
        linked.small_bytes < TARGET_BYTES.div_ceil(2) && linked.large_bytes >= TARGET_BYTES / 2,
        "a file of one row is small and one of {LARGE_FILE_ROWS} rows is large: {} and {} bytes",
        linked.small_bytes,
        linked.large_bytes
    );
    let database = directory.path().join("database");
    std::fs::create_dir_all(&database).unwrap();

    let dispatch = Dispatch::spin_up(arguments.workers, 512, None);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let table_metadata_cache = TableMetadataCache::new(DEFAULT_TABLE_METADATA_CACHE_BYTES);
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(&database).unwrap());
    let datastore = PivotlakeDatastore::from_store(
        store,
        dispatch.dispatcher(),
        None,
        Some(arguments.format),
        Arc::clone(&table_metadata_cache),
    )
    .unwrap();
    create_table(&runtime, &datastore, &dispatch, "t", &arguments);
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let id = datastore.table_handle(&name).unwrap().id();
    let (compacter, actor) = CompacterHandle::new(
        TARGET_BYTES,
        SMALL_FILES_PER_MERGE * linked.small_bytes,
        datastore_pivotlake::DEFAULT_MIN_FILES_TO_MERGE,
        1,
        datastore_pivotlake::DEFAULT_LAYOUT_CLIQUE_SIZE,
        None,
        Arc::clone(&datastore),
    );
    let actor = runtime.spawn(actor.run());
    let harness = Harness {
        runtime,
        datastore,
        compacter,
        name,
        id,
    };
    let table_directory = database.join(harness.find_table().location());
    let files = TableFiles {
        table_directory: table_directory.clone(),
        linked: &linked,
        arguments: &arguments,
    };

    // Every large file but the last commit's worth, which is committed with
    // the small files.
    let build_started = Instant::now();
    let mut next_number = 0;
    let mut files_per_commit = FIRST_COMMIT_FILES.min(arguments.files);
    let mut commits = 0;
    while arguments.files - next_number > files_per_commit || commits == 0 {
        let commit_files = files_per_commit.min(arguments.files - next_number);
        let entries: Vec<FileEntry> = (next_number..next_number + commit_files)
            .map(|number| files.link_large(number))
            .collect();
        harness.commit(&[], &entries);
        if commits == 0 {
            files_per_commit = match find_largest_manifest_bytes(&table_directory) {
                Some(manifest_bytes) => {
                    let bytes_per_file = manifest_bytes / commit_files as u64;
                    (MANIFEST_TARGET_BYTES / bytes_per_file.max(1)).max(1) as usize
                }
                None => FILES_PER_COMMIT_WITHOUT_MANIFESTS,
            };
        }
        next_number += commit_files;
        commits += 1;
    }
    let committed_files = next_number;
    let held_back = committed_files..arguments.files;
    println!(
        "table: {} format, {} large files of {:.1} KiB in {} partitions, {} of them in {} commits so far, the largest manifest {:.1} MiB, built in {:.1} s",
        arguments.format,
        arguments.files,
        linked.large_bytes as f64 / KIB as f64,
        arguments.partitions,
        next_number,
        commits,
        find_largest_manifest_bytes(&table_directory).unwrap_or(0) as f64 / MIB as f64,
        build_started.elapsed().as_secs_f64(),
    );
    println!("each time: the median (the least to the greatest)");

    if arguments.measures_held {
        let entries: Vec<FileEntry> = held_back.map(|number| files.link_large(number)).collect();
        harness.commit(&[], &entries);
        table_metadata_cache.remove_all_tables();
        let live_before_rounds = read_live_bytes();
        for _ in 0..4 {
            harness.run_round();
        }
        table_metadata_cache.remove_all_tables();
        let held_bytes = read_live_bytes().saturating_sub(live_before_rounds);
        println!(
            "held:                                {:.1} MiB, {:.0} bytes for each of {} files",
            held_bytes as f64 / MIB as f64,
            held_bytes as f64 / arguments.files as f64,
            arguments.files,
        );
        actor.abort();
        return;
    }

    let (first_round, _) = harness.run_round();
    println!(
        "first round:                         {:.3} ms",
        first_round.as_secs_f64() * 1000.0
    );

    let unchanged: Vec<Duration> = (0..arguments.repeats * 4)
        .map(|_| harness.run_round().0)
        .collect();
    println!(
        "unchanged round:                     {}",
        summarize(unchanged)
    );

    // Each rewrite replaces a large file of the commit after the first, a
    // manifest's worth of files when the table has that many, with a file
    // past every other.
    let first_replaced = if committed_files > FIRST_COMMIT_FILES + 2 * arguments.repeats {
        FIRST_COMMIT_FILES
    } else {
        0
    };
    let mut next_replacement = arguments.files;
    let mut replace_one_large_file = |position: usize| {
        let replaced_number = first_replaced + position;
        let replaced = ObjectPath::new(format!("large-{replaced_number:08}.parquet"));
        let replacement =
            files.link_large_into(next_replacement, replaced_number % arguments.partitions);
        next_replacement += 1;
        harness.commit(&[replaced], &[replacement]);
    };
    let after_rewrite: Vec<Duration> = (0..arguments.repeats)
        .map(|repeat| {
            replace_one_large_file(repeat);
            harness.run_round().0
        })
        .collect();
    println!(
        "round after a rewrite, cold:         {}",
        summarize(after_rewrite)
    );
    let after_rewrite_and_scan: Vec<Duration> = (0..arguments.repeats)
        .map(|repeat| {
            replace_one_large_file(arguments.repeats + repeat);
            harness.scan_past_every_file(&arguments);
            harness.run_round().0
        })
        .collect();
    println!(
        "round after a rewrite, after a scan: {}",
        summarize(after_rewrite_and_scan)
    );

    let mut entries: Vec<FileEntry> = held_back.map(|number| files.link_large(number)).collect();
    entries.extend((0..arguments.small_files).map(|number| files.link_small(number)));
    harness.commit(&[], &entries);
    let (merging_round, passes) = harness.run_round();
    let commits = passes - 1;
    println!(
        "merging round:                       {:.3} ms for {} commits of merges of {} small files, {:.3} ms each",
        merging_round.as_secs_f64() * 1000.0,
        commits,
        arguments.small_files,
        merging_round.as_secs_f64() * 1000.0 / commits.max(1) as f64,
    );
    let (settled_round, _) = harness.run_round();
    println!(
        "round after the merging round:       {:.3} ms",
        settled_round.as_secs_f64() * 1000.0
    );

    actor.abort();
}
