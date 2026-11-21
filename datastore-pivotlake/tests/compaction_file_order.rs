//! The order maintenance holds a partition's files in, whichever format
//! keeps the table: by where each file starts on the table's sort key, read
//! off the bounds the table's metadata records for the file, then by its
//! commit. No format lists its files in that order, so the files are
//! ordered as the compacter orders the records it takes of them.

mod iceberg_support;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use datastore_pivotlake::{CompacterHandle, FileRef, TableFormat};
use iceberg_support::Database;
use object_storage::ObjectPath;
use planner::catalog::SchemaQualifiedTableName;

const FORMATS: [TableFormat; 2] = [TableFormat::Delta, TableFormat::Iceberg];

/// The files of the table `t`, in the order the table lists them.
fn list_files(database: &Database) -> Vec<FileRef> {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    database.datastore.table_files(&name).unwrap().unwrap()
}

/// The files of the table `t`, in the order maintenance holds them.
fn list_files_in_order(database: &Database) -> Vec<ObjectPath> {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let files = database.datastore.table_files_in_maintenance_order(&name);
    let files = files.unwrap().unwrap();
    files.into_iter().map(|file| file.path).collect()
}

/// Run each of `inserts`, which writes one file, and name the file each
/// wrote, in the order given.
fn insert_each(database: &mut Database, inserts: &[&str]) -> Vec<ObjectPath> {
    let mut held = list_files_in_order(database);
    let mut written = Vec::new();
    for insert in inserts {
        database.execute(insert);
        let files = list_files_in_order(database);
        let added = files.iter().find(|path| !held.contains(path)).unwrap();
        written.push(added.clone());
        held = files;
    }
    written
}

/// Which of `written` each file of the table `t` is, in the order
/// maintenance holds them.
fn order_written(database: &Database, written: &[ObjectPath]) -> Vec<usize> {
    list_files_in_order(database)
        .iter()
        .map(|path| written.iter().position(|file| file == path).unwrap())
        .collect()
}

/// The order maintenance holds the files of `inserts` in, in a table kept
/// in `format` and created by `create`, each named by the position of its
/// insert.
fn order_inserts(format: TableFormat, create: &str, inserts: &[&str]) -> Vec<usize> {
    let mut database = Database::in_format(format);
    database.execute(create);
    let written = insert_each(&mut database, inserts);
    order_written(&database, &written)
}

#[test]
fn files_stand_by_where_they_start_on_the_sort_key() {
    for format in FORMATS {
        let create = "CREATE TABLE t (id BIGINT) SORTED BY (id)";
        let inserts = ["INSERT INTO t VALUES (9)", "INSERT INTO t VALUES (2)"];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [1, 0], "{format}");
    }
}

#[test]
fn an_unsorted_tables_files_stand_in_the_order_they_were_committed_in() {
    for format in FORMATS {
        let create = "CREATE TABLE t (id BIGINT)";
        let inserts = [
            "INSERT INTO t VALUES (9)",
            "INSERT INTO t VALUES (2)",
            "INSERT INTO t VALUES (5)",
            "INSERT INTO t VALUES (1)",
            "INSERT INTO t VALUES (7)",
        ];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [0, 1, 2, 3, 4], "{format}");
    }
}

#[test]
fn a_file_of_false_stands_before_a_file_of_true() {
    for format in FORMATS {
        let create = "CREATE TABLE t (flag BOOLEAN) SORTED BY (flag)";
        let inserts = [
            "INSERT INTO t VALUES (true)",
            "INSERT INTO t VALUES (false)",
        ];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [1, 0], "{format}");
    }
}

#[test]
fn a_file_starting_at_a_prefix_stands_before_the_file_starting_at_what_extends_it() {
    for format in FORMATS {
        let create = "CREATE TABLE t (name VARCHAR) SORTED BY (name)";
        let inserts = [
            "INSERT INTO t VALUES ('ab\u{e9}')",
            "INSERT INTO t VALUES ('ab')",
            "INSERT INTO t VALUES ('aa\u{65e5}')",
        ];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [2, 1, 0], "{format}");
    }
}

#[test]
fn files_tied_on_the_first_sort_column_stand_by_the_second() {
    for format in FORMATS {
        let create = "CREATE TABLE t (id BIGINT, name VARCHAR) SORTED BY (id, name)";
        let inserts = [
            "INSERT INTO t VALUES (7, 'b')",
            "INSERT INTO t VALUES (7, 'a')",
            "INSERT INTO t VALUES (3, 'z')",
        ];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [2, 1, 0], "{format}");
    }
}

#[test]
fn a_file_without_a_bound_on_the_sort_column_stands_before_the_files_with_one() {
    for format in FORMATS {
        let create = "CREATE TABLE t (id BIGINT) SORTED BY (id)";
        let inserts = [
            "INSERT INTO t VALUES (5)",
            "INSERT INTO t VALUES (NULL)",
            "INSERT INTO t VALUES (-9000000000000000000)",
        ];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [1, 2, 0], "{format}");
    }
}

#[test]
fn files_starting_at_negative_floats_stand_before_those_starting_at_positive_ones() {
    for format in FORMATS {
        let create = "CREATE TABLE t (reading DOUBLE) SORTED BY (reading)";
        let inserts = [
            "INSERT INTO t VALUES (0.25)",
            "INSERT INTO t VALUES (-1e300)",
            "INSERT INTO t VALUES (0.0)",
            "INSERT INTO t VALUES (-0.5)",
        ];

        let order = order_inserts(format, create, &inserts);

        assert_eq!(order, [1, 3, 2, 0], "{format}");
    }
}

#[test]
fn a_file_whose_unsigned_bounds_were_read_back_stands_by_value_among_new_ones() {
    for format in FORMATS {
        let mut database = Database::in_format(format);
        database.execute("CREATE TABLE t (count UINTEGER) SORTED BY (count)");
        let mut written = insert_each(&mut database, &["INSERT INTO t VALUES (5)"]);
        let mut reopened = Database::reopen_in_format(database.close(), format);
        let inserts = [
            "INSERT INTO t VALUES (4000000000)",
            "INSERT INTO t VALUES (3)",
        ];

        written.extend(insert_each(&mut reopened, &inserts));

        assert_eq!(order_written(&reopened, &written), [2, 0, 1], "{format}");
    }
}

/// A payload of `length` characters no compression shortens much, its own
/// for each `seed`.
fn build_payload(seed: u64, length: usize) -> String {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut payload = String::with_capacity(length);
    while payload.len() < length {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        payload.push_str(&format!("{state:016x}"));
    }
    payload.truncate(length);
    payload
}

/// Run one routine compaction round over the table `t` of `database` in
/// which any two small files are a merge, and a file larger than the
/// largest the table holds now is small no longer.
fn merge_small_files_in_pairs(database: &Database) {
    let largest_file = list_files(database).iter().map(|file| file.size).max();
    let (compacter, actor) = CompacterHandle::new(
        2 * largest_file.unwrap() + 1,
        1,
        2,
        1,
        datastore_pivotlake::DEFAULT_LAYOUT_CLIQUE_SIZE,
        None,
        Arc::clone(&database.datastore),
    );
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let id = database.datastore.table_handle(&name).unwrap().id();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let running = runtime.spawn(actor.run());
    runtime
        .block_on(compacter.compact(name, id, false))
        .unwrap();
    running.abort();
}

#[test]
fn a_small_file_merge_takes_the_files_next_to_each_other_on_the_sort_key() {
    for format in FORMATS {
        let mut database = Database::in_format(format);
        database.execute("CREATE TABLE t (id BIGINT, payload VARCHAR) SORTED BY (id)");
        for id in [30, 10, 50, 20, 60, 40] {
            let payload = build_payload(id, 16 * 1024);
            database.execute(&format!("INSERT INTO t VALUES ({id}, '{payload}')"));
        }

        merge_small_files_in_pairs(&database);

        let name = SchemaQualifiedTableName::in_default_schema("t");
        let mut table = database.datastore.table_handle(&name).unwrap();
        table.refresh().unwrap();
        let entries = table.live_entries().unwrap();
        let mut id_ranges: Vec<(i64, i64)> = entries
            .iter()
            .map(|entry| {
                let stats = entry.stats.as_ref().unwrap();
                let read_id =
                    |bound: &arrow_array::ArrayRef| bound.as_primitive::<Int64Type>().value(0);
                (
                    read_id(&stats.min_values["id"]),
                    read_id(&stats.max_values["id"]),
                )
            })
            .collect();
        id_ranges.sort_unstable();
        assert_eq!(id_ranges, [(10, 20), (30, 40), (50, 60)], "{format}");
    }
}
