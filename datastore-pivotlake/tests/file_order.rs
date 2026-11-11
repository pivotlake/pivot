//! The order a sorted table holds a partition's files in, whichever format
//! keeps the table: by where each file starts on the sort key, read off the
//! bounds the table's metadata records for the file.

mod iceberg_support;

use datastore_pivotlake::TableFormat;
use iceberg_support::Database;
use object_storage::ObjectPath;
use planner::catalog::SchemaQualifiedTableName;

const FORMATS: [TableFormat; 2] = [TableFormat::Delta, TableFormat::Iceberg];

/// The files of the table `t`, in the order the table holds them.
fn list_files(database: &Database) -> Vec<ObjectPath> {
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let files = database.datastore.table_files(&name).unwrap().unwrap();
    files.into_iter().map(|file| file.path).collect()
}

/// Run each of `inserts`, which writes one file, and name the file each
/// wrote, in the order given.
fn insert_each(database: &mut Database, inserts: &[&str]) -> Vec<ObjectPath> {
    let mut held = list_files(database);
    let mut written = Vec::new();
    for insert in inserts {
        database.execute(insert);
        let files = list_files(database);
        let added = files.iter().find(|path| !held.contains(path)).unwrap();
        written.push(added.clone());
        held = files;
    }
    written
}

/// Which of `written` each file of the table `t` is, in the order the
/// table holds them.
fn order_written(database: &Database, written: &[ObjectPath]) -> Vec<usize> {
    list_files(database)
        .iter()
        .map(|path| written.iter().position(|file| file == path).unwrap())
        .collect()
}

/// The order a table created by `create` in a database of `format` holds
/// the files of `inserts` in, each named by the position of its insert.
fn order_inserts(format: TableFormat, create: &str, inserts: &[&str]) -> Vec<usize> {
    let mut database = Database::in_format(format);
    database.execute(create);
    let written = insert_each(&mut database, inserts);
    order_written(&database, &written)
}

#[test]
fn a_file_starting_at_a_prefix_stands_before_the_file_starting_at_what_extends_it() {
    let create = "CREATE TABLE t (name VARCHAR) SORTED BY (name)";
    let inserts = [
        "INSERT INTO t VALUES ('ab\u{e9}')",
        "INSERT INTO t VALUES ('ab')",
        "INSERT INTO t VALUES ('aa\u{65e5}')",
    ];

    let orders = FORMATS.map(|format| order_inserts(format, create, &inserts));

    assert_eq!(orders, [[2, 1, 0], [2, 1, 0]]);
}

#[test]
fn files_tied_on_the_first_sort_column_stand_by_the_second() {
    let create = "CREATE TABLE t (id BIGINT, name VARCHAR) SORTED BY (id, name)";
    let inserts = [
        "INSERT INTO t VALUES (7, 'b')",
        "INSERT INTO t VALUES (7, 'a')",
        "INSERT INTO t VALUES (3, 'z')",
    ];

    let orders = FORMATS.map(|format| order_inserts(format, create, &inserts));

    assert_eq!(orders, [[2, 1, 0], [2, 1, 0]]);
}

#[test]
fn a_file_without_a_bound_on_the_sort_column_stands_before_the_files_with_one() {
    let create = "CREATE TABLE t (id BIGINT) SORTED BY (id)";
    let inserts = [
        "INSERT INTO t VALUES (5)",
        "INSERT INTO t VALUES (NULL)",
        "INSERT INTO t VALUES (-9000000000000000000)",
    ];

    let orders = FORMATS.map(|format| order_inserts(format, create, &inserts));

    assert_eq!(orders, [[1, 2, 0], [1, 2, 0]]);
}

#[test]
fn files_starting_at_negative_floats_stand_before_those_starting_at_positive_ones() {
    let create = "CREATE TABLE t (reading DOUBLE) SORTED BY (reading)";
    let inserts = [
        "INSERT INTO t VALUES (0.25)",
        "INSERT INTO t VALUES (-1e300)",
        "INSERT INTO t VALUES (0.0)",
        "INSERT INTO t VALUES (-0.5)",
    ];

    let orders = FORMATS.map(|format| order_inserts(format, create, &inserts));

    assert_eq!(orders, [[1, 3, 2, 0], [1, 3, 2, 0]]);
}

#[test]
fn a_file_whose_bounds_were_read_back_from_the_tables_metadata_stands_by_value_among_new_ones() {
    let orders = FORMATS.map(|format| {
        let mut database = Database::in_format(format);
        database.execute("CREATE TABLE t (count UINTEGER) SORTED BY (count)");
        let mut written = insert_each(&mut database, &["INSERT INTO t VALUES (5)"]);
        let mut reopened = Database::reopen_in_format(database.close(), format);

        let inserts = [
            "INSERT INTO t VALUES (4000000000)",
            "INSERT INTO t VALUES (3)",
        ];
        written.extend(insert_each(&mut reopened, &inserts));

        order_written(&reopened, &written)
    });

    assert_eq!(orders, [[2, 0, 1], [2, 0, 1]]);
}
