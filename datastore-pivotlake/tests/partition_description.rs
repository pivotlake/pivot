//! A file's partition as the catalog describes it: the same text whichever
//! format the table is kept in.

mod iceberg_support;

use std::sync::Arc;

use catalog::datastore::DatastoreTransaction as _;
use datastore_pivotlake::TableFormat;
use iceberg_support::Database;

/// The partition of every file of a table kept in `format` and partitioned
/// by `region`, holding a region with a space and a slash in it and a NULL
/// one, as the datastore describes the files, sorted.
fn describe_partitions(format: TableFormat) -> Vec<String> {
    let mut database = Database::in_format(format);
    database.execute("CREATE TABLE t (id BIGINT, region VARCHAR) WITH (partition_by = 'region')");
    database.execute("INSERT INTO t VALUES (1, 'eu west/1'), (2, NULL)");
    let transaction = Arc::clone(&database.datastore).begin_transaction();
    let tables = transaction.tables().unwrap();
    let mut partitions: Vec<String> = tables[0]
        .files
        .iter()
        .map(|file| file.partition.clone())
        .collect();
    partitions.sort();
    partitions
}

#[test]
fn a_files_partition_is_described_alike_in_a_delta_and_an_iceberg_table() {
    let delta = describe_partitions(TableFormat::Delta);

    let iceberg = describe_partitions(TableFormat::Iceberg);

    assert_eq!(delta, ["region=", "region=eu west/1"]);
    assert_eq!(iceberg, delta);
}
