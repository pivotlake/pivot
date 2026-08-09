//! Blackbox integration tests for the object-store backends, local filesystem,
//! S3 (MinIO) and GCS (`fake-gcs-server`), each a short Setup / Execute / Assert
//! against the public datastore + store API.
//!
//! Every behaviour is written once (in [`bodies`]) over a `&Backend` and run on
//! each backend by the [`backend_tests!`] macro. The local case always runs; the
//! S3/GCS cases bring up containers via [`harness`] and skip when Docker is
//! absent, so `cargo test` stays green offline.
//!
//! Covered: store contract (round-trip through `source` and `sink`, one-level
//! `list`, the `put_if_absent` CAS), and the table lifecycle end to end — `CREATE TABLE`,
//! reopen, **appending a file** (out-of-band registration), and **compaction**
//! (replacing files) — all over object storage.

mod common;

use datastore_delta::test_support as harness;

use std::collections::HashMap;
use std::sync::Arc;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use common::{
    DispatchGuard, collect_i64s, commit_datastore_transaction, current_parquet,
    dispatch_with_buffers, strings_and_ints,
};
use datastore::DatastoreTransaction;
use datastore_delta::parquet::table_input;
use datastore_delta::store::ObjectPath;
use datastore_delta::{DeltaDatastore, FileRef};
use dispatch::Projection;
use harness::Backend;
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};
use planner::types::Type;

// --- helpers (not tests) ---------------------------------------------------

/// (name Utf8, value Int64) — matches [`strings_and_ints`].
fn columns() -> Vec<Column> {
    vec![
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
        Column {
            name: "value".to_string(),
            col_type: Type::Int64,
        },
    ]
}

/// `CREATE TABLE <name> (cols) WITH (path = '<path>')`, `path` store-relative.
fn path_request(name: &str, path: &str) -> CreateTableRequest {
    CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns: columns(),
        options: HashMap::from([("path".to_string(), path.to_string())]),
        if_not_exists: false,
    }
}

/// Snappy Parquet bytes for rows carrying the given `value`s (names are filler).
fn pq(values: &[i64]) -> Vec<u8> {
    let names: Vec<&str> = values.iter().map(|_| "x").collect();
    let batch = strings_and_ints(&names, values);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buf
}

/// Put `files` (name → values) under `events/` and `CREATE TABLE events` over them.
fn create_events(d: &DispatchGuard, b: &Backend, files: &[(&str, &[i64])]) -> Arc<DeltaDatastore> {
    for (name, values) in files {
        b.store
            .put(&ObjectPath::new(format!("events/{name}")), &pq(values))
            .unwrap();
    }
    let datastore = DeltaDatastore::open(&b.root, d).unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(path_request("events", "events"))
        .unwrap()
        .compile(d)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    datastore
}

/// Scan the table's `value` column, sorted.
fn scan(d: &DispatchGuard, datastore: &DeltaDatastore, name: &str) -> Vec<i64> {
    let parquet = current_parquet(datastore, name);
    let out = table_input(d, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    let mut values = collect_i64s(&out, 1);
    values.sort();
    values
}

fn row_groups(datastore: &DeltaDatastore, name: &str) -> usize {
    current_parquet(datastore, name).row_groups().len()
}

// --- behaviours (run on every backend) -------------------------------------

mod bodies {
    use super::*;

    /// `CREATE TABLE` over a Parquet file in the store; the rows scan back.
    pub fn create_and_scan(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        b.store
            .put(&ObjectPath::new("events/p1.parquet"), &pq(&[1, 2, 3]))
            .unwrap();
        let datastore = DeltaDatastore::open(&b.root, &d).unwrap();

        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(path_request("events", "events"))
            .unwrap()
            .compile(&d)
            .unwrap()
            .execute()
            .collect()
            .unwrap();
        commit_datastore_transaction(transaction).unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![1, 2, 3]);
    }

    /// The table and its data survive reopening the datastore, a server restart.
    pub fn survives_reopen(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        drop(create_events(&d, b, &[("p1.parquet", &[1, 2, 3])]));

        let reopened = DeltaDatastore::open(&b.root, &d).unwrap();

        assert_eq!(scan(&d, &reopened, "events"), vec![1, 2, 3]);
    }

    /// Registering a new file (an out-of-band append) makes its rows visible to the
    /// next bind, on top of the existing ones.
    pub fn append_registers_new_file(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        let datastore = create_events(&d, b, &[("p1.parquet", &[1, 2, 3])]);

        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .append_data_file(ObjectPath::new("p2.parquet"), &pq(&[4, 5, 6]), None, None)
            .unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![1, 2, 3, 4, 5, 6]);
    }

    /// Compaction swaps the small files for one merged file in a single version:
    /// the rows are unchanged but now live in one row group.
    pub fn compaction_replaces_files(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        let datastore = create_events(&d, b, &[("p1.parquet", &[1, 2, 3]), ("p2.parquet", &[4])]);
        let merged = pq(&[1, 2, 3, 4]);
        b.store
            .put(&ObjectPath::new("events/merged.parquet"), &merged)
            .unwrap();

        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .replace_data_files(
                &[ObjectPath::new("p1.parquet"), ObjectPath::new("p2.parquet")],
                &[datastore_delta::DeltaFileEntry::new(FileRef {
                    path: ObjectPath::new("merged.parquet"),
                    size: merged.len() as u64,
                })],
            )
            .unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![1, 2, 3, 4]);
        assert_eq!(row_groups(&datastore, "events"), 1);
    }

    /// A stored object reads back through `source` — the read source the ring is
    /// handed (a presigned S3 URL, a GCS media URL, or a local path).
    pub fn source_reads_object_back(b: &Backend) {
        b.store
            .put(&ObjectPath::new("s/o.bin"), b"payload")
            .unwrap();

        let bytes = harness::read_via_source(b.store.as_ref(), &ObjectPath::new("s/o.bin"));

        assert_eq!(bytes, b"payload");
    }

    /// An object written through `sink` — the upload destination the ring is
    /// handed — lands under the key the store was asked for, and reads back
    /// through the ordinary `get`.
    pub fn sink_writes_object_back(b: &Backend) {
        b.store.create_dir(&ObjectPath::new("w")).unwrap();

        harness::write_via_sink(b.store.as_ref(), &ObjectPath::new("w/o.bin"), b"uploaded");

        assert_eq!(
            b.store.get(&ObjectPath::new("w/o.bin")).unwrap().unwrap(),
            b"uploaded"
        );
    }

    /// `list` is one level only: a nested object is not returned.
    pub fn list_is_one_level(b: &Backend) {
        b.store.put(&ObjectPath::new("d/x.bin"), b"abc").unwrap();
        b.store.put(&ObjectPath::new("d/sub/y.bin"), b"z").unwrap();

        let names: Vec<String> = b
            .store
            .list(&ObjectPath::new("d"))
            .unwrap()
            .into_iter()
            .map(|o| o.file.path.as_str().to_string())
            .collect();

        assert_eq!(names, vec!["x.bin".to_string()]);
    }

    /// `put_if_absent` is a CAS: the second writer loses and the first's bytes
    /// stay — the primitive the manifest commit is built on.
    pub fn put_if_absent_is_a_cas(b: &Backend) {
        b.store
            .put_if_absent(&ObjectPath::new("cas.bin"), b"first")
            .unwrap();

        let won = b
            .store
            .put_if_absent(&ObjectPath::new("cas.bin"), b"second")
            .unwrap();

        assert!(!won);
        assert_eq!(
            b.store.get(&ObjectPath::new("cas.bin")).unwrap().unwrap(),
            b"first"
        );
    }
}

// --- backend matrix --------------------------------------------------------

/// Emit `local` / `s3` / `gcs` tests for a [`bodies`] behaviour. S3/GCS skip
/// when Docker is absent; each gets its own bucket prefix for isolation.
macro_rules! backend_tests {
    ($name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn local() {
                let (_dir, b) = harness::local();
                bodies::$name(&b);
            }
            #[test]
            fn s3() {
                if let Some(b) = harness::s3(concat!(stringify!($name), "-s3")) {
                    bodies::$name(&b);
                }
            }
            #[test]
            fn gcs() {
                if let Some(b) = harness::gcs(concat!(stringify!($name), "-gcs")) {
                    bodies::$name(&b);
                }
            }
        }
    };
}

backend_tests!(create_and_scan);
backend_tests!(survives_reopen);
backend_tests!(append_registers_new_file);
backend_tests!(compaction_replaces_files);
backend_tests!(source_reads_object_back);
backend_tests!(sink_writes_object_back);
backend_tests!(list_is_one_level);

/// CAS-conflict tests, for backends that enforce the precondition. The
/// `fake-gcs-server` emulator ignores `ifGenerationMatch=0`, so GCS is excluded
/// (real GCS enforces it — the gap is the emulator's, not the datastore's).
macro_rules! local_s3_tests {
    ($name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn local() {
                let (_dir, b) = harness::local();
                bodies::$name(&b);
            }
            #[test]
            fn s3() {
                if let Some(b) = harness::s3(concat!(stringify!($name), "-s3")) {
                    bodies::$name(&b);
                }
            }
        }
    };
}

local_s3_tests!(put_if_absent_is_a_cas);
