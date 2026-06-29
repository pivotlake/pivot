//! `ingest` runs the background **compacter** inside the pivotdb server: it
//! merges a table's small Parquet files into target-sized ones, so a table that
//! is written in many small appends (inserts, or any external writer) does not
//! accumulate a long tail of tiny files that slows every scan.
//!
//! The merge is one scan->encode dataflow on the dispatch pool, swapped in with a
//! single table-log commit. The compacter only watches the log, so the bundled
//! background task is a convenience: the same loop can run in a separate process
//! over the same database root.
//!
//! [`Ingestor`] is the lifecycle handle the server holds: [`Ingestor::start`]
//! launches the compacter; [`Ingestor::shutdown`] stops it, joining an in-flight
//! merge *before* the dispatch workers are torn down (the merge encodes on them).

mod compact;

// The Parquet write pipeline lives in `catalog` (beside the read pipeline) so the
// catalog can reuse it for INSERT; the compacter drives it to re-encode merges.
use catalog::parquet::writing as parquet_writing;

use std::sync::Arc;

use catalog::ParquetCatalog;
use tokio::sync::watch;
use tokio::task::JoinHandle;

pub use compact::{
    Compacter, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL, DEFAULT_MIN_FILES_TO_MERGE,
};

/// Lifecycle handle for the bundled background compacter.
pub struct Ingestor {
    shutdown_tx: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
}

impl Ingestor {
    /// Start the bundled, catalog-wide compacter. `compact_bytes` of `0` skips it
    /// (e.g. when a dedicated compacter process owns the job); `compact_min_files`
    /// is its count trigger for sub-target partitions.
    pub fn start(
        catalog: Arc<ParquetCatalog>,
        compact_bytes: u64,
        compact_min_files: usize,
    ) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        let mut tasks = Vec::new();
        if compact_bytes > 0 {
            let compacter = Arc::new(Compacter::new(
                compact_bytes,
                compact_min_files,
                DEFAULT_COMPACT_POLL,
                catalog,
            ));
            tasks.push(tokio::spawn(compacter.run(shutdown_tx.subscribe())));
        }
        Self { shutdown_tx, tasks }
    }

    /// Stop the compacter, joining an in-flight merge. Must be awaited *before*
    /// the dispatch workers shut down: the merge encodes on a worker via
    /// `run_on_worker`, which needs them alive.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        for task in self.tasks {
            let _ = task.await;
        }
    }

    /// Stop the compacter without joining. Used on the fatal path (a dispatch
    /// worker died), where awaiting a merge would hang on a dead worker.
    pub fn abort(self) {
        let _ = self.shutdown_tx.send(true);
        for task in self.tasks {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};
    use catalog::parquet::{ParquetTable, table_input};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch, Projection, values_input};
    use planner::catalog::{Catalog as _, Column, CreateTableRequest};
    use planner::types::Type;
    use std::collections::HashMap;
    use std::path::Path;

    const RING_BUFFERS: usize = 4 * 64 * 1024 * 1024 / BUFFER_SIZE;
    const NUM_COLUMNS: usize = 1;

    /// A one-column `(Timestamp Int64)` batch.
    fn timestamp_batch(values: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("Timestamp", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()
    }

    /// A `(ServiceName Utf8, Timestamp Int64)` batch, for the partitioned table.
    fn service_batch(service: &str, times: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ServiceName", DataType::Utf8View, false),
            Field::new("Timestamp", DataType::Int64, false),
        ]));
        let names = StringViewArray::from(vec![service; times.len()]);
        RecordBatch::try_new(schema, vec![Arc::new(names), Arc::new(Int64Array::from(times))])
            .unwrap()
    }

    /// `CREATE TABLE <name> (...)` against `catalog` the way the server would,
    /// with the given `columns` and `options` (path / partition_by / sort_by).
    fn create_table(
        catalog: &Arc<ParquetCatalog>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        columns: Vec<Column>,
        options: HashMap<String, String>,
    ) {
        let request = CreateTableRequest {
            name: name.to_string(),
            columns,
            options,
            if_not_exists: false,
        };
        catalog
            .create_table(request, dispatcher)
            .unwrap()
            .execute()
            .collect()
            .unwrap();
    }

    /// Write `batch` as one committed file in `table` through the same write path
    /// `INSERT` uses (encode on the pool, commit one file). Each call lands one
    /// small file, so repeated calls build the tiny-file backlog compaction merges.
    fn write_one_file(
        catalog: &Arc<ParquetCatalog>,
        dispatcher: &DataFlowDispatcher,
        table: &str,
        batch: RecordBatch,
    ) {
        let rows = values_input(dispatcher, [batch]).record_batches();
        catalog.insert(table.to_string(), rows, dispatcher).unwrap();
    }

    /// Refresh `name` to the latest committed manifest (a writer evolves a
    /// cloned-out handle, so the catalog's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(catalog: &ParquetCatalog, name: &str) -> Arc<ParquetTable> {
        let mut table = catalog.table_handle(name).expect("table exists");
        table.refresh().expect("manifest reload");
        table.parquet(&[]).expect("build scan view")
    }

    fn parquet_file_count(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_file())
            .count()
    }

    fn options_with(path: &Path, extra: &[(&str, &str)]) -> HashMap<String, String> {
        let mut options =
            HashMap::from([("path".to_string(), path.to_str().unwrap().to_string())]);
        for (key, value) in extra {
            options.insert(key.to_string(), value.to_string());
        }
        options
    }

    /// End-to-end compaction: several registered small files merge into one (a
    /// single scan->encode dataflow on the dispatch pool), the catalog swaps to
    /// the merged file in one log commit, and the merged file decodes back to all
    /// the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_catalog() {
        let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
        let columns = vec![Column { name: "Timestamp".into(), col_type: Type::Int64 }];
        create_table(&catalog, dispatch.dispatcher(), "events", columns, options_with(dir.path(), &[]));

        write_one_file(&catalog, dispatch.dispatcher(), "events", timestamp_batch(vec![1, 2, 3]));
        write_one_file(&catalog, dispatch.dispatcher(), "events", timestamp_batch(vec![4, 5]));
        write_one_file(&catalog, dispatch.dispatcher(), "events", timestamp_batch(vec![6, 7, 8, 9]));
        assert_eq!(parquet_file_count(dir.path()), 3);

        // Target = the three files' combined size: each is smaller than it
        // (candidate) and together they reach it (trigger).
        let total: u64 = catalog
            .table_files("events")
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(
            total,
            DEFAULT_MIN_FILES_TO_MERGE,
            std::time::Duration::from_secs(1),
            catalog.clone(),
        );
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // The merge is a 3->1 swap in the catalog; the inputs linger on disk as
        // deferred-deletion orphans until their version is pruned.
        assert_eq!(catalog.table_files("events").unwrap().len(), 1);
        let parquet = fresh_parquet(&catalog, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert!(
            catalog
                .table_files("events")
                .unwrap()
                .iter()
                .any(|f| f.path.as_str().contains("compacted"))
        );

        // The merged file's contents round-trip through the engine's scan. Read
        // via the live binding (the swapped-in merged file), not the directory,
        // which under deferred deletion still holds the un-pruned input orphans.
        let batches = table_input(dispatch.dispatcher(), &parquet, Projection::all(NUM_COLUMNS), false)
            .collect()
            .unwrap();
        let mut got: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                let a = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>()
            })
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        dispatch.exit();
    }

    /// Compacting a partitioned + sorted table preserves each merged file's
    /// partition tuple and recomputes its sort bounds: it merges within one
    /// partition and re-applies the table's spec rather than dropping the metadata.
    #[test]
    fn compaction_preserves_partition_and_sort_metadata() {
        let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
        let columns = vec![
            Column { name: "ServiceName".into(), col_type: Type::Utf8 },
            Column { name: "Timestamp".into(), col_type: Type::Int64 },
        ];
        create_table(
            &catalog,
            dispatch.dispatcher(),
            "events",
            columns,
            options_with(dir.path(), &[("partition_by", "ServiceName"), ("sort_by", "Timestamp")]),
        );

        // Three flushes, all service "svc" -> three files in the one partition.
        write_one_file(&catalog, dispatch.dispatcher(), "events", service_batch("svc", vec![3, 1, 2]));
        write_one_file(&catalog, dispatch.dispatcher(), "events", service_batch("svc", vec![5, 4]));
        write_one_file(&catalog, dispatch.dispatcher(), "events", service_batch("svc", vec![9, 6]));
        assert_eq!(parquet_file_count(dir.path()), 3);

        let total: u64 = catalog
            .table_files("events")
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(
            total,
            DEFAULT_MIN_FILES_TO_MERGE,
            std::time::Duration::from_secs(1),
            catalog.clone(),
        );
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // The three merged into one file that still carries the "svc" partition
        // tuple and has recomputed (non-empty) sort bounds.
        assert_eq!(catalog.table_files("events").unwrap().len(), 1);
        let mut table = catalog.table_handle("events").unwrap();
        table.refresh().unwrap();
        let partitions = table.file_partitions();
        assert_eq!(partitions.len(), 1);
        assert_eq!(
            partitions[0]
                .1
                .as_ref()
                .expect("merged file keeps a partition tuple")["ServiceName"],
            "svc"
        );
        assert!(
            table.file_sort_bounds()[0].1.is_some(),
            "merged file keeps recomputed sort bounds"
        );

        dispatch.exit();
    }

    /// Compaction is location-agnostic: a **store-relative** table (data under
    /// the database root, resolved through the store, the same path a remote
    /// `s3://` root takes) compacts through the exact same code, with writes and
    /// deletes going through the table's store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        use parquet::arrow::ArrowWriter;

        let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let table_dir = db.path().join("events");
        std::fs::create_dir_all(&table_dir).unwrap();

        // Two small files under the table's prefix inside the database root.
        let schema = Arc::new(Schema::new(vec![Field::new("Timestamp", DataType::Int64, false)]));
        for (name, values) in [("a.parquet", vec![1i64, 2]), ("b.parquet", vec![3i64])] {
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values)) as _])
                    .unwrap();
            let file = std::fs::File::create(table_dir.join(name)).unwrap();
            // SNAPPY, like every pivot-written file: the engine's decompressor
            // expects it.
            let props = parquet::file::properties::WriterProperties::builder()
                .set_compression(parquet::basic::Compression::SNAPPY)
                .build();
            let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let catalog = Arc::new(
            ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap(),
        );
        // No `path` option: the table lives at `events` under the root.
        create_table(
            &catalog,
            dispatch.dispatcher(),
            "events",
            vec![Column { name: "Timestamp".into(), col_type: Type::Int64 }],
            HashMap::new(),
        );

        let total: u64 = catalog
            .table_files("events")
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(
            total,
            DEFAULT_MIN_FILES_TO_MERGE,
            std::time::Duration::from_secs(1),
            catalog.clone(),
        );
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // One merged object replaced the two inputs, in the store and the catalog.
        let files = catalog.table_files("events").unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.as_str().contains("compacted"));
        let rows = fresh_parquet(&catalog, "events")
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows)
            .sum::<i64>();
        assert_eq!(rows, 3);

        dispatch.exit();
    }

    /// The compacter can live in a **separate process**: it holds nothing but a
    /// catalog handle, and each poll round reloads the table from its log. Here
    /// the "server" registers writes through one catalog instance while the
    /// compacter works through a second over the same database root, and the
    /// server sees the swap at its next bind.
    #[test]
    fn compacter_in_another_process_compacts_the_servers_writes() {
        let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();

        // "Server" process: creates the table and registers three writes.
        let server_catalog = Arc::new(
            ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap(),
        );
        create_table(
            &server_catalog,
            dispatch.dispatcher(),
            "events",
            vec![Column { name: "Timestamp".into(), col_type: Type::Int64 }],
            options_with(data_dir.path(), &[]),
        );
        write_one_file(&server_catalog, dispatch.dispatcher(), "events", timestamp_batch(vec![1, 2, 3]));
        write_one_file(&server_catalog, dispatch.dispatcher(), "events", timestamp_batch(vec![4, 5]));
        write_one_file(&server_catalog, dispatch.dispatcher(), "events", timestamp_batch(vec![6, 7, 8, 9]));

        // "Compacter" process: a separate catalog over the same root. Its poll
        // round reloads the table from the log before scanning.
        let compacter_catalog = Arc::new(
            ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap(),
        );
        let total: u64 = server_catalog
            .table_files("events")
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(
            total,
            DEFAULT_MIN_FILES_TO_MERGE,
            std::time::Duration::from_secs(1),
            compacter_catalog,
        );
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // The server's next query reloads to the compacted version.
        let parquet = fresh_parquet(&server_catalog, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert!(
            server_catalog
                .table_files("events")
                .unwrap()
                .iter()
                .any(|f| f.path.as_str().contains("compacted"))
        );
        assert_eq!(server_catalog.table_files("events").unwrap().len(), 1);

        dispatch.exit();
    }
}
