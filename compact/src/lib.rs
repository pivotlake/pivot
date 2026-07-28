//! `compact` merges a table's small Parquet files into target-sized ones,
//! running *inside* the pivotdb server (or as a separate process over the same
//! database root).
//!
//! Frequent writes keep a table's tail full of small files, and every scan pays
//! per file. The [`Compacter`] watches each table's log and, whenever a
//! partition's sub-target files add up to one target-sized output, re-encodes
//! them into a single file and swaps it in with one log commit. The CPU-heavy
//! part -- decoding the inputs and re-encoding the merged file -- runs on the
//! [`dispatch`] worker pool (the same thread-per-core pool that executes
//! queries), not the async runtime.
//!
//! [`Compaction`] is the lifecycle handle the server holds:
//! [`Compaction::start`] launches the background loop and
//! [`Compaction::shutdown`] stops it, letting an in-flight merge finish
//! **before** the dispatch workers are torn down (the merge encodes on them).

mod compact;

use std::sync::Arc;

use catalog::ParquetCatalog;
use tokio::sync::watch;
use tokio::task::JoinHandle;

pub use compact::{
    CompactSnapshot, CompactStatsSnapshot, Compacter, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL,
    DEFAULT_MIN_FILES_TO_MERGE,
};

/// Lifecycle handle for the background compaction loop the server runs.
///
/// [`start`](Self::start) spawns the loop (unless it is disabled);
/// [`shutdown`](Self::shutdown) stops it after any in-flight merge finishes --
/// that merge encodes on the dispatch workers, so it must complete before they
/// are torn down.
pub struct Compaction {
    shutdown_tx: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
    stats: CompactionStats,
}

/// A cheap, cloneable handle for reading live compaction counters (held by the
/// server's introspection API).
#[derive(Clone, Default)]
pub struct CompactionStats {
    compacter: Option<Arc<Compacter>>,
}

impl CompactionStats {
    /// Current per-table compaction counters, or `None` if no compacter runs.
    pub fn compaction(&self) -> Option<CompactSnapshot> {
        self.compacter
            .as_ref()
            .map(|compacter| compacter.snapshot())
    }
}

impl Compaction {
    /// Start the background compacter over `catalog`. `compact_bytes` sizes the
    /// target output and the candidate threshold (`0` disables compaction --
    /// e.g. when a dedicated compacter process owns the job); `compact_min_files`
    /// is the count trigger for a sub-target partition's pile.
    pub fn start(
        catalog: Arc<ParquetCatalog>,
        compact_bytes: u64,
        compact_min_files: usize,
    ) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        let (task, compacter) = if compact_bytes > 0 {
            let compacter = Arc::new(Compacter::new(
                compact_bytes,
                compact_min_files,
                DEFAULT_COMPACT_POLL,
                catalog,
            ));
            let task = tokio::spawn(compacter.clone().run(shutdown_tx.subscribe()));
            (Some(task), Some(compacter))
        } else {
            (None, None)
        };
        Self {
            shutdown_tx,
            task,
            stats: CompactionStats { compacter },
        }
    }

    /// A cloneable handle for reading live compaction counters.
    pub fn stats(&self) -> CompactionStats {
        self.stats.clone()
    }

    /// Stop the compaction loop, waiting for an in-flight merge to finish.
    ///
    /// Must be awaited **before** the dispatch workers shut down: a merge encodes
    /// on a worker via the write pipeline, so it must complete first.
    pub async fn shutdown(mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }

    /// Stop the loop without waiting for the in-flight merge. Used on the fatal
    /// path (a dispatch worker died), where waiting on a merge would hang on the
    /// dead worker.
    pub fn abort(mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use catalog::parquet::ParquetTable;
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
    use parquet::arrow::ArrowWriter;
    use std::path::Path;

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

    /// Write `values` as a small SNAPPY Parquet file (single Int64 `Timestamp`
    /// column) into `dir` under `file_name`. A table created over `dir` picks it
    /// up as a small input for the compacter to merge.
    fn write_parquet_file(dir: &Path, file_name: &str, values: Vec<i64>) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "Timestamp",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values)) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        // SNAPPY, like every pivot-written file -- the decompressor expects it.
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// `CREATE TABLE <name> (Timestamp Int64) WITH (path = dir)` against
    /// `catalog`, the way the server would run it -- or, when `dir` is `None`, at
    /// `<name>` under the database root (a store-relative table).
    fn create_table(
        catalog: &Arc<catalog::ParquetCatalog>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: Option<&Path>,
    ) {
        use planner::catalog::{Catalog as _, Column, CreateTableRequest};
        let options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "path".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        let request = CreateTableRequest {
            name: name.to_string(),
            columns: vec![Column::new("Timestamp", planner::types::Type::Int64)],
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

    /// Refresh `name` to the latest committed manifest (a writer evolves a
    /// cloned-out handle, so the catalog's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(catalog: &catalog::ParquetCatalog, name: &str) -> Arc<ParquetTable> {
        let mut table = catalog.table_handle(name).expect("table exists");
        table.refresh().expect("manifest reload");
        table.build_scan_view(&[]).expect("build scan view")
    }

    /// Drive one full compaction sweep to completion on a temporary runtime.
    fn run_one_sweep(compacter: &Compacter) {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());
    }

    /// End-to-end compaction: several registered small files merge into one, the
    /// catalog swaps to the merged file in one log commit, and it decodes back to
    /// all the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_catalog() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let table_dir = db.path().join("events");
        std::fs::create_dir_all(&table_dir).unwrap();
        write_parquet_file(&table_dir, "a.parquet", vec![1, 2, 3]);
        write_parquet_file(&table_dir, "b.parquet", vec![4, 5]);
        write_parquet_file(&table_dir, "c.parquet", vec![6, 7, 8, 9]);

        let catalog = Arc::new(
            catalog::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        create_table(&catalog, dispatch.dispatcher(), "events", None);
        assert_eq!(catalog.table_files("events").unwrap().len(), 3);

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
        run_one_sweep(&compacter);

        assert_eq!(catalog.table_files("events").unwrap().len(), 1);
        let parquet = fresh_parquet(&catalog, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert_eq!(parquet.row_groups()[0].num_rows, 9);
        assert!(
            catalog
                .table_files("events")
                .unwrap()
                .iter()
                .all(|f| f.path.as_str().starts_with("pivot-"))
        );

        dispatch.exit();
    }

    /// Compaction is location-agnostic: a **store-relative** table (data under
    /// the database root, resolved through the store -- the same path a remote
    /// `s3://` root takes) compacts through the exact same code, with writes and
    /// deletes going through the table's store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let table_dir = db.path().join("events");
        std::fs::create_dir_all(&table_dir).unwrap();

        // Two small files under the table's prefix inside the database root.
        write_parquet_file(&table_dir, "a.parquet", vec![1, 2]);
        write_parquet_file(&table_dir, "b.parquet", vec![3]);

        let catalog = Arc::new(
            catalog::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        // No `path` option: the table lives at `events` under the root.
        create_table(&catalog, dispatch.dispatcher(), "events", None);

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
        run_one_sweep(&compacter);

        // One merged object replaced the two inputs, in the store and in the
        // catalog.
        let files = catalog.table_files("events").unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.as_str().starts_with("pivot-"));
        let on_disk: Vec<_> = std::fs::read_dir(&table_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        // Deferred deletion: the merged object is written to the store; the two
        // inputs remain as orphans until their version is pruned.
        assert!(on_disk.contains(&files[0].path.as_str().to_string()));

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
    /// one catalog instance registers files while the compacter works through a
    /// second instance over the same database root -- and the first sees the swap
    /// at its next bind.
    #[test]
    fn compacter_in_another_process_compacts_registered_files() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let table_dir = db.path().join("events");
        std::fs::create_dir_all(&table_dir).unwrap();
        write_parquet_file(&table_dir, "a.parquet", vec![1, 2, 3]);
        write_parquet_file(&table_dir, "b.parquet", vec![4, 5]);
        write_parquet_file(&table_dir, "c.parquet", vec![6, 7, 8, 9]);

        // "Writer" process: creates the table over the pre-written files.
        let writer_catalog = Arc::new(
            catalog::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        create_table(&writer_catalog, dispatch.dispatcher(), "events", None);

        // "Compacter" process: a separate catalog over the same root. Its poll
        // round reloads the table from the log before scanning.
        let compacter_catalog = Arc::new(
            catalog::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        let total: u64 = writer_catalog
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
        run_one_sweep(&compacter);

        // The writer's next query reloads to the compacted version.
        let parquet = fresh_parquet(&writer_catalog, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert_eq!(parquet.row_groups()[0].num_rows, 9);
        assert!(
            writer_catalog
                .table_files("events")
                .unwrap()
                .iter()
                .all(|f| f.path.as_str().starts_with("pivot-"))
        );
        assert_eq!(writer_catalog.table_files("events").unwrap().len(), 1);

        dispatch.exit();
    }
}
