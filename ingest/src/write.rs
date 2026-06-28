//! The shared encode-and-append write path: turn a flush's [`ToRecordBatch`]
//! items into Parquet files and append each to a catalog table.
//!
//! Both ingest sources drive it; the difference is only what they do with the
//! returned [`Result`]:
//!
//! - The OTLP [`ParquetSink`](crate::sink::ParquetSink) ignores it - the
//!   drop-under-pressure stance OTLP exporters expect.
//! - The Kafka consumer gates its **offset commit** on it: offsets advance only
//!   when the whole flush committed, so a failed flush is reprocessed on restart
//!   (at-least-once).
//!
//! The CPU-heavy encode runs on the dispatch worker pool (via
//! [`parquet_writing::encode_items`]); the store write + manifest commit run on a
//! blocking task, never on a pinned worker.

use std::sync::Arc;

use catalog::ParquetCatalog;
use catalog::store::ObjectPath;
use dispatch::{DataFlowDispatcher, DataFlowError};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::parquet_writing::{self, EncodedFile, ToRecordBatch};
use crate::sink::{ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE};

/// Finished files buffered between the encode pipeline and the writer: small, so
/// it bounds in-flight files and backpressures the pipeline onto the writer.
const IN_FLIGHT_FILES: usize = 4;

/// A flush failed to fully land. The Kafka path turns this into "don't commit
/// offsets"; the OTLP path logs and drops it.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WriteError {
    #[error("table `{0}` no longer exists")]
    MissingTable(String),
    #[error("parquet encode pipeline failed: {0}")]
    Pipeline(#[from] DataFlowError),
    #[error("appending file to table failed: {0}")]
    Append(#[from] catalog::Error),
    #[error("write task panicked")]
    Panicked,
}

/// Encode `items` into Parquet files and append each to catalog `table`,
/// returning `Ok(())` only if **every** file committed. Files are written under
/// opaque uuid names so concurrent ingestors never collide. The table's
/// partition/sort spec is read per call (it may have been dropped out from under
/// us, which is [`WriteError::MissingTable`]).
///
/// On a per-file append failure the remaining files are still drained and
/// written (so the pipeline, which blocks on the channel, never deadlocks, and
/// the OTLP path still writes what it can), but the first error is returned.
pub(crate) async fn encode_and_append<T: ToRecordBatch>(
    catalog: &Arc<ParquetCatalog>,
    table: &str,
    dispatcher: &DataFlowDispatcher,
    items: Vec<T>,
) -> Result<(), WriteError> {
    let (partition_by, sort_by) = match catalog.table_handle(table) {
        Some(handle) => (handle.partition_by().to_vec(), handle.sort_by().to_vec()),
        None => return Err(WriteError::MissingTable(table.to_string())),
    };

    let disp = dispatcher.clone();
    let (tx, mut rx) = mpsc::channel::<EncodedFile>(IN_FLIGHT_FILES);
    let pipeline = tokio::task::spawn_blocking(move || -> Result<(), DataFlowError> {
        for file in parquet_writing::encode_items(
            &disp,
            items,
            partition_by.into(),
            sort_by.into(),
            ROW_GROUP_ROWS,
            ROW_GROUPS_PER_FILE,
        ) {
            if tx.blocking_send(file?).is_err() {
                return Ok(());
            }
        }
        Ok(())
    });

    let mut first_err: Option<WriteError> = None;
    while let Some(encoded) = rx.recv().await {
        let file_name = format!("pivot-{}.parquet", uuid::Uuid::new_v4());
        if let Err(e) = append_file(catalog, table, file_name, encoded).await {
            error!(table, error = %e, "appending parquet to table failed");
            first_err.get_or_insert(e);
        }
    }

    match pipeline.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            first_err.get_or_insert(WriteError::Pipeline(e));
        }
        Err(_) => {
            first_err.get_or_insert(WriteError::Panicked);
        }
    }

    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Append one finished file (`file_name`, relative to the table's location) to
/// `table`: write the bytes and commit the manifest entry. Runs on a blocking
/// thread - both the store write and the footer-reading commit block.
async fn append_file(
    catalog: &Arc<ParquetCatalog>,
    table: &str,
    file_name: String,
    encoded: EncodedFile,
) -> Result<(), WriteError> {
    let catalog = catalog.clone();
    let table = table.to_string();
    let path = ObjectPath::new(file_name.clone());
    let result = tokio::task::spawn_blocking({
        let table = table.clone();
        move || match catalog.table_handle(&table) {
            Some(mut handle) => handle
                .append_data_file(path, &encoded.bytes, encoded.partition, encoded.sort_bounds)
                .map_err(WriteError::Append),
            None => Err(WriteError::MissingTable(table)),
        }
    })
    .await;
    match result {
        Ok(Ok(())) => {
            info!(table = %table, file = %file_name, "appended parquet to table");
            Ok(())
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(WriteError::Panicked),
    }
}
