//! The merge: one dataflow that decodes a group of a table's files, repacks
//! their rows into target-sized row groups and files, and uploads those
//! under the table's directory, handing back what a commit records.

use std::sync::Arc;

use arrow_array::RecordBatch;
use crossbeam_deque::{Injector, Steal};
use dispatch::Projection;
use object_storage::FileRef;
use parquet_engine::file_entry::FileEntry;
use parquet_engine::writing::upload::{UploadedFile, upload_files_spec};
use parquet_engine::writing::{encode_compaction_batches_spec, unshred_batches_spec};
use parquet_engine::{ParquetTable, RowGroupMetadata};

use crate::{CompactionTable, Error, TableFile};

/// Merge `inputs`, live files of `table` sharing one partition tuple, into
/// fresh files of at most `max_output_file_size` compressed bytes, each row
/// group holding `target_rows_per_group` rows, written under the table's
/// directory. The outputs are returned with their footers for the caller to
/// commit. Nothing here touches the metadata, so it runs off any read copy
/// that holds the inputs, and a merge of a large batch never stalls another
/// writer's commit. An input the copy no longer lists is a commit conflict:
/// another writer swapped it out first.
///
/// A merge that fails part way has uploaded some of its outputs. They will
/// never be committed, so they are deleted rather than left as orphans.
pub fn merge_files<T: CompactionTable>(
    table: &T,
    inputs: &[FileRef],
    target_rows_per_group: usize,
    max_output_file_size: u64,
) -> Result<Vec<TableFile>, T::Error> {
    let input_entries: Vec<&FileEntry> = inputs
        .iter()
        .map(|input| {
            table
                .entries()
                .iter()
                .find(|entry| entry.file.path == input.path)
                .ok_or_else(|| {
                    Error::CommitConflict {
                        location: table.location(),
                        file: input.path.to_string(),
                    }
                    .into()
                })
        })
        .collect::<Result<_, T::Error>>()?;
    let row_groups: Vec<Arc<RowGroupMetadata>> = table
        .row_groups(&input_entries)?
        .into_iter()
        .flatten()
        .collect();
    let parquet = Arc::new(ParquetTable::new(row_groups));
    let column_count = parquet.schema().fields().len();
    let scan = parquet_engine::table_input(
        table.dispatcher(),
        &parquet,
        Projection::all(column_count),
        false,
    );
    // The selector guarantees that all inputs share one partition, so its
    // recorded tuple can key every decoded batch without repartitioning it.
    let partition = input_entries
        .first()
        .and_then(|entry| entry.partition.clone());
    // A table that names its columns differently from how its files carry
    // them gets its outputs stamped with the declared schema: the inputs are
    // reassembled to the declared shape (an adopted file may carry no field
    // ids, and a variant column is re-shredded for the file it lands in) and
    // encoded under it. Otherwise the outputs keep the inputs' own schema.
    let (scan, output_schema) = match table.written_schema() {
        None => (scan, parquet.schema().clone()),
        Some(schema) => {
            let projected = unshred_batches_spec(scan).project({
                let schema = schema.clone();
                move || {
                    let schema = schema.clone();
                    move |batch: RecordBatch| {
                        Ok(RecordBatch::try_new(
                            schema.clone(),
                            batch.columns().to_vec(),
                        )?)
                    }
                }
            });
            (projected, schema)
        }
    };
    let max_output_file_size = usize::try_from(max_output_file_size).unwrap_or(usize::MAX);
    let uploaded_files = Arc::new(Injector::new());
    let encoded = encode_compaction_batches_spec(
        scan,
        output_schema,
        partition,
        Arc::from(table.partition_by()),
        Arc::from(table.sort_by()),
        target_rows_per_group,
        max_output_file_size,
    );
    let spec = upload_files_spec(
        table.store(),
        table.directory(),
        table.file_prefix(),
        table.columns().into(),
        uploaded_files.clone(),
        None,
        encoded,
    );
    // Drive encode -> upload to completion; the emitted row-count batch is
    // ignored, and the uploaded files arrive on `uploaded_files`.
    let outcome = spec.collect().map(|_| ()).map_err(Error::Merge);

    let mut added: Vec<TableFile> = Vec::new();
    loop {
        match uploaded_files.steal() {
            Steal::Success(uploaded) => {
                let UploadedFile {
                    file,
                    partition,
                    row_groups,
                } = uploaded;
                let stats = Some(Arc::new(parquet_engine::aggregate_file_stats(&row_groups)));
                let entry = FileEntry {
                    file,
                    partition,
                    stats,
                };
                added.push(TableFile::new(entry, row_groups));
            }
            Steal::Retry => continue,
            Steal::Empty => break,
        }
    }

    if let Err(error) = outcome {
        for file in &added {
            if let Err(cleanup_error) = table.delete_data_file(&file.entry.file.path) {
                tracing::warn!(
                    merge_error = %error,
                    cleanup_error = %cleanup_error,
                    file = %file.entry.file.path,
                    "compaction: deleting an output of a merge that will not commit failed (orphan left)"
                );
            }
        }
        return Err(error.into());
    }
    Ok(added)
}
