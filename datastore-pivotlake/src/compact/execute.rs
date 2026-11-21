//! Carrying out the merges of one commit: the batch a round selected, the
//! writing of its merges at the same time, and the one commit that swaps
//! their outputs into the table.

use std::sync::Arc;

use object_storage::{FileRef, ObjectPath};
use planner::catalog::SchemaQualifiedTableName;
use tracing::{info, warn};
use uuid::Uuid;

use super::MergeSelection;
use crate::table_format::{CompactionFile, PartitionHandle, RemovedFile};
use crate::{CatalogTable, Error, PivotlakeDatastore, Result, TableFile};

/// The files the merges of one commit rewrite, all of one partition.
pub(super) struct MergeBatch {
    /// The partition the inputs sit in.
    pub(super) partition: PartitionHandle,
    /// The inputs of each merge, each file as the lists it was selected
    /// from hold it. No file is among those of two.
    merges: Vec<Vec<Arc<dyn CompactionFile>>>,
    kind: MergeKind,
    /// The version of the table the inputs were selected from.
    selected_at_version: u64,
}

/// What the merges of one commit wrote, before the commit.
struct WrittenMerges {
    /// The inputs of every merge written, each as the lists it was selected
    /// from hold it.
    inputs: Vec<Arc<dyn CompactionFile>>,
    /// The outputs of every merge written.
    outputs: Vec<TableFile>,
    /// Why a merge was not written, the first one when several were not.
    failure: Option<Error>,
}

impl MergeBatch {
    /// The merges of one commit, selected from `table`.
    pub(super) fn new(
        table: &CatalogTable,
        partition: PartitionHandle,
        merges: Vec<Vec<Arc<dyn CompactionFile>>>,
        kind: MergeKind,
    ) -> Self {
        Self {
            partition,
            merges,
            kind,
            selected_at_version: table.version(),
        }
    }

    /// The files no other merge may claim until this batch is committed:
    /// the inputs of every one of its merges.
    pub(super) fn list_reserved_paths(&self) -> Vec<ObjectPath> {
        self.merges
            .iter()
            .flatten()
            .map(|file| file.file().path.clone())
            .collect()
    }

    /// Write the merges off `table`, a read copy that holds their inputs,
    /// all at the same time, each on a thread of its own. Nothing is
    /// committed.
    ///
    /// A merge that fails leaves no output, and the others are written all
    /// the same: what is handed back holds the inputs and outputs of the
    /// merges written, and the failure of the first that was not.
    ///
    /// Blocking store I/O, so callers run it on the blocking pool.
    fn write_merges(
        self,
        table: &CatalogTable,
        selection: &MergeSelection,
        name: &SchemaQualifiedTableName,
    ) -> WrittenMerges {
        let MergeBatch {
            partition,
            merges,
            kind,
            ..
        } = self;
        // Each merge blocks its thread on the dataflow that writes it, and
        // reads the table's store through the runtime of the calling thread.
        let runtime = tokio::runtime::Handle::current();
        let merged: Vec<Result<Vec<TableFile>>> = std::thread::scope(|scope| {
            let writers: Vec<_> = merges
                .iter()
                .map(|inputs| {
                    let (partition, runtime) = (&partition, &runtime);
                    scope.spawn(move || {
                        let _in_runtime = runtime.enter();
                        let input_files: Vec<FileRef> =
                            inputs.iter().map(|file| file.file().clone()).collect();
                        table.merge_files(partition, &input_files, selection.target_bytes)
                    })
                })
                .collect();
            writers
                .into_iter()
                .map(|writer| {
                    writer.join().unwrap_or_else(|_| {
                        Err(Error::CompactionJobPanicked("a merge panicked".to_string()))
                    })
                })
                .collect()
        });

        let mut written = WrittenMerges {
            inputs: Vec::new(),
            outputs: Vec::new(),
            failure: None,
        };
        for (mut inputs, outputs) in merges.into_iter().zip(merged) {
            let outputs = match outputs {
                Ok(outputs) => outputs,
                Err(error) => {
                    written.failure.get_or_insert(error);
                    continue;
                }
            };
            let input_sizes: Vec<u64> = inputs.iter().map(|file| file.file().size).collect();
            let output_sizes: Vec<u64> = outputs.iter().map(|file| file.file_ref().size).collect();
            info!(
                table = %name,
                kind = kind.label(),
                inputs = input_sizes.len(),
                input_sizes = ?input_sizes,
                input_bytes = input_sizes.iter().sum::<u64>(),
                outputs = output_sizes.len(),
                output_sizes = ?output_sizes,
                output_bytes = output_sizes.iter().sum::<u64>(),
                "compacted batch"
            );
            written.inputs.append(&mut inputs);
            written.outputs.extend(outputs);
        }
        written
    }
}

/// Delete `outputs`, files merges wrote that `error` keeps out of `table`.
fn delete_uncommitted_outputs(table: &CatalogTable, outputs: &[TableFile], error: &Error) {
    for file in outputs {
        let path = &file.file_ref().path;
        if let Err(cleanup_error) = table.delete_data_file(path) {
            warn!(
                error = %error,
                cleanup_error = %cleanup_error,
                file = %path,
                "compaction: deleting an output that will not be committed failed (orphan left)"
            );
        }
    }
}

/// What the merges of one commit hand back.
pub(super) struct MergeOutcome {
    /// The partition the inputs sit in.
    pub(super) partition: PartitionHandle,
    /// The paths the batch held reserved.
    pub(super) reserved_paths: Vec<ObjectPath>,
    /// The version of the table the inputs were selected from.
    pub(super) selected_at_version: u64,
    /// Whether every merge was written and their commit swapped them in,
    /// or why not. The merges written beside one that failed are committed
    /// all the same.
    pub(super) committed: Result<()>,
}

/// Write the merges of `batch` and swap them into the table in one commit
/// ([`write_and_commit_merges`]), on the blocking pool. Returns the
/// partition and the reserved paths of the batch beside the outcome so the
/// round can release the reservations however the merges went, and the
/// version the inputs were selected from so the round can tell a conflict
/// it cannot get past.
pub(super) async fn compact_batch(
    datastore: Arc<PivotlakeDatastore>,
    name: SchemaQualifiedTableName,
    id: Uuid,
    batch: MergeBatch,
    selection: MergeSelection,
) -> MergeOutcome {
    let partition = batch.partition.clone();
    let reserved_paths = batch.list_reserved_paths();
    let selected_at_version = batch.selected_at_version;
    // `spawn_blocking` needs a `'static` closure, so the merges take the
    // datastore handle rather than a borrow of it.
    let committed = tokio::task::spawn_blocking(move || {
        write_and_commit_merges(datastore.as_ref(), id, &name, batch, &selection)
    })
    .await
    .unwrap_or_else(|error| Err(Error::CompactionJobPanicked(error.to_string())));
    MergeOutcome {
        partition,
        reserved_paths,
        selected_at_version,
        committed,
    }
}

#[derive(Clone, Copy)]
pub(super) enum MergeKind {
    SmallFiles,
    LayoutOptimization,
}

impl MergeKind {
    fn label(self) -> &'static str {
        match self {
            Self::SmallFiles => "small files",
            Self::LayoutOptimization => "layout optimization",
        }
    }
}

/// Write the merges of `batch`, files of one partition of the table with
/// identity `id`, known as `name` ([`MergeBatch::write_merges`]), and swap
/// the outputs of all of them in for their inputs in one commit labelled a
/// rearrangement, which publishes the version it wrote.
///
/// The merges themselves (decode, re-encode, upload) run off a plain read
/// copy of the table with no commit lock held, so they never block a
/// concurrent INSERT's commit; only the swap that follows takes it. When a
/// merge fails, the merges written beside it are swapped in all the same,
/// its own inputs stay in the table, and its failure is returned once the
/// swap is made. If the swap fails for any reason, including another
/// writer having swapped an input out first, every output the merges wrote
/// is deleted and the swap's error is returned; a swap whose outcome is
/// unknown may have succeeded, so its outputs stay.
///
/// `selection.target_bytes` is a compressed row-group body target for the
/// outputs. Parquet headers and footers are excluded, and an individually
/// oversized row group is emitted alone.
///
/// Blocking store I/O, so callers run it on the blocking pool.
fn write_and_commit_merges(
    datastore: &PivotlakeDatastore,
    id: Uuid,
    name: &SchemaQualifiedTableName,
    batch: MergeBatch,
    selection: &MergeSelection,
) -> Result<()> {
    let table = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
    let partition = batch.partition.clone();
    let WrittenMerges {
        inputs,
        outputs,
        failure,
    } = batch.write_merges(&table, selection, name);
    let all_written = failure.map_or(Ok(()), Err);
    if inputs.is_empty() {
        return all_written;
    }

    let removed: Vec<RemovedFile<'_>> = inputs
        .iter()
        .map(|file| RemovedFile {
            partition: &partition,
            path: &file.file().path,
        })
        .collect();
    // The swap goes through the datastore's one in-process write path, under
    // the table's commit lock.
    let committed = datastore.commit_to_table(id, &removed, outputs.clone(), false);
    if let Err(error) = &committed
        && !error.may_have_committed()
    {
        delete_uncommitted_outputs(&table, &outputs, error);
    }
    committed.and(all_written)
}
