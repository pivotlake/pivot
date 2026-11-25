//! The new compaction: rewriting several sorted input files of a table as
//! fewer, larger sorted files while holding only what is about to be used.
//!
//! [`compact_files_spec`] builds its dataflow in two halves. The read half
//! reads the inputs in the order of the output and produces the decoded
//! rows in merge order, a column at a time. The write half encodes them
//! and produces the output's column chunks, in the order of the output
//! file, for a writer to append to files. One [`memory_budget`] counts what both halves hold.
//!
//! The compaction that tables run by default is not in this module. It
//! decodes every input row group whole and merges the decoded batches (see
//! `encode_compaction_batches_spec` in the parent module).
//!
//! The module is divided as the compaction is:
//!
//! - [`read`]: the read half. It has the diagram of its operators, and is
//!   divided again into the order of the output's rows
//!   ([`read::row_order`]), the input files and their reads
//!   ([`read::input`]), and the operators it adds to the scan's pipeline
//!   ([`read::pipeline`]).
//! - [`mod@write`]: the write half, with the diagram of its operators.
//! - [`memory_budget`]: the count of every byte both halves hold.
//! - [`settings`]: the tunables.

mod memory_budget;
mod read;
mod settings;
mod write;

use std::sync::Arc;

use dispatch::{DataFlowDispatcher, OperatorFactory, OperatorSpec, Projection};

use crate::writing::error::WriteError;
use crate::writing::shredding::{self, FileShredding};
use memory_budget::MemoryBudget;

pub use memory_budget::MemoryReservation;
pub use settings::MergeSortSettings;
pub use write::column_chunk_batcher::EncodedColumnChunks;

/// How many input row groups a compaction's shredding plan reads.
const SHREDDING_SAMPLE_ROW_GROUPS: usize = 8;

/// A compaction of input files as a stream of encoded column chunks in the
/// order of the output, with what its writer needs to lay them out as
/// files.
pub struct FileCompaction<OF: OperatorFactory<EncodedColumnChunks>> {
    pub column_chunks: OperatorSpec<EncodedColumnChunks, OF>,
    /// The worker the column chunks are put in file order on.
    pub write_worker: usize,
    /// How many rows the output holds in all.
    pub row_count: usize,
}

fn into_dataflow_error(error: WriteError) -> dispatch::DataFlowError {
    dispatch::DataFlowError::Operation(dispatch::UnaryError::from(error).into())
}

/// Plan how the output of a compaction of `table` is shredded, before the
/// compaction writes its first row group. Only variant columns are
/// shredded, so a table without any reads nothing. Otherwise the plan comes
/// from the variant columns of a few row groups spread evenly over the
/// input.
pub fn plan_compaction_shredding(
    dispatcher: &DataFlowDispatcher,
    table: &crate::ParquetTable,
) -> Result<Arc<FileShredding>, dispatch::DataFlowError> {
    let schema = table.schema();
    let variant_columns: Vec<usize> = (0..schema.fields().len())
        .filter(|&column| crate::is_variant_field(schema.field(column)))
        .collect();
    let row_groups_with_rows: Vec<_> = table
        .row_groups()
        .iter()
        .filter(|row_group| row_group.num_rows > 0)
        .collect();
    let mut sample = Vec::new();
    if !variant_columns.is_empty() && !row_groups_with_rows.is_empty() {
        let sampled_count = SHREDDING_SAMPLE_ROW_GROUPS.min(row_groups_with_rows.len());
        let sampled_row_groups = (0..sampled_count)
            .map(|sampled| {
                row_groups_with_rows[sampled * row_groups_with_rows.len() / sampled_count].clone()
            })
            .collect();
        let sampled_table = Arc::new(crate::ParquetTable::new(sampled_row_groups));
        let scan = crate::table_input(
            dispatcher,
            &sampled_table,
            Projection::columns(variant_columns.clone()),
            false,
        );
        sample = crate::writing::unshred_batches_spec(scan).collect()?;
    }
    let shredding = shredding::plan_shredding_from_sample(schema, &variant_columns, &sample)
        .map_err(into_dataflow_error)?;
    Ok(Arc::new(shredding))
}

/// Builds the dataflow that merges the files of `table`, each already
/// sorted by `sort_column_names`, into the encoded column chunks of row
/// groups of `output_row_group_rows` rows in that order, shredded as
/// `shredding` says
/// (see [`plan_compaction_shredding`]). [`read`] and [`mod@write`] have the
/// diagrams of the operators chained here.
///
/// A table without sort columns has its files' rows passed through in file
/// order. Fails for a table that has deleted rows.
pub fn compact_files_spec(
    dispatcher: &DataFlowDispatcher,
    table: Arc<crate::ParquetTable>,
    shredding: Arc<FileShredding>,
    sort_column_names: &[String],
    output_row_group_rows: usize,
    settings: MergeSortSettings,
) -> Result<
    FileCompaction<impl OperatorFactory<EncodedColumnChunks> + 'static>,
    dispatch::DataFlowError,
> {
    // The encoder copies a fixed-width leaf of a row group into one block.
    debug_assert!(
        output_row_group_rows <= dispatch::BUFFER_SIZE / 16,
        "a row group's widest column must fit one block"
    );
    let compaction_worker = dispatcher.next_worker();
    let write_worker = (compaction_worker + 1) % dispatcher.worker_count();
    let memory_budget = MemoryBudget::new(
        dispatcher.waker_set(),
        settings.memory_bytes,
        compaction_worker,
    );
    let row_count = table.total_rows() as usize;
    let merged_batch_columns = read::read_in_merge_order_spec(
        dispatcher,
        table,
        sort_column_names,
        output_row_group_rows,
        settings,
        memory_budget.clone(),
        compaction_worker,
    )?;
    let column_chunks = write::encode_column_chunks_spec(
        merged_batch_columns,
        &shredding,
        settings,
        &memory_budget,
        compaction_worker,
        write_worker,
    );
    Ok(FileCompaction {
        column_chunks,
        write_worker,
        row_count,
    })
}
