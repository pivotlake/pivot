//! The write half of a compaction: the rows the read sends on, encoded as the
//! output's column chunks, in the order of the output file.
//!
//! The write half takes each column of each merged batch as the read
//! sends it, so a column of an output row group is encoded, and its rows
//! are let go of, while the read is still on the row group's other
//! columns.
//!
//! ```text
//!                 ┌──────────────────────────┐
//!                 │    MergeSortCompacter    │  the compaction worker
//!                 └──────────────────────────┘
//!                              │ MergedBatchColumn
//!                              ▼
//!                 ┌──────────────────────────┐  any other worker; copies the
//!                 │       LeafGatherer       │  rows into output order, shreds
//!                 └──────────────────────────┘  variants, splits into leaves
//!                              │ GatheredLeaves
//!                              ▼
//!                 ┌──────────────────────────┐  the write worker; joins the
//!                 │      LeafCollector       │  merged batches of a column of
//!                 └──────────────────────────┘  an output row group
//!                              │ RowGroupLeaf
//!                              ▼
//!                 ┌──────────────────────────┐
//!                 │   RowGroupLeafEncoder    │  any other worker
//!                 └──────────────────────────┘
//!                              │ EncodedRowGroupLeaf
//!                              ▼
//!                 ┌──────────────────────────┐  the write worker; puts the
//!                 │    ColumnChunkBatcher    │  leaves in file order, a batch
//!                 └──────────────────────────┘  of bytes at a time
//!                              │ EncodedColumnChunks, in file order
//!                              ▼
//! ```
//!
//! - [`leaf_gatherer`]: one column of one merged batch becomes that
//!   column's leaves with the rows in output order. A variant column
//!   arrives the way each input file shredded it and leaves the way the
//!   output is shredded, which is decided before the compaction starts
//!   ([`plan_compaction_shredding`](super::plan_compaction_shredding)).
//! - [`leaf_collector`]: the leaves of a column of an output row group are
//!   sent to be encoded once every merged batch of the column is there.
//! - [`leaf_encoder`]: one leaf of one output row group is encoded with
//!   the encoder every write uses.
//! - [`column_chunk_batcher`]: the encoded leaves are put in the order of
//!   the output file and passed on a batch of bytes at a time, so a row
//!   group's first columns are written while its last are still being read.
//!
//! Gathering and encoding are the heavy steps, and any worker but the
//! compaction worker does them. The two steps that only wait for pieces
//! and pass them on run on one worker, the write worker, which is not the
//! compaction worker either when there are two workers or more.

pub(crate) mod column_chunk_batcher;
pub(crate) mod leaf_collector;
pub(crate) mod leaf_encoder;
pub(crate) mod leaf_gatherer;

use std::sync::Arc;

use dispatch::{OperatorFactory, OperatorSpec, shared_work_queue, to_single_worker_mpsc};

use super::memory_budget::MemoryBudget;
use super::read::output::MergedBatchColumn;
use super::settings::MergeSortSettings;
use crate::writing::shredding::FileShredding;
use column_chunk_batcher::{ColumnChunkBatcher, EncodedColumnChunks};
use leaf_collector::{LeafCollector, RowGroupLeaf};
use leaf_encoder::{EncodedRowGroupLeaf, RowGroupLeafEncoder};
use leaf_gatherer::{GatheredLeaves, LeafGatherer};

/// Builds the write half of the dataflow: it encodes the columns of merged
/// batches that `merged_batch_columns` produces as the output's column
/// chunks, shredded as `shredding` says and batched as `settings` says, and
/// counts what it holds in `memory_budget`. The column chunks are put in
/// file order on `write_worker`, and `compaction_worker` and `write_worker`
/// take none of the gathering or encoding while other workers can.
pub(super) fn encode_column_chunks_spec<OF>(
    merged_batch_columns: OperatorSpec<MergedBatchColumn, OF>,
    shredding: &Arc<FileShredding>,
    settings: MergeSortSettings,
    memory_budget: &Arc<MemoryBudget>,
    compaction_worker: usize,
    write_worker: usize,
) -> OperatorSpec<EncodedColumnChunks, impl OperatorFactory<EncodedColumnChunks> + 'static + use<OF>>
where
    OF: OperatorFactory<MergedBatchColumn> + 'static,
{
    let worker_count = merged_batch_columns.dispatcher().worker_count();
    let gathers_and_encodes =
        list_workers_that_gather_and_encode(worker_count, compaction_worker, write_worker);
    merged_batch_columns
        .chain(
            shared_work_queue::<MergedBatchColumn>(worker_count)
                .into_iter()
                .collect(),
            LeafGatherer::create_for_workers(shredding, memory_budget, &gathers_and_encodes),
        )
        .chain(
            to_single_worker_mpsc::<GatheredLeaves>(worker_count, write_worker)
                .into_iter()
                .collect(),
            LeafCollector::create_for_workers(shredding, worker_count),
        )
        .chain(
            shared_work_queue::<RowGroupLeaf>(worker_count)
                .into_iter()
                .collect(),
            RowGroupLeafEncoder::create_for_workers(memory_budget, &gathers_and_encodes),
        )
        .chain(
            to_single_worker_mpsc::<EncodedRowGroupLeaf>(worker_count, write_worker)
                .into_iter()
                .collect(),
            ColumnChunkBatcher::create_for_workers(
                shredding,
                settings.write_batch_bytes,
                memory_budget,
                worker_count,
            ),
        )
}

/// For each of `worker_count` workers, whether it takes gathering and
/// encoding work.
///
/// The compaction worker's time goes to the read, and the write worker's to
/// passing pieces on and issuing writes: a leaf can take long to encode, and
/// everything that waits for either worker waits that long with it. So
/// neither takes that work when there is a third worker to do it. With two
/// workers the write worker takes it, and a single worker does everything.
fn list_workers_that_gather_and_encode(
    worker_count: usize,
    compaction_worker: usize,
    write_worker: usize,
) -> Vec<bool> {
    (0..worker_count)
        .map(|worker| match worker_count {
            1 => true,
            2 => worker != compaction_worker,
            _ => worker != compaction_worker && worker != write_worker,
        })
        .collect()
}
