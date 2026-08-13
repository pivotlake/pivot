//! The write pipeline's collector: turns the stream of per-partition sorted
//! pieces into files, each cut the moment enough bytes are in hand.
//!
//! Every piece of a write funnels to one worker's collector (the
//! single-worker channel in [`super`]), each holding one partition's rows in
//! key order. The collector groups pieces by partition and cuts a file once
//! a partition gathers `target_bytes_per_file` of them, or when the stream
//! finishes; a partition larger than the file target becomes several files,
//! each internally in key order, left for the compacter to merge. The byte
//! threshold is THE write path's memory bound: upstream buffers nothing, so
//! however wide or heavy the rows, a file cut is the same number of bytes.
//!
//! The collector never touches row data. Cutting a file fixes its identity
//! (its [`FilePlan`]) and, when the pieces need merging, plans the file's
//! k-way merge into stealable slices; the [`sorter`](super::sorter) stage
//! does the moving of rows. A file of one piece (or of a table with no sort
//! keys) skips the merge and ships whole.

use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

use arrow_array::RecordBatch;
use arrow_row::OwnedRow;
use dispatch::{OrderBy, Sender, SortedPiece, Unary, UnaryFactory, UnaryResult};

use super::error::WriteError;
use super::types::{FileJob, FileMerge, FilePlan, SortedFile};
use crate::scalar_values_from_row;

/// Builds one worker's [`Collector`]. Every worker gets one, but only the
/// worker the single-worker channel targets ever receives a piece.
pub(super) struct CollectorFactory {
    partition_by: Arc<[String]>,
    sort_keys: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_bytes_per_file: usize,
    worker_count: usize,
}

pub(super) fn factories(
    partition_by: Arc<[String]>,
    sort_keys: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    target_bytes_per_file: usize,
    worker_count: usize,
) -> Vec<CollectorFactory> {
    (0..worker_count)
        .map(|_| CollectorFactory {
            partition_by: partition_by.clone(),
            sort_keys: sort_keys.clone(),
            target_rows_per_group,
            target_bytes_per_file,
            worker_count,
        })
        .collect()
}

impl UnaryFactory<SortedPiece, FileJob> for CollectorFactory {
    type Unary = Collector;

    fn build_unary(self) -> Collector {
        Collector {
            partition_by: self.partition_by,
            sort_keys: self.sort_keys,
            target_rows_per_group: self.target_rows_per_group,
            target_bytes_per_file: self.target_bytes_per_file,
            worker_count: self.worker_count,
            open_partitions: HashMap::new(),
            next_file_id: 0,
            next_row_group_id: 0,
            total_pieces: 0,
            total_rows: 0,
            total_bytes: 0,
        }
    }
}

/// One partition's pieces still gathering toward a file. Each piece is one
/// sorted chunk, and each becomes one run of the file's merge.
#[derive(Default)]
struct OpenPartition {
    pieces: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
}

pub(super) struct Collector {
    partition_by: Arc<[String]>,
    sort_keys: Arc<[OrderBy]>,
    target_rows_per_group: usize,
    /// Raw bytes at which a partition's gathered pieces cut into a file.
    /// Compaction passes `usize::MAX` here, keeping one file per partition
    /// (its whole point is merging a partition's files into one globally
    /// sorted file).
    target_bytes_per_file: usize,
    worker_count: usize,
    open_partitions: HashMap<Option<OwnedRow>, OpenPartition>,
    next_file_id: u64,
    next_row_group_id: u64,
    /// Lifetime tallies of everything consumed, for the flow log each cut
    /// emits.
    total_pieces: usize,
    total_rows: usize,
    total_bytes: usize,
}

impl Collector {
    /// Cut one file from a partition's gathered pieces: fix its plan, and
    /// either ship it whole (nothing to merge) or plan its k-way merge and
    /// fan the slices out.
    fn cut_file(
        &mut self,
        tuple: Option<OwnedRow>,
        partition: OpenPartition,
        sender: &mut dyn Sender<FileJob>,
    ) -> UnaryResult<()> {
        debug_assert!(partition.rows > 0, "a partition only opens with rows");
        let file_id = self.next_file_id;
        tracing::debug!(
            file_id,
            file_rows = partition.rows,
            file_bytes = partition.bytes,
            total_pieces = self.total_pieces,
            total_rows = self.total_rows,
            total_bytes = self.total_bytes,
            "collector cut a file"
        );
        self.next_file_id += 1;
        let n_row_groups = partition.rows.div_ceil(self.target_rows_per_group);
        let base_row_group_id = self.next_row_group_id;
        self.next_row_group_id += n_row_groups as u64;
        let partition_values = match &tuple {
            Some(_) => Some(
                scalar_values_from_row(&partition.pieces[0], &self.partition_by, 0)
                    .map_err(WriteError::from)?,
            ),
            None => None,
        };
        let plan = FilePlan {
            file_id,
            base_row_group_id,
            // The assembler gathers a whole file on one worker; spread files
            // across workers round-robin by id.
            dest_worker: (file_id as usize) % self.worker_count,
            partition: partition_values,
            target_rows_per_group: self.target_rows_per_group,
        };

        if self.sort_keys.is_empty() || partition.pieces.len() == 1 {
            sender.send(FileJob::Sorted(SortedFile {
                plan,
                chunks: partition.pieces,
                rows: partition.rows,
            }))?;
            return Ok(());
        }

        // Each piece is sorted within itself, so each is one run of the
        // file's merge.
        let runs: Vec<Vec<RecordBatch>> = partition
            .pieces
            .into_iter()
            .map(|piece| vec![piece])
            .collect();
        let slices = dispatch::plan_multiway_merge_slices(&self.sort_keys, &runs)
            .map_err(WriteError::from)?;
        let merge = Arc::new(FileMerge {
            plan,
            sort_keys: self.sort_keys.clone(),
            rows: partition.rows,
            merged: (0..slices.len()).map(|_| OnceLock::new()).collect(),
            slices_remaining: AtomicUsize::new(slices.len()),
            slices,
            runs,
        });
        for slice in 0..merge.slices.len() {
            sender.send(FileJob::MergeSlice {
                merge: merge.clone(),
                slice,
            })?;
        }
        Ok(())
    }
}

impl Unary<SortedPiece, FileJob> for Collector {
    fn consume(&mut self, piece: SortedPiece, sender: &mut dyn Sender<FileJob>) -> UnaryResult<()> {
        if piece.chunk.num_rows() == 0 {
            return Ok(());
        }
        self.total_pieces += 1;
        self.total_rows += piece.chunk.num_rows();
        self.total_bytes += piece.bytes;
        let open = self.open_partitions.entry(piece.tuple.clone()).or_default();
        open.rows += piece.chunk.num_rows();
        open.bytes += piece.bytes;
        open.pieces.push(piece.chunk);
        if open.bytes >= self.target_bytes_per_file {
            let full = self
                .open_partitions
                .remove(&piece.tuple)
                .expect("the partition was open just above");
            self.cut_file(piece.tuple, full, sender)?;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<FileJob>) -> UnaryResult<bool> {
        for (tuple, partition) in std::mem::take(&mut self.open_partitions) {
            self.cut_file(tuple, partition, sender)?;
        }
        Ok(true)
    }

    /// Per-piece work here is a vec push; the backlog must never wait on the
    /// host worker's other operators.
    fn drains_greedily(&self) -> bool {
        true
    }
}
