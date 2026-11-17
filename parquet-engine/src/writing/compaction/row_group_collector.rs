//! Cuts the merged rows into output row groups and sends each one out as
//! column jobs.
//!
//! [`RowGroupCollector`] puts the merged runs in order and cuts them where
//! output row groups end. A row group whose rows are all there goes out as
//! one job per column. A job carries
//! the column's part of every batch the row group takes rows from. The rows
//! of those batches interleave in the output, so each job says which row of
//! which batch each of its rows is ([`GatherRows`]), by which the worker
//! that takes the job copies its column into output order. A variant column
//! goes out as several jobs, each a slice of the row group's rows. A row group that is made of consecutive rows of its
//! batches, one batch after the other, goes out as it is, without the copy.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use arrow_array::{Array, ArrayRef};
use arrow_buffer::ScalarBuffer;
use dispatch::{
    BuildContext, Built, BuiltUnary, Sender, Topology, Unary, UnaryFactory, UnaryResult,
};

use super::super::leaves;
use super::super::shredding::FileShredding;
use super::super::types::{ColumnChunkJob, ColumnSlice, FileAssemblyInfo, RowGroupContext};
use super::VARIANT_COLUMN_SLICE_ROWS;
use super::input_batch::InputBatch;
use super::memory_budget::MemoryBudget;
use super::merge::{MergedRows, MergedRun};

/// What a source batch that a part takes no rows of has in place of a chunk.
const NOT_A_CHUNK: u32 = u32::MAX;

/// Which row of which chunk each row of a part is. The chunks are numbered
/// within the part until the part joins an output row group's column jobs,
/// and within the row group's chunks from then on.
enum PartRows {
    /// Row `i` of the part is row `rows[i]` of the chunk that source batch
    /// `sources[i]` of the part's run is, which `chunk_of_source` gives.
    Listed {
        sources: ScalarBuffer<u32>,
        rows: ScalarBuffer<u32>,
        chunk_of_source: Vec<u32>,
    },
    /// The part is the rows of one chunk, in order.
    Consecutive { chunk: u32 },
}

/// The rows of a merged run that go to one output row group, with the
/// batches those rows are in and no others. A run's batches hold the rows of
/// several output row groups, so a part keeps only its own.
struct RowGroupPart {
    /// The batches the part takes rows of.
    chunks: Vec<InputBatch>,
    rows: PartRows,
    row_count: usize,
}

impl RowGroupPart {
    /// Rows `rows` of `run`.
    fn new(run: &MergedRun, rows: Range<usize>) -> Self {
        let row_count = rows.len();
        match &run.rows {
            MergedRows::Listed {
                sources,
                rows: source_rows,
            } => {
                let sources = sources.values().slice(rows.start, row_count);
                let mut chunk_of_source = vec![NOT_A_CHUNK; run.sources.len()];
                let mut chunks = Vec::new();
                for &source in sources.iter() {
                    if chunk_of_source[source as usize] == NOT_A_CHUNK {
                        chunk_of_source[source as usize] = chunks.len() as u32;
                        chunks.push(run.sources[source as usize].clone());
                    }
                }
                Self {
                    chunks,
                    rows: PartRows::Listed {
                        sources,
                        rows: source_rows.values().slice(rows.start, row_count),
                        chunk_of_source,
                    },
                    row_count,
                }
            }
            MergedRows::Consecutive {
                source, first_row, ..
            } => Self {
                chunks: vec![
                    run.sources[*source as usize]
                        .slice_rows(*first_row as usize + rows.start, row_count),
                ],
                rows: PartRows::Consecutive { chunk: 0 },
                row_count,
            },
        }
    }
}

/// The rows of one column job: a range of an output row group's rows, with
/// where each of them is among the chunks the job carries.
#[derive(Clone)]
pub(crate) struct GatherRows {
    order: Arc<GatherOrder>,
    /// Which of the row group's rows.
    rows: Range<usize>,
}

impl GatherRows {
    /// How many rows the job holds.
    pub(crate) fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// For each of the job's rows, in output order, the chunk that holds it
    /// and its row there.
    pub(crate) fn row_sources(&self) -> &[(u32, u32)] {
        &self.order.row_sources()[self.rows.clone()]
    }
}

/// Where each row of one output row group is among the chunks its column
/// jobs carry. Every job of the row group shares one.
struct GatherOrder {
    row_count: usize,
    /// The row group's parts, in output order: for each, how many rows it
    /// holds, and which row of which of the row group's chunks each is.
    parts: Vec<(usize, PartRows)>,
    row_sources: OnceLock<Vec<(u32, u32)>>,
}

impl GatherOrder {
    /// For each row of the row group, in output order, the chunk that holds
    /// it and its row there. Worked out on the first call, by whichever
    /// column job makes it, and shared by the rest.
    fn row_sources(&self) -> &[(u32, u32)] {
        self.row_sources.get_or_init(|| {
            let mut row_sources = Vec::with_capacity(self.row_count);
            for (part_row_count, part_rows) in &self.parts {
                match part_rows {
                    PartRows::Listed {
                        sources,
                        rows,
                        chunk_of_source,
                    } => {
                        row_sources.extend(
                            sources
                                .iter()
                                .zip(rows.iter())
                                .map(|(&source, &row)| (chunk_of_source[source as usize], row)),
                        );
                    }
                    PartRows::Consecutive { chunk } => {
                        row_sources.extend((0..*part_row_count as u32).map(|row| (*chunk, row)));
                    }
                }
            }
            assert_eq!(
                row_sources.len(),
                self.row_count,
                "the parts of an output row group hold its rows"
            );
            row_sources
        })
    }
}

/// Cuts the merged runs into output row groups and sends each out as column
/// jobs. The runs are routed to one worker, the compaction worker; the other
/// workers' collectors receive nothing.
pub(super) struct RowGroupCollector {
    /// Runs that arrived ahead of their turn.
    arrived: BTreeMap<usize, MergedRun>,
    next_sequence: usize,
    /// The parts of the output row group being filled.
    filling: Vec<RowGroupPart>,
    rows_filling: usize,
    /// How many rows the runs taken so far hold.
    rows_taken: usize,
    row_count: usize,
    rows_per_group: usize,
    /// How many output row groups have been sent.
    row_groups_sent: usize,
    /// Where the row groups sent to be encoded are counted, until they are
    /// written.
    memory_budget: Arc<MemoryBudget>,
    /// The worker that assembles the row groups.
    assembly_worker: usize,
    topology: Topology,
    shredding: Arc<FileShredding>,
    /// Leaves per top-level column, numbered depth-first across the columns.
    leaf_counts: Vec<usize>,
    file_info: Arc<FileAssemblyInfo>,
}

impl RowGroupCollector {
    /// One collector per worker of `topology`. The compaction worker's
    /// collector sends `row_count` rows out in row groups of
    /// `target_rows_per_group`, which that worker also assembles.
    pub(super) fn factories(
        topology: Topology,
        shredding: Arc<FileShredding>,
        row_count: usize,
        memory_budget: Arc<MemoryBudget>,
        compaction_worker: usize,
        target_rows_per_group: usize,
        sort_column_names: Arc<[String]>,
    ) -> Vec<Self> {
        // A fixed-width leaf is materialized into one ring slab by the encoder.
        debug_assert!(
            target_rows_per_group <= dispatch::BUFFER_SIZE / 16,
            "a row group's widest column must fit one slab"
        );
        let leaf_counts: Vec<usize> = shredding
            .schema
            .fields()
            .iter()
            .map(|field| leaves::count_leaves(field))
            .collect();
        let file_info = Arc::new(FileAssemblyInfo {
            file_id: 0,
            row_group_count: row_count.div_ceil(target_rows_per_group),
            partition: None,
            max_file_size: None,
            sorted_by: sort_column_names,
        });
        (0..topology.total_workers())
            .map(|_| Self {
                arrived: BTreeMap::new(),
                next_sequence: 0,
                filling: Vec::new(),
                rows_filling: 0,
                rows_taken: 0,
                row_count,
                rows_per_group: target_rows_per_group,
                row_groups_sent: 0,
                memory_budget: memory_budget.clone(),
                assembly_worker: compaction_worker,
                topology,
                leaf_counts: leaf_counts.clone(),
                shredding: shredding.clone(),
                file_info: file_info.clone(),
            })
            .collect()
    }

    /// Send the column jobs of the next output row group, which is the rows
    /// of `parts`. The jobs hold the memory reserved for the batches the row
    /// group takes rows from, for as long as it is being encoded.
    fn send_column_jobs_of_row_group(
        &self,
        parts: Vec<RowGroupPart>,
        sender: &mut dyn Sender<ColumnChunkJob>,
    ) -> UnaryResult<()> {
        let row_count: usize = parts.iter().map(|part| part.row_count).sum();
        // The batches the row group takes rows from, which become the
        // chunks of every column job. Several parts take rows of the same
        // batch, which is one chunk. A batch is told apart by the rows it
        // holds: its row group, its first row there, and how many.
        let mut chunks: Vec<InputBatch> = Vec::new();
        let mut chunk_of_batch: HashMap<(u32, u32, usize), u32> = HashMap::new();
        let mut gather_parts = Vec::with_capacity(parts.len());
        let mut is_in_output_order = true;
        for part in parts {
            // Where each of the part's chunks is among the row group's.
            let mut row_group_chunks = Vec::with_capacity(part.chunks.len());
            for chunk in part.chunks {
                let batch = (chunk.row_group, chunk.first_row, chunk.row_count());
                row_group_chunks.push(*chunk_of_batch.entry(batch).or_insert_with(|| {
                    chunks.push(chunk);
                    chunks.len() as u32 - 1
                }));
            }
            let rows = match part.rows {
                PartRows::Listed {
                    sources,
                    rows,
                    mut chunk_of_source,
                } => {
                    is_in_output_order = false;
                    for chunk in &mut chunk_of_source {
                        if *chunk != NOT_A_CHUNK {
                            *chunk = row_group_chunks[*chunk as usize];
                        }
                    }
                    PartRows::Listed {
                        sources,
                        rows,
                        chunk_of_source,
                    }
                }
                PartRows::Consecutive { chunk } => PartRows::Consecutive {
                    chunk: row_group_chunks[chunk as usize],
                },
            };
            gather_parts.push((part.row_count, rows));
        }
        // Chunks that are each consecutive rows, one after the other, are
        // the row group as they stand.
        let gather = (!is_in_output_order).then(|| {
            Arc::new(GatherOrder {
                row_count,
                parts: gather_parts,
                row_sources: OnceLock::new(),
            })
        });

        let mut rows_by_node = vec![0usize; self.topology.node_count];
        for chunk in &chunks {
            rows_by_node[self.topology.node_of_worker(chunk.worker)] += chunk.row_count();
        }
        let target_node = dispatch::dominant_node(&rows_by_node);

        let mut held_reservations = HashSet::new();
        let context = Arc::new(RowGroupContext {
            row_group_id: self.row_groups_sent as u64,
            assembly_worker: self.assembly_worker,
            schema: self.shredding.schema.clone(),
            leaf_count: self.leaf_counts.iter().sum(),
            file_info: self.file_info.clone(),
            // Chunks that are parts of one decoded batch share its
            // reservation, which is held once.
            memory_reservations: chunks
                .iter()
                .map(|chunk| &chunk.memory)
                .filter(|memory| held_reservations.insert(Arc::as_ptr(memory)))
                .cloned()
                .collect(),
        });
        self.memory_budget.record_row_group_sent();
        let mut first_leaf_index = 0;
        for (column_index, &column_leaf_count) in self.leaf_counts.iter().enumerate() {
            let column_chunks: Arc<[ArrayRef]> = chunks
                .iter()
                .map(|chunk| chunk.batch.column(column_index).clone())
                .collect();
            // A variant column is the heavy one: its rows are folded back
            // from the input files' shredding and shredded anew. It goes out
            // in slices of rows, one job each, so that several workers do it
            // at once. Any other column is one job.
            let slice_rows = if crate::is_variant_field(self.shredding.schema.field(column_index)) {
                VARIANT_COLUMN_SLICE_ROWS
            } else {
                row_count
            };
            let slice_count = row_count.div_ceil(slice_rows);
            for slice_index in 0..slice_count {
                let rows =
                    slice_index * slice_rows..((slice_index + 1) * slice_rows).min(row_count);
                let (batches, gather) = match &gather {
                    Some(order) => (
                        column_chunks.clone(),
                        Some(GatherRows {
                            order: order.clone(),
                            rows,
                        }),
                    ),
                    None if slice_count == 1 => (column_chunks.clone(), None),
                    // The chunks are the row group's rows in order, so a
                    // slice is the parts of them its rows are in.
                    None => {
                        let mut slice_chunks = Vec::new();
                        let mut chunk_start = 0;
                        for chunk in column_chunks.iter() {
                            let start = rows.start.max(chunk_start);
                            let end = rows.end.min(chunk_start + chunk.len());
                            if start < end {
                                slice_chunks.push(chunk.slice(start - chunk_start, end - start));
                            }
                            chunk_start += chunk.len();
                        }
                        (slice_chunks.into(), None)
                    }
                };
                sender.send(ColumnChunkJob {
                    context: context.clone(),
                    column_index,
                    first_leaf_index,
                    batches,
                    gather,
                    slice: ColumnSlice {
                        index: slice_index,
                        count: slice_count,
                    },
                    shredding: self.shredding.column_shredding[column_index].clone(),
                    target_node,
                })?;
            }
            first_leaf_index += column_leaf_count;
        }
        Ok(())
    }
}

impl UnaryFactory<MergedRun, ColumnChunkJob> for RowGroupCollector {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<MergedRun, ColumnChunkJob> for RowGroupCollector {
    fn consume(
        &mut self,
        run: MergedRun,
        sender: &mut dyn Sender<ColumnChunkJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        // The runs are merged on any worker and arrive in any order. Keep
        // this one until its turn, then take every run whose turn has come
        // and cut it where output row groups end.
        self.arrived.insert(run.sequence, run);
        while let Some(run) = self.arrived.remove(&self.next_sequence) {
            self.next_sequence += 1;
            let mut first_row = 0;
            while first_row < run.row_count() {
                let rows =
                    (self.rows_per_group - self.rows_filling).min(run.row_count() - first_row);
                self.filling
                    .push(RowGroupPart::new(&run, first_row..first_row + rows));
                first_row += rows;
                self.rows_filling += rows;
                self.rows_taken += rows;
                // The last output row group ends with the rows.
                if self.rows_filling == self.rows_per_group || self.rows_taken == self.row_count {
                    let parts = std::mem::take(&mut self.filling);
                    self.rows_filling = 0;
                    self.send_column_jobs_of_row_group(parts, sender)?;
                    self.row_groups_sent += 1;
                }
            }
        }
        Ok(())
    }
}
