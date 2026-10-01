//! Streams a sorted file into row groups as its rows are laid out.
//!
//! A stage that knows the file's final row order hands out [`GatherJob`]s: one
//! stretch of the order, naming each row in the decoded batch that holds it.
//! Any worker gathers a stretch into one batch. A single
//! [`StreamingRowGroupPlanner`] puts the stretches back in sequence, cuts the
//! file into row groups and emits a row group's column jobs as soon as its
//! rows are all in, so the decoded input is let go of as the file is encoded
//! instead of being held whole beside a sorted copy. The shredder, encoder and
//! assembler are the ones every write uses. The file's shredding plan is
//! settled before the first row group, from the rows a write of the whole
//! file samples (see
//! [`plan_compaction_shredding`](super::plan_compaction_shredding)).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use dispatch::arrays::take_chunked;
use dispatch::memory::SlabAllocator;
use dispatch::{
    OperatorFactory, OperatorSpec, Sender, Topology, Unary, UnaryFactory, UnaryResult,
    node_work_queue, return_to_worker_mpsc, stealable, to_single_worker_mpsc,
};

use super::error::WriteError;
use super::shredding::FileShredding;
use super::types::{
    AssembledFile, ColumnChunkJob, EncodedLeafChunk, FileAssemblyInfo, LeafChunkJob,
    RowGroupContext,
};
use super::{assembler, encoder, leaves, shredder};

/// One stretch of a file's row order, ready to be copied into a batch.
pub(super) struct GatherJob {
    /// The stretch's place in the file: stretches are laid out in sequence.
    pub(super) sequence: usize,
    /// How many rows the whole file has.
    pub(super) file_rows: usize,
    /// The gathered batch's columns, which lead the source batches' columns.
    pub(super) schema: SchemaRef,
    pub(super) sources: Vec<RecordBatch>,
    /// Each row of the stretch as `(source, row in it)`.
    pub(super) mapping: Vec<(u32, u32)>,
}

pub(super) struct SortedStretch {
    sequence: usize,
    file_rows: usize,
    batch: RecordBatch,
    node_id: usize,
}

struct Gatherer {
    node_id: usize,
    allocator: Option<SlabAllocator>,
}

struct GathererFactory {
    node_id: usize,
}

impl UnaryFactory<GatherJob, SortedStretch> for GathererFactory {
    type Unary = Gatherer;

    fn build_unary(self) -> Gatherer {
        Gatherer {
            node_id: self.node_id,
            allocator: None,
        }
    }
}

impl Unary<GatherJob, SortedStretch> for Gatherer {
    fn consume(
        &mut self,
        job: GatherJob,
        sender: &mut dyn Sender<SortedStretch>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        // Built on the first job: a slab allocator takes a write buffer from
        // the running worker's memory context.
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let columns = (0..job.schema.fields().len())
            .map(|column| {
                let sources: Vec<ArrayRef> = job
                    .sources
                    .iter()
                    .map(|batch| batch.column(column).clone())
                    .collect();
                take_chunked(allocator, &sources, &job.mapping)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(WriteError::from)?;
        let batch = RecordBatch::try_new(job.schema, columns).map_err(WriteError::from)?;
        sender.send(SortedStretch {
            sequence: job.sequence,
            file_rows: job.file_rows,
            batch,
            node_id: self.node_id,
        })?;
        Ok(())
    }
}

pub(super) struct StreamingRowGroupPlanner {
    assembly_worker: usize,
    node_count: usize,
    target_rows_per_group: usize,
    shredding: Arc<FileShredding>,
    /// Leaves per top-level column, numbered depth-first across the columns.
    leaf_counts: Vec<usize>,
    partition: Option<crate::PartitionValues>,
    max_file_size: usize,
    /// Stretches that arrived ahead of their turn.
    arrived: BTreeMap<usize, SortedStretch>,
    next_sequence: usize,
    /// Stretches in sequence, not yet cut into a row group, with their nodes.
    pending: VecDeque<(RecordBatch, usize)>,
    pending_rows: usize,
    /// How many rows the file has, from its first stretch.
    file_rows: Option<usize>,
    /// Set with the first row group, once the file's rows are known.
    file_info: Option<Arc<FileAssemblyInfo>>,
    row_groups_emitted: usize,
}

struct StreamingRowGroupPlannerFactory {
    assembly_worker: usize,
    node_count: usize,
    target_rows_per_group: usize,
    shredding: Arc<FileShredding>,
    partition: Option<crate::PartitionValues>,
    max_file_size: usize,
}

impl UnaryFactory<SortedStretch, ColumnChunkJob> for StreamingRowGroupPlannerFactory {
    type Unary = StreamingRowGroupPlanner;

    fn build_unary(self) -> StreamingRowGroupPlanner {
        StreamingRowGroupPlanner {
            assembly_worker: self.assembly_worker,
            node_count: self.node_count,
            target_rows_per_group: self.target_rows_per_group,
            leaf_counts: self
                .shredding
                .schema
                .fields()
                .iter()
                .map(|field| leaves::count_leaves(field))
                .collect(),
            shredding: self.shredding,
            partition: self.partition,
            max_file_size: self.max_file_size,
            arrived: BTreeMap::new(),
            next_sequence: 0,
            pending: VecDeque::new(),
            pending_rows: 0,
            file_rows: None,
            file_info: None,
            row_groups_emitted: 0,
        }
    }
}

impl StreamingRowGroupPlanner {
    /// Cut the next `rows` pending rows into one row group and emit its
    /// column jobs.
    fn emit_row_group(
        &mut self,
        rows: usize,
        sender: &mut dyn Sender<ColumnChunkJob>,
    ) -> UnaryResult<()> {
        let mut batches = Vec::new();
        let mut rows_by_node = vec![0usize; self.node_count];
        let mut rows_left = rows;
        while rows_left > 0 {
            let (batch, node_id) = self.pending.front().expect("the pending rows are counted");
            let taken = batch.num_rows().min(rows_left);
            rows_by_node[*node_id] += taken;
            if taken == batch.num_rows() {
                batches.push(self.pending.pop_front().unwrap().0);
            } else {
                batches.push(batch.slice(0, taken));
                let rest = batch.slice(taken, batch.num_rows() - taken);
                self.pending.front_mut().unwrap().0 = rest;
            }
            rows_left -= taken;
        }
        self.pending_rows -= rows;
        let target_node = dispatch::dominant_node(&rows_by_node);

        let file_info = match &self.file_info {
            Some(file_info) => file_info.clone(),
            None => {
                let file_rows = self.file_rows.expect("a stretch arrived");
                self.file_info
                    .insert(Arc::new(FileAssemblyInfo {
                        file_id: 0,
                        row_group_count: file_rows.div_ceil(self.target_rows_per_group),
                        partition: self.partition.clone(),
                        max_file_size: Some(self.max_file_size),
                    }))
                    .clone()
            }
        };
        let context = Arc::new(RowGroupContext {
            row_group_id: self.row_groups_emitted as u64,
            assembly_worker: self.assembly_worker,
            schema: self.shredding.schema.clone(),
            leaf_count: self.leaf_counts.iter().sum(),
            file_info,
        });
        self.row_groups_emitted += 1;
        let mut first_leaf_index = 0;
        for (column_index, &column_leaf_count) in self.leaf_counts.iter().enumerate() {
            let column_batches: Arc<[ArrayRef]> = batches
                .iter()
                .map(|batch| batch.column(column_index).clone())
                .collect();
            sender.send(ColumnChunkJob {
                context: context.clone(),
                column_index,
                first_leaf_index,
                batches: column_batches,
                shredding: self.shredding.column_shredding[column_index].clone(),
                target_node,
            })?;
            first_leaf_index += column_leaf_count;
        }
        Ok(())
    }
}

impl Unary<SortedStretch, ColumnChunkJob> for StreamingRowGroupPlanner {
    fn consume(
        &mut self,
        stretch: SortedStretch,
        sender: &mut dyn Sender<ColumnChunkJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        // A fixed-width leaf is materialized into one ring slab by the encoder.
        debug_assert!(
            self.target_rows_per_group <= dispatch::BUFFER_SIZE / 16,
            "a row group's widest column must fit one slab"
        );
        self.file_rows = Some(stretch.file_rows);
        self.arrived.insert(stretch.sequence, stretch);
        while let Some(entry) = self.arrived.first_entry() {
            if *entry.key() != self.next_sequence {
                break;
            }
            let stretch = entry.remove();
            self.pending_rows += stretch.batch.num_rows();
            self.pending.push_back((stretch.batch, stretch.node_id));
            self.next_sequence += 1;
        }
        while self.pending_rows >= self.target_rows_per_group {
            self.emit_row_group(self.target_rows_per_group, sender)?;
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<ColumnChunkJob>) -> UnaryResult<bool> {
        assert!(
            self.arrived.is_empty(),
            "every stretch of the file arrived before it finished"
        );
        if self.pending_rows > 0 {
            self.emit_row_group(self.pending_rows, sender)?;
        }
        if let Some(file_info) = &self.file_info {
            assert_eq!(
                self.row_groups_emitted, file_info.row_group_count,
                "the file's row groups match its rows"
            );
        }
        Ok(true)
    }
}

/// Gather the stretches, plan the file's row groups as they complete, and
/// shred, encode and assemble them.
pub(super) fn encode_gathered_spec<OF>(
    jobs: OperatorSpec<GatherJob, OF>,
    shredding: Arc<FileShredding>,
    partition: Option<crate::PartitionValues>,
    target_rows_per_group: usize,
    max_file_size: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static>
where
    OF: OperatorFactory<GatherJob> + 'static,
{
    let worker_count = jobs.dispatcher().worker_count();
    let topology: Topology = jobs.dispatcher().topology();
    let planner_worker = jobs.dispatcher().next_worker();
    let gatherers: Vec<_> = (0..worker_count)
        .map(|worker| GathererFactory {
            node_id: topology.node_of_worker(worker),
        })
        .collect();
    let planners: Vec<_> = (0..worker_count)
        .map(|_| StreamingRowGroupPlannerFactory {
            assembly_worker: planner_worker,
            node_count: topology.node_count,
            target_rows_per_group,
            shredding: shredding.clone(),
            partition: partition.clone(),
            max_file_size,
        })
        .collect();
    jobs.chain(
        stealable::<GatherJob>(topology).into_iter().collect(),
        gatherers,
    )
    .chain(
        to_single_worker_mpsc::<SortedStretch>(worker_count, planner_worker)
            .into_iter()
            .collect(),
        planners,
    )
    .chain(
        node_work_queue::<ColumnChunkJob>(topology)
            .into_iter()
            .collect(),
        shredder::factories(worker_count),
    )
    .chain(
        node_work_queue::<LeafChunkJob>(topology)
            .into_iter()
            .collect(),
        encoder::factories(worker_count),
    )
    .chain(
        return_to_worker_mpsc::<EncodedLeafChunk>(worker_count)
            .into_iter()
            .collect(),
        assembler::factories(worker_count),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::test_utils::feed_unary;

    fn stretch(sequence: usize, values: &[i64]) -> SortedStretch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        SortedStretch {
            sequence,
            file_rows: 4,
            batch: RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))])
                .unwrap(),
            node_id: 0,
        }
    }

    fn rows_per_batch(job: &ColumnChunkJob) -> Vec<usize> {
        job.batches.iter().map(|batch| batch.len()).collect()
    }

    #[test]
    fn cuts_stretches_into_row_groups_in_sequence() {
        let shredding =
            super::super::shredding::plan_file_shredding(&[stretch(0, &[1]).batch]).unwrap();
        let mut planner = StreamingRowGroupPlannerFactory {
            assembly_worker: 0,
            node_count: 1,
            target_rows_per_group: 3,
            shredding: Arc::new(shredding),
            partition: None,
            max_file_size: usize::MAX,
        }
        .build_unary();

        let mut jobs = feed_unary(&mut planner, vec![stretch(1, &[3, 4]), stretch(0, &[1, 2])]);
        let mut sender = dispatch::test_utils::CollectSender::new();
        assert!(planner.finish(&mut sender).unwrap());
        jobs.extend(sender.items);

        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].context.row_group_id, 0);
        assert_eq!(rows_per_batch(&jobs[0]), vec![2, 1]);
        assert_eq!(jobs[1].context.row_group_id, 1);
        assert_eq!(rows_per_batch(&jobs[1]), vec![1]);
        assert_eq!(jobs[1].context.file_info.row_group_count, 2);
    }
}
