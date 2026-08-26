//! Splits each row-group column into the primitive leaves Parquet stores,
//! shredding variant columns on the way.
//!
//! A [`ColumnChunkJob`] holds one top-level column of one row group. Shredding
//! a variant column's documents into typed leaves is the heaviest step of a
//! write, so the shredder does not take such a column in one call. It holds
//! one column at a time and advances a cursor over its rows, shredding and
//! converting at most [`RECORD_BATCH_SIZE`] rows to Parquet leaves per worker
//! turn. Between turns the worker serves its other dataflows, and the columns
//! still queued stay available for peer workers to steal. Once the cursor
//! reaches the end, the column's leaves go out as one [`LeafChunkJob`] each.
//!
//! A column with nothing to shred (a plain column, or a variant the planner
//! left unshredded) only has to be split into its leaves, which costs at most
//! a pass over its null bitmap. That is not worth spreading over turns, so it
//! is split and sent on within the call that received it.

use arrow_array::{Array, ArrayRef};
use dispatch::memory::SlabAllocator;
use dispatch::{DefaultUnaryFactory, RECORD_BATCH_SIZE, Sender, Unary, UnaryResult, WorkStatus};

use super::error::WriteResult;
use super::leaves::{self, Leaf};
use super::shredding;
use super::types::{ColumnChunkJob, LeafChunkJob};

pub(super) type ShredderFactory = DefaultUnaryFactory<Shredder>;

pub(super) fn factories(worker_count: usize) -> Vec<ShredderFactory> {
    DefaultUnaryFactory::create_for_workers(worker_count)
}

#[derive(Default)]
pub(super) struct Shredder {
    /// The column being stepped through. While one is held, no other is taken.
    held: Option<HeldColumn>,
    /// Backs the shredded columns. Initialized on first use so an inactive
    /// shredder holds no ring buffer.
    allocator: Option<SlabAllocator>,
}

impl Unary<ColumnChunkJob, LeafChunkJob> for Shredder {
    fn consume(
        &mut self,
        job: ColumnChunkJob,
        sender: &mut dyn Sender<LeafChunkJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        debug_assert!(
            self.held.is_none(),
            "a shredder holding a column is not fed another"
        );
        let mut column = HeldColumn::new(job);
        if column.job.shredding.is_none() {
            while !column.is_exhausted() {
                column.convert_next_slice(usize::MAX, &mut self.allocator)?;
            }
            return send_leaves(column, sender);
        }
        self.advance(column, sender)
    }

    fn run(&mut self, sender: &mut dyn Sender<LeafChunkJob>) -> UnaryResult<WorkStatus> {
        let Some(column) = self.held.take() else {
            return Ok(WorkStatus::Pending);
        };
        self.advance(column, sender)?;
        Ok(WorkStatus::Ran)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.held.is_none()
    }

    /// A held column keeps the stage from finishing until its leaves are out.
    fn has_pending_work(&self) -> bool {
        self.held.is_some()
    }
}

impl Shredder {
    /// Shred one record batch of `column`: send its leaves if that was the
    /// last, and keep holding it otherwise.
    fn advance(
        &mut self,
        mut column: HeldColumn,
        sender: &mut dyn Sender<LeafChunkJob>,
    ) -> UnaryResult<()> {
        column.convert_next_slice(RECORD_BATCH_SIZE, &mut self.allocator)?;
        if !column.is_exhausted() {
            self.held = Some(column);
            return Ok(());
        }
        send_leaves(column, sender)
    }
}

fn send_leaves(column: HeldColumn, sender: &mut dyn Sender<LeafChunkJob>) -> UnaryResult<()> {
    for job in column.into_leaf_jobs() {
        sender.send(job)?;
    }
    Ok(())
}

/// A column part-way through splitting: the cursor over its batches, and the
/// values and levels gathered so far for each of its leaves.
struct HeldColumn {
    job: ColumnChunkJob,
    batch_index: usize,
    rows_consumed_from_batch: usize,
    /// One per leaf of the column, depth-first, created from the first slice.
    leaves: Vec<LeafAccumulator>,
}

/// One leaf's values and definition levels, gathered slice by slice.
struct LeafAccumulator {
    path: Vec<String>,
    value_chunks: Vec<ArrayRef>,
    def_levels: Option<Vec<i16>>,
    max_def_level: i16,
}

impl HeldColumn {
    fn new(job: ColumnChunkJob) -> Self {
        debug_assert!(!job.batches.is_empty(), "a row group column has rows");
        Self {
            job,
            batch_index: 0,
            rows_consumed_from_batch: 0,
            leaves: Vec::new(),
        }
    }

    fn is_exhausted(&self) -> bool {
        self.batch_index == self.job.batches.len()
    }

    /// Shred (if the column is shredded) and convert the next slice of at most
    /// `row_cap` rows to Parquet leaves, appending their values and levels to
    /// the column's. Shredded values are built on slabs from `allocator`,
    /// which is created the first time a slice needs it.
    fn convert_next_slice(
        &mut self,
        row_cap: usize,
        allocator: &mut Option<SlabAllocator>,
    ) -> WriteResult<()> {
        let slice = self.next_slice(row_cap);
        let values = match &self.job.shredding {
            Some(shredding) => {
                let allocator = allocator.get_or_insert_with(|| SlabAllocator::new(false));
                shredding::shred_column(&slice, shredding, allocator)?
            }
            None => slice,
        };
        let field = self.job.context.schema.field(self.job.column_index);
        let pieces = leaves::to_parquet_leaves(field, &values)?;
        if self.leaves.is_empty() {
            self.leaves = pieces.into_iter().map(LeafAccumulator::start).collect();
            return Ok(());
        }
        debug_assert_eq!(pieces.len(), self.leaves.len());
        for (leaf, piece) in self.leaves.iter_mut().zip(pieces) {
            leaf.append(piece);
        }
        Ok(())
    }

    /// Move the cursor past the next slice and return it: the rest of the
    /// current batch, capped at `row_cap` rows.
    fn next_slice(&mut self, row_cap: usize) -> ArrayRef {
        let batch = &self.job.batches[self.batch_index];
        let rows = (batch.len() - self.rows_consumed_from_batch).min(row_cap);
        let slice = if self.rows_consumed_from_batch == 0 && rows == batch.len() {
            batch.clone()
        } else {
            batch.slice(self.rows_consumed_from_batch, rows)
        };
        self.rows_consumed_from_batch += rows;
        if self.rows_consumed_from_batch == batch.len() {
            self.batch_index += 1;
            self.rows_consumed_from_batch = 0;
        }
        slice
    }

    /// One job per leaf, numbered on from the column's first leaf.
    fn into_leaf_jobs(self) -> impl Iterator<Item = LeafChunkJob> {
        let HeldColumn { job, leaves, .. } = self;
        leaves
            .into_iter()
            .enumerate()
            .map(move |(leaf_offset, leaf)| LeafChunkJob {
                context: job.context.clone(),
                leaf_index: job.first_leaf_index + leaf_offset,
                path: leaf.path,
                value_chunks: leaf.value_chunks,
                def_levels: leaf.def_levels,
                max_def_level: leaf.max_def_level,
                target_node: job.target_node,
            })
    }
}

impl LeafAccumulator {
    fn start(piece: Leaf) -> Self {
        Self {
            path: piece.path,
            value_chunks: vec![piece.values],
            def_levels: piece.def_levels.map(|levels| levels.to_vec()),
            max_def_level: piece.max_def_level,
        }
    }

    fn append(&mut self, piece: Leaf) {
        debug_assert_eq!(piece.path, self.path);
        debug_assert_eq!(piece.max_def_level, self.max_def_level);
        self.value_chunks.push(piece.values);
        if let Some(levels) = &mut self.def_levels {
            let more = piece
                .def_levels
                .expect("a leaf's level depth is fixed by its field");
            levels.extend_from_slice(&more);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use dispatch::TestOperatorIO;
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::CollectSender;
    use parquet_variant_compute::{
        ShreddedSchemaBuilder, VariantArray, json_to_variant, shred_variant,
    };

    use super::*;
    use crate::writing::types::{FileAssemblyInfo, RowGroupContext};

    /// A job over a one-column row group of `field`, whose rows arrive as
    /// `batches`, with the column's leaves numbered from `first_leaf_index`.
    fn column_job(field: Field, batches: Vec<ArrayRef>, first_leaf_index: usize) -> ColumnChunkJob {
        let leaf_count = leaves::count_leaves(&field);
        ColumnChunkJob {
            context: Arc::new(RowGroupContext {
                row_group_id: 0,
                assembly_worker: 0,
                schema: Arc::new(Schema::new(vec![field])),
                leaf_count,
                file_info: Arc::new(FileAssemblyInfo {
                    file_id: 0,
                    row_group_count: 1,
                    partition: None,
                    max_file_size: None,
                }),
            }),
            column_index: 0,
            first_leaf_index,
            batches: batches.into(),
            shredding: None,
            target_node: 0,
        }
    }

    fn int_batch(values: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(values))
    }

    /// A job over a variant column of JSON documents, one batch per entry of
    /// `batches`, planned to shred the documents' `id` into a typed leaf. The
    /// shredder builds its output on slabs, so this also installs a test ring.
    fn shredded_variant_job(batches: Vec<Vec<String>>) -> ColumnChunkJob {
        init_test_free_pool(16);
        let shredding = ShreddedSchemaBuilder::new()
            .with_path("id", &DataType::Int64)
            .unwrap()
            .build();
        let batches: Vec<ArrayRef> = batches
            .into_iter()
            .map(|documents| {
                let json: ArrayRef = Arc::new(StringArray::from(documents));
                Arc::new(json_to_variant(&json).unwrap().into_inner()) as ArrayRef
            })
            .collect();
        let no_rows = VariantArray::try_new(batches[0].slice(0, 0).as_ref()).unwrap();
        let widened = shred_variant(&no_rows, &shredding).unwrap().field("attrs");
        let mut job = column_job(widened, batches, 0);
        job.shredding = Some(Arc::new(shredding));
        job
    }

    fn documents(ids: impl Iterator<Item = i64>) -> Vec<String> {
        ids.map(|id| format!("{{\"id\": {id}}}")).collect()
    }

    fn present_rows(job: &LeafChunkJob) -> usize {
        job.value_chunks.iter().map(|chunk| chunk.len()).sum()
    }

    /// Feed `job` and then take turns until the shredder accepts input again,
    /// returning how many turns the column took.
    fn drive(
        shredder: &mut Shredder,
        job: ColumnChunkJob,
        sender: &mut CollectSender<LeafChunkJob>,
    ) -> usize {
        let mut test_io = TestOperatorIO::default();
        shredder.consume(job, sender, &mut test_io.io()).unwrap();
        let mut turns = 1;
        while !shredder.ready_for_more_work() {
            assert!(shredder.has_pending_work());
            shredder.run(sender).unwrap();
            turns += 1;
        }
        turns
    }

    /// A column with nothing to shred is split into its leaf in the turn that
    /// received it, however many batches it spans.
    #[test]
    fn a_column_with_nothing_to_shred_is_split_in_one_turn() {
        let mut shredder = Shredder::default();
        let mut sender = CollectSender::new();
        let job = column_job(
            Field::new("n", DataType::Int64, true),
            vec![
                int_batch(vec![Some(1), None]),
                int_batch(vec![Some(3)]),
                int_batch(vec![None, Some(5)]),
            ],
            0,
        );

        let turns = drive(&mut shredder, job, &mut sender);

        assert_eq!(turns, 1);
        assert_eq!(sender.items.len(), 1);
        let leaf = &sender.items[0];
        assert_eq!(present_rows(leaf), 3);
        assert_eq!(leaf.def_levels.as_deref(), Some([1, 0, 1, 0, 1].as_slice()));
    }

    /// Each turn shreds one batch, and the column's leaves go out whole once
    /// the last batch is done.
    #[test]
    fn a_shredded_column_takes_one_turn_per_batch_and_emits_its_leaves_once_done() {
        let mut shredder = Shredder::default();
        let mut sender = CollectSender::new();
        let job = shredded_variant_job(vec![documents(0..2), documents(2..3), documents(3..5)]);

        let turns = drive(&mut shredder, job, &mut sender);

        assert_eq!(turns, 3);
        assert!(!shredder.has_pending_work());
        let typed_id = sender
            .items
            .iter()
            .find(|leaf| leaf.path == ["attrs", "typed_value", "id", "typed_value"])
            .expect("the id leaf is one of the column's leaves");
        assert_eq!(typed_id.value_chunks.len(), 3);
        assert_eq!(present_rows(typed_id), 5);
    }

    /// A batch wider than a record batch is stepped through in record-batch
    /// sized slices rather than in one turn.
    #[test]
    fn a_wide_shredded_batch_is_stepped_in_record_batch_sized_slices() {
        let mut shredder = Shredder::default();
        let mut sender = CollectSender::new();
        let rows = 2 * RECORD_BATCH_SIZE + 1;
        let job = shredded_variant_job(vec![documents(0..rows as i64)]);

        let turns = drive(&mut shredder, job, &mut sender);

        assert_eq!(turns, 3);
        assert!(sender.items.iter().all(|leaf| {
            leaf.def_levels
                .as_ref()
                .is_none_or(|levels| levels.len() == rows)
        }));
    }

    /// A struct column comes out as one job per leaf, in depth-first order and
    /// numbered on from the column's first leaf.
    #[test]
    fn a_struct_column_emits_one_job_per_leaf() {
        let mut shredder = Shredder::default();
        let mut sender = CollectSender::new();
        let inner = Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]);
        let batch: ArrayRef = Arc::new(StructArray::new(
            inner.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["x", "y"])),
            ],
            None,
        ));
        let job = column_job(
            Field::new("s", DataType::Struct(inner), false),
            vec![batch],
            3,
        );

        drive(&mut shredder, job, &mut sender);

        let numbered: Vec<(usize, Vec<String>)> = sender
            .items
            .iter()
            .map(|leaf| (leaf.leaf_index, leaf.path.clone()))
            .collect();
        assert_eq!(
            numbered,
            vec![
                (3, vec!["s".to_string(), "a".to_string()]),
                (4, vec!["s".to_string(), "b".to_string()]),
            ]
        );
    }
}
