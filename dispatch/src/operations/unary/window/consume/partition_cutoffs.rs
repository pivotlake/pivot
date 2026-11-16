//! Rejects rows that cannot pass a `row_number()` limit as they arrive,
//! before they are held, sorted or copied.
//!
//! Once a worker holds `limit` rows of a partition, the last of them is the
//! partition's cutoff: a later row that sorts at or after it can never be
//! among the partition's first `limit` rows, so it is dropped on arrival.
//! (A row that ties with the cutoff could take its place, but which of two
//! tied rows gets the last number is unspecified, so dropping it is valid.)
//!
//! ```text
//!  incoming row ──► look its partition key up ──► not found: keep
//!                        │ found: the partition's cutoff row
//!                        ▼
//!                  compare order keys ──► row sorts before the cutoff: keep
//!                                    └──► otherwise: drop
//! ```
//!
//! The lookup is the hash table and row key encoding that GROUP BY uses: the
//! table maps a partition key to where that partition's cutoff row is held.
//! Cutoff rows are kept per bucket, in one small batch that is replaced each
//! time the bucket's rows have all been merged and cut, which is when the
//! worker knows the bucket's cutoffs exactly.
//!
//! Each worker has its own cutoffs, from the rows it has seen itself.

use std::cmp::Ordering;
use std::sync::Arc;

use ahash::RandomState;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, Schema};

use crate::arrays::take_chunked;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::{
    ArenaKey, DEFAULT_CAPACITY, HashTable, KeyExtractor, RowKeyExtractor, RowKeySchema, RowScratch,
    StoredValue,
};
use crate::operations::unary::order_by::RowComparer;
use crate::operations::unary::order_by_limit::OrderBy;
use crate::operations::unary::window::WindowLayout;

/// Where a partition's cutoff row is held: which bucket's cutoff batch, and
/// which row of it. This is the value the table stores for each partition.
#[derive(Clone, Copy, Default)]
struct CutoffRow {
    bucket: u32,
    row: u32,
}

impl StoredValue for CutoffRow {
    type StorageMetadata = ();

    fn stored_size(_metadata: ()) -> usize {
        size_of::<Self>()
    }

    fn stored_align() -> usize {
        align_of::<Self>()
    }

    #[inline(always)]
    unsafe fn from_entry<'a>(ptr: *const u8, _metadata: ()) -> &'a Self {
        unsafe { &*(ptr as *const Self) }
    }

    #[inline(always)]
    unsafe fn from_entry_mut<'a>(ptr: *mut u8, _metadata: ()) -> &'a mut Self {
        unsafe { &mut *(ptr as *mut Self) }
    }
}

/// One worker's cutoff row for every partition it holds `limit` rows of (see
/// the module docs).
pub struct PartitionCutoffs {
    partition_by: Vec<usize>,
    order_by: Vec<OrderBy>,
    /// How GROUP BY's row key encoding reads the partition columns.
    key_schema: RowKeySchema,
    key_hash: RandomState,
    /// Partition key to the position of the partition's cutoff row.
    table: HashTable<ArenaKey, CutoffRow>,
    table_capacity: usize,
    /// Holds the partition keys too long to sit inside a table entry.
    key_arena: WorkerArena,
    /// Ring memory for the table and the cutoff rows.
    allocator: SlabAllocator,
    /// Per bucket, the cutoff row of each of its partitions that has one. A
    /// bucket without cutoffs has an empty batch.
    cutoff_rows_by_bucket: Vec<RecordBatch>,
    /// Scratch space reused from batch to batch.
    key_scratch: RowScratch,
    key_hashes: Vec<u64>,
}

impl PartitionCutoffs {
    /// Creates empty cutoffs for rows of `schema`. `None` when a partition
    /// key is of a type GROUP BY's row key encoding does not cover (a float
    /// or a boolean, say). The window then runs without cutoffs.
    pub fn try_new(layout: &WindowLayout, schema: &Arc<Schema>) -> Option<Self> {
        let partition_by = layout.spec.partition_by.clone();
        let key_types = partition_by
            .iter()
            .map(|&column| match schema.field(column).data_type() {
                // Dates and times are keyed as the integers they are stored as.
                DataType::Date32 | DataType::Time32(_) => Some(DataType::Int32),
                DataType::Date64
                | DataType::Timestamp(_, _)
                | DataType::Time64(_)
                | DataType::Duration(_) => Some(DataType::Int64),
                key_type @ (DataType::Decimal64(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Utf8View) => Some(key_type.clone()),
                key_type if key_type.is_integer() => Some(key_type.clone()),
                _ => None,
            })
            .collect::<Option<Vec<DataType>>>()?;
        let key_nullability: Vec<bool> = partition_by
            .iter()
            .map(|&column| schema.field(column).is_nullable())
            .collect();

        let mut allocator = SlabAllocator::new(false);
        let table = HashTable::with_storage_metadata(&mut allocator, DEFAULT_CAPACITY, 0, ());
        Some(Self {
            partition_by,
            order_by: layout.spec.order_by.clone(),
            key_schema: RowKeySchema::new(key_types, key_nullability),
            key_hash: RandomState::new(),
            table,
            table_capacity: DEFAULT_CAPACITY,
            key_arena: WorkerArena::new(SharedArena::new(layout.arena_buffers)),
            allocator,
            cutoff_rows_by_bucket: vec![
                RecordBatch::new_empty(schema.clone());
                layout.bucket_count
            ],
            key_scratch: RowScratch::default(),
            key_hashes: Vec::new(),
        })
    }

    /// How many partitions have a cutoff.
    pub fn partition_count(&self) -> usize {
        self.table.len()
    }

    /// Replaces the cutoffs of `bucket` with the rows of `run` at
    /// `cutoff_positions`, each a `(batch, row)` position.
    ///
    /// `run` must hold everything the worker has for the bucket, and the
    /// positions must name the `limit`-th row of every partition that has
    /// that many. Every partition that had a cutoff before then has one
    /// again, so no table entry is left pointing into the replaced batch.
    pub fn replace_bucket_cutoffs(
        &mut self,
        bucket: usize,
        run: &[RecordBatch],
        cutoff_positions: &[(u32, u32)],
    ) -> Result<(), ArrowError> {
        if cutoff_positions.is_empty() {
            return Ok(());
        }
        let schema = run[0].schema();
        let columns = (0..schema.fields().len())
            .map(|column| {
                let run_column: Vec<ArrayRef> = run
                    .iter()
                    .map(|batch| batch.column(column).clone())
                    .collect();
                take_chunked(&mut self.allocator, &run_column, cutoff_positions)
            })
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        let cutoff_rows = RecordBatch::try_new(schema, columns)?;

        self.key_hashes.clear();
        self.key_hashes.resize(cutoff_rows.num_rows(), 0);
        let mut key_reader = RowKeyExtractor::make_reader(
            &cutoff_rows,
            &self.partition_by,
            &self.key_schema,
            &mut self.key_scratch,
        );
        RowKeyExtractor::prepare_and_hash(&mut key_reader, &self.key_hash, &mut self.key_hashes);
        let mut prober = self.table.prober();
        for (row, &key_hash) in self.key_hashes.iter().enumerate() {
            prober.grow_if_full(&mut self.allocator, &mut self.table_capacity);
            let key = RowKeyExtractor::live_key(&key_reader, row, &mut self.key_arena);
            let cutoff = CutoffRow {
                bucket: bucket as u32,
                row: row as u32,
            };
            prober.probe_fold::<false, _, _, _, _>(
                key_hash,
                key,
                cutoff,
                |cutoff, stored| *stored = cutoff,
                |cutoff, stored| *stored = cutoff,
            );
        }
        drop(key_reader);
        self.cutoff_rows_by_bucket[bucket] = cutoff_rows;
        Ok(())
    }

    /// Replaces `kept_rows` with the rows of `batch` that are not past their
    /// partition's cutoff, in order.
    pub fn collect_rows_before_cutoffs(
        &mut self,
        batch: &RecordBatch,
        kept_rows: &mut Vec<u32>,
    ) -> Result<(), ArrowError> {
        let row_count = batch.num_rows();
        kept_rows.clear();
        if self.table.len() == 0 {
            kept_rows.extend(0..row_count as u32);
            return Ok(());
        }

        self.key_hashes.clear();
        self.key_hashes.resize(row_count, 0);
        let mut key_reader = RowKeyExtractor::make_reader(
            batch,
            &self.partition_by,
            &self.key_schema,
            &mut self.key_scratch,
        );
        RowKeyExtractor::prepare_and_hash(&mut key_reader, &self.key_hash, &mut self.key_hashes);
        let mut order_comparer = RowComparer::try_new(
            &self.order_by,
            &self.cutoff_rows_by_bucket,
            std::slice::from_ref(batch),
        )?;
        let prober = self.table.prober();
        for (row, &key_hash) in self.key_hashes.iter().enumerate() {
            let key = RowKeyExtractor::live_key(&key_reader, row, &mut self.key_arena);
            let is_past_cutoff = match prober.find(key_hash, &key) {
                None => false,
                Some(cutoff) => {
                    let cutoff_position = (cutoff.bucket as usize, cutoff.row as usize);
                    order_comparer.compare(cutoff_position, (0, row)) != Ordering::Greater
                }
            };
            if !is_past_cutoff {
                kept_rows.push(row as u32);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::window::{WindowFunction, WindowRowLimit, WindowSpec};
    use arrow_array::{Float64Array, Int64Array};
    use arrow_schema::Field;

    /// A batch of `(team, score)` rows.
    fn team_score_batch(rows: &[(Option<i64>, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("team", DataType::Int64, true),
            Field::new("score", DataType::Int64, false),
        ]));
        let teams: Int64Array = rows.iter().map(|&(team, _)| team).collect();
        let scores: Int64Array = rows.iter().map(|&(_, score)| Some(score)).collect();
        RecordBatch::try_new(schema, vec![Arc::new(teams), Arc::new(scores)]).unwrap()
    }

    /// The layout of `row_number() OVER (PARTITION BY team ORDER BY score)`
    /// limited to the first two rows.
    fn lowest_two_scores_per_team() -> WindowLayout {
        let spec = WindowSpec {
            partition_by: vec![0],
            order_by: vec![OrderBy::new(1, false, false)],
            functions: vec![WindowFunction::RowNumber],
            row_limit: Some(WindowRowLimit {
                function_index: 0,
                limit: 2,
            }),
        };
        WindowLayout::new(spec, 1, 64)
    }

    #[test]
    fn rows_at_or_after_their_partitions_cutoff_are_dropped() {
        init_test_free_pool(16);
        let layout = lowest_two_scores_per_team();
        let kept_run = team_score_batch(&[(Some(1), 10), (Some(1), 20), (None, 7), (None, 8)]);
        let mut cutoffs = PartitionCutoffs::try_new(&layout, &kept_run.schema()).unwrap();
        cutoffs
            .replace_bucket_cutoffs(0, &[kept_run], &[(0, 1), (0, 3)])
            .unwrap();
        let arriving = team_score_batch(&[
            (Some(1), 15),
            (Some(1), 20),
            (Some(1), 30),
            (Some(2), 99),
            (None, 9),
            (None, 1),
        ]);
        let mut kept_rows = Vec::new();

        cutoffs
            .collect_rows_before_cutoffs(&arriving, &mut kept_rows)
            .unwrap();

        assert_eq!(kept_rows, vec![0, 3, 5]);
    }

    #[test]
    fn a_partition_key_the_table_cannot_hold_gets_no_cutoffs() {
        init_test_free_pool(16);
        let layout = lowest_two_scores_per_team();
        let schema = Arc::new(Schema::new(vec![
            Field::new("team", DataType::Float64, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.5])),
                Arc::new(Int64Array::from(vec![10])),
            ],
        )
        .unwrap();

        let cutoffs = PartitionCutoffs::try_new(&layout, &batch.schema());

        assert!(cutoffs.is_none());
    }
}
