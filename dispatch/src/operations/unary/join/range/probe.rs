//! Probe a build table sorted by the range-join key.
//!
//! Each probe key divides the build table at a single boundary. Depending on
//! the comparison, its matches are either every build row before that boundary
//! or every row from it onward. Locating the boundary takes two binary searches:
//! one over chunk fences and one within the selected chunk. Matching rows are
//! then appended as whole Arrow ranges while the corresponding probe row is
//! repeated beside them.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Schema, SchemaRef};

use crate::RECORD_BATCH_SIZE;
use crate::arrays::accumulator::BatchAccumulator;
use crate::memory::SlabAllocator;
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::build::filter_null_keys;
use crate::operations::unary::join::range::{RangeCompare, RangeJoinSpec, RangeTable, SortedChunk};

pub struct RangeProbe<T: ArrowPrimitiveType> {
    build_table: RangeTable<T>,
    spec: Arc<RangeJoinSpec>,
    batch_builder: JoinedBatchBuilder,
}

impl<T: ArrowPrimitiveType> RangeProbe<T>
where
    T::Native: Ord,
{
    pub(crate) fn new(table: RangeTable<T>, spec: Arc<RangeJoinSpec>) -> Self {
        let batch_builder = JoinedBatchBuilder::new(&spec);
        Self {
            build_table: table,
            spec,
            batch_builder,
        }
    }
}

impl<T: ArrowPrimitiveType> Unary<RecordBatch, RecordBatch> for RangeProbe<T>
where
    T::Native: Ord + Send,
{
    fn consume(
        &mut self,
        probe_batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        // The build operator publishes this table with a Release store before
        // the probe gate's Acquire load allows this operator to run.
        let build_chunks = unsafe { &*self.build_table.chunks.get() };
        if build_chunks.is_empty() {
            return Ok(());
        }

        let probe_batch = filter_null_keys(probe_batch, &[self.spec.probe_key_index]);
        if probe_batch.num_rows() == 0 {
            return Ok(());
        }

        let projected_probe = probe_batch.project(&self.spec.probe_output_indices)?;
        let probe_keys = probe_batch
            .column(self.spec.probe_key_index)
            .as_primitive::<T>();

        for probe_row_index in 0..probe_batch.num_rows() {
            // The loop bounds the index, and NULL probe keys were removed above.
            let probe_key = unsafe { probe_keys.value_unchecked(probe_row_index) };
            let probe_row = probe_row_index as u32;
            let boundary = find_build_boundary(build_chunks, probe_key, self.spec.compare);
            let (earlier_chunks, boundary_and_later) = build_chunks.split_at(boundary.chunk_index);
            let (boundary_chunk, later_chunks) = boundary_and_later
                .split_first()
                .expect("a non-empty build table always has a boundary chunk");

            if self.spec.compare.matches_key_above_boundary() {
                self.batch_builder.append_build_range(
                    probe_row,
                    &projected_probe,
                    boundary_chunk,
                    boundary.row_index..boundary_chunk.keys.len(),
                    sender,
                )?;
                self.batch_builder.append_build_chunks(
                    probe_row,
                    &projected_probe,
                    later_chunks,
                    sender,
                )?;
            } else {
                self.batch_builder.append_build_chunks(
                    probe_row,
                    &projected_probe,
                    earlier_chunks,
                    sender,
                )?;
                self.batch_builder.append_build_range(
                    probe_row,
                    &projected_probe,
                    boundary_chunk,
                    0..boundary.row_index,
                    sender,
                )?;
            }
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        self.batch_builder.flush_pending(sender)?;
        Ok(true)
    }
}

/// A position in the build table's concatenated key order.
struct BuildBoundary {
    chunk_index: usize,
    row_index: usize,
}

/// Find the first build key on the upper side of the comparison's boundary.
/// Rows before the returned position are below the boundary; rows at or after
/// it are above. The position may be at the end of its chunk.
fn find_build_boundary<T: ArrowPrimitiveType>(
    build_chunks: &[SortedChunk<T>],
    probe_key: T::Native,
    comparison: RangeCompare,
) -> BuildBoundary
where
    T::Native: Ord,
{
    let build_key_is_before_boundary = |build_key: T::Native| {
        if comparison.splits_at_lower_bound() {
            build_key < probe_key
        } else {
            build_key <= probe_key
        }
    };

    // The last chunk starting before the boundary is the only one that can
    // contain it. If the boundary precedes the table, search the first chunk.
    let chunk_index = build_chunks
        .partition_point(|chunk| build_key_is_before_boundary(chunk.first_key()))
        .saturating_sub(1);
    let row_index = build_chunks[chunk_index]
        .keys
        .values()
        .partition_point(|&build_key| build_key_is_before_boundary(build_key));

    BuildBoundary {
        chunk_index,
        row_index,
    }
}

/// Accumulates joined probe/build rows and emits bounded record batches.
struct JoinedBatchBuilder {
    joined_schema: SchemaRef,
    probe_columns: BatchAccumulator,
    build_columns: BatchAccumulator,
    allocator: SlabAllocator,
    /// Scratch space for repeating one probe row beside many build rows.
    repeated_probe_row_indices: Vec<u32>,
}

impl JoinedBatchBuilder {
    fn new(spec: &RangeJoinSpec) -> Self {
        let mut allocator = SlabAllocator::new(false);
        let joined_fields: Vec<_> = spec
            .probe_fields
            .iter()
            .chain(&spec.build_fields)
            .cloned()
            .collect();
        Self {
            joined_schema: Arc::new(Schema::new(joined_fields)),
            probe_columns: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(spec.probe_fields.clone())),
                &mut allocator,
            ),
            build_columns: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(spec.build_fields.clone())),
                &mut allocator,
            ),
            allocator,
            repeated_probe_row_indices: vec![0; RECORD_BATCH_SIZE],
        }
    }

    /// Pair one probe row with every row in each complete build chunk.
    fn append_build_chunks<T: ArrowPrimitiveType>(
        &mut self,
        probe_row_index: u32,
        projected_probe: &RecordBatch,
        build_chunks: &[SortedChunk<T>],
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        for build_chunk in build_chunks {
            self.append_build_range(
                probe_row_index,
                projected_probe,
                build_chunk,
                0..build_chunk.keys.len(),
                sender,
            )?;
        }
        Ok(())
    }

    /// Pair one probe row with the selected rows from a build chunk, flushing
    /// each time the accumulators reach the target output batch size.
    fn append_build_range<T: ArrowPrimitiveType>(
        &mut self,
        probe_row_index: u32,
        projected_probe: &RecordBatch,
        build_chunk: &SortedChunk<T>,
        mut build_rows: Range<usize>,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        while !build_rows.is_empty() {
            let available_rows = RECORD_BATCH_SIZE - self.probe_columns.len();
            let rows_to_append = available_rows.min(build_rows.len());
            self.repeated_probe_row_indices[..rows_to_append].fill(probe_row_index);
            self.probe_columns.append_batch_by_indices(
                projected_probe,
                &self.repeated_probe_row_indices[..rows_to_append],
                &mut self.allocator,
            );
            self.build_columns.append_range(
                &build_chunk.output,
                build_rows.start,
                rows_to_append,
                &mut self.allocator,
            );
            build_rows.start += rows_to_append;

            if self.probe_columns.has_full_batch() {
                self.flush(sender)?;
            }
        }
        Ok(())
    }

    /// Combine and send the rows currently held by both accumulators.
    fn flush(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<()> {
        let probe_batch = self.probe_columns.take_batch(&mut self.allocator)?;
        let build_batch = self.build_columns.take_batch(&mut self.allocator)?;
        debug_assert_eq!(probe_batch.num_rows(), build_batch.num_rows());

        let joined_columns = probe_batch
            .columns()
            .iter()
            .chain(build_batch.columns())
            .cloned()
            .collect();
        let options = RecordBatchOptions::new().with_row_count(Some(probe_batch.num_rows()));
        sender.send(RecordBatch::try_new_with_options(
            self.joined_schema.clone(),
            joined_columns,
            &options,
        )?)?;
        Ok(())
    }

    fn flush_pending(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<()> {
        if !self.probe_columns.is_empty() {
            self.flush(sender)?;
        }
        Ok(())
    }
}
