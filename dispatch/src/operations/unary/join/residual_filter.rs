//! The join's residual predicate at probe time: the pass that weighs each
//! collected key match and discards the ones the predicate rejects, before
//! any match is flagged or emitted.

use crate::arrays::accumulator::{BatchAccumulator, ChunkedColumn};
use crate::memory::SlabAllocator;
use crate::operations::unary;
use crate::operations::unary::join::build_rows::{self, BuildRows};
use crate::operations::unary::join::row_arena::RowId;
use crate::operations::unary::join::{JoinResidualFn, JoinResidualSpec};
use arrow_array::{Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use std::sync::Arc;

/// Applies a join's residual predicate to the collected matches of one
/// drain: gathers the matched rows of both sides into one combined batch,
/// evaluates, and compacts the index lists to the passing matches.
pub(super) struct ResidualFilter {
    spec: JoinResidualSpec,
    eval: JoinResidualFn,
    /// The gatherers of the combined batch, shaped by the first drain's
    /// columns and reused after (every drain gathers the same columns): the
    /// probe side copies rows at the probe indices, the build side gathers
    /// them by row id from the stored batches, so a drain costs its matches
    /// and never a walk over every stored build batch.
    gatherers: Option<Gatherers>,
    allocator: SlabAllocator,
    /// Whether each probe row is emitted at most once (a semi join): with a
    /// residual, the match loop records every candidate match rather than
    /// exiting on the first, so duplicates are dropped here.
    semi: bool,
    /// The probe row this batch's earlier drains last emitted, so a semi
    /// probe row whose matches straddle a drain boundary still comes out once.
    last_emitted_probe_row: Option<u32>,
}

/// The two accumulators a drain gathers into and the schema of the batch
/// their columns combine into.
struct Gatherers {
    probe: BatchAccumulator,
    build: BatchAccumulator,
    combined_schema: SchemaRef,
}

impl ResidualFilter {
    pub(super) fn new(spec: &JoinResidualSpec, semi: bool) -> Self {
        Self {
            spec: spec.clone(),
            eval: (spec.evaluator)(),
            gatherers: None,
            allocator: SlabAllocator::new(false),
            semi,
            last_emitted_probe_row: None,
        }
    }

    /// Forget the previous probe batch's emitted-row tracking; its row indices
    /// mean nothing in the batch about to be probed.
    pub(super) fn begin_probe_batch(&mut self) {
        self.last_emitted_probe_row = None;
    }

    /// Evaluate the predicate over the collected matches and compact both
    /// index lists to the passing ones, returning how many remain. A NULL
    /// verdict rejects, matching SQL's treatment of a non-TRUE join condition.
    pub(super) fn filter_combined_batch<R: RowId>(
        &mut self,
        probe_batch: &RecordBatch,
        build_rows: &BuildRows,
        probe_indices: &mut [u32],
        build_indices: &mut [R],
        matched: usize,
    ) -> unary::Result<usize> {
        let combined = self.gather_combined_batch(
            probe_batch,
            build_rows,
            &probe_indices[..matched],
            &build_indices[..matched],
        )?;
        let mask = (self.eval)(&combined);
        let mut kept = 0;
        for i in 0..matched {
            probe_indices[kept] = probe_indices[i];
            build_indices[kept] = build_indices[i];
            kept += (mask.is_valid(i) && mask.value(i)) as usize;
        }
        if !self.semi {
            return Ok(kept);
        }
        // A probe row's matches are recorded consecutively and probe rows in
        // increasing order, so duplicates to drop are always runs.
        let mut unique = 0;
        for i in 0..kept {
            if self.last_emitted_probe_row != Some(probe_indices[i]) {
                self.last_emitted_probe_row = Some(probe_indices[i]);
                probe_indices[unique] = probe_indices[i];
                build_indices[unique] = build_indices[i];
                unique += 1;
            }
        }
        Ok(unique)
    }

    /// The compact batch of matched rows: the residual's selected probe
    /// columns copied at the probe indices, followed by its selected build
    /// columns gathered at the build row ids. Its refs were rebound to this
    /// layout once by the caller.
    fn gather_combined_batch<R: RowId>(
        &mut self,
        probe_batch: &RecordBatch,
        build_rows: &BuildRows,
        probe_indices: &[u32],
        build_indices: &[R],
    ) -> unary::Result<RecordBatch> {
        let probe_selected = probe_batch.project(&self.spec.probe_column_indices)?;
        if self.gatherers.is_none() {
            self.gatherers = Some(Gatherers::new(
                &probe_selected,
                &build_rows.residual_columns,
                &mut self.allocator,
            ));
        }
        let gatherers = self
            .gatherers
            .as_mut()
            .expect("the gatherers were just shaped");
        gatherers.probe.append_batch_by_indices(
            &probe_selected,
            probe_indices,
            &mut self.allocator,
        );
        gatherers.build.append_from_batches(
            &build_rows.residual_columns,
            build_rows::row_id_shift(),
            R::encode(build_indices),
            &mut self.allocator,
        );
        let mut columns = gatherers
            .probe
            .take_batch(&mut self.allocator)?
            .columns()
            .to_vec();
        columns.extend_from_slice(gatherers.build.take_batch(&mut self.allocator)?.columns());
        let options = RecordBatchOptions::new().with_row_count(Some(probe_indices.len()));
        Ok(RecordBatch::try_new_with_options(
            gatherers.combined_schema.clone(),
            columns,
            &options,
        )?)
    }
}

impl Gatherers {
    /// Accumulators shaped like the selected probe columns and the residual's
    /// build columns, plus the schema of the batch that lays one after the
    /// other.
    fn new(
        probe_selected: &RecordBatch,
        residual_columns: &[ChunkedColumn],
        allocator: &mut SlabAllocator,
    ) -> Self {
        let combined_field = |index: usize, data_type: &DataType| {
            Field::new(format!("combined_{index}"), data_type.clone(), true)
        };
        let probe_fields: Vec<Field> = probe_selected
            .schema()
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| combined_field(index, field.data_type()))
            .collect();
        let build_fields: Vec<Field> = residual_columns
            .iter()
            .enumerate()
            .map(|(index, column)| combined_field(probe_fields.len() + index, column.data_type()))
            .collect();
        let combined_schema = Arc::new(Schema::new(
            probe_fields
                .iter()
                .chain(build_fields.iter())
                .cloned()
                .collect::<Vec<_>>(),
        ));
        Self {
            probe: BatchAccumulator::new(Arc::new(Schema::new(probe_fields)), allocator),
            build: BatchAccumulator::new(Arc::new(Schema::new(build_fields)), allocator),
            combined_schema,
        }
    }
}
