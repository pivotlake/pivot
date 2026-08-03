//! The join's residual predicate at probe time: the pass that weighs each
//! collected key match and discards the ones the predicate rejects, before
//! any match is flagged or emitted.

use crate::operations::unary;
use crate::operations::unary::join::build_rows;
use crate::operations::unary::join::{JoinResidual, JoinResidualFn};
use arrow_array::{Array, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::{Field, Schema, SchemaRef};
use std::sync::Arc;

/// Applies a join's residual predicate to the collected matches of one
/// drain: gathers the matched rows of both sides into one combined batch,
/// evaluates, and compacts the index lists to the passing matches.
pub(super) struct ResidualFilter {
    eval: JoinResidualFn,
    /// The combined batch's schema, shaped by the first drain's arrays and
    /// reused after (every drain gathers the same columns).
    schema: Option<SchemaRef>,
    /// Whether each probe row is emitted at most once (a semi join): with a
    /// residual, the match loop records every candidate match rather than
    /// exiting on the first, so duplicates are dropped here.
    semi: bool,
    /// The probe row this batch's earlier drains last emitted, so a semi
    /// probe row whose matches straddle a drain boundary still comes out once.
    last_emitted_probe_row: Option<u32>,
}

impl ResidualFilter {
    pub(super) fn new(residual_filters: &JoinResidual, semi: bool) -> Self {
        Self {
            eval: (residual_filters.0)(),
            schema: None,
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
    pub(super) fn filter_combined_batch(
        &mut self,
        probe_batch: &RecordBatch,
        build_batches: &[RecordBatch],
        probe_indices: &mut [u32],
        build_indices: &mut [u32],
        matched: usize,
    ) -> unary::Result<usize> {
        let combined = self.gather_combined_batch(
            probe_batch,
            build_batches,
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

    /// The combined batch of the matched rows: every probe input column taken
    /// at the probe indices, then every build input column gathered at the
    /// build row ids. That column order is the layout the predicate's column
    /// refs were bound against.
    fn gather_combined_batch(
        &mut self,
        probe_batch: &RecordBatch,
        build_batches: &[RecordBatch],
        probe_indices: &[u32],
        build_indices: &[u32],
    ) -> unary::Result<RecordBatch> {
        let build_column_count = build_batches[0].num_columns();
        let mut columns = Vec::with_capacity(probe_batch.num_columns() + build_column_count);
        let take_indices = UInt32Array::from(probe_indices.to_vec());
        for column in probe_batch.columns() {
            columns.push(arrow::compute::take(column, &take_indices, None)?);
        }
        let locations: Vec<(usize, usize)> = build_indices
            .iter()
            .map(|&row_id| build_rows::split_row_id(row_id))
            .collect();
        for column_idx in 0..build_column_count {
            let arrays: Vec<&dyn Array> = build_batches
                .iter()
                .map(|batch| batch.column(column_idx).as_ref())
                .collect();
            columns.push(arrow::compute::interleave(&arrays, &locations)?);
        }
        let schema = self
            .schema
            .get_or_insert_with(|| {
                Arc::new(Schema::new(
                    columns
                        .iter()
                        .enumerate()
                        .map(|(i, column)| {
                            Field::new(format!("combined_{i}"), column.data_type().clone(), true)
                        })
                        .collect::<Vec<_>>(),
                ))
            })
            .clone();
        let options = RecordBatchOptions::new().with_row_count(Some(probe_indices.len()));
        Ok(RecordBatch::try_new_with_options(
            schema, columns, &options,
        )?)
    }
}
