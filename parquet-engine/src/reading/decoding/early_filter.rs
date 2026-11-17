//! Drops the rows of a batch that fail the scan's pushed-down conditions
//! before the variant paths their file keeps no typed leaf for are extracted.
//!
//! Extracting such a path parses every row's binary variant, which costs far
//! more than deciding a comparison on plain columns and on the paths that do
//! have a typed leaf. A row group that extracts one therefore reads its other
//! output columns first, evaluates on them the conditions that need nothing
//! else, and extracts the path only for the rows that passed.

use crate::reading::decoding::column_decoder::{ColumnDecoder, Result};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, NullArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use arrow_select::filter::FilterBuilder;
use planner::compile::{ExprEvalFn, ExprFn};
use std::sync::Arc;

/// What parsing a path out of one row's binary variant costs, counted in
/// element copies. Dropping a batch's failing rows copies every kept row once
/// per array, so it pays only when the parsing it spares outweighs those
/// copies. Measured, the parse takes about 24 ns a row, and copying one
/// fixed-width value or string view about 1.5 ns.
const EXTRACT_COST_IN_ELEMENT_COPIES: usize = 16;

/// How many of a row group's next batches pass unfiltered after a batch whose
/// conditions kept too many rows for dropping the rest to pay. The query's
/// `Filter` evaluates the same conditions again, so evaluating them here on
/// rows they do not narrow is wasted; such a row group is only probed now and
/// then for a stretch they do narrow.
const BATCHES_BETWEEN_PROBES: usize = 7;

/// A pushed-down filter condition a scan evaluates over its own output
/// columns. Every row the scan emits has to pass it for the query to keep the
/// row, so a batch's rows that fail it can be dropped as soon as the columns
/// it reads are decoded.
pub struct ScanCondition {
    /// Builds a worker's evaluator of the condition, over a batch laid out as
    /// the scan's projection.
    evaluator: ExprFn,
    /// The output columns the condition reads.
    output_columns: Vec<usize>,
}

impl ScanCondition {
    pub(crate) fn new(evaluator: ExprFn, output_columns: Vec<usize>) -> Self {
        Self {
            evaluator,
            output_columns,
        }
    }
}

/// Reads one row group's batches on one worker, filtering each by the
/// conditions its cheap columns decide before the costly ones are read.
pub(crate) struct EarlyFilter {
    /// The evaluators of the conditions that read no deferred column.
    evaluators: Vec<ExprEvalFn>,
    /// The output columns that parse binary variant rows, ascending. They
    /// are read last, for the rows that passed.
    deferred_columns: Vec<usize>,
    /// The decoded leaves the deferred columns fold, ascending.
    deferred_leaves: Vec<usize>,
    /// The output schema with every deferred column as a NULL placeholder: the
    /// layout the conditions are evaluated over.
    evaluation_schema: SchemaRef,
    /// Batches still to pass unfiltered, after one the conditions did not
    /// narrow enough.
    batches_until_probe: usize,
}

impl EarlyFilter {
    /// The filter for a row group whose output columns are `column_decoders`,
    /// emitted as `schema`. `None` when none of them parses binary variant
    /// rows, or when no condition can be decided without one that does.
    pub fn new(
        column_decoders: &[ColumnDecoder],
        schema: &Schema,
        conditions: &[ScanCondition],
    ) -> Option<Self> {
        let deferred_columns: Vec<usize> = column_decoders
            .iter()
            .enumerate()
            .filter(|(_, decoder)| decoder.parses_binary_variant_rows())
            .map(|(output_idx, _)| output_idx)
            .collect();
        if deferred_columns.is_empty() {
            return None;
        }
        let evaluators: Vec<ExprEvalFn> = conditions
            .iter()
            .filter(|condition| {
                condition
                    .output_columns
                    .iter()
                    .all(|column| !deferred_columns.contains(column))
            })
            .map(|condition| (condition.evaluator)())
            .collect();
        if evaluators.is_empty() {
            return None;
        }
        let mut deferred_leaves: Vec<usize> = deferred_columns
            .iter()
            .flat_map(|&column| column_decoders[column].leaf_positions().iter().copied())
            .collect();
        deferred_leaves.sort_unstable();
        deferred_leaves.dedup();
        let evaluation_fields: Fields = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(output_idx, field)| {
                if deferred_columns.contains(&output_idx) {
                    Arc::new(Field::new(field.name(), DataType::Null, true))
                } else {
                    field.clone()
                }
            })
            .collect();
        Some(Self {
            evaluators,
            deferred_columns,
            deferred_leaves,
            evaluation_schema: Arc::new(Schema::new(evaluation_fields)),
            batches_until_probe: 0,
        })
    }

    /// Reads the output columns of a batch of `row_count` rows from its
    /// `decoded` leaves, holding only the rows that pass the conditions when
    /// dropping the others saves more extraction than it costs in copies.
    pub fn read_columns(
        &mut self,
        column_decoders: &[ColumnDecoder],
        decoded: &[ArrayRef],
        row_count: usize,
    ) -> Result<Vec<ArrayRef>> {
        if self.batches_until_probe > 0 {
            self.batches_until_probe -= 1;
            return column_decoders
                .iter()
                .map(|decoder| decoder.read(decoded))
                .collect();
        }
        let mut columns = column_decoders
            .iter()
            .enumerate()
            .map(|(output_idx, decoder)| {
                if self.deferred_columns.contains(&output_idx) {
                    Ok(Arc::new(NullArray::new(row_count)) as ArrayRef)
                } else {
                    decoder.read(decoded)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let passing_mask = self.select_passing_rows(RecordBatch::try_new(
            self.evaluation_schema.clone(),
            columns.clone(),
        )?);
        let passing_count = passing_mask.count_set_bits();
        let extraction_saved = (row_count - passing_count)
            * self.deferred_columns.len()
            * EXTRACT_COST_IN_ELEMENT_COPIES;
        // Each kept row is copied once per cheap column and once per leaf of
        // the deferred ones.
        let arrays_copied =
            columns.len() - self.deferred_columns.len() + self.deferred_leaves.len();
        if extraction_saved <= passing_count * arrays_copied {
            self.batches_until_probe = BATCHES_BETWEEN_PROBES;
            for &deferred_column in &self.deferred_columns {
                columns[deferred_column] = column_decoders[deferred_column].read(decoded)?;
            }
            return Ok(columns);
        }
        let passing_rows = FilterBuilder::new(&BooleanArray::new(passing_mask, None))
            .optimize()
            .build();
        let mut passing_leaves = decoded.to_vec();
        for &leaf in &self.deferred_leaves {
            passing_leaves[leaf] = passing_rows.filter(&decoded[leaf])?;
        }
        for (output_idx, column) in columns.iter_mut().enumerate() {
            *column = if self.deferred_columns.contains(&output_idx) {
                column_decoders[output_idx].read(&passing_leaves)?
            } else {
                passing_rows.filter(column)?
            };
        }
        Ok(columns)
    }

    /// The rows of `batch` every condition holds for, a bit per row. A
    /// condition that is NULL for a row fails it, as in the query's `Filter`.
    fn select_passing_rows(&mut self, batch: RecordBatch) -> BooleanBuffer {
        let mut passing_mask: Option<BooleanBuffer> = None;
        for evaluator in &mut self.evaluators {
            let result = evaluator(&batch).into_array(batch.num_rows());
            let result = result.as_boolean();
            let condition_mask = match result.nulls() {
                Some(nulls) => result.values() & nulls.inner(),
                None => result.values().clone(),
            };
            passing_mask = Some(match passing_mask {
                Some(passing_mask) => &passing_mask & &condition_mask,
                None => condition_mask,
            });
        }
        passing_mask.expect("an early filter has a condition to evaluate")
    }
}
