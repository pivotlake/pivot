//! Filters a batch by the scan's pushed-down conditions while it is decoded:
//! the output columns the conditions read are decoded first, and the rest of
//! the batch only for the rows that passed.
//!
//! The rest is every leaf no condition column folds, which is then decoded
//! through the passing rows' positions, and every variant path the file keeps
//! no typed leaf for. Extracting such a path parses each row's binary variant,
//! which costs far more than deciding a comparison, so a condition that reads
//! one is left to the query's `Filter`.
//!
//! The passing rows are carried as ascending positions, not as runs of kept
//! and dropped rows. A decoder walking runs pays per run, and a filter's
//! passing rows come a few at a time, while reading by position costs per
//! passing row however they are spread.

use crate::reading::decoding::column_decoder::{ColumnDecoder, Result, read_columns};
use crate::reading::decoding::leaf_decoders::LeafDecoder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, NullArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use dispatch::arrays::take;
use dispatch::memory::SlabAllocator;
use planner::compile::{ExprEvalFn, ExprFn};
use std::sync::Arc;

/// What parsing a path out of one row's binary variant costs, counted in
/// element copies. Measured, the parse takes about 24 ns a row, and copying
/// one fixed-width value or string view about 1.5 ns.
const EXTRACT_COST_IN_ELEMENT_COPIES: usize = 16;

/// What decoding one row of one leaf through the passing rows' positions
/// costs, counted in element copies. A row decoded with its whole batch costs
/// one.
const SELECTED_DECODE_COST_IN_ELEMENT_COPIES: usize = 2;

/// How many of a row group's next batches are decoded whole after a batch
/// whose conditions kept too many rows for dropping the rest to pay. The
/// query's `Filter` evaluates the same conditions again, so evaluating them
/// here on rows they do not narrow is wasted; such a row group is only probed
/// now and then for a stretch they do narrow.
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
/// conditions before the columns they do not read are decoded.
pub(crate) struct EarlyFilter {
    /// The evaluators of the conditions that read no column parsing binary
    /// variant rows.
    evaluators: Vec<ExprEvalFn>,
    /// Per output column, whether one of those conditions reads it. Such a
    /// column is read first, over every row.
    is_condition_column: Vec<bool>,
    /// The decoded leaves the condition columns fold, ascending.
    condition_leaves: Vec<usize>,
    /// The condition leaves another output column folds too, ascending. They
    /// are narrowed to the passing rows for that column to fold.
    shared_leaves: Vec<usize>,
    /// Every other leaf, ascending: decoded for the passing rows only.
    late_leaves: Vec<usize>,
    /// The output schema with every column no condition reads as a NULL
    /// placeholder: the layout the conditions are evaluated over.
    evaluation_schema: SchemaRef,
    /// What a row costs when its batch is decoded whole and when only the
    /// batch's passing rows are, counted in element copies and leaving out
    /// the condition columns, which cost the same either way.
    whole_row_cost: usize,
    passing_row_cost: usize,
    /// Batches still to decode whole, after one the conditions did not narrow
    /// enough.
    batches_until_probe: usize,
}

impl EarlyFilter {
    /// The filter for a row group whose output columns are `column_decoders`,
    /// folding `leaf_count` decoded leaves and emitted as `schema`. `None`
    /// when no condition can be decided without parsing binary variant rows,
    /// or when the conditions read everything the row group decodes, which
    /// leaves nothing to spare.
    pub fn new(
        column_decoders: &[ColumnDecoder],
        leaf_count: usize,
        schema: &Schema,
        conditions: &[ScanCondition],
    ) -> Option<Self> {
        let parses_binary_variant_rows: Vec<bool> = column_decoders
            .iter()
            .map(ColumnDecoder::parses_binary_variant_rows)
            .collect();
        let mut is_condition_column = vec![false; column_decoders.len()];
        let mut evaluators = Vec::new();
        for condition in conditions {
            if condition
                .output_columns
                .iter()
                .any(|&column| parses_binary_variant_rows[column])
            {
                continue;
            }
            for &column in &condition.output_columns {
                is_condition_column[column] = true;
            }
            evaluators.push((condition.evaluator)());
        }
        if evaluators.is_empty() {
            return None;
        }

        let mut is_condition_leaf = vec![false; leaf_count];
        let mut is_shared_leaf = vec![false; leaf_count];
        for (decoder, _) in column_decoders
            .iter()
            .zip(&is_condition_column)
            .filter(|(_, is_condition_column)| **is_condition_column)
        {
            for &leaf in decoder.leaf_positions() {
                is_condition_leaf[leaf] = true;
            }
        }
        for (decoder, _) in column_decoders
            .iter()
            .zip(&is_condition_column)
            .filter(|(_, is_condition_column)| !**is_condition_column)
        {
            for &leaf in decoder.leaf_positions() {
                is_shared_leaf[leaf] = is_condition_leaf[leaf];
            }
        }
        let leaves_where = |flags: &[bool], wanted: bool| -> Vec<usize> {
            (0..leaf_count)
                .filter(|&leaf| flags[leaf] == wanted)
                .collect()
        };
        let late_leaves = leaves_where(&is_condition_leaf, false);
        let shared_leaves = leaves_where(&is_shared_leaf, true);
        let parsed_column_count = parses_binary_variant_rows
            .iter()
            .filter(|parses| **parses)
            .count();
        if late_leaves.is_empty() && parsed_column_count == 0 {
            return None;
        }

        let condition_column_count = is_condition_column
            .iter()
            .filter(|is_condition_column| **is_condition_column)
            .count();
        let extract_cost = parsed_column_count * EXTRACT_COST_IN_ELEMENT_COPIES;
        let evaluation_fields: Fields = schema
            .fields()
            .iter()
            .zip(&is_condition_column)
            .map(|(field, &is_condition_column)| {
                if is_condition_column {
                    field.clone()
                } else {
                    Arc::new(Field::new(field.name(), DataType::Null, true))
                }
            })
            .collect();
        Some(Self {
            evaluators,
            is_condition_column,
            condition_leaves: leaves_where(&is_condition_leaf, true),
            whole_row_cost: late_leaves.len() + extract_cost,
            // A passing row is also copied once per condition column and
            // shared leaf, to line up with the columns read after it.
            passing_row_cost: late_leaves.len() * SELECTED_DECODE_COST_IN_ELEMENT_COPIES
                + extract_cost
                + condition_column_count
                + shared_leaves.len(),
            shared_leaves,
            late_leaves,
            evaluation_schema: Arc::new(Schema::new(evaluation_fields)),
            batches_until_probe: 0,
        })
    }

    /// Decodes the next `row_count` rows into the output columns, holding
    /// only the rows that pass the conditions when leaving the others
    /// undecoded saves more than narrowing the batch costs.
    pub fn read_columns(
        &mut self,
        column_decoders: &[ColumnDecoder],
        leaf_decoders: &mut [Box<dyn LeafDecoder>],
        allocator: &mut SlabAllocator,
        row_count: usize,
    ) -> Result<Vec<ArrayRef>> {
        if self.batches_until_probe > 0 {
            self.batches_until_probe -= 1;
            return read_columns(column_decoders, leaf_decoders, allocator, row_count);
        }
        let not_decoded: ArrayRef = Arc::new(NullArray::new(0));
        let mut decoded = vec![not_decoded; leaf_decoders.len()];
        for &leaf in &self.condition_leaves {
            decoded[leaf] = leaf_decoders[leaf].read(allocator, row_count)?;
        }
        let mut columns = column_decoders
            .iter()
            .zip(&self.is_condition_column)
            .map(|(decoder, &is_condition_column)| {
                if is_condition_column {
                    decoder.read(&decoded)
                } else {
                    Ok(Arc::new(NullArray::new(row_count)) as ArrayRef)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let passing_mask = self.select_passing_rows(RecordBatch::try_new(
            self.evaluation_schema.clone(),
            columns.clone(),
        )?);
        let passing_count = passing_mask.count_set_bits();

        if passing_count * self.passing_row_cost < row_count * self.whole_row_cost {
            let mut passing_positions = Vec::with_capacity(passing_count);
            passing_positions.extend(passing_mask.set_indices_u32());
            for &leaf in &self.late_leaves {
                decoded[leaf] =
                    leaf_decoders[leaf].read_selected(allocator, row_count, &passing_positions)?;
            }
            for &leaf in &self.shared_leaves {
                decoded[leaf] = take(allocator, &decoded[leaf], &passing_positions)?;
            }
            for (column, _) in columns
                .iter_mut()
                .zip(&self.is_condition_column)
                .filter(|(_, is_condition_column)| **is_condition_column)
            {
                *column = take(allocator, column, &passing_positions)?;
            }
        } else {
            self.batches_until_probe = BATCHES_BETWEEN_PROBES;
            for &leaf in &self.late_leaves {
                decoded[leaf] = leaf_decoders[leaf].read(allocator, row_count)?;
            }
        }
        for ((column, decoder), _) in columns
            .iter_mut()
            .zip(column_decoders)
            .zip(&self.is_condition_column)
            .filter(|(_, is_condition_column)| !**is_condition_column)
        {
            *column = decoder.read(&decoded)?;
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
