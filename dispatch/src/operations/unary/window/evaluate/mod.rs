//! Evaluates window functions over a sorted sequence of batches: a whole
//! sorted bucket in the output phase, or one sorted run when the row limit
//! is applied during the consume phase.
//!
//! ```text
//!  WindowEvaluator::for_output(batches)
//!      find where every partition and tie group starts   (boundaries)
//!
//!  evaluate_next_batch(), evaluate_next_batch(), ...       one call per batch
//!      compute each function for the batch's rows         (functions)
//!      mark the rows that pass the row limit
//! ```
//!
//! The boundaries are found for the whole sequence before the first batch is
//! evaluated, so a function can use the end of a partition that lies in a
//! later batch.

mod boundaries;
mod functions;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::ArrowError;

use super::{WindowLayout, WindowRowLimit};
use crate::memory::SlabAllocator;
use boundaries::SequenceBoundaries;
use functions::FunctionEvaluator;
pub use functions::WindowFunction;

/// Evaluates a set of window functions over one sorted sequence of batches.
pub struct WindowEvaluator {
    functions: Vec<Box<dyn FunctionEvaluator>>,
    row_limit: Option<WindowRowLimit>,
    boundaries: SequenceBoundaries,
    /// The batch the next call to `evaluate_next_batch` evaluates.
    next_batch: usize,
}

impl WindowEvaluator {
    /// An evaluator of every function of the window, applying its row limit.
    /// `batches` hold all the rows of the sequence, sorted, in order, and
    /// none of them is empty.
    pub fn for_output(layout: &WindowLayout, batches: &[RecordBatch]) -> Result<Self, ArrowError> {
        Self::try_new(
            layout,
            &layout.spec.functions,
            layout.spec.row_limit,
            batches,
        )
    }

    /// An evaluator of only the function `row_limit` applies to, for
    /// dropping rows before the output phase. `batches` are as in
    /// [`for_output`](Self::for_output).
    pub fn for_row_limit(
        layout: &WindowLayout,
        row_limit: WindowRowLimit,
        batches: &[RecordBatch],
    ) -> Result<Self, ArrowError> {
        let limited_function = layout.spec.functions[row_limit.function_index];
        let only_function_limit = WindowRowLimit {
            function_index: 0,
            limit: row_limit.limit,
        };
        Self::try_new(
            layout,
            &[limited_function],
            Some(only_function_limit),
            batches,
        )
    }

    fn try_new(
        layout: &WindowLayout,
        functions: &[WindowFunction],
        row_limit: Option<WindowRowLimit>,
        batches: &[RecordBatch],
    ) -> Result<Self, ArrowError> {
        // The order keys are compared only when a function depends on ties.
        let needs_tie_groups = functions.iter().any(|function| function.needs_tie_groups());
        let order_columns: Option<Vec<usize>> = needs_tie_groups.then(|| {
            layout
                .spec
                .order_by
                .iter()
                .map(|key| key.column_idx())
                .collect()
        });
        let boundaries =
            SequenceBoundaries::find(batches, &layout.spec.partition_by, order_columns.as_deref())?;
        let functions = functions
            .iter()
            .map(|function| function.build_evaluator(&boundaries, batches))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            functions,
            row_limit,
            boundaries,
            next_batch: 0,
        })
    }

    /// How many partitions the sequence holds.
    pub fn count_partitions(&self) -> usize {
        self.boundaries.partitions.starts().count_set_bits()
    }

    /// Evaluates the next batch of the sequence: the first batch on the
    /// first call, the second on the next, and so on. The caller makes one
    /// call per batch it built the evaluator with. Returns one column per
    /// evaluated function, built on slab memory, and one bit per row that is
    /// set for the rows that pass the row limit. The bits are `None` when
    /// the evaluator applies no limit.
    pub fn evaluate_next_batch(
        &mut self,
        allocator: &mut SlabAllocator,
    ) -> Result<(Vec<ArrayRef>, Option<BooleanBuffer>), ArrowError> {
        let batch_index = self.next_batch;
        self.next_batch += 1;
        let boundaries = &self.boundaries;
        let function_columns: Vec<ArrayRef> = self
            .functions
            .iter_mut()
            .map(|function| function.evaluate_batch(boundaries, batch_index, allocator))
            .collect::<Result<_, _>>()?;

        let rows_within_limit = self.row_limit.map(|row_limit| {
            let limit = i64::try_from(row_limit.limit).unwrap_or(i64::MAX);
            let limited_values = function_columns[row_limit.function_index]
                .as_primitive::<Int64Type>()
                .values();
            BooleanBuffer::collect_bool(limited_values.len(), |row| limited_values[row] <= limit)
        });
        Ok((function_columns, rows_within_limit))
    }
}
