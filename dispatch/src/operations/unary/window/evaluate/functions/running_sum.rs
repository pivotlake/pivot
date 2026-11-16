//! `sum(x)` over the rows from the partition's first row up to each row.
//!
//! It is the one function that carries a total from batch to batch.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_schema::{ArrowError, DataType};

use super::FunctionEvaluator;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::window::evaluate::boundaries::SequenceBoundaries;

pub struct RunningSumEvaluator {
    /// The argument column of every batch of the sequence.
    argument_batches: Vec<ArrayRef>,
    /// Whether a row's sum also covers the rows after it that tie with it
    /// on the order keys.
    through_tie_group: bool,
    running_sum: RunningSum,
    /// Scratch space reused from batch to batch.
    tie_group_ends: Vec<u64>,
}

impl RunningSumEvaluator {
    /// An evaluator of the sum of column `column` of `batches`, or an error
    /// when the column's type is one a running sum does not cover.
    pub fn try_new(
        batches: &[RecordBatch],
        column: usize,
        through_tie_group: bool,
    ) -> Result<Self, ArrowError> {
        Ok(Self {
            running_sum: RunningSum::try_new(batches[0].column(column).data_type())?,
            argument_batches: batches
                .iter()
                .map(|batch| batch.column(column).clone())
                .collect(),
            through_tie_group,
            tie_group_ends: Vec::new(),
        })
    }
}

impl FunctionEvaluator for RunningSumEvaluator {
    fn evaluate_batch(
        &mut self,
        boundaries: &SequenceBoundaries,
        batch_index: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let rows = boundaries.batch_rows(batch_index);
        let frame_ends = if self.through_tie_group {
            boundaries.tie_groups.fill_group_ends(
                batch_index,
                rows.clone(),
                &mut self.tie_group_ends,
            );
            Some(self.tie_group_ends.as_slice())
        } else {
            None
        };
        match &mut self.running_sum {
            RunningSum::Wide { state, scale } => {
                let mut sums =
                    PrimitiveBuilder::<Decimal128Type>::with_capacity(allocator, rows.len());
                let sum_is_valid = state.fill_batch(
                    boundaries,
                    batch_index,
                    frame_ends,
                    &self.argument_batches,
                    sums.spare_mut(rows.len()),
                )?;
                let sums = sums.into_array(Some(sum_is_valid.into_inner()));
                let sums = sums
                    .as_primitive::<Decimal128Type>()
                    .clone()
                    .with_precision_and_scale(38, *scale)?;
                Ok(Arc::new(sums))
            }
            RunningSum::Float(state) => {
                let mut sums =
                    PrimitiveBuilder::<Float64Type>::with_capacity(allocator, rows.len());
                let sum_is_valid = state.fill_batch(
                    boundaries,
                    batch_index,
                    frame_ends,
                    &self.argument_batches,
                    sums.spare_mut(rows.len()),
                )?;
                Ok(sums.into_array(Some(sum_is_valid.into_inner())))
            }
        }
    }
}

/// A number a running sum can accumulate in.
trait SumValue: Copy {
    const ZERO: Self;

    fn add(self, value: Self) -> Result<Self, ArrowError>;
}

impl SumValue for i128 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn add(self, value: Self) -> Result<Self, ArrowError> {
        self.checked_add(value)
            .ok_or_else(|| ArrowError::ComputeError("a running sum overflowed".to_string()))
    }
}

impl SumValue for f64 {
    const ZERO: Self = 0.0;

    #[inline(always)]
    fn add(self, value: Self) -> Result<Self, ArrowError> {
        Ok(self + value)
    }
}

/// How far a running sum has got through the sequence, accumulating
/// in `Total`.
struct RunningSumState<Total> {
    /// Reads row `row` of an argument batch as a `Total`, given the address
    /// of the batch's first value.
    read_argument: fn(*const u8, usize) -> Total,
    /// The sum of the non-NULL values of rows `partition begin..summed_rows`
    /// of the current partition.
    total: Total,
    /// Whether any of those rows held a value. The sum is NULL until one
    /// does.
    has_value: bool,
    /// The first row of the sequence not yet added to `total`.
    summed_rows: usize,
    /// The batch that row is in.
    summed_batch: usize,
}

impl<Total: SumValue> RunningSumState<Total> {
    fn new(read_argument: fn(*const u8, usize) -> Total) -> Self {
        Self {
            read_argument,
            total: Total::ZERO,
            has_value: false,
            summed_rows: 0,
            summed_batch: 0,
        }
    }

    /// Writes the running sum of every row of batch `batch_index` into
    /// `sums`, and returns one bit per row that is set where the sum is not
    /// NULL. Batches must be handed over in order.
    ///
    /// For each row the sum has to cover the partition's rows up to the end
    /// of the row's frame: the row itself, or `frame_ends[row]` when given.
    /// That end only moves forward, so each row of the sequence is added
    /// once, wherever it is, and the total restarts at each partition's
    /// first row.
    fn fill_batch(
        &mut self,
        boundaries: &SequenceBoundaries,
        batch_index: usize,
        frame_ends: Option<&[u64]>,
        argument_batches: &[ArrayRef],
        sums: &mut [Total],
    ) -> Result<BooleanBuffer, ArrowError> {
        let rows = boundaries.batch_rows(batch_index);
        let batch_first_rows = boundaries.batch_first_rows();
        let starts_partition = boundaries.partitions.starts().slice(rows.start, rows.len());
        // The first value and the validity of the argument batch the sum
        // has reached. Bound once per batch, not once per row.
        let bind_arguments = |batch: usize| {
            let arguments = &argument_batches[batch];
            let argument_data = arguments.to_data();
            let width = argument_data
                .data_type()
                .primitive_width()
                .expect("a running sum's argument is a fixed-width number");
            // SAFETY: buffer 0 holds `offset + len` values of `width` bytes,
            // so the first value's address is inside it. The buffer stays
            // alive with `arguments`.
            let first_value = unsafe {
                argument_data.buffers()[0]
                    .as_ptr()
                    .add(argument_data.offset() * width)
            };
            let validity = arguments.nulls().filter(|nulls| nulls.null_count() > 0);
            (first_value, validity)
        };
        let (mut first_value, mut validity) = bind_arguments(self.summed_batch);

        let mut sum_is_valid = BooleanBufferBuilder::new(rows.len());
        for (((row_in_batch, row), starts_partition), sum) in rows
            .enumerate()
            .zip(starts_partition.iter())
            .zip(sums.iter_mut())
        {
            if starts_partition {
                self.total = Total::ZERO;
                self.has_value = false;
                self.summed_rows = row;
            }
            let frame_end = match frame_ends {
                Some(frame_ends) => frame_ends[row_in_batch] as usize,
                None => row + 1,
            };
            while self.summed_rows < frame_end {
                if self.summed_rows >= batch_first_rows[self.summed_batch + 1] {
                    while self.summed_rows >= batch_first_rows[self.summed_batch + 1] {
                        self.summed_batch += 1;
                    }
                    (first_value, validity) = bind_arguments(self.summed_batch);
                }
                let row_in_arguments = self.summed_rows - batch_first_rows[self.summed_batch];
                self.summed_rows += 1;
                if validity.is_none_or(|validity| validity.is_valid(row_in_arguments)) {
                    let value = (self.read_argument)(first_value, row_in_arguments);
                    self.total = self.total.add(value)?;
                    self.has_value = true;
                }
            }
            *sum = self.total;
            sum_is_valid.append(self.has_value);
        }
        Ok(sum_is_valid.finish())
    }
}

/// A running sum's state, in the type its argument adds up in.
enum RunningSum {
    /// Integers and decimals add up as 128-bit integers. The result is a
    /// 38-digit decimal with the argument's scale.
    Wide {
        state: RunningSumState<i128>,
        scale: i8,
    },
    Float(RunningSumState<f64>),
}

impl RunningSum {
    /// The state for summing a column of `argument_type`, or an error for a
    /// type a running sum does not cover.
    fn try_new(argument_type: &DataType) -> Result<Self, ArrowError> {
        // SAFETY of both readers: the caller hands the first value of a
        // column of `T` and an in-bounds row. Unaligned reads tolerate
        // sliced offsets.
        fn read_wide<T: Copy + Into<i128>>(values: *const u8, row: usize) -> i128 {
            unsafe { (values as *const T).add(row).read_unaligned() }.into()
        }
        fn read_float<T: Copy + Into<f64>>(values: *const u8, row: usize) -> f64 {
            unsafe { (values as *const T).add(row).read_unaligned() }.into()
        }
        let read_wide: fn(*const u8, usize) -> i128 = match argument_type {
            DataType::Int8 => read_wide::<i8>,
            DataType::Int16 => read_wide::<i16>,
            DataType::Int32 => read_wide::<i32>,
            DataType::Int64 | DataType::Decimal64(_, _) => read_wide::<i64>,
            DataType::UInt8 => read_wide::<u8>,
            DataType::UInt16 => read_wide::<u16>,
            DataType::UInt32 => read_wide::<u32>,
            DataType::UInt64 => read_wide::<u64>,
            DataType::Decimal128(_, _) => read_wide::<i128>,
            DataType::Float32 => return Ok(Self::Float(RunningSumState::new(read_float::<f32>))),
            DataType::Float64 => return Ok(Self::Float(RunningSumState::new(read_float::<f64>))),
            other => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "a running sum over a {other} column"
                )));
            }
        };
        let scale = match argument_type {
            DataType::Decimal64(_, scale) | DataType::Decimal128(_, scale) => *scale,
            _ => 0,
        };
        Ok(Self::Wide {
            state: RunningSumState::new(read_wide),
            scale,
        })
    }
}
