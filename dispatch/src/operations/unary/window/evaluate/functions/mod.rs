//! The window functions. Each function has a module with the struct that
//! computes it over one sorted sequence of rows:
//!
//! ```text
//!  row_number      RowNumberEvaluator     row - partition begin + 1
//!  rank            RankEvaluator          tie group begin - partition begin + 1
//!  dense_rank      DenseRankEvaluator     tie groups so far in the partition
//!  percent_rank    PercentRankEvaluator   (rank - 1) / (partition rows - 1)
//!  lag, lead       LagLeadEvaluator       the argument, an offset of rows away
//!  sum             RunningSumEvaluator    the argument, added up from the
//!                                         partition's first row
//! ```
//!
//! Every one of them works from where each row's partition and tie group
//! begin and end in the whole sequence (see [`SequenceBoundaries`]), one
//! batch at a time.

mod dense_rank;
mod lag_lead;
mod percent_rank;
mod rank;
mod row_number;
mod running_sum;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};

use super::boundaries::SequenceBoundaries;
use crate::memory::SlabAllocator;
use dense_rank::DenseRankEvaluator;
use lag_lead::LagLeadEvaluator;
use percent_rank::PercentRankEvaluator;
use rank::RankEvaluator;
use row_number::RowNumberEvaluator;
use running_sum::RunningSumEvaluator;

/// A window function the operator can compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowFunction {
    /// `row_number()`: 1, 2, 3, ... down the partition.
    RowNumber,
    /// `rank()`: tied rows share the row number of the first of them, so the
    /// values skip after a tie: 1, 1, 3.
    Rank,
    /// `dense_rank()`: tied rows share a value and the next group gets the
    /// next integer: 1, 1, 2.
    DenseRank,
    /// `percent_rank()`: the rank scaled to run from 0 to 1 over the
    /// partition, `(rank - 1) / (partition rows - 1)`. A partition of one
    /// row has 0.
    PercentRank,
    /// `lag(column, offset)`: the value of input column `column` in the row
    /// `offset` rows earlier in the partition, or NULL when the partition has
    /// no such row.
    Lag { column: usize, offset: u64 },
    /// `lead(column, offset)`: as [`Lag`](Self::Lag), `offset` rows later.
    Lead { column: usize, offset: u64 },
    /// `sum(column)` over the rows from the partition's first row up to the
    /// current one. With `through_tie_group` the sum also includes the rows
    /// after the current one that tie with it on the order keys, which is
    /// what SQL's default frame does. Tied rows then share one value.
    RunningSum {
        column: usize,
        through_tie_group: bool,
    },
}

impl WindowFunction {
    /// The function's name in SQL.
    pub fn sql_name(self) -> &'static str {
        match self {
            WindowFunction::RowNumber => "row_number",
            WindowFunction::Rank => "rank",
            WindowFunction::DenseRank => "dense_rank",
            WindowFunction::PercentRank => "percent_rank",
            WindowFunction::Lag { .. } => "lag",
            WindowFunction::Lead { .. } => "lead",
            WindowFunction::RunningSum { .. } => "sum",
        }
    }

    /// The output field of the function's column, given the schema of the
    /// operator's input.
    ///
    /// `lag` and `lead` return their argument's type. A running sum returns
    /// what the aggregate `sum` returns: a 128-bit integer for integers, a
    /// double for floats, and a 38-digit decimal of the same scale for
    /// decimals.
    pub fn output_field(self, input_schema: &Schema) -> Result<Field, ArrowError> {
        let (data_type, nullable) = match self {
            WindowFunction::RowNumber | WindowFunction::Rank | WindowFunction::DenseRank => {
                (DataType::Int64, false)
            }
            WindowFunction::PercentRank => (DataType::Float64, false),
            WindowFunction::Lag { column, .. } | WindowFunction::Lead { column, .. } => {
                (input_schema.field(column).data_type().clone(), true)
            }
            WindowFunction::RunningSum { column, .. } => {
                let sum_type = match input_schema.field(column).data_type() {
                    DataType::Float32 | DataType::Float64 => DataType::Float64,
                    DataType::Decimal64(_, scale) | DataType::Decimal128(_, scale) => {
                        DataType::Decimal128(38, *scale)
                    }
                    argument_type if argument_type.is_integer() => DataType::Decimal128(38, 0),
                    argument_type => {
                        return Err(ArrowError::NotYetImplemented(format!(
                            "a running sum over a {argument_type} column"
                        )));
                    }
                };
                (sum_type, true)
            }
        };
        Ok(Field::new(self.sql_name(), data_type, nullable))
    }

    /// Whether the function is a whole number that starts at 1 in every
    /// partition and never decreases down it. A filter `function <= N` on
    /// such a function keeps the first rows of each partition, which is what
    /// lets the operator apply it as a row limit.
    pub fn counts_up_from_one(self) -> bool {
        matches!(
            self,
            WindowFunction::RowNumber | WindowFunction::Rank | WindowFunction::DenseRank
        )
    }

    /// Whether the function's value depends on which rows tie on the order
    /// keys.
    pub fn needs_tie_groups(self) -> bool {
        match self {
            WindowFunction::Rank | WindowFunction::DenseRank | WindowFunction::PercentRank => true,
            WindowFunction::RunningSum {
                through_tie_group, ..
            } => through_tie_group,
            WindowFunction::RowNumber
            | WindowFunction::Lag { .. }
            | WindowFunction::Lead { .. } => false,
        }
    }

    /// Builds the evaluator of this function over the sequence whose rows
    /// `batches` hold and `boundaries` describes.
    pub fn build_evaluator(
        self,
        boundaries: &SequenceBoundaries,
        batches: &[RecordBatch],
    ) -> Result<Box<dyn FunctionEvaluator>, ArrowError> {
        Ok(match self {
            WindowFunction::RowNumber => Box::new(RowNumberEvaluator::default()),
            WindowFunction::Rank => Box::new(RankEvaluator::default()),
            WindowFunction::DenseRank => Box::new(DenseRankEvaluator::new(boundaries)),
            WindowFunction::PercentRank => Box::new(PercentRankEvaluator::default()),
            WindowFunction::Lag { column, offset } => {
                Box::new(LagLeadEvaluator::new(batches, column, offset, false))
            }
            WindowFunction::Lead { column, offset } => {
                Box::new(LagLeadEvaluator::new(batches, column, offset, true))
            }
            WindowFunction::RunningSum {
                column,
                through_tie_group,
            } => Box::new(RunningSumEvaluator::try_new(
                batches,
                column,
                through_tie_group,
            )?),
        })
    }
}

/// Computes one window function over the batches of one sorted sequence.
pub trait FunctionEvaluator {
    /// Computes the function for every row of batch `batch_index` of the
    /// sequence, into a column on slab memory. The batches are handed over
    /// in order, each once.
    fn evaluate_batch(
        &mut self,
        boundaries: &SequenceBoundaries,
        batch_index: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal128Type, Float64Type, Int64Type};

    use super::*;
    use crate::memory::init_test_free_pool;

    /// A batch of `(team, score)` rows.
    fn team_score_batch(rows: &[(i64, Option<i64>)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("team", DataType::Int64, false),
            Field::new("score", DataType::Int64, true),
        ]));
        let teams = Int64Array::from_iter_values(rows.iter().map(|&(team, _)| team));
        let scores: Int64Array = rows.iter().map(|&(_, score)| score).collect();
        RecordBatch::try_new(schema, vec![Arc::new(teams), Arc::new(scores)]).unwrap()
    }

    /// Evaluates `function` over `batches`, sorted by team then score, and
    /// returns its column for each batch.
    fn evaluate_sequence(function: WindowFunction, batches: &[RecordBatch]) -> Vec<ArrayRef> {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let boundaries = SequenceBoundaries::find(batches, &[0], Some(&[1])).unwrap();
        let mut evaluator = function.build_evaluator(&boundaries, batches).unwrap();
        (0..batches.len())
            .map(|batch_index| {
                evaluator
                    .evaluate_batch(&boundaries, batch_index, &mut allocator)
                    .unwrap()
            })
            .collect()
    }

    /// The values of integer columns, one after another.
    fn int_values(columns: &[ArrayRef]) -> Vec<Option<i64>> {
        columns
            .iter()
            .flat_map(|column| {
                column
                    .as_primitive::<Int64Type>()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The values of 128-bit sums, one after another.
    fn sum_values(columns: &[ArrayRef]) -> Vec<Option<i128>> {
        columns
            .iter()
            .flat_map(|column| {
                column
                    .as_primitive::<Decimal128Type>()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Team 1 scored 3, 9, 9 and 12, split over two batches after the first
    /// 9. Team 2 scored 5.
    fn two_batches_with_a_tie_across_them() -> [RecordBatch; 2] {
        [
            team_score_batch(&[(1, Some(3)), (1, Some(9))]),
            team_score_batch(&[(1, Some(9)), (1, Some(12)), (2, Some(5))]),
        ]
    }

    #[test]
    fn row_number_counts_up_and_restarts_at_each_partition() {
        let batches = two_batches_with_a_tie_across_them();

        let values = evaluate_sequence(WindowFunction::RowNumber, &batches);

        assert_eq!(
            int_values(&values),
            vec![Some(1), Some(2), Some(3), Some(4), Some(1)]
        );
    }

    #[test]
    fn rank_repeats_within_a_tie_group_and_skips_after_it() {
        let batches = two_batches_with_a_tie_across_them();

        let values = evaluate_sequence(WindowFunction::Rank, &batches);

        assert_eq!(
            int_values(&values),
            vec![Some(1), Some(2), Some(2), Some(4), Some(1)]
        );
    }

    #[test]
    fn dense_rank_repeats_within_a_tie_group_without_skipping() {
        let batches = two_batches_with_a_tie_across_them();

        let values = evaluate_sequence(WindowFunction::DenseRank, &batches);

        assert_eq!(
            int_values(&values),
            vec![Some(1), Some(2), Some(2), Some(3), Some(1)]
        );
    }

    #[test]
    fn percent_rank_runs_from_zero_to_one_over_a_partition() {
        let batches = two_batches_with_a_tie_across_them();

        let values = evaluate_sequence(WindowFunction::PercentRank, &batches);

        let values: Vec<f64> = values
            .iter()
            .flat_map(|column| column.as_primitive::<Float64Type>().values().to_vec())
            .collect();
        assert_eq!(values, vec![0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 0.0]);
    }

    #[test]
    fn lag_reads_an_earlier_row_of_the_partition_across_batches() {
        let batches = two_batches_with_a_tie_across_them();
        let lag_of_score = WindowFunction::Lag {
            column: 1,
            offset: 2,
        };

        let values = evaluate_sequence(lag_of_score, &batches);

        assert_eq!(
            int_values(&values),
            vec![None, None, Some(3), Some(9), None]
        );
    }

    #[test]
    fn lead_reads_a_later_row_of_the_partition_across_batches() {
        let batches = two_batches_with_a_tie_across_them();
        let lead_of_score = WindowFunction::Lead {
            column: 1,
            offset: 1,
        };

        let values = evaluate_sequence(lead_of_score, &batches);

        assert_eq!(
            int_values(&values),
            vec![Some(9), Some(9), Some(12), None, None]
        );
    }

    #[test]
    fn a_running_sum_through_each_row_restarts_at_each_partition() {
        let batches = two_batches_with_a_tie_across_them();
        let running_sum_of_score = WindowFunction::RunningSum {
            column: 1,
            through_tie_group: false,
        };

        let values = evaluate_sequence(running_sum_of_score, &batches);

        assert_eq!(
            sum_values(&values),
            vec![Some(3), Some(12), Some(21), Some(33), Some(5)]
        );
    }

    #[test]
    fn a_running_sum_through_the_tie_group_gives_tied_rows_one_value() {
        let batches = two_batches_with_a_tie_across_them();
        let running_sum_of_score = WindowFunction::RunningSum {
            column: 1,
            through_tie_group: true,
        };

        let values = evaluate_sequence(running_sum_of_score, &batches);

        assert_eq!(
            sum_values(&values),
            vec![Some(3), Some(21), Some(21), Some(33), Some(5)]
        );
    }

    #[test]
    fn a_running_sum_is_null_until_the_partition_has_a_value() {
        let batches = [team_score_batch(&[(1, None), (1, Some(4)), (2, None)])];
        let running_sum_of_score = WindowFunction::RunningSum {
            column: 1,
            through_tie_group: false,
        };

        let values = evaluate_sequence(running_sum_of_score, &batches);

        assert_eq!(sum_values(&values), vec![None, Some(4), None]);
    }
}
