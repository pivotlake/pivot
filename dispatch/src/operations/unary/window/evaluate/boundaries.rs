//! Where partitions and tie groups start and end in a sorted sequence of
//! rows.
//!
//! A sorted bucket holds each partition as one contiguous stretch of rows,
//! and within a partition each group of rows tied on the order keys is
//! contiguous too. Both kinds of stretch start at a row whose key columns
//! differ from the row before it.
//!
//! [`SequenceBoundaries`] finds those rows for a whole sequence of batches
//! at once, before any function is evaluated. A function can then ask, for
//! every row of a batch, where the row's partition or tie group begins and
//! ends, as row numbers counted from the start of the sequence:
//!
//! ```text
//!  row   team  score     partition begin/end     tie group begin/end
//!  0     a     3         0 / 3                   0 / 1
//!  1     a     9         0 / 3                   1 / 3
//!  2     a     9         0 / 3                   1 / 3
//!  3     b     1         3 / 4                   3 / 4
//! ```
//!
//! A group's end is known from the start, even when it lies in a later
//! batch, so a function needs no state carried from one batch to the next.
//!
//! The comparison runs one column at a time over a whole batch, through
//! Arrow's `distinct` kernel, which treats two NULLs as equal, as SQL does
//! for both PARTITION BY and ties.

use std::ops::Range;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_ord::cmp::distinct;
use arrow_schema::ArrowError;

/// The rows at which a group (a partition, or a tie group) starts in a
/// sequence of batches, with what each batch needs to know about the groups
/// that cross its edges.
pub struct GroupStarts {
    /// One bit per row of the sequence, set where a group starts.
    starts: BooleanBuffer,
    /// Per batch, the first row of the group its first row belongs to.
    group_begin_at_batch_start: Vec<usize>,
    /// Per batch, the first group start after its last row, or the
    /// sequence's row count when no group starts after it.
    next_start_after_batch: Vec<usize>,
    /// Per batch, the last group start inside it, if any.
    last_start_in_batch: Vec<Option<usize>>,
}

impl GroupStarts {
    /// `starts` holds one bit per row of the sequence, and `batch_first_rows`
    /// each batch's first row followed by the sequence's row count.
    fn new(starts: BooleanBuffer, batch_first_rows: &[usize]) -> Self {
        let batch_count = batch_first_rows.len() - 1;
        let row_count = batch_first_rows[batch_count];
        let mut first_start_in_batch = Vec::with_capacity(batch_count);
        let mut last_start_in_batch = Vec::with_capacity(batch_count);
        for batch in 0..batch_count {
            let first_row = batch_first_rows[batch];
            let batch_starts = starts.slice(first_row, batch_first_rows[batch + 1] - first_row);
            let mut start_rows = batch_starts.set_indices().map(|row| first_row + row);
            let first_start = start_rows.next();
            first_start_in_batch.push(first_start);
            last_start_in_batch.push(start_rows.last().or(first_start));
        }

        let mut group_begin_at_batch_start = Vec::with_capacity(batch_count);
        let mut latest_start = 0;
        for batch in 0..batch_count {
            if first_start_in_batch[batch] == Some(batch_first_rows[batch]) {
                latest_start = batch_first_rows[batch];
            }
            group_begin_at_batch_start.push(latest_start);
            if let Some(last_start) = last_start_in_batch[batch] {
                latest_start = last_start;
            }
        }

        let mut next_start_after_batch = vec![row_count; batch_count];
        let mut next_start = row_count;
        for batch in (0..batch_count).rev() {
            next_start_after_batch[batch] = next_start;
            if let Some(first_start) = first_start_in_batch[batch] {
                next_start = first_start;
            }
        }

        Self {
            starts,
            group_begin_at_batch_start,
            next_start_after_batch,
            last_start_in_batch,
        }
    }

    /// One bit per row of the sequence, set where a group starts.
    pub fn starts(&self) -> &BooleanBuffer {
        &self.starts
    }

    /// The last group start inside batch `batch_index`, if any.
    pub fn last_start_in_batch(&self, batch_index: usize) -> Option<usize> {
        self.last_start_in_batch[batch_index]
    }

    /// Replaces `group_begins` with, for every row of batch `batch_index`
    /// (rows `rows` of the sequence), the first row of the row's group.
    pub fn fill_group_begins(
        &self,
        batch_index: usize,
        rows: Range<usize>,
        group_begins: &mut Vec<u64>,
    ) {
        let mut group_begin = self.group_begin_at_batch_start[batch_index] as u64;
        let batch_starts = self.starts.slice(rows.start, rows.len());
        group_begins.clear();
        group_begins.extend(batch_starts.iter().zip(rows).map(|(starts_group, row)| {
            if starts_group {
                group_begin = row as u64;
            }
            group_begin
        }));
    }

    /// Replaces `group_ends` with, for every row of batch `batch_index` (rows
    /// `rows` of the sequence), the row after the last row of the row's
    /// group.
    pub fn fill_group_ends(
        &self,
        batch_index: usize,
        rows: Range<usize>,
        group_ends: &mut Vec<u64>,
    ) {
        let mut group_end = self.next_start_after_batch[batch_index] as u64;
        group_ends.clear();
        group_ends.resize(rows.len(), 0);
        // Walk backward: a group ends where the next one starts.
        for (group_end_of_row, row) in group_ends.iter_mut().zip(rows.clone()).rev() {
            *group_end_of_row = group_end;
            if self.starts.value(row) {
                group_end = row as u64;
            }
        }
    }
}

/// The partitions and tie groups of one sorted sequence of batches (see the
/// module docs).
pub struct SequenceBoundaries {
    /// Each batch's first row, followed by the sequence's row count.
    batch_first_rows: Vec<usize>,
    pub partitions: GroupStarts,
    /// A row that starts a partition also starts a tie group.
    pub tie_groups: GroupStarts,
}

impl SequenceBoundaries {
    /// Finds the boundaries of `batches`, which hold one sorted sequence in
    /// order and are not empty. `order_columns` is `None` when nothing will
    /// ask about tie groups, which saves comparing the order keys.
    pub fn find(
        batches: &[RecordBatch],
        partition_columns: &[usize],
        order_columns: Option<&[usize]>,
    ) -> Result<Self, ArrowError> {
        let mut batch_first_rows = Vec::with_capacity(batches.len() + 1);
        let mut row_count = 0;
        for batch in batches {
            batch_first_rows.push(row_count);
            row_count += batch.num_rows();
        }
        batch_first_rows.push(row_count);

        let starts_partition = Self::mark_changes(batches, partition_columns, row_count)?;
        let starts_tie_group = match order_columns {
            None => starts_partition.clone(),
            Some(order_columns) => {
                &starts_partition | &Self::mark_changes(batches, order_columns, row_count)?
            }
        };
        Ok(Self {
            partitions: GroupStarts::new(starts_partition, &batch_first_rows),
            tie_groups: GroupStarts::new(starts_tie_group, &batch_first_rows),
            batch_first_rows,
        })
    }

    /// Returns one bit per row of `batches`, set when the row's values in
    /// `columns` differ from those of the row before it. The first row has no
    /// row before it and is always marked.
    fn mark_changes(
        batches: &[RecordBatch],
        columns: &[usize],
        row_count: usize,
    ) -> Result<BooleanBuffer, ArrowError> {
        let mut changes = BooleanBufferBuilder::new(row_count);
        // The watched columns of the previous batch's last row, each a
        // one-row array.
        let mut last_row: Option<Vec<ArrayRef>> = None;
        for batch in batches {
            let batch_row_count = batch.num_rows();
            let first_row_changes = match &last_row {
                None => true,
                Some(last_row) => {
                    let mut first_row_changes = false;
                    for (&column, last_value) in columns.iter().zip(last_row) {
                        let first_value = batch.column(column).slice(0, 1);
                        first_row_changes |= distinct(last_value, &first_value)?.value(0);
                    }
                    first_row_changes
                }
            };
            changes.append(first_row_changes);

            // Compare rows 1.. against rows ..n-1, a column at a time.
            let mut later_rows_change: Option<BooleanBuffer> = None;
            if batch_row_count > 1 {
                for &column in columns {
                    let values = batch.column(column);
                    let previous = values.slice(0, batch_row_count - 1);
                    let current = values.slice(1, batch_row_count - 1);
                    let column_changes = distinct(&current, &previous)?.into_parts().0;
                    later_rows_change = Some(match later_rows_change {
                        None => column_changes,
                        Some(changes) => &changes | &column_changes,
                    });
                }
            }
            match later_rows_change {
                Some(later_rows_change) => changes.append_buffer(&later_rows_change),
                None => changes.append_n(batch_row_count - 1, false),
            }

            last_row = Some(
                columns
                    .iter()
                    .map(|&column| batch.column(column).slice(batch_row_count - 1, 1))
                    .collect(),
            );
        }
        Ok(changes.finish())
    }

    /// Each batch's first row, followed by the sequence's row count.
    pub fn batch_first_rows(&self) -> &[usize] {
        &self.batch_first_rows
    }

    /// The rows of the sequence that batch `batch_index` holds.
    pub fn batch_rows(&self, batch_index: usize) -> Range<usize> {
        self.batch_first_rows[batch_index]..self.batch_first_rows[batch_index + 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

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

    /// The `(begin, end)` of each row's group, for every batch in turn.
    fn group_ranges(boundaries: &SequenceBoundaries, groups: &GroupStarts) -> Vec<(u64, u64)> {
        let batch_count = boundaries.batch_first_rows.len() - 1;
        let mut ranges = Vec::new();
        for batch_index in 0..batch_count {
            let rows = boundaries.batch_rows(batch_index);
            let (mut begins, mut ends) = (Vec::new(), Vec::new());
            groups.fill_group_begins(batch_index, rows.clone(), &mut begins);
            groups.fill_group_ends(batch_index, rows, &mut ends);
            ranges.extend(begins.into_iter().zip(ends));
        }
        ranges
    }

    #[test]
    fn a_partition_spanning_batches_has_one_begin_and_one_end() {
        let batches = [
            team_score_batch(&[(Some(1), 3), (Some(1), 9)]),
            team_score_batch(&[(Some(1), 9), (None, 1)]),
            team_score_batch(&[(None, 2)]),
        ];

        let boundaries = SequenceBoundaries::find(&batches, &[0], None).unwrap();

        assert_eq!(
            group_ranges(&boundaries, &boundaries.partitions),
            vec![(0, 3), (0, 3), (0, 3), (3, 5), (3, 5)]
        );
    }

    #[test]
    fn tie_groups_end_where_the_order_key_or_the_partition_changes() {
        let batches = [
            team_score_batch(&[(Some(1), 3), (Some(1), 9)]),
            team_score_batch(&[(Some(1), 9), (Some(2), 9)]),
        ];

        let boundaries = SequenceBoundaries::find(&batches, &[0], Some(&[1])).unwrap();

        assert_eq!(
            group_ranges(&boundaries, &boundaries.tie_groups),
            vec![(0, 1), (1, 3), (1, 3), (3, 4)]
        );
    }

    #[test]
    fn with_no_partition_columns_the_whole_sequence_is_one_partition() {
        let batches = [
            team_score_batch(&[(Some(1), 3), (Some(2), 9)]),
            team_score_batch(&[(Some(3), 9)]),
        ];

        let boundaries = SequenceBoundaries::find(&batches, &[], None).unwrap();

        assert_eq!(
            group_ranges(&boundaries, &boundaries.partitions),
            vec![(0, 3), (0, 3), (0, 3)]
        );
    }
}
