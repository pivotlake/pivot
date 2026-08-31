//! In-scan late materialization: evaluate pushed predicates on their columns
//! first, then fetch the full projection only for the rows that survive.
//!
//! A gated scan runs in two waves. The first wave scans only the predicate
//! ("gate") columns with row-group metadata attached. Its batches stream
//! through a [`Gate`], which evaluates the pushed comparisons and, the moment
//! a row group's rows have all been seen, emits one [`RowGroupRequest`] for
//! the full projection: with the surviving row indices when the mask is
//! sparse enough to pay off, without them (a plain dense decode) when it is
//! not, and nothing at all when no row survives. The second wave is the
//! ordinary request-driven read pipeline, which page-skips decompression and
//! decode for fully-masked pages.
//!
//! The SQL filter above the scan still evaluates every condition on the rows
//! that come back, so the gate only ever has to be conservative, never exact:
//! a dense request is always correct.

use crate::reading::record_batch_metadata::{global_row_group, row_index};
use crate::types::metadata::QueryRowGroupMetadata;
use crate::types::projection::Projection;
use crate::{ParquetTable, RowGroupRequest, op_err};
use ahash::HashMap;
use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray, Datum, RecordBatch, Scalar, UInt32Array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::ArrowError;
use dispatch::Sender;
use dispatch::{Unary, UnaryFactory};
use planner::expression::CompareType;
use std::sync::Arc;

/// Keep the surviving-row indices on a request only when they select at most
/// this fraction of the row group. Above it, masked decode's per-run overhead
/// outweighs the skipped work, so the request decodes densely and the filter
/// above drops the rest.
const MAX_SPARSE_KEEP_FRACTION: f64 = 1.0 / 32.0;

/// One pushed comparison the gate evaluates, resolved to its position in the
/// gate scan's batches.
#[derive(Clone)]
pub struct GatePredicate {
    /// The predicate column's position among the gate projection's columns.
    pub batch_column_idx: usize,
    pub compare_type: CompareType,
    pub value: Scalar<arrow_array::ArrayRef>,
}

/// The plan for a gated scan: which columns the first wave reads and the
/// comparisons the gate evaluates on them.
pub struct GatedScanPlan {
    pub gate_projection: Projection,
    pub predicates: Vec<GatePredicate>,
}

/// Factory for creating [`Gate`] instances within the unary pipeline.
pub struct GateFactory {
    predicates: Vec<GatePredicate>,
    full_projection: Projection,
    table: Arc<ParquetTable>,
}

impl GateFactory {
    pub fn new(
        predicates: Vec<GatePredicate>,
        full_projection: Projection,
        table: Arc<ParquetTable>,
    ) -> Self {
        Self {
            predicates,
            full_projection,
            table,
        }
    }
}

impl UnaryFactory<RecordBatch, RowGroupRequest> for GateFactory {
    type Unary = Gate;

    fn build_unary(self) -> Gate {
        Gate {
            predicates: self.predicates,
            full_projection: self.full_projection,
            table: self.table,
            states: HashMap::default(),
        }
    }
}

/// A row group's progress through the gate: how many of its rows have been
/// seen and which of them survived the predicates.
#[derive(Default)]
struct GroupState {
    seen_rows: u32,
    survivors: Vec<u32>,
}

/// Evaluates pushed predicates on gate-column batches and emits one
/// [`RowGroupRequest`] per row group as soon as the group completes.
///
/// A row group is decoded entirely by its claiming worker and this operator
/// sits behind a worker-local channel, so each instance sees every batch of
/// the row groups its worker claimed and nothing else.
pub struct Gate {
    predicates: Vec<GatePredicate>,
    full_projection: Projection,
    table: Arc<ParquetTable>,
    /// Row group ID -> progress. An entry exists only while the group is
    /// partially seen; completion removes it and emits the request.
    states: HashMap<u32, GroupState>,
}

/// The Arrow kernel implementing one [`CompareType`].
fn compare_kernel(
    compare_type: CompareType,
) -> fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError> {
    match compare_type {
        CompareType::Equal => arrow_ord::cmp::eq,
        CompareType::NotEqual => arrow_ord::cmp::neq,
        CompareType::Less => arrow_ord::cmp::lt,
        CompareType::Greater => arrow_ord::cmp::gt,
        CompareType::LessEqual => arrow_ord::cmp::lt_eq,
        CompareType::GreaterEqual => arrow_ord::cmp::gt_eq,
    }
}

/// Evaluate every predicate over the batch and AND the results into one
/// survivor bitmap, with comparison nulls counting as non-survivors.
fn survivor_bits(
    batch: &RecordBatch,
    predicates: &[GatePredicate],
) -> Result<BooleanBuffer, ArrowError> {
    let mut combined: Option<BooleanBuffer> = None;
    for predicate in predicates {
        let column: &dyn Array = batch.column(predicate.batch_column_idx).as_ref();
        let result = compare_kernel(predicate.compare_type)(&column, &predicate.value)?;
        let (values, nulls) = result.into_parts();
        let bits = match nulls {
            Some(nulls) => &values & nulls.inner(),
            None => values,
        };
        combined = Some(match combined {
            Some(combined) => &combined & &bits,
            None => bits,
        });
    }
    Ok(combined.expect("a gate always has at least one predicate"))
}

impl Gate {
    /// Emit the completed group's request: sparse when the mask pays for
    /// itself, dense otherwise, nothing when no row survives.
    fn emit_request(
        &self,
        group: u32,
        state: GroupState,
        sender: &mut dyn Sender<RowGroupRequest>,
    ) -> dispatch::UnaryResult<()> {
        if state.survivors.is_empty() {
            return Ok(());
        }
        let total_rows = self.table.row_groups()[group as usize].num_rows as usize;
        debug_assert!(state.survivors.is_sorted());
        let indices = (state.survivors.len() as f64
            <= total_rows as f64 * MAX_SPARSE_KEEP_FRACTION)
            .then_some(state.survivors);
        sender.send(RowGroupRequest::from(
            QueryRowGroupMetadata::new(&self.table, group as usize, indices),
            &self.full_projection,
        ))?;
        Ok(())
    }
}

impl Unary<RecordBatch, RowGroupRequest> for Gate {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RowGroupRequest>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let bits = survivor_bits(&batch, &self.predicates).map_err(op_err)?;

        // Walk the row-group runs the way the materializer does: logical run
        // ends against logically-indexed row indices (see `Materializer`).
        let groups = global_row_group(&batch);
        let run_ends = groups.run_ends();
        let physical_start = run_ends.get_start_physical_index();
        let group_values = groups
            .values()
            .as_primitive::<arrow_array::types::UInt32Type>();
        let row_indices: &UInt32Array = row_index(&batch);

        let mut logical = 0usize;
        for (run_offset, logical_end) in run_ends.sliced_values().enumerate() {
            let logical_end = logical_end as usize;
            let group = group_values.value(physical_start + run_offset);
            let state = self.states.entry(group).or_default();
            while logical < logical_end {
                if bits.value(logical) {
                    state.survivors.push(row_indices.value(logical));
                }
                state.seen_rows += 1;
                logical += 1;
            }
            let total_rows = self.table.row_groups()[group as usize].num_rows as u32;
            if state.seen_rows == total_rows {
                let state = self.states.remove(&group).unwrap();
                self.emit_request(group, state, sender)?;
            }
        }
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<RowGroupRequest>) -> dispatch::UnaryResult<bool> {
        // A clean scan completes every group in `consume`. If the dataflow
        // winds down early with groups partially seen, a dense request keeps
        // every produced row correct; the filter above drops the rest.
        debug_assert!(
            self.states.is_empty(),
            "gate finished with partial row groups"
        );
        for (group, _state) in std::mem::take(&mut self.states) {
            sender.send(RowGroupRequest::from(
                QueryRowGroupMetadata::new(&self.table, group as usize, None),
                &self.full_projection,
            ))?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{GatePredicate, survivor_bits};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use planner::expression::CompareType;
    use std::sync::Arc;

    fn value_batch(values: Vec<Option<i64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()
    }

    fn predicate(compare_type: CompareType, constant: i64) -> GatePredicate {
        GatePredicate {
            batch_column_idx: 0,
            compare_type,
            value: Scalar::new(Arc::new(Int64Array::from(vec![constant])) as ArrayRef),
        }
    }

    // A comparison against a null row yields a null, which must count as a
    // non-survivor, not survive by accident.
    #[test]
    fn null_rows_never_survive() {
        let batch = value_batch(vec![Some(1), None, Some(3)]);

        let bits = survivor_bits(&batch, &[predicate(CompareType::LessEqual, 3)]).unwrap();

        assert_eq!(bits.iter().collect::<Vec<_>>(), [true, false, true]);
    }

    // Two pushed comparisons intersect: only rows passing both survive.
    #[test]
    fn multiple_predicates_intersect() {
        let batch = value_batch(vec![Some(2), Some(5), Some(7), Some(9)]);

        let bits = survivor_bits(
            &batch,
            &[
                predicate(CompareType::Greater, 2),
                predicate(CompareType::Less, 9),
            ],
        )
        .unwrap();

        assert_eq!(bits.iter().collect::<Vec<_>>(), [false, true, true, false]);
    }
}
