//! A `VALUES` list compiled as a nullary source.
//!
//! DuckDB binds a `VALUES` clause as a `LogicalExpressionGet` over a
//! `LogicalDummyScan`. In DuckDB's own physical planner, a foldable expression
//! list is evaluated eagerly and replaced with a `COLUMN_DATA_SCAN` over the
//! resulting in-memory collection. Pivot consumes the logical plan instead of
//! DuckDB's physical plan, so the plan builder removes that dummy child and
//! this operator performs the equivalent job as a source: one worker evaluates
//! the complete expression matrix and emits one [`RecordBatch`].
//!
//! The expression interface still accepts a batch because it is shared with
//! projections. The source therefore creates a private zero-column, one-row
//! evaluation context. That batch is an implementation detail, not an input
//! operator and not a dispatched batch.

use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RecordBatchOperatorSpec, Sender,
    WorkStatus,
};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub struct Values {
    pub rows: Vec<Vec<Expression>>,
}

impl fmt::Display for Values {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = self
            .rows
            .iter()
            .map(|row| {
                let cells = row
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({cells})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(formatter, "Values({rows})")
    }
}

impl Values {
    /// Compile the complete `VALUES` matrix into a source that emits one batch.
    ///
    /// Every worker receives a source factory and private expression evaluators,
    /// but a shared atomic claim ensures that exactly one worker evaluates and
    /// emits the matrix. Cells are concatenated by column, with DuckDB's bound
    /// type for the first cell used to normalize compatible literals such as
    /// mixed-width integers.
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let column_count = self.rows.first().map_or(0, Vec::len);
        let builders: Arc<Vec<Vec<ExprFn>>> = Arc::new(
            self.rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(Expression::compile)
                        .collect::<Result<_, _>>()
                })
                .collect::<Result<_, _>>()?,
        );

        let claimed = Arc::new(AtomicBool::new(false));
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| ValuesSourceFactory {
                builders: builders.clone(),
                claimed: claimed.clone(),
                column_count,
            }),
        ))
    }
}

/// Builds one worker's source and its private expression evaluators.
struct ValuesSourceFactory {
    builders: Arc<Vec<Vec<ExprFn>>>,
    claimed: Arc<AtomicBool>,
    column_count: usize,
}

impl NullaryFactory<RecordBatch> for ValuesSourceFactory {
    type Nullary = ValuesSource;

    fn build_nullary(self) -> Self::Nullary {
        let evaluators = self
            .builders
            .iter()
            .map(|row| row.iter().map(|builder| builder()).collect())
            .collect();
        ValuesSource {
            evaluators,
            claimed: self.claimed,
            column_count: self.column_count,
            ran: false,
        }
    }
}

/// One worker's source. Only the worker that claims `claimed` emits a batch.
struct ValuesSource {
    evaluators: Vec<Vec<ExprEvalFn>>,
    claimed: Arc<AtomicBool>,
    column_count: usize,
    ran: bool,
}

impl Nullary<RecordBatch> for ValuesSource {
    fn run<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }
        self.ran = true;

        if !self.claimed.swap(true, Ordering::SeqCst) {
            sender.send(evaluate_values(&mut self.evaluators, self.column_count))?;
        }
        Ok(WorkStatus::Ran)
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<bool> {
        Ok(self.ran)
    }
}

/// Evaluate one complete expression matrix into columns and concatenate its
/// cells down the rows. The private single-row context satisfies the shared
/// expression API without introducing a `DummyScan` into the dataflow.
fn evaluate_values(evaluators: &mut [Vec<ExprEvalFn>], column_count: usize) -> RecordBatch {
    let context = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .expect("empty single-row VALUES context is always well-formed");
    let columns: Vec<ArrayRef> = (0..column_count)
        .map(|column_index| {
            let cells = evaluators
                .iter_mut()
                .map(|row| row[column_index](&context).into_array(1))
                .collect::<Vec<_>>();
            let target = cells[0].data_type().clone();
            let cast_cells = cells
                .iter()
                .map(|cell| {
                    arrow::compute::cast(cell, &target)
                        .expect("VALUES cells in one column share a bound type")
                })
                .collect::<Vec<_>>();
            let arrays = cast_cells
                .iter()
                .map(|array| array.as_ref())
                .collect::<Vec<_>>();
            arrow::compute::concat(&arrays).expect("VALUES cells have one type")
        })
        .collect();
    let fields = columns
        .iter()
        .enumerate()
        .map(|(index, column)| Field::new(format!("col{index}"), column.data_type().clone(), true))
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}
