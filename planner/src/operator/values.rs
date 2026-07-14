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
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::Schema;
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
    /// Builders are arranged by output column so evaluation can concatenate
    /// each column without transposing the row-major logical representation.
    /// Every worker receives the shared builders, but only the worker that
    /// claims the source constructs evaluators and emits the matrix.
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let column_count = self.rows.first().map_or(0, Vec::len);
        let builders: Arc<Vec<Vec<ExprFn>>> = Arc::new(
            (0..column_count)
                .map(|column_index| {
                    self.rows
                        .iter()
                        .map(|row| row[column_index].compile())
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
            }),
        ))
    }
}

/// Builds one worker's source over the shared expression builders.
struct ValuesSourceFactory {
    builders: Arc<Vec<Vec<ExprFn>>>,
    claimed: Arc<AtomicBool>,
}

impl NullaryFactory<RecordBatch> for ValuesSourceFactory {
    type Nullary = ValuesSource;

    fn build_nullary(self) -> Self::Nullary {
        ValuesSource {
            builders: self.builders,
            claimed: self.claimed,
            ran: false,
        }
    }
}

/// One worker's source. Only the worker that claims `claimed` emits a batch.
struct ValuesSource {
    builders: Arc<Vec<Vec<ExprFn>>>,
    claimed: Arc<AtomicBool>,
    ran: bool,
}

impl Nullary<RecordBatch> for ValuesSource {
    fn run<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }
        self.ran = true;

        if self
            .claimed
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            let mut evaluators = self
                .builders
                .iter()
                .map(|column| column.iter().map(|builder| builder()).collect())
                .collect::<Vec<_>>();
            sender.send(evaluate_values(&mut evaluators))?;
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
fn evaluate_values(evaluators: &mut [Vec<ExprEvalFn>]) -> RecordBatch {
    let context = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .expect("empty single-row VALUES context is always well-formed");
    let columns = evaluators
        .iter_mut()
        .enumerate()
        .map(|(column_index, column)| {
            let cells = column
                .iter_mut()
                .map(|evaluate| evaluate(&context).into_array(1))
                .collect::<Vec<_>>();
            let arrays = cells.iter().map(|array| array.as_ref()).collect::<Vec<_>>();
            let column = arrow::compute::concat(&arrays)
                .expect("DuckDB-bound VALUES cells in one column have one type");
            (format!("col{column_index}"), column, true)
        });
    RecordBatch::try_from_iter_with_nullable(columns)
        .expect("VALUES columns must have equal lengths")
}
