//! [`Explain`]: renders a query's plan as text instead of running it.
//!
//! DuckDB parses `EXPLAIN <query>` into a `LOGICAL_EXPLAIN` wrapping the
//! optimized plan; the bridge and [`TryFrom`] carry that across as this
//! operator with the explained plan as its single input. Compilation never
//! lowers that input: [`compile`](Explain::compile) takes the already-formatted
//! plan tree and emits it, so the explained query never runs. The single output
//! column is named `QUERY PLAN`, matching PostgreSQL's `EXPLAIN`, with one row
//! per plan line.

use crate::compile::Error;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RecordBatchOperatorSpec, Sender,
    WorkStatus,
};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The output column header for `EXPLAIN`, matching PostgreSQL.
const PLAN_COLUMN: &str = "QUERY PLAN";

/// Emits a planned query's operator tree as text (the `EXPLAIN <query>`
/// statement). Carries no payload: the plan it explains is its child input,
/// which `compile` formats rather than lowers, so the explained
/// query never runs.
#[derive(Debug)]
pub struct Explain;

impl Explain {
    /// Compile into a source that emits `plan_text` (the formatted child plan,
    /// one row per line). `plan_text` is rendered by the caller from the child
    /// plan node, since this operator never lowers its input.
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        plan_text: String,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // One nullary per worker sharing a flag, so exactly one emits the plan
        // rows (mirrors `DummyScan`).
        let emitted = Arc::new(AtomicBool::new(false));
        let plan_text = Arc::new(plan_text);
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| ExplainNullaryFactory {
                emitted: emitted.clone(),
                plan_text: plan_text.clone(),
            }),
        ))
    }
}

impl fmt::Display for Explain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Explain")
    }
}

struct ExplainNullaryFactory {
    /// Shared across every worker's nullary so exactly one emits the plan.
    emitted: Arc<AtomicBool>,
    plan_text: Arc<String>,
}

impl NullaryFactory<RecordBatch> for ExplainNullaryFactory {
    type Nullary = ExplainDispatchOperator;

    fn build_nullary(self) -> Self::Nullary {
        ExplainDispatchOperator {
            emitted: self.emitted,
            plan_text: self.plan_text,
            ran: false,
        }
    }
}

struct ExplainDispatchOperator {
    emitted: Arc<AtomicBool>,
    plan_text: Arc<String>,
    ran: bool,
}

impl Nullary<RecordBatch> for ExplainDispatchOperator {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::io::OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }
        self.ran = true;

        // The first worker to win the swap emits the plan (one row per line);
        // the rest emit nothing.
        if !self.emitted.swap(true, Ordering::SeqCst) {
            let lines: Vec<&str> = self.plan_text.lines().collect();
            let schema = Arc::new(Schema::new(vec![Field::new(
                PLAN_COLUMN,
                DataType::Utf8,
                false,
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(lines))])
                .expect("single-column string batch is always well-formed");
            sender.send(batch)?;
        }

        Ok(WorkStatus::Ran)
    }

    // No IO: `next_*_requests` / `process_*_response` use the `Nullary` trait
    // defaults.

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.ran)
    }
}
