//! [`DummyScan`] — the single-row source under a `FROM`-less `SELECT`.
//!
//! The operator (DuckDB's single-row input under e.g. `SELECT 1`,
//! `SELECT drop_cache()`) compiles to the [`Nullary`] source below, which emits
//! exactly one empty (zero-column) row on a single worker so the projection
//! above it evaluates its expressions once and produces one output row.

use crate::compile::Error;
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::Schema;
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RecordBatchOperatorSpec, Sender,
    WorkStatus,
};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The single-row source under a `FROM`-less `SELECT`.
#[derive(Debug)]
pub struct DummyScan;

impl fmt::Display for DummyScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DummyScan")
    }
}

impl DummyScan {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // One nullary per worker sharing a flag, so exactly one emits the single
        // dummy row the parent projection runs over.
        let emitted = Arc::new(AtomicBool::new(false));
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| DummyScanNullaryFactory::new(emitted.clone())),
        ))
    }
}

struct DummyScanNullaryFactory {
    /// Shared across every worker's nullary so exactly one emits the row.
    emitted: Arc<AtomicBool>,
}

impl DummyScanNullaryFactory {
    fn new(emitted: Arc<AtomicBool>) -> Self {
        Self { emitted }
    }
}

impl NullaryFactory<RecordBatch> for DummyScanNullaryFactory {
    type Nullary = DummyScanDispatchOperator;

    fn build_nullary(self) -> Self::Nullary {
        DummyScanDispatchOperator {
            emitted: self.emitted,
        }
    }
}

struct DummyScanDispatchOperator {
    emitted: Arc<AtomicBool>,
}

impl Nullary<RecordBatch> for DummyScanDispatchOperator {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::io::OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        if !self.emitted.swap(true, Ordering::Relaxed) {
            let batch = RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(1)),
            )
            .expect("empty single-row batch is always well-formed");
            sender.send(batch)?;
            Ok(WorkStatus::Ran)
        } else {
            Ok(WorkStatus::Pending)
        }
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.emitted.load(Ordering::Relaxed))
    }
}
