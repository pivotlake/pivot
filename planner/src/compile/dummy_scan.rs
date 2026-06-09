//! Dispatch [`Nullary`] source for `DummyScan` — DuckDB's single-row input
//! under a `FROM`-less `SELECT` (e.g. `SELECT drop_cache()`, `SELECT 1`).
//!
//! Emits exactly one empty (zero-column) row, on a single worker, so the
//! projection above it evaluates its expressions once and produces one output
//! row.

use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::Schema;
use dispatch::{Nullary, NullaryFactory, NullaryResult, Sender, WorkStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct DummyScanNullaryFactory {
    /// Shared across every worker's nullary so exactly one emits the row.
    emitted: Arc<AtomicBool>,
}

impl DummyScanNullaryFactory {
    pub(crate) fn new(emitted: Arc<AtomicBool>) -> Self {
        Self { emitted }
    }
}

impl NullaryFactory<RecordBatch> for DummyScanNullaryFactory {
    type Nullary = DummyScanDispatchOperator;

    fn build_nullary(self) -> Self::Nullary {
        DummyScanDispatchOperator {
            emitted: self.emitted,
            ran: false,
        }
    }
}

pub(crate) struct DummyScanDispatchOperator {
    emitted: Arc<AtomicBool>,
    ran: bool,
}

impl Nullary<RecordBatch> for DummyScanDispatchOperator {
    fn run<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }
        self.ran = true;

        // The first worker to win the swap emits the single empty row the dummy
        // scan stands for; the rest emit nothing, so the parent projection runs
        // exactly once.
        if !self.emitted.swap(true, Ordering::SeqCst) {
            let batch = RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(1)),
            )
            .expect("empty single-row batch is always well-formed");
            sender.send(batch)?;
        }

        Ok(WorkStatus::Ran)
    }

    // No IO: `next_*_requests` / `process_*_response` use the `Nullary` trait
    // defaults (none / unreachable).

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<bool> {
        Ok(self.ran)
    }
}
