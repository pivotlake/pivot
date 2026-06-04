//! Dispatch [`Nullary`] operator for CREATE TABLE.
//!
//! Unlike the other operators (which compile to existing dispatch primitives),
//! CREATE TABLE needs its own nullary dispatch operator that invokes the
//! catalog exactly once across all workers.

use arrow_array::RecordBatch;
use dispatch::{Nullary, NullaryFactory, NullaryResult, Sender, WorkStatus};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct CreateTableDispatchOperator {
    catalog: Arc<dyn crate::catalog::Catalog>,
    request: crate::catalog::CreateTableRequest,
    already_created: Arc<AtomicBool>,
    ran: bool,
}

impl CreateTableDispatchOperator {
    fn new(
        catalog: Arc<dyn crate::catalog::Catalog>,
        request: crate::catalog::CreateTableRequest,
        already_created: Arc<AtomicBool>,
    ) -> Self {
        Self {
            catalog,
            request,
            already_created,
            ran: false,
        }
    }
}

pub(crate) struct CreateTableNullaryFactory {
    catalog: Arc<dyn crate::catalog::Catalog>,
    request: crate::catalog::CreateTableRequest,
    already_created: Arc<AtomicBool>,
}

impl CreateTableNullaryFactory {
    pub(crate) fn new(
        catalog: Arc<dyn crate::catalog::Catalog>,
        request: crate::catalog::CreateTableRequest,
        already_created: Arc<AtomicBool>,
    ) -> Self {
        Self {
            catalog,
            request,
            already_created,
        }
    }
}

impl NullaryFactory<RecordBatch> for CreateTableNullaryFactory {
    type Nullary = CreateTableDispatchOperator;

    fn build_nullary(self) -> Self::Nullary {
        CreateTableDispatchOperator::new(self.catalog, self.request, self.already_created)
    }
}

impl Nullary<RecordBatch> for CreateTableDispatchOperator {
    fn run<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }

        self.ran = true;
        if !self.already_created.swap(true, Ordering::SeqCst) {
            self.catalog
                .create_table(self.request.clone())
                .map_err(Box::from)?;
        }

        Ok(WorkStatus::Ran)
    }

    // No IO: `next_fs_requests` / `next_http_requests` / `process_io_response`
    // use the `Nullary` trait defaults (none / unreachable).

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<bool> {
        Ok(self.ran)
    }
}
