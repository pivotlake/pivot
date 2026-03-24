//! An [`Operator`] wrapper that blocks all activity until a shared gate opens.
//!
//! Used to prevent a sub-pipeline from running until a dependency completes.
//! For example, the probe side of a hash join must not start until the build
//! side has finished constructing the hash table.
//!
//! When the gate is closed (the [`AtomicBool`] is `false`):
//! - [`run_cpu_work`](Operator::run_cpu_work) → `Pending`
//! - [`next_io_requests`](Operator::next_io_requests) → empty
//! - [`try_finish`](Operator::try_finish) → `false` (prevents premature finishing)
//! - [`try_steal_work`](Operator::try_steal_work) → `Pending`
//!
//! Once the gate opens, all methods delegate to the inner operator.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::data_flow::WorkStatus;
use crate::io::IORequest;
use crate::memory::ReadBuffer;

use super::{Operator, Result};

pub struct GatedOperator {
    inner: Box<dyn Operator>,
    gate: Arc<AtomicBool>,
}

impl GatedOperator {
    pub fn new(inner: Box<dyn Operator>, gate: Arc<AtomicBool>) -> Self {
        Self { inner, gate }
    }

    fn is_open(&self) -> bool {
        self.gate.load(Ordering::Acquire)
    }
}

impl Operator for GatedOperator {
    fn run_cpu_work(&mut self) -> Result<WorkStatus> {
        if !self.is_open() {
            return Ok(WorkStatus::Pending);
        }
        self.inner.run_cpu_work()
    }

    fn next_io_requests(&mut self) -> Result<Vec<IORequest>> {
        if !self.is_open() {
            return Ok(vec![]);
        }
        self.inner.next_io_requests()
    }

    fn process_disk_response(&mut self, buffer: ReadBuffer, request: IORequest) -> Result<()> {
        self.inner.process_disk_response(buffer, request)
    }

    fn try_finish(&mut self) -> Result<bool> {
        if !self.is_open() {
            return Ok(false);
        }
        self.inner.try_finish()
    }

    fn try_steal_work(&mut self) -> Result<WorkStatus> {
        if !self.is_open() {
            return Ok(WorkStatus::Pending);
        }
        self.inner.try_steal_work()
    }
}
