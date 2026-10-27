//! Dataflows compiled ahead of time for the next run of a cached plan.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use dispatch::RecordBatchOperatorSpec;

/// How many plans keep a dataflow ready at once; the least recently prepared
/// one gives its dataflow up first.
const READY_DATAFLOW_CAPACITY: usize = 16;

/// Compiled dataflows waiting for the next run of their plan.
///
/// Compiling a plan builds every worker's operators and channels on one
/// thread, while the pool waits for it. A cached plan's query usually runs
/// again, so once a run of it is done, the dataflow for the next run is
/// compiled off the query's path and kept here, and that run starts it at
/// once. The plan cache hands a plan out only while every table it reads is
/// at the revision it was planned at, so a dataflow kept for that very plan
/// reads exactly what compiling it at the next run would.
pub(super) struct ReadyDataflows {
    ready: Mutex<VecDeque<(Arc<planner::Plan>, RecordBatchOperatorSpec)>>,
    capacity: usize,
}

impl Default for ReadyDataflows {
    fn default() -> Self {
        Self::new(READY_DATAFLOW_CAPACITY)
    }
}

impl ReadyDataflows {
    fn new(capacity: usize) -> Self {
        Self {
            ready: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Takes the dataflow compiled for `plan`, if one is ready.
    pub(super) fn take(&self, plan: &Arc<planner::Plan>) -> Option<RecordBatchOperatorSpec> {
        let mut ready = self.ready.lock().unwrap();
        let position = ready
            .iter()
            .position(|(prepared, _)| Arc::ptr_eq(prepared, plan))?;
        ready.remove(position).map(|(_, spec)| spec)
    }

    /// Keeps `spec` for the next run of `plan`, replacing a dataflow already
    /// kept for it.
    pub(super) fn store(&self, plan: Arc<planner::Plan>, spec: RecordBatchOperatorSpec) {
        let mut ready = self.ready.lock().unwrap();
        ready.retain(|(prepared, _)| !Arc::ptr_eq(prepared, &plan));
        if ready.len() == self.capacity {
            ready.pop_front();
        }
        ready.push_back((plan, spec));
    }
}
