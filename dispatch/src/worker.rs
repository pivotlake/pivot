//! Per-core worker threads that execute dataflows.
//!
//! Each worker is pinned to a CPU core and runs an event loop that:
//! - Receives [`DataFlowBuilder`]s from the [`Dispatcher`](crate::Dispatcher) via a channel
//! - Builds them into [`DataFlow`]s (creating operators, channels, etc. on the worker thread)
//! - Drives execution by interleaving CPU work, IO, and work stealing
//!
//! Workers are the execution backbone of the system. A query built with
//! [`RecordBatchOperatorSpec`](crate::api::RecordBatchOperatorSpec)
//! doesn't run until [`.collect()`](crate::api::RecordBatchOperatorSpec::collect)
//! sends one `DataFlowBuilder` to each worker. The worker then:
//!
//! 1. Calls [`DataFlowBuilder::build`] to construct the operator chain — this happens
//!    here because some resources (`Rc<Worker<T>>`, thread-local buffers) are not `Send`
//!    and must be created on the thread that uses them.
//! 2. Runs the resulting `DataFlow` through its event loop, which prioritizes keeping IO
//!    saturated (to hide disk latency), then runs CPU work, then attempts to steal work
//!    from sibling workers when idle.
//!
//! Workers coordinate with each other only at the operator level — e.g. a `Count` operator
//! uses shared atomics so the last worker to finish can emit the total. The worker itself
//! has no direct knowledge of other workers; synchronization is encapsulated in the
//! operators.

use crate::Identifier;
use crate::api::DataFlowBuilder;
use crate::data_flow::{DataFlow, WorkStatus};
use crate::io::IORequester;
use crate::memory::{init_free_pool, pop_dirty_buffer};
use core_affinity::CoreId;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier};
use std::thread::{JoinHandle, sleep};
use std::time::Duration;
use std::{result, thread};
use thiserror::Error;
use tracing::{debug, info, instrument};

thread_local! {
    pub static WORKER_IDX: Cell<usize> = const { Cell::new(usize::MAX) };
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Pipeline(#[from] crate::data_flow::Error),
    #[error("{0}")]
    IORequester(#[from] crate::io::IORequesterError),
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// A Worker is spun up per CPU core. The worker's main entry-point, `create`, spins up a
/// thread with affinity to a CPU which continuously requests work from the dispatcher and does it.
///
/// The main idea of a Worker is to keep everything possible "local" to it, to prevent
/// context-switching/CPU cache-invalidation and in the future allow NUMA optimizations etc. We try
/// to make our physical CPU cores first class citizens.
///
/// The Worker receives dataflows from the Dispatcher and runs them- the logic is outlined is as
/// follows:
///
/// 1. Process any outstanding IO completions — we want to always have IO running in background, so
///    we clear any pending if possible
/// 2. Saturate IO - if we have space now, submit new IO
/// 3. Run any CPU work available - this will run a single operation if available
/// 4. Try running trough all dataflows and finishing then
///
/// If no work was done (no CPU work or finish work), then
/// - Wait on existing IO - if none exists, then:
/// - Try stealing work - if none exists, then:
/// - Sleep!
///
/// And forever back to 1!
pub struct Worker {
    io: IORequester,
    id: Identifier,
    data_flow_queue: Receiver<DataFlowBuilder>,
    data_flows: HashMap<Identifier, DataFlow>,
    did_work_last_iteration: bool,
    sleeps_between_clean: usize,
}

impl Worker {
    /// Spawn a worker thread pinned to `core`. Blocks on `ready_barrier` before
    /// entering the event loop, so all workers start roughly together.
    pub fn create(
        idx: usize,
        core: CoreId,
        receiver: Receiver<DataFlowBuilder>,
        ready_barrier: Arc<Barrier>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            WORKER_IDX.set(idx);
            let mut worker = Self {
                io: IORequester::new(),
                id: core.id,
                data_flows: HashMap::new(),
                data_flow_queue: receiver,
                did_work_last_iteration: false,
                sleeps_between_clean: 0,
            };
            init_free_pool(idx);
            core_affinity::set_for_current(core);
            ready_barrier.wait();
            debug!("Starting worker {:?}", idx);
            worker.run().expect("Worker failed!");
        })
    }

    /// Submit IO requests from dataflows until the IO queue is full or no more requests remain.
    fn saturate_io(&mut self) -> Result<()> {
        for flow in self.data_flows.values_mut() {
            if self.io.has_pending() {
                break;
            }

            while let Some(requests) = flow.get_next_io_request()? {
                for r in requests {
                    self.io.request(r)?;
                }

                if self.io.has_pending() {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Run one unit of CPU work from the first dataflow that has work ready.
    fn step_run_ready_cpu_work(&mut self) -> Result<()> {
        for flow in self.data_flows.values_mut() {
            if let WorkStatus::Ran = flow.run_ready_cpu_work()? {
                self.did_work_last_iteration = true;
                return Ok(());
            }
        }

        Ok(())
    }

    /// Deliver completed IO buffers back to the operators that requested them.
    fn process_io_completions(&mut self) -> Result<()> {
        for (buffer, request) in self.io.completions()? {
            let data_flow = self.data_flows.get_mut(&request.data_flow_id).unwrap();
            debug!("Received IO for request {:?}", request.request.location);
            data_flow.process_io(request.operator_idx, request.request, buffer)?;
        }
        Ok(())
    }

    /// Check each dataflow for completion and remove finished ones.
    fn try_finishing_dataflows(&mut self) -> Result<()> {
        let mut to_remove = Vec::new();
        for (id, flow) in &mut self.data_flows {
            if flow.maybe_finish()? {
                // If nobody broke out- everybody is finished!
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            info!("Finished data flow {:?}", id);
            self.data_flows.remove(&id);
        }
        Ok(())
    }

    /// Attempt to steal work from sibling workers' channels when this worker is idle.
    fn try_steal_cpu_work(&mut self) -> Result<()> {
        for flow in self.data_flows.values_mut() {
            if let WorkStatus::Ran = flow.try_stealing_cpu_work()? {
                self.did_work_last_iteration = true;
                return Ok(());
            }
        }
        Ok(())
    }

    fn clear_dirty_buffer_or_sleep(&mut self) {
        if self.sleeps_between_clean >= 1
            && let Some(b) = pop_dirty_buffer()
        {
            self.sleeps_between_clean = 0;
            b.zero_out();
            return;
        }

        self.sleeps_between_clean += 1;
        sleep(Duration::from_millis(1));
    }

    #[instrument(skip(self), fields(worker_id = %self.id))]
    pub fn run(&mut self) -> Result<()> {
        loop {
            self.did_work_last_iteration = false;
            if let Ok(builder) = self.data_flow_queue.try_recv() {
                info!("Received data flow...");
                let data_flow = builder.build();
                self.data_flows.insert(data_flow.id(), data_flow);
            }

            self.process_io_completions()?;
            self.saturate_io()?;
            self.step_run_ready_cpu_work()?;
            self.try_finishing_dataflows()?;

            if !self.did_work_last_iteration {
                if self.io.has_pending() {
                    debug!("Waiting for IO...");
                    self.io.wait()?;
                    continue;
                }

                self.try_steal_cpu_work()?;

                if !self.did_work_last_iteration {
                    self.clear_dirty_buffer_or_sleep();
                }
            }
        }
    }
}
