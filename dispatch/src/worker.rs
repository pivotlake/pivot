use crate::Dispatcher;
use crate::env::get_env_var_with_default;
use core_affinity::CoreId;
use crossbeam_deque::Steal;
use parquetd::BufferPool;
use std::sync::{Arc, Barrier, LazyLock};
use std::thread;
use std::thread::{JoinHandle, sleep};
use std::time::Duration;

static PARQUET_POOL_SIZE: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("PARQUET_POOL_SIZE", 8));

static PARQUET_POOL_BUFFER_SIZE: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("PARQUET_POOL_BUFFER_SIZE", 64 * 1024 * 1024));

/// A Worker is spun up per CPU core. The worker's main entry-point, `create`, spins up a
/// thread with affinity to a CPU which continuously requests work from the dispatcher and does it.
///
/// The main idea of a Worker is to keep everything possible "local" to it, to prevent
/// context-switching/CPU cache-invalidation and in the future allow NUMA optimizations etc.
///
/// The Worker receives pipelines from the Dispatcher and runs them.
pub struct Worker {
    /// The dispatcher the worker is using- there should ideally be one per the lifetime of the
    /// process
    dispatcher: Arc<Dispatcher>,
    /// A pool of buffers to be re-used when reading parquet files using `parquetd`. The buffers
    /// are pre-allocated to prevent continuous deallocation/allocation of buffers whenever
    /// a parquet needs to be read, and the buffers are saved between runs of different pipelines
    parquet_pool: BufferPool,
}

impl Worker {
    /// Spin up a worker on a given core.
    /// The `ready_barrier` is used to synchronize the spinning up of different workers
    pub fn create(
        core: CoreId,
        dispatcher: Arc<Dispatcher>,
        ready_barrier: Arc<Barrier>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let worker = Self {
                dispatcher,
                parquet_pool: BufferPool::new(*PARQUET_POOL_SIZE, *PARQUET_POOL_BUFFER_SIZE),
            };
            core_affinity::set_for_current(core);
            ready_barrier.wait();
            worker.run().expect("Worker failed!");
        })
    }

    /// The inner "forever-loop" of the Worker. Continuously request pipelines from the dispatcher
    /// and execute them
    fn run(mut self) -> crate::pipeline::Result<()> {
        loop {
            match self.dispatcher.pipelines.steal() {
                Steal::Success(p) => {
                    let pipeline = p.into_pipeline(self.parquet_pool)?;
                    self.parquet_pool = pipeline.execute()?;
                }
                _ => sleep(Duration::from_millis(1)),
            }
        }
    }
}
