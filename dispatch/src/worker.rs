use crate::env::get_env_var_with_default;
use crate::identified::Identifier;
use crate::io::{IORequest, PipelineIORequest, RowGroupFetch};
use crate::operations::ConsumeContext;
use crate::pipeline;
use crate::pipeline::Pipeline;
use crate::record_batch_metadata::with_row_group_metadata;
use arrow_schema::ArrowError;
use core_affinity::CoreId;
use crossbeam_deque::{Steal, Stealer};
use parquet::errors::ParquetError;
use parquetd::{BufferPool, ParquetReader, ParquetReaderBuilder, RawRowGroup};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier, LazyLock};
use std::thread::{JoinHandle, sleep};
use std::time::Duration;
use std::{result, thread};
use thiserror::Error;
use tracing::{debug, info, instrument, trace};

/// The size of the pool of buffers for pulling parquets. Only this many IO requests per worker may
/// be active at once
static PARQUET_BUFFER_POOL_SIZE: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("PARQUET_POOL_SIZE", 1));

/// The size of a buffer in the Parquet pool
static PARQUET_POOL_BUFFER_SIZE: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("PARQUET_POOL_BUFFER_SIZE", 256 * 1024 * 1024));

pub static IO_URING_SIZE: LazyLock<u32> =
    LazyLock::new(|| get_env_var_with_default("IO_URING_SIZE", 8));

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Parquet(#[from] ParquetError),
    #[error("{0}")]
    ParquetD(#[from] parquetd::Error),
    #[error("{0}")]
    Pipeline(#[from] pipeline::Error),
    #[error("{0}")]
    PipelineMissing(Identifier),
}

pub type Result<T, E = Error> = result::Result<T, E>;

pub struct PipelineHandle {
    pipeline: Pipeline,
    // IO pending right now within the parquet reader
    io_pending: usize,
    io_request_queue: crossbeam_deque::Worker<PipelineIORequest>,
    io_stealers: Vec<Stealer<PipelineIORequest>>,
}

impl PipelineHandle {
    pub fn new(
        pipeline: Pipeline,
        io_request_queue: crossbeam_deque::Worker<PipelineIORequest>,
        io_stealers: Vec<Stealer<PipelineIORequest>>,
    ) -> Self {
        Self {
            pipeline,
            io_pending: 0,
            io_request_queue,
            io_stealers,
        }
    }
}

impl Deref for PipelineHandle {
    type Target = Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl DerefMut for PipelineHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pipeline
    }
}

/// A Worker is spun up per CPU core. The worker's main entry-point, `create`, spins up a
/// thread with affinity to a CPU which continuously requests work from the dispatcher and does it.
///
/// The main idea of a Worker is to keep everything possible "local" to it, to prevent
/// context-switching/CPU cache-invalidation and in the future allow NUMA optimizations etc.
///
/// The Worker receives pipelines from the Dispatcher and runs them- the logic is outlined is as
/// follows:
///
/// 1. Look for work in Inputs:
///     - Look for batches ready in memory (`poll_record_batch`)- if there is, run them through
///       the entire pipeline
///     - If there's any pending work, wait on it (then run it through the pipeline),
///       and then resaturate
/// 2. If no work is waiting/ready, attempt to saturate IO- this will send any new row groups
///    (together with the given projections) to our `ParquetReader`
/// 3. Check if any Pipelines are finished and run their pipeline breakers (to output their data)
/// 4. Check if any new pipelines exist
///
/// And forever back to 1!
pub struct Worker {
    id: Identifier,
    pipeline_queue: Receiver<PipelineHandle>,
    parquet_reader: ParquetReader<PipelineIORequest>,
    /// A mapping from the pipeline identifier, to the pipeline and the number of pending IO
    /// operations
    pipelines: HashMap<Identifier, PipelineHandle>,
    did_work_on_iteration: bool,
}

impl Worker {
    /// Spin up a worker on a given core.
    /// The `ready_barrier` is used to synchronize the spinning up of different workers
    pub fn create(
        core: CoreId,
        receiver: Receiver<PipelineHandle>,
        ready_barrier: Arc<Barrier>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let pool = BufferPool::new(*PARQUET_BUFFER_POOL_SIZE, *PARQUET_POOL_BUFFER_SIZE);
            let reader = ParquetReaderBuilder::new()
                .ring_size(*IO_URING_SIZE)
                .buffer_pool(pool)
                .build()
                .expect("Cannot create reader");

            let worker = Self {
                id: core.id,
                parquet_reader: reader,
                pipelines: HashMap::new(),
                pipeline_queue: receiver,
                did_work_on_iteration: false,
            };
            core_affinity::set_for_current(core);
            ready_barrier.wait();
            worker.run().expect("Worker failed!");
        })
    }

    /// Process a raw row group received from the `ParquetReader` - this is an IO op that has
    /// returned, and the context (`PipelineIOContext`) will tell us where it's from and where to
    /// send the resulting `RecordBatch`s
    fn process_raw_row_group(
        &mut self,
        raw_row_group: RawRowGroup<PipelineIORequest>,
    ) -> Result<()> {
        let buffer_index = raw_row_group.buffer_index();

        let mut reader = raw_row_group.create_reader()?;
        let pipeline_context = raw_row_group.take_user_data();

        let pipeline_handle = self
            .pipelines
            .get_mut(&pipeline_context.pipeline_identifier)
            .ok_or(Error::PipelineMissing(pipeline_context.pipeline_identifier))?;

        // We received an IO operation back-
        pipeline_handle.io_pending -= 1;
        let (operations, row_group_handle, ctx) = match pipeline_context.request {
            IORequest::Operation(id, r, o) => (vec![id], r, ConsumeContext::IORequest(o)),
            // TODO: can we somehow get rid of `to_vec` here?
            IORequest::Input(id, r) => (
                pipeline_handle.subscribers(id).to_vec(),
                r,
                ConsumeContext::Publisher,
            ),
        };

        let mut offset = 0;
        for batch in reader.by_ref() {
            let record_batch = with_row_group_metadata(
                batch?,
                row_group_handle.row_group_metadata_handle.index(),
                offset,
            );
            offset += record_batch.num_rows();
            self.did_work_on_iteration = true;
            for operation_id in &operations {
                pipeline_handle.pipeline.run_operation(
                    &ctx,
                    *operation_id,
                    &mut pipeline_handle.io_request_queue,
                    &record_batch,
                )?;
            }
        }

        drop(reader);
        self.parquet_reader.return_buffer(buffer_index);
        Ok(())
    }

    /// Process any input batches that are available in memory- this will only call
    /// `poll_record_batch` on inputs
    fn process_pending_input_batches_from_memory(&mut self) -> Result<()> {
        for pipeline_handle in self.pipelines.values_mut() {
            for input in pipeline_handle.inputs() {
                if let Some(r) = input.poll_record_batch() {
                    let identifier = input.id();
                    self.did_work_on_iteration = true;
                    for subscriber in pipeline_handle.subscribers(identifier).to_vec() {
                        pipeline_handle.pipeline.run_operation(
                            &ConsumeContext::Publisher,
                            subscriber,
                            &mut pipeline_handle.io_request_queue,
                            &r,
                        )?;
                    }
                    // We return here to continue running on same input if possible
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn process_pending_reads(&mut self) -> Result<()> {
        trace!("Processing reads...");
        let completed = self.parquet_reader.wait_completed()?;
        trace!("Have {:?} completed", completed.len());

        for raw_row_group in completed {
            self.process_raw_row_group(raw_row_group)?;
            self.saturate_io()?;
        }
        Ok(())
    }

    fn current_io_limit(&self) -> usize {
        self.parquet_reader.buffer_pool().available_count()
    }

    /// Continuously attempt to add IO to the parquet reader until it is full (no more buffers
    /// available).
    ///
    /// Strategy-wise, this will attempt to exhaust each input before moving on to the next. This
    /// is purposefully done to allow anything hot in a particular segment of a pipeline to be run
    /// again and again.
    fn saturate_pending_io_work_from_inputs(&mut self) -> Result<()> {
        let limit = self.current_io_limit();
        let mut remaining = limit - self.pipelines.values().map(|p| p.io_pending).sum::<usize>();

        for (id, pipeline_handle) in &mut self.pipelines {
            for input in pipeline_handle
                .pipeline
                .inputs()
                .iter()
                .filter(|f| !f.source_finished())
            {
                // Exhaust this input until it's empty OR the global pool is full
                while remaining > 0 {
                    let Some((handle, projection)) = input.poll_io() else {
                        break;
                    };

                    remaining -= 1;
                    pipeline_handle.io_request_queue.push(PipelineIORequest {
                        pipeline_identifier: *id,
                        request: IORequest::Input(
                            input.id(),
                            RowGroupFetch {
                                row_group_metadata_handle: handle,
                                projection,
                            },
                        ),
                    });
                }

                if remaining == 0 {
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn saturate_parquet_reader_from_pending_io_work(&mut self) -> Result<()> {
        let mut did_input = false;
        let mut remaining = self.current_io_limit();

        'outer: for pipeline_handle in self.pipelines.values_mut() {
            while remaining > 0 {
                match pipeline_handle.io_request_queue.pop() {
                    Some(i) => {
                        // TODO: can we avoid cloning here?
                        let row_group_fetch = i.request.row_group_fetch();
                        let row_group_metadata =
                            row_group_fetch.row_group_metadata_handle.get().clone();
                        let projection = row_group_fetch.projection.clone();

                        self.parquet_reader.submit_projected_row_group_read(
                            i,
                            row_group_metadata,
                            projection,
                        )?;

                        pipeline_handle.io_pending += 1;
                        remaining -= 1;
                        did_input = true;
                    }
                    None => break,
                }
            }

            if remaining == 0 {
                break 'outer;
            }
        }

        if did_input {
            self.did_work_on_iteration = true;
            self.parquet_reader.submit()?;
        }
        Ok(())
    }

    fn saturate_io(&mut self) -> Result<()> {
        self.saturate_pending_io_work_from_inputs()?;
        self.saturate_parquet_reader_from_pending_io_work()?;
        Ok(())
    }

    fn try_saturate_pending_work_from_sibling_workers(&mut self) -> Result<()> {
        let mut remaining = self.current_io_limit();

        for pipeline_handle in self.pipelines.values_mut() {
            for stealer in &mut pipeline_handle.io_stealers {
                // Exhaust this input until it's empty OR the global pool is full
                while remaining > 0 {
                    let res = stealer.steal();
                    let Steal::Success(request) = res else {
                        trace!("No pending work from sibling worker!");
                        break;
                    };

                    // Given we're stealing from another worker, this pipeline may have notified that
                    // it was finished - let's notify our brethren that we are now resuming
                    pipeline_handle
                        .pipeline
                        .maybe_increment_shared_pipeline_workers();

                    pipeline_handle.io_pending += 1;
                    remaining -= 1;
                    // TODO: can we avoid cloning here?
                    let row_group_metadata = request
                        .request
                        .row_group_fetch()
                        .row_group_metadata_handle
                        .get()
                        .clone();
                    let projection = request.request.row_group_fetch().projection.clone();
                    self.did_work_on_iteration = true;
                    self.parquet_reader.submit_projected_row_group_read(
                        request,
                        row_group_metadata,
                        projection,
                    )?;
                }

                if remaining == 0 {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// The inner "forever-loop" of the Worker. Continuously request pipelines from the dispatcher
    /// and execute them
    #[instrument(skip(self), fields(worker_id = %self.id))]
    fn run(mut self) -> Result<()> {
        loop {
            self.did_work_on_iteration = false;
            // Look for work in Inputs - we're going to attempt to find work either in pending
            // record batches or IO
            self.process_pending_input_batches_from_memory()?;

            if self.parquet_reader.has_pending() {
                // We have pending data in our uring! Let's wait on it and the run on the data
                self.process_pending_reads()?;
            } else {
                // No work available within pipelines - let's try saturating our pipelines
                self.saturate_io()?;
            }

            // Check if any pipeline is finished
            for (_id, pipeline_handle) in self.pipelines.extract_if(|_, p| {
                if p.io_pending == 0 && p.sources_finished() {
                    // We notify our siblings that we're finished (if we didn't), and return whether we were the last sibling to finish (if so, run breakers)
                    debug!("Notifying siblings that pipeline {:?} finished...", p.id());
                    p.maybe_decrement_shared_pipeline_workers()
                } else {
                    false
                }
            }) {
                info!("Finished pipeline {:?}", pipeline_handle.id());
                pipeline_handle.pipeline.finish_pipeline_breakers()?;
            }

            // Check if we want to steal IO
            if self.pipelines.iter_mut().all(|(_, p)| p.sources_finished())
                && !self.parquet_reader.has_pending()
                && self.pipelines.values().all(|p| p.io_pending == 0)
            {
                debug!("Trying to steal IO...");
                self.try_saturate_pending_work_from_sibling_workers()?;
            }

            // Try collecting any new pipelines
            if let Ok(p) = self.pipeline_queue.try_recv() {
                debug!("Received pipeline!");
                self.pipelines.insert(p.id(), p);
            }

            // If nothing to do - let's sleep :) Save the environment! One millisecond at a time.
            if !self.did_work_on_iteration {
                sleep(Duration::from_millis(1));
            }
        }
    }
}
