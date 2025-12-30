use crate::env::get_env_var_with_default;
use crate::identified::Identifier;
use crate::io::{PipelineIO, PipelineIOContext};
use crate::pipeline::Pipeline;
use crate::{ConsumeContext, pipeline};
use arrow_schema::ArrowError;
use core_affinity::CoreId;
use parquet::errors::ParquetError;
use parquetd::{
    BufferPool, ParquetReader, ParquetReaderBuilder, RawRowGroup,
};
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier, LazyLock};
use std::thread::{JoinHandle, sleep};
use std::time::Duration;
use std::{result, thread};
use thiserror::Error;
use tracing::{debug, trace};
use crate::record_batch_metadata::with_row_group_metadata;

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
    pipeline_queue: Receiver<Pipeline>,
    parquet_reader: ParquetReader<PipelineIOContext>,
    /// A mapping from the pipeline identifier, to the pipeline and the number of running IO
    /// operations
    pipelines: HashMap<Identifier, (Pipeline, usize)>,
}

impl Worker {
    /// Spin up a worker on a given core.
    /// The `ready_barrier` is used to synchronize the spinning up of different workers
    pub fn create(
        core: CoreId,
        receiver: Receiver<Pipeline>,
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
        raw_row_group: RawRowGroup<PipelineIOContext>,
    ) -> Result<()> {
        let buffer_index = raw_row_group.buffer_index();

        let mut reader = raw_row_group.create_reader()?;
        let pipeline_context = raw_row_group.take_user_data();

        let (pipeline, io_count) = self
            .pipelines
            .get_mut(&pipeline_context.pipeline_identifier)
            .ok_or(Error::PipelineMissing(pipeline_context.pipeline_identifier))?;

        // We received an IO operation back-
        *io_count -= 1;
        let (operations, row_group_handle, ctx) = match pipeline_context.context {
            PipelineIO::Operation(id, r, o) => (vec![id], r, ConsumeContext::IORequest(o)),
            // TODO: can we somehow get rid of `to_vec` here?
            PipelineIO::Input(id, r) => (pipeline.subscribers(id).to_vec(), r, ConsumeContext::Publisher)
        };

        let mut offset = 0;
        for batch in reader.by_ref() {
            let record_batch = with_row_group_metadata(batch?, row_group_handle.index(), offset);
            offset += record_batch.num_rows();
            for operation_id in &operations {
                pipeline.run_operation(
                    &ctx,
                    *operation_id,
                    &mut self.parquet_reader,
                    io_count,
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
        for (pipeline, io_count) in self.pipelines.values_mut() {
            for input in pipeline.inputs() {
                if let Some(r) = input.poll_record_batch() {
                    let identifier = input.id();
                    pipeline.run_input_operation(
                        identifier,
                        r,
                        &mut self.parquet_reader,
                        io_count,
                    )?;
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
            self.saturate_parquet_reader()?;
        }
        Ok(())
    }

    /// Continuously attempt to add IO to the parquet reader until it is full (no more buffers
    /// available).
    ///
    /// Strategy-wise, this will attempt to exhaust each input before moving on to the next. This
    /// is purposefully done to allow anything hot in a particular segment of a pipeline to be run
    /// again and again.
    fn saturate_parquet_reader(&mut self) -> Result<()> {
        let limit = self.parquet_reader.buffer_pool().available_count();
        let mut did_input = false;

        'saturation: for (p, counter) in self.pipelines.values_mut() {
            let p_id = p.id();

            for input in p.inputs().iter().filter(|f| !f.source_finished()) {
                // Exhaust this input until it's empty OR the global pool is full
                while self.parquet_reader.pending_count() < limit {
                    let Some((handle, projection)) = input.poll_io() else {
                        break;
                    };

                    let row_group_metadata = handle.get().clone();
                    self.parquet_reader.submit_projected_row_group_read(
                        PipelineIOContext {
                            pipeline_identifier: p_id,
                            context: PipelineIO::Input(input.id(), handle),
                        },
                        row_group_metadata,
                        projection.as_ref(),
                    )?;

                    *counter += 1;
                    did_input = true;
                }

                if self.parquet_reader.pending_count() >= limit {
                    break 'saturation;
                }
            }
        }

        if did_input {
            self.parquet_reader.submit()?;
        }
        Ok(())
    }

    /// The inner "forever-loop" of the Worker. Continuously request pipelines from the dispatcher
    /// and execute them
    fn run(mut self) -> Result<()> {
        loop {
            // Look for work in Inputs - we're going to attempt to find work either in pending
            // record batches or IO
            self.process_pending_input_batches_from_memory()?;

            if self.parquet_reader.has_pending() {
                // We have pending data in our uring! Let's wait on it and the run on the data
                self.process_pending_reads()?;
            } else {
                // No work available within pipelines - let's try saturating our pipelines
                self.saturate_parquet_reader()?;
            }

            // Check if any pipeline is finished
            for (_id, (pipeline, _)) in self
                .pipelines
                .extract_if(|_, (pipe, cnt)| *cnt == 0 && pipe.sources_finished())
            {
                debug!("{:?} Finished pipeline!", self.id);
                pipeline.output_pipeline_breakers();
            }

            // Try collecting any new pipelines
            if let Ok(p) = self.pipeline_queue.try_recv() {
                debug!("{:?} Received pipeline!", self.id);
                self.pipelines.insert(p.id(), (p, 0));
            }

            // If nothing to do - let's sleep :) Save the environment! One millisecond at a time.
            if self.pipelines.is_empty() {
                sleep(Duration::from_millis(1));
            }
        }
    }
}
