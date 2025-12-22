use crate::env::get_env_var_with_default;
use crate::operations::{Identifier, Operation};
use crate::source::Source;
use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use parquet::errors::ParquetError;
use parquetd::{BufferPool, ParquetReader, ParquetReaderBuilder, Projection};
use std::collections::HashMap;
use std::result;
use std::sync::{Arc, Barrier, LazyLock};
use thiserror::Error;
use tracing::{debug, trace};

/// What is the minimum amount of work that should be in the io_uring? If there is an amount below
/// this, the pipeline will "replenish" up to `MAX_IN_DISK_QUEUE`
static MIN_IN_DISK_QUEUE: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("MIN_IN_DISK_QUEUE", 1usize));

/// What is the maximum amount of work there should be in the io_uring? Note that the maximum can
/// be passed if a single path brings the number of pending reads above the maximum.
static MAX_IN_DISK_QUEUE: LazyLock<usize> =
    LazyLock::new(|| get_env_var_with_default("MAX_IN_DISK_QUEUE", 1usize));

static IO_URING_SIZE: LazyLock<u32> =
    LazyLock::new(|| get_env_var_with_default("IO_URING_SIZE", 32));

#[derive(Debug, Error)]
pub enum Error {
    #[error("parquet-uring error: {0}")]
    ParquetD(#[from] parquetd::Error),
    #[error("parquet error: {0}")]
    Parquet(#[from] ParquetError),
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// A configuration of a Pipeline: this is the main API to be used to create a Pipeline. To see
/// more about what a Pipeline is, look at docs of `Pipeline`
pub struct PipelineConfig {
    /// What is the Pipeline's source (for parquets)?
    pub source: Arc<Source>,
    /// What is the initial projection on the parquet files?
    pub projection: Option<Projection>,
    /// A mapping of identifiers to non-input operations
    pub operations: HashMap<Identifier, Box<dyn Operation>>,
    /// All input operations, which will be directly fed with RecordBatch's from the soruce
    pub initial_operations: Vec<Box<dyn Operation>>,
    /// A mapping of publishing identifiers to subscriber identifiers
    pub publishers_to_subscribers: HashMap<Identifier, Vec<Identifier>>,
    /// The barrier for finishing the pipeline (so that multiple pipelines on different workers
    /// can synchronize their ending together)
    /// TODO: we probably don't want this, just a nice way to know the query ended
    pub barrier: Arc<Barrier>,
}

impl PipelineConfig {
    pub fn into_pipeline(self, buffer_pool: BufferPool) -> Result<Pipeline> {
        let reader = ParquetReaderBuilder::new()
            .projection(self.projection)
            .ring_size(*IO_URING_SIZE)
            .buffer_pool(buffer_pool)
            .build()?;
        Ok(Pipeline {
            source: self.source,
            parquet_reader: reader,
            operations: self.operations,
            publishers_to_subscribers: self.publishers_to_subscribers,
            barrier: self.barrier,
            initial_operations: self.initial_operations,
        })
    }
}

/// A `Pipeline` is the main unit of work of a `Worker`. A Pipeline consists of multiple operations
/// chained together (their connections are described through `publishers_to_subscribers`), and is
/// meant to be executed by a single worker (and thus a core).
///
/// The Pipeline looks for work in three places (given here by order of priority):
/// 1. Are there any operations that have output ready to be consumed? If so, consume the output
///    and run its subscribers on it
/// 2. Does the parquet reader have any pending work (in its io_uring)? If so, wait for it to be
///    ready, read it (decompressing & parsing) and send it to the inputs of the pipeline
/// 3. Are there any new files waiting in the `self.source`? If so, send them to the io_uring to be
///    loaded asynchronously
pub struct Pipeline {
    source: Arc<Source>,
    parquet_reader: ParquetReader,
    operations: HashMap<Identifier, Box<dyn Operation>>,
    publishers_to_subscribers: HashMap<Identifier, Vec<Identifier>>,
    initial_operations: Vec<Box<dyn Operation>>,
    barrier: Arc<Barrier>,
}

impl Pipeline {
    /// Execute the pipeline continuously until no more work exists in the source, and all operations
    /// have been exhausted (i.e., have no more output to be consumed)
    pub fn execute(mut self) -> Result<BufferPool> {
        debug!("Running worker!");
        loop {
            if let Some((batch, id)) = self.find_local_operation() {
                // We have work locally! Let's do it.
                let subscribers = self.publishers_to_subscribers.get(&id).unwrap();
                for identifier in subscribers {
                    let operation = self.operations.get_mut(identifier).unwrap();
                    operation.run(&batch);
                }
            } else {
                // No work available; let's get some from either our parquet_reader or source
                // First, check if parquet_reader has any available
                if self.parquet_reader.has_pending() {
                    // We have parquets pending- let's wait for them to be loaded & process them
                    self.decode_ready_parquets()?;
                } else {
                    // Nothing to do, let's get from our source!
                    // We want to drain a few at a time to ensure we always have waiting work
                    if !self.source.is_empty() {
                        debug!("Source isn't empty, saturating worker...");
                        self.saturate_parquet_reader()?;
                    }
                    // It's possible we're finished at this point- let's check
                    if !self.parquet_reader.has_pending() {
                        return Ok(self.finish_pipeline());
                    }
                }
            }
        }
    }

    /// Decode all ready parquets that have been made ready by the parquet_reader (this happens
    /// asynchronously to the execution of the pipeline by the kernel)
    ///
    /// The `reader` will be re-saturated from the source upon finishing to read all parquets, to
    /// ensure any IO work is done asynchronously to CPU execution.
    fn decode_ready_parquets(&mut self) -> Result<()> {
        let completed = self.parquet_reader.wait_completed()?;

        for raw_row_group in completed {
            let buffer_index = raw_row_group.buffer_index();
            let mut reader = raw_row_group.create_reader()?;
            for r in reader.by_ref() {
                let batch = r?;
                for operation in &mut self.initial_operations {
                    operation.run(&batch);
                }
            }
            drop(reader);
            self.parquet_reader.return_buffer(buffer_index);
        }
        self.saturate_parquet_reader()?;
        self.parquet_reader.drain_pending_reads()?;
        Ok(())
    }

    fn saturate_parquet_reader(&mut self) -> Result<()> {
        if self.parquet_reader.pending_count() < *MIN_IN_DISK_QUEUE {
            let mut collected = false;
            while self.parquet_reader.pending_count() < *MAX_IN_DISK_QUEUE {
                match self.source.pop_parquet_path() {
                    None => {
                        // Nothing left, pipeline is finished.
                        break;
                    }
                    Some(p) => {
                        collected = true;
                        trace!("Submitting path {:?}", p);
                        self.parquet_reader.open_and_submit(p)?;
                    }
                }
            }
            if collected {
                self.parquet_reader.submit()?;
            }
        }
        Ok(())
    }

    fn finish_pipeline(mut self) -> BufferPool {
        let mut operations = self.initial_operations;

        while !operations.is_empty() {
            let mut next_operations = vec![];
            for operation in &mut operations {
                let last_batch_opt = operation.finish();

                if let Some(subscribers) = self.publishers_to_subscribers.remove(&operation.id()) {
                    // Get a vector of all child operations
                    let mut operations = subscribers
                        .iter()
                        .map(|i| self.operations.remove(i).unwrap())
                        .collect::<Vec<_>>();
                    // If we have a last batch outputting on the operation's finish, let's do it!
                    if let Some(batch) = last_batch_opt {
                        for operation in &mut operations {
                            operation.run(&batch);
                        }
                    }
                    next_operations.extend(operations);
                }
            }
            operations = next_operations;
        }

        self.barrier.wait();
        self.parquet_reader.buffer_pool()
    }

    fn find_local_operation(&mut self) -> Option<(RecordBatch, Identifier)> {
        for operation in self
            .operations
            .values_mut()
            .chain(&mut self.initial_operations)
        {
            if let Some(batch) = operation.consume_output_batch() {
                return Some((batch, operation.id()));
            }
        }
        None
    }
}
