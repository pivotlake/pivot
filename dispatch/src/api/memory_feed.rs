use crate::input::Input;
use crate::memory_source::{MemoryOutput, MemorySource};
use crate::operations::Output;
use crate::{OutputSpec, dispatcher};
use arrow_array::RecordBatch;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::channel;
use std::sync::{Arc, mpsc};
use tracing::debug;

/// In-memory buffer for sending output from a `PipelineBreaker`.
/// This allows us to create a sort of in-memory "bridge" between pipelines, or simply to collect
/// the output of a specific `PipelineBreaker`
///
/// Use `output()` to get a spec for pipeline breakers, and `collect()` to
/// wait for completion and retrieve results. For multi-stage pipelines,
/// use `source()` to feed into the next stage.
///
/// For simply collection output:
/// ```ignore
/// let feed = MemoryFeed::new();
///
/// PipelineSpec::new()
///     .table_input(table, None)
///     .count(feed.output())
///     .execute();
///
/// let results = feed.collect();  // blocks until done
/// ```
///
/// For creating a "bridge"
/// ```ignore
/// let feed = MemoryFeed::new();
///
/// PipelineSpec::new()
///     .table_input(table, None)
///     .group_by(..., feed.output())
///     .execute();
///
/// PipelineSpec::new()
///     .memory_input(feed.source())
///     ...
/// ```
pub struct MemoryFeed {
    source: Arc<MemorySource>,
    active: Arc<AtomicUsize>,
    tx: Arc<mpsc::Sender<()>>,
    rx: Option<mpsc::Receiver<()>>,
}

/// Output specification for [`MemoryFeed`].
///
/// Created via `MemoryFeed::output()`. Passed to pipeline breakers.
pub struct MemoryOutputSpec {
    source: Arc<MemorySource>,
    active: Arc<AtomicUsize>,
    tx: Arc<mpsc::Sender<()>>,
}

impl Default for MemoryFeed {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryFeed {
    /// Create a new memory feed.
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            source: Arc::new(Default::default()),
            active: Arc::new(AtomicUsize::new(dispatcher().workers())),
            tx: Arc::new(tx),
            rx: Some(rx),
        }
    }

    /// Get the underlying source for multi-stage pipelines.
    ///
    /// Pass to `PipelineSpec::memory_input()` in the next stage.
    pub fn source(&self) -> Arc<MemorySource> {
        self.source.clone()
    }

    /// Get an output spec for pipeline breakers.
    pub fn output(&self) -> MemoryOutputSpec {
        MemoryOutputSpec {
            source: self.source.clone(),
            active: self.active.clone(),
            tx: self.tx.clone(),
        }
    }

    /// Wait for the pipeline to complete and collect all output batches.
    ///
    /// Consumes the feed. Blocks until all workers have finished.
    pub fn collect(mut self) -> Vec<RecordBatch> {
        // Wait while there are still active writers
        self.rx
            .take()
            .expect("Output was already collected!")
            .recv()
            .expect("Output died!");

        debug!("Source is empty? {:?}", self.source.is_empty());

        let mut batches = vec![];
        while !self.source.source_finished() {
            while let Some(s) = self.source.poll_record_batch() {
                batches.push(s);
            }
        }

        batches
    }
}

impl OutputSpec for MemoryOutputSpec {
    fn build_output(&self) -> Box<dyn Output> {
        Box::new(MemoryOutput::new(
            self.active.clone(),
            self.source.clone(),
            self.tx.clone(),
        ))
    }
}
