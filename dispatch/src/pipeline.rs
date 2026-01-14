use crate::identified::{Identified, Identifier};
use crate::input::Input;
use crate::io::OperationIOSubmitter;
use crate::io::PipelineIORequest;
use crate::operations;
use crate::operations::{ConsumeContext, Operation, PipelineBreaker};
use arrow_array::RecordBatch;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::result;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Cannot find operation {0}")]
    CannotFindOperation(Identifier),
    #[error("{0}")]
    Operation(#[from] operations::Error),
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// A `Pipeline` is the main unit of work of a `Worker`. A Pipeline consists of multiple operations
/// chained together (their connections are described through `publishers_to_subscribers`), and is
/// meant to be executed by a single worker (and thus a core).
pub struct Pipeline {
    id: Identifier,
    inputs: Vec<Identified<Box<dyn Input>>>,
    operations: HashMap<Identifier, Box<dyn Operation>>,
    pipeline_breakers: HashMap<Identifier, Box<dyn PipelineBreaker>>,
    /// A mapping of publishing identifiers to subscriber identifiers
    publishers_to_subscribers: HashMap<Identifier, Vec<Identifier>>,
    /// Amount of workers still running on this pipeline. Once this goes down to 0, all instances
    /// of pipelines will run their breakers
    workers_remaining: Arc<AtomicUsize>,
    /// Did we decrement `workers_remaining`?
    did_decrement_workers_remaining: bool,
}

impl Debug for Pipeline {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Pipeline");
        debug_struct.field("id", &self.id).finish()
    }
}

impl Pipeline {
    pub fn new(
        id: Identifier,
        inputs: Vec<Identified<Box<dyn Input>>>,
        operations: Vec<Identified<Box<dyn Operation>>>,
        pipeline_breakers: Vec<Identified<Box<dyn PipelineBreaker>>>,
        publishers_to_subscribers: HashMap<Identifier, Vec<Identifier>>,
        siblings_remaining: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            id,
            inputs,
            operations: operations.into_iter().map(|o| (o.id(), o.take())).collect(),
            pipeline_breakers: pipeline_breakers
                .into_iter()
                .map(|o| (o.id(), o.take()))
                .collect(),
            publishers_to_subscribers,
            did_decrement_workers_remaining: false,
            workers_remaining: siblings_remaining,
        }
    }

    pub fn id(&self) -> Identifier {
        self.id
    }

    pub fn sources_finished(&self) -> bool {
        self.inputs.iter().all(|i| i.source_finished())
    }

    /// Maybe decrement the shared Atomic for the number of workers remaining working the pipeline,
    /// if the given instance of the pipeline has not already done so (i.e., if the worker has not
    /// previously finished this pipeline). This is used to notify other workers (sibling
    /// pipelines) that this instance of the pipeline is done. Once the counter reaches 0, nobody
    /// is working on the pipeline anymore.
    ///
    /// Once decremented, `maybe_increment_shared_pipeline_workers` will increment the counter.
    ///
    /// # Returns whether there are no more workers left working on this pipeline
    pub fn maybe_decrement_shared_pipeline_workers(&mut self) -> bool {
        if !self.did_decrement_workers_remaining {
            self.did_decrement_workers_remaining = true;
            self.workers_remaining
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
                == 1
        } else {
            self.workers_remaining
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
        }
    }

    /// Maybe increment the shared Atomic for the number of workers remaining working the pipeline,
    /// if the given instance of the pipeline has previously notified that it finished (i.e., called
    /// `maybe_decrement_shared_pipeline_workers`). This can happen if a worker steals work from
    /// a sibling worker, at which point the pipeline would "come back to life".
    /// Upon the beginning of work on a pipeline, this
    /// function would do nothing.
    ///
    /// Once incremented, `maybe_decrement_shared_pipeline_workers` will decrement the counter.
    pub fn maybe_increment_shared_pipeline_workers(&mut self) {
        if self.did_decrement_workers_remaining {
            self.did_decrement_workers_remaining = false;
            self.workers_remaining
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn finish_pipeline_breakers(self) -> Result<()> {
        for (_, pipline_breaker) in self.pipeline_breakers {
            pipline_breaker.finish()?;
        }
        Ok(())
    }

    pub fn subscribers(&mut self, identifier: Identifier) -> &[Identifier] {
        self.publishers_to_subscribers.get(&identifier).unwrap()
    }

    pub fn inputs(&mut self) -> &mut [Identified<Box<dyn Input>>] {
        self.inputs.as_mut_slice()
    }

    pub fn run_operation(
        &mut self,
        context: &ConsumeContext,
        operation_id: Identifier,
        io_request_queue: &mut crossbeam_deque::Worker<PipelineIORequest>,
        batch: &RecordBatch,
    ) -> Result<()> {
        let mut batches = vec![(vec![operation_id], batch.clone())];
        while let Some((ids, b)) = batches.pop() {
            for id in ids {
                let operation: &mut dyn Operation = self
                    .operations
                    .get_mut(&id)
                    .map(|boxed| boxed.as_mut())
                    .or_else(|| {
                        self.pipeline_breakers
                            .get_mut(&id)
                            .map(|boxed| boxed.as_mut() as &mut dyn Operation)
                    })
                    .ok_or(Error::CannotFindOperation(id))?;
                let next_batch = operation.consume(
                    context,
                    OperationIOSubmitter::new(self.id, id, io_request_queue),
                    &b,
                )?;
                if let Some(n) = next_batch {
                    let subscribers = self.publishers_to_subscribers.get(&id).cloned();
                    if let Some(subscribers) = subscribers {
                        batches.push((subscribers, n))
                    }
                }
            }
        }
        Ok(())
    }
}
