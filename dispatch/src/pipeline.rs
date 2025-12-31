use crate::identified::{Identified, Identifier};
use crate::input::Input;
use crate::io::OperationIOSubmitter;
use crate::io::PipelineIOContext;
use crate::operations::Operation;
use crate::{ConsumeContext, PipelineBreaker, operations};
use arrow_array::RecordBatch;
use parquetd::ParquetReader;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::result;
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
        }
    }

    pub fn id(&self) -> Identifier {
        self.id
    }

    pub fn sources_finished(&self) -> bool {
        self.inputs.iter().all(|i| i.source_finished())
    }

    pub fn output_pipeline_breakers(self) -> Result<()> {
        for (_, pipline_breaker) in self.pipeline_breakers {
            pipline_breaker.output()?;
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
        parquet_reader: &mut ParquetReader<PipelineIOContext>,
        counter: &mut usize,
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
                    OperationIOSubmitter::new(parquet_reader, counter, id, self.id),
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

    pub fn run_input_operation(
        &mut self,
        identifier: Identifier,
        batch: RecordBatch,
        reader: &mut ParquetReader<PipelineIOContext>,
        counter: &mut usize,
    ) -> Result<()> {
        // TODO: remove this clone
        let subscribers = self
            .publishers_to_subscribers
            .get(&identifier)
            .unwrap()
            .clone();
        for subscriber in subscribers {
            self.run_operation(
                &ConsumeContext::Publisher,
                subscriber,
                reader,
                counter,
                &batch,
            )?;
        }
        Ok(())
    }
}
