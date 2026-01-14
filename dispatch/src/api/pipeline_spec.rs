use crate::api::InputSpec;
use crate::api::node::Node;
use crate::api::operation_spec::OperationSpec;
use crate::api::pipeline_breaker_spec::PipelineBreakerSpec;
use crate::dispatcher;
use crate::identified::{Identified, Identifier};
use crate::memory_source::MemorySource;
use crate::pipeline::Pipeline;
use crate::table::{Table, TableSource};
use parquetd::Projection;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

static ID: LazyLock<AtomicUsize> = LazyLock::new(|| AtomicUsize::new(0));

pub fn next_pipeline_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

struct Inner {
    id: usize,
    breakers: Vec<Identified<PipelineBreakerSpec>>,
    operations: Vec<Identified<OperationSpec>>,
    inputs: Vec<Identified<InputSpec>>,
    topology: HashMap<Identifier, Vec<Identifier>>,
}

/// A declarative specification for building parallel pipelines.
///
/// `PipelineSpec` stores the blueprint for a pipeline: inputs, operations,
/// and pipeline breakers. When `execute()` is called, it instantiates the
/// actual pipeline components for each worker thread.
///
/// Use `table_input()` or `memory_input()` to start building. Chain operations
/// on the returned [`Node`], and finish with pipeline breakers like
/// `count()`, `order_by_limit()`, or `group_by_count()`.
///
/// Note that every pipeline can have *multiple* pipeline breakers. This is the reason the pipeline
/// is edited in place, instead of continuously returning itself with every new operation.
#[derive(Clone)]
pub struct PipelineSpec {
    id: usize,
    inner: Rc<RefCell<Inner>>,
}

impl Default for PipelineSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineSpec {
    /// Create a new empty pipeline specification.
    pub fn new() -> Self {
        Self {
            id: next_pipeline_id(),
            inner: Rc::new(RefCell::new(Inner {
                id: 0,
                breakers: vec![],
                operations: vec![],
                inputs: vec![],
                topology: Default::default(),
            })),
        }
    }

    fn next_id(&self) -> usize {
        let mut inner = self.inner.borrow_mut();
        let id = inner.id;
        inner.id += 1;
        id
    }

    /// Execute the pipeline across all workers.
    ///
    /// This instantiates the pipeline for each worker thread and dispatches
    /// work. The call returns immediately; use `MemoryFeed::collect()` to
    /// wait for results.
    pub fn execute(&self) {
        let inner = self.inner.borrow();
        let topology = inner.topology.clone();
        let operation_builders: Vec<_> = inner
            .operations
            .iter()
            .map(|o| (o.id(), o.build_operation()))
            .collect();
        let breaker_builders: Vec<_> = inner
            .breakers
            .iter()
            .map(|o| (o.id(), o.build_breaker()))
            .collect();
        let inputs: Vec<_> = inner
            .inputs
            .iter()
            .map(|o| (o.id(), o.build_input()))
            .collect();

        let workers_remaining = Arc::new(AtomicUsize::new(dispatcher().workers()));

        dispatcher().push_pipeline(|| {
            Pipeline::new(
                self.id,
                inputs
                    .iter()
                    .map(|(id, f)| Identified::new(*id, f()))
                    .collect(),
                operation_builders
                    .iter()
                    .map(|(id, f)| Identified::new(*id, f()))
                    .collect(),
                breaker_builders
                    .iter()
                    .map(|(id, f)| Identified::new(*id, f()))
                    .collect(),
                topology.clone(),
                workers_remaining.clone(),
            )
        })
    }

    /// Add a parquet table as input.
    ///
    /// Use `projection` to specify which columns to read (column indices).
    /// Returns a [`Node`] to chain operations.
    pub fn table_input(&self, table: Arc<Table>, projection: Option<Projection>) -> Node {
        let next_id = self.next_id();
        self.inner.borrow_mut().inputs.push(Identified::new(
            next_id,
            InputSpec::Table {
                table_source: Arc::new(TableSource::from(&table)),
                projection,
            },
        ));
        Node::new(next_id, self.clone())
    }

    /// Add an in-memory source as input (for multi-stage pipelines).
    ///
    /// Use with `MemoryFeed::source()` to chain pipeline stages.
    pub fn memory_input(&self, memory_source: Arc<MemorySource>) -> Node {
        let next_id = self.next_id();
        self.inner
            .borrow_mut()
            .inputs
            .push(Identified::new(next_id, InputSpec::Memory(memory_source)));
        Node::new(next_id, self.clone())
    }

    pub(crate) fn push_operation(&self, id: Identifier, spec: OperationSpec) -> Node {
        let next_id = self.next_id();
        let mut inner = self.inner.borrow_mut();
        inner.topology.entry(id).or_default().push(next_id);
        inner.operations.push(Identified::new(next_id, spec));
        inner.id += 1;
        Node::new(next_id, self.clone())
    }

    pub(crate) fn push_breaker(&self, id: Identifier, spec: PipelineBreakerSpec) -> Node {
        let next_id = self.next_id();
        let mut inner = self.inner.borrow_mut();
        inner.topology.entry(id).or_default().push(next_id);
        inner.breakers.push(Identified::new(next_id, spec));
        Node::new(next_id, self.clone())
    }
}
