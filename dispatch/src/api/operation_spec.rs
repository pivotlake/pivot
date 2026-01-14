use crate::operations::{Filter, Materializer, Operation, Project};
use crate::table::Table;
use arrow_array::{BooleanArray, RecordBatch};
use parquetd::Projection;
use std::sync::Arc;

/// Factory type for filter predicates.
pub type FilterBuild =
    Box<dyn Fn() -> Box<dyn FnMut(&RecordBatch) -> BooleanArray + Send + Sync + 'static>>;

/// Factory type for projection functions.
pub type ProjectBuild =
    Box<dyn Fn() -> Box<dyn FnMut(&RecordBatch) -> RecordBatch + Send + 'static>>;

/// Specification for intermediate operations (non-terminal).
///
/// Created internally by `Node::filter_builder()`, `project()`, and `materialize()`.
pub enum OperationSpec {
    /// Filter rows using a predicate builder.
    FilterBuilder(FilterBuild),
    /// Project columns using a projection builder.
    ProjectBuilder(ProjectBuild),
    /// Materialize deferred columns from the table.
    Materializer {
        projection: Option<Projection>,
        table: Arc<Table>,
    },
}

impl OperationSpec {
    /// Create an operation factory. Called once per worker.
    pub fn build_operation<'a>(&'a self) -> Box<dyn Fn() -> Box<dyn Operation> + 'a> {
        match self {
            OperationSpec::Materializer { projection, table } => {
                Box::new(|| Box::new(Materializer::new(projection.clone(), table.clone())))
            }
            OperationSpec::FilterBuilder(f) => Box::new(|| Box::new(Filter::new(f()))),
            OperationSpec::ProjectBuilder(f) => Box::new(|| Box::new(Project::new(f()))),
        }
    }
}
