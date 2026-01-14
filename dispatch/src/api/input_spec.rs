use crate::input::Input;
use crate::memory_source::MemorySource;
use crate::table::{TableInput, TableSource};
use parquetd::Projection;
use std::sync::Arc;

/// Specification for pipeline inputs.
///
/// Created internally by `PipelineSpec::table_input()` and `memory_input()`.
pub enum InputSpec {
    /// Parquet table input with optional column projection.
    Table {
        table_source: Arc<TableSource>,
        projection: Option<Projection>,
    },
    /// In-memory input from a previous pipeline stage.
    Memory(Arc<MemorySource>),
}

impl InputSpec {
    /// Create an input factory. Called once per worker.
    pub fn build_input<'a>(&'a self) -> Box<dyn Fn() -> Box<dyn Input> + 'a> {
        match self {
            InputSpec::Table {
                table_source,
                projection,
            } => Box::new(|| Box::new(TableInput::new(table_source.clone(), projection.clone()))),
            InputSpec::Memory(m) => Box::new(|| Box::new(m.clone())),
        }
    }
}
