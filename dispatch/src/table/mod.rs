mod input;
mod source;

use parquetd::{RowGroupMetadata, create_row_groups_metadata_from_path};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

pub use input::TableInput;
pub use source::TableSource;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    ParquetD(#[from] parquetd::Error),
    #[error("{0}")]
    IO(#[from] std::io::Error),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Represents a logical queryable Table from a list of parquet row groups.
/// To query a table, it must be transformed into a `TableSource`
pub struct Table {
    row_groups: Vec<RowGroupMetadata>,
}

impl Table {
    pub fn new(row_groups: Vec<RowGroupMetadata>) -> Self {
        Self { row_groups }
    }

    /// Create a table from a directory. This will automatically load all files metadata in the
    /// directory and load them into memory, which can be a "costly" operation.
    pub fn from_directory(path: &Path) -> Result<Self> {
        let row_groups: Vec<_> = fs::read_dir(path)?
            .flatten()
            .filter_map(|d| {
                if d.path().is_file() {
                    Some(create_row_groups_metadata_from_path(d.path()).map_err(Error::from))
                } else {
                    None
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Self::new(row_groups))
    }
}

/// A handle to a particular RowGroupMetadata within a table
pub struct RowGroupMetadataHandle {
    table: Arc<Table>,
    /// The index of the row group within the table's Vec<RowGroupMetadata>. Note that this should
    /// not be conflated with the row group number within a particular parquet file.
    row_group_index: usize,
}

impl RowGroupMetadataHandle {
    pub fn new(table: Arc<Table>, index: usize) -> Self {
        Self {
            table,
            row_group_index: index,
        }
    }

    /// Get the corresponding RowGroupMetadata from the table
    pub fn get(&self) -> &RowGroupMetadata {
        &self.table.row_groups[self.row_group_index]
    }

    pub fn index(&self) -> usize {
        self.row_group_index
    }
}
