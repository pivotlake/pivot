mod factory;

pub use factory::TableInputFactory;
use std::collections::HashSet;
pub mod input;
mod source;

use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fmt::{Debug, Formatter};
use std::fs;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

use parquet::file::metadata::ParquetMetaData;
pub use source::TableSource;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("{0}")]
    IO(#[from] std::io::Error),
    #[error("Column not found {0}")]
    ColumnNotFound(String),
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Represents a logical queryable Table from a list of parquet row groups.
/// To query a table, it must be transformed into a `TableSource`
pub struct Table {
    row_groups: Vec<RowGroupMetadata>,
}

impl Debug for Table {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("table")
    }
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
#[derive(Debug, Clone)]
pub struct RowGroupMetadataHandle {
    table: Arc<Table>,
    /// The global index of the row group within the table's Vec<RowGroupMetadata>. Note that this
    /// should not be conflated with the row group number within a particular parquet file.
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

#[derive(Clone)]
pub struct RowGroupMetadata {
    pub file: Arc<File>,
    pub arrow_metadata: ArrowReaderMetadata,
    pub row_group: usize,
}

fn open_direct_read(path: &Path) -> std::io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        return OpenOptions::new()
            .read(true)
            // .custom_flags(libc::O_DIRECT)
            .open(path);
    }

    #[cfg(target_os = "macos")]
    {
        let file = OpenOptions::new().read(true).open(path)?;
        // Best-effort "no cache" on macOS (not the same as O_DIRECT)
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if rc == -1 {
            return Err(std::io::Error::last_os_error());
        }
        return Ok(file);
    }

    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    {
        // Fallback: normal cached reads (or add per-OS handling here)
        return OpenOptions::new().read(true).open(path);
    }

    #[cfg(not(unix))]
    {
        return OpenOptions::new().read(true).open(path);
    }
}

pub fn create_row_groups_metadata_from_path(
    path: impl AsRef<Path>,
) -> Result<Vec<RowGroupMetadata>> {
    let path = path.as_ref();
    let file = File::open(path)?;

    let reader = SerializedFileReader::new(file)?;
    let metadata = Arc::new(reader.metadata().clone());

    let file = open_direct_read(path)?;
    let file = Arc::new(file);

    // Create arrow metadata from parquet metadata
    let arrow_metadata = ArrowReaderMetadata::try_new(metadata.clone(), Default::default())?;
    Ok((0..metadata.row_groups().len())
        .map(|i| RowGroupMetadata {
            file: file.clone(),
            arrow_metadata: arrow_metadata.clone(),
            row_group: i,
        })
        .collect())
}

/// A projection specifying which columns to read
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    /// Column indices to read
    pub column_indices: HashSet<usize>,
}

impl Projection {
    /// Create a projection for specific column indices
    pub fn columns(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            column_indices: indices.into_iter().collect(),
        }
    }

    /// Create a projection for specific column names
    pub fn columns_by_name(metadata: &ParquetMetaData, names: &[&str]) -> Result<Self> {
        let schema = metadata.file_metadata().schema_descr();
        let mut indices = HashSet::new();

        for name in names {
            let mut found = false;
            for i in 0..schema.num_columns() {
                if schema.column(i).name() == *name {
                    indices.insert(i);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Error::ColumnNotFound(name.to_string()));
            }
        }

        Ok(Self {
            column_indices: indices,
        })
    }

    /// Check if this projection includes a specific column
    pub fn includes(&self, column_idx: usize) -> bool {
        self.column_indices.contains(&column_idx)
    }
}
