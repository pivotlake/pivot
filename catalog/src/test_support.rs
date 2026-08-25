//! Delta-specific test helpers layered on the reusable object-store harness.

pub use object_storage::test_support::*;

use crate::delta::DeltaFileEntry;
use crate::delta::{CatalogTable, PartitionValues};
use object_storage::{FileRef, ObjectPath};

impl CatalogTable {
    /// Write `bytes` as a new data file and commit it into the table.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<PartitionValues>,
    ) -> crate::delta::Result<()> {
        if self.file_refs().iter().any(|file| file.path == path) {
            return Ok(());
        }
        self.store()
            .put(&self.object_location().resolve(&path), bytes)?;
        let entry = DeltaFileEntry {
            file: FileRef {
                path,
                size: bytes.len() as u64,
            },
            partition,
            stats: None,
        };
        self.commit_entries(&[], &[entry], true)?;
        Ok(())
    }

    /// Atomically replace `removed` files with `added` files in one version.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[DeltaFileEntry],
    ) -> crate::delta::Result<()> {
        self.commit_entries(removed, added, false)
    }
}
