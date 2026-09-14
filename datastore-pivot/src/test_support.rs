//! Delta-specific test helpers layered on the reusable object-store harness.

use std::sync::{Arc, LazyLock};

pub use object_storage::test_support::*;

use crate::DeltaFileEntry;
use crate::{CatalogTable, PartitionValues, PivotDatastore};
use dispatch::DataFlowDispatcher;
use object_storage::{FileRef, ObjectPath};

/// The runtime a sync test opens its datastores under. A datastore's Delta
/// engine runs its log I/O on the ambient multi-thread tokio runtime and fails
/// to build without one, and a plain `#[test]` has none. A `static` so it is
/// never dropped: dropping a runtime from inside an async context panics.
static TEST_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build the test runtime")
});

/// Enter the shared test runtime for the guard's lifetime, for a sync test that
/// builds something needing the ambient runtime itself.
pub fn enter_test_runtime() -> tokio::runtime::EnterGuard<'static> {
    TEST_RUNTIME.enter()
}

/// [`PivotDatastore::open`] from a sync test: the open runs under the shared
/// test runtime, which the datastore's Delta engine then keeps for its I/O.
pub fn open_datastore(
    uri: &str,
    dispatcher: &DataFlowDispatcher,
) -> crate::Result<Arc<PivotDatastore>> {
    let _runtime = enter_test_runtime();
    PivotDatastore::open(uri, dispatcher)
}

impl CatalogTable {
    /// Write `bytes` as a new data file and commit it into the table.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<PartitionValues>,
    ) -> crate::Result<()> {
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
    ) -> crate::Result<()> {
        self.commit_entries(removed, added, false)
    }
}
