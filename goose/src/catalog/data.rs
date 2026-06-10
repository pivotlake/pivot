//! Write access to one table's data location, wherever it is.

use std::sync::Arc;

use crate::store::{self, ObjectStore, join_prefix};

/// One table's data location as a writable store: an absolute local directory
/// is wrapped in a [`LocalStore`](crate::store::LocalStore) rooted at it, a
/// store-relative location addresses the database's own store under that
/// prefix. Callers (the compacter) write and delete data files through this
/// one interface — local and remote are the same code path.
pub struct TableStore {
    pub(super) store: Arc<dyn ObjectStore>,
    pub(super) prefix: String,
}

impl TableStore {
    /// Write a data file (whole). On a local directory this lands via a temp
    /// file + rename; on a remote store a PUT is atomic per object. Either
    /// way a reader never sees a partial file — and an unfinished file is
    /// invisible regardless, because only log-committed names are read.
    pub fn put(&self, name: &str, bytes: &[u8]) -> store::Result<()> {
        self.store.put(&self.key(name), bytes)
    }

    /// Delete a data file (idempotent).
    pub fn delete(&self, name: &str) -> store::Result<()> {
        self.store.delete(&self.key(name))
    }

    fn key(&self, name: &str) -> String {
        join_prefix(&self.prefix, name)
    }
}
