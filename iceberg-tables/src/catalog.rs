//! The seam between a table's content and whatever keeps track of its
//! current version.
//!
//! Building a new version is the same everywhere: write the data files, the
//! manifests and the manifest list, and describe them in a metadata
//! document. Making that version current is what differs. A table in a
//! directory numbers its metadata files and makes a version current by
//! creating the next number, which exactly one writer can do. A table in a
//! catalog service makes a version current by asking the service to advance
//! its pointer, under the condition that the pointer still names the base
//! the version was built on.

use std::sync::Arc;

use iceberg::spec::{Snapshot, TableMetadata};

use crate::Result;
use crate::state::CurrentMetadata;

/// What a commit changes, named for a catalog that applies changes rather
/// than storing whole metadata documents: the snapshot it adds, if it adds
/// one, and the snapshots it expires.
pub struct CommitChanges {
    pub snapshot: Option<Snapshot>,
    pub expired: Vec<i64>,
}

/// Where a table's current metadata pointer lives, and how it advances.
pub trait TableCatalog: Send + Sync {
    /// The table's current metadata and where it is.
    fn load(&self) -> Result<CurrentMetadata>;

    /// A newer metadata than `held`, or `None` when `held` is still current.
    fn refresh(&self, held: &CurrentMetadata) -> Result<Option<CurrentMetadata>>;

    /// Make `metadata` the table's current version, as the successor of
    /// `base` (`None` for a table's first version). Its manifests and
    /// manifest list are already in the store. `Ok(None)` means `base` is no
    /// longer current: another writer landed first, and the caller withdraws
    /// its objects, reloads and builds the commit again on the winner.
    fn publish(
        &self,
        base: Option<&CurrentMetadata>,
        metadata: Arc<TableMetadata>,
        changes: &CommitChanges,
    ) -> Result<Option<CurrentMetadata>>;

    /// Whether a commit through this catalog also expires the snapshots that
    /// have outlived the table's retention window. A directory has nothing
    /// else to expire them; a catalog service has its own maintenance, which
    /// a writer that only appends leaves alone.
    fn expires_snapshots(&self) -> bool;
}
