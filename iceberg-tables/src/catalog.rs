//! The seam between a table's content and whatever keeps track of its
//! current version.
//!
//! Building a new version is the same everywhere: the spec crate's
//! transaction writes the manifests and the manifest list of the snapshot it
//! adds and describes them as changes to the metadata document. Making that
//! version current is what differs. A table in a directory numbers its
//! metadata files and makes a version current by creating the next number,
//! which exactly one writer can do. A table in a catalog service makes a
//! version current by asking the service to advance its pointer.

use std::sync::Arc;

use crate::Result;
use crate::state::CurrentMetadata;

/// Where a table's current metadata pointer lives, and how it advances.
pub trait TableCatalog: Send + Sync {
    /// The table's current metadata and where it is.
    fn load(&self) -> Result<CurrentMetadata>;

    /// A newer metadata than `held`, or `None` when `held` is still current.
    fn refresh(&self, held: &CurrentMetadata) -> Result<Option<CurrentMetadata>>;

    /// The catalog a transaction built on `base` commits through: it answers
    /// `base` as the table's current version and lands the commit as its
    /// successor, or refuses it with a commit conflict when another writer
    /// landed one first.
    fn commit_catalog(&self, base: iceberg::table::Table) -> Result<Arc<dyn iceberg::Catalog>>;
}
