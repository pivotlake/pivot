//! Iceberg tables as Pivot reads and writes them, independently of what
//! keeps track of the table's current version.
//!
//! An Iceberg table is a chain of metadata files, each naming a snapshot whose
//! manifest list names manifests whose entries name data files. This crate
//! holds everything about that chain that is the same wherever the table is
//! kept, behind one [`Table`]:
//!
//! - its declared shape as Iceberg metadata and back ([`schema`], [`columns`]);
//! - its data files as manifest entries and back ([`data_files`]), and the
//!   manifests and manifest lists that record them ([`manifest_writer`]);
//! - reading a version: the manifests and files a scan over some filters can
//!   skip ([`manifests`]), the deletion vectors of the rest, and their footers
//!   ([`table`]);
//! - building the next version on top of one ([`commit`]);
//! - which snapshots have expired ([`retention`]) and what the retained ones
//!   still reach ([`reachability`]);
//! - the partition keys an INSERT's files are cut by, computed from the
//!   spec's transforms ([`partitioning`]), and the dataflow that writes the
//!   files ([`sink`]).
//!
//! What differs is only where the pointer to the current metadata lives and
//! how a new version is made current: that is the [`TableCatalog`] seam. A
//! table in a directory keeps numbered metadata files and lands a version
//! with one conditional write ([`DirectoryCatalog`]); a table in a catalog
//! service lands it with the service's commit call, which a datastore built
//! on that service implements.

pub mod catalog;
pub mod columns;
pub mod commit;
pub mod data_files;
mod deletes;
pub mod directory;
mod error;
pub mod layout;
pub mod load;
pub mod manifest_writer;
pub mod manifests;
pub mod partitioning;
mod pruning;
pub mod reachability;
pub mod retention;
mod runtime;
pub mod schema;
pub mod sink;
pub mod state;
pub mod store;
pub mod table;

pub use catalog::{CommitChanges, TableCatalog};
pub use columns::{LOGICAL_TYPES_PROPERTY, read_logical_types, table_columns};
pub use directory::DirectoryCatalog;
pub use error::{Error, Result};
pub use layout::{TableLocation, split_location};
pub use load::identity_columns;
pub use manifests::{DecodedManifest, FileDescriptor, ManifestList};
pub use partitioning::PartitionKeys;
pub use runtime::block_on;
pub use state::CurrentMetadata;
pub use store::TableStore;
pub use table::Table;

/// The spec crate this crate's types come from, so a dependant names one
/// version of it.
pub use iceberg;
