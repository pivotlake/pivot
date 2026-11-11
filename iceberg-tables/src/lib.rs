//! Iceberg tables as Pivot reads and writes them, independently of what
//! keeps track of the table's current version.
//!
//! An Iceberg table is a chain of metadata files, each naming a snapshot whose
//! manifest list names manifests whose entries name data files. This crate
//! holds everything about that chain that is the same wherever the table is
//! kept, behind one [`Table`]:
//!
//! - its declared shape as Iceberg metadata and back ([`schema`], [`columns`]);
//! - its data files as manifest entries and back ([`data_files`]);
//! - a version's live files held once, as their manifest entries in an
//!   order, which every scan selects its files from and a datastore plans,
//!   compacts and commits against ([`file_index`]);
//! - reading a version: the delete manifests a scan over some filters can
//!   skip ([`manifests`]), the deletion vectors of the files it reads, and
//!   their footers ([`table`]);
//! - building the next version on top of one ([`commit`]);
//! - which snapshots and previous metadata files have expired
//!   ([`retention`]), what the retained ones still use ([`objects_in_use`])
//!   and what an expiry leaves to delete ([`expired_objects`]);
//! - the partition keys an INSERT's files are cut by, computed from the
//!   spec's transforms ([`partitioning`]), and the dataflow that writes the
//!   files ([`sink`]).
//!
//! What differs is only where the pointer to the current metadata lives and
//! how a new version is made current: that is the [`MetadataPointer`] seam.
//! A table in a directory keeps numbered metadata files and commits a version
//! with one conditional write ([`DirectoryMetadataPointer`]); a table in a
//! catalog service commits it with the service's commit call, which a
//! datastore built on that service implements.

pub mod columns;
pub mod commit;
pub mod data_files;
mod datum;
mod deletes;
pub mod directory;
mod error;
pub mod expired_objects;
pub mod file_index;
pub mod layout;
pub mod manifests;
pub mod metadata_pointer;
pub mod objects_in_use;
pub mod partitioning;
mod pruning;
pub mod retention;
mod runtime;
pub mod schema;
pub mod sink;
pub mod state;
pub mod storage;
pub mod store;
pub mod table;
#[cfg(test)]
mod test_support;

pub use columns::{LOGICAL_TYPES_PROPERTY, identity_columns, read_table_columns};
pub use directory::DirectoryMetadataPointer;
pub use error::{Error, Result};
pub use file_index::{FileIndex, IndexedFile};
pub use layout::{TableLocation, split_location};
pub use manifests::{FileDescriptor, ManifestList};
pub use metadata_pointer::MetadataPointer;
pub use partitioning::PartitionKeys;
pub use runtime::block_on;
pub use state::CurrentMetadata;
pub use store::TableStore;
pub use table::Table;

/// The spec crate this crate's types come from, so a dependant names one
/// version of it.
pub use iceberg;
