//! Iceberg tables as Pivot reads and writes them, independently of what
//! keeps track of the table's current version.
//!
//! An Iceberg table is a chain of metadata files, each naming a snapshot whose
//! manifest list names manifests whose entries name data files. This crate
//! holds everything about that chain that is the same wherever the table is
//! kept:
//!
//! - its declared shape as Iceberg metadata and back ([`schema`], [`columns`]);
//! - its data files as manifest entries and back ([`data_files`]), and the
//!   manifests and manifest lists that record them ([`manifest_writer`]);
//! - loading a version's live files from its manifests ([`load`]) and
//!   building the next version on top of it ([`commit`]);
//! - which snapshots have expired ([`retention`]) and what the retained ones
//!   still reach ([`reachability`]);
//! - the Parquet upload sink that writes a table's data files ([`sink`]).
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
pub mod directory;
mod error;
pub mod layout;
pub mod load;
pub mod manifest_writer;
pub mod reachability;
pub mod retention;
mod runtime;
pub mod schema;
pub mod sink;
pub mod state;
pub mod store;

pub use catalog::{CommitChanges, TableCatalog};
pub use columns::{LOGICAL_TYPES_PROPERTY, read_logical_types, table_columns};
pub use directory::DirectoryCatalog;
pub use error::{Error, Result};
pub use layout::{TableLocation, split_location};
pub use load::identity_columns;
pub use runtime::block_on;
pub use state::{CurrentMetadata, TableState};
pub use store::TableStore;

/// The spec crate this crate's types come from, so a dependant names one
/// version of it.
pub use iceberg;
