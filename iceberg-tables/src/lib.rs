//! Iceberg tables as Pivot reads and writes them, independently of what
//! keeps track of the table's current version.
//!
//! An Iceberg table is a chain of metadata files, each naming a snapshot whose
//! manifest list names manifests whose entries name data files. This crate
//! holds everything about that chain that is the same wherever the table is
//! kept, behind one [`Table`]:
//!
//! - its declared shape as Iceberg metadata and back ([`types::schema`],
//!   [`types::columns`]);
//! - its data files as manifest entries and back ([`files::data_files`]);
//! - a version's live files, as what each data manifest records of them
//!   cut into parts by partition and by column, which a scan selects its
//!   files through, a commit and a merge find theirs in, and maintenance
//!   takes a manifest at a time; the parts of every table are held in one
//!   cache with one budget of memory, which whoever owns the tables makes
//!   and hands to each
//!   ([`table_metadata_cache::TableMetadataCache`]); which manifests a
//!   partition's files are looked for in is read off the manifest list's
//!   bounds every time, and nothing of a decoded manifest is kept outside
//!   that cache ([`table`]);
//! - reading a version: the delete manifests a scan over some filters can
//!   skip (`ManifestList`), the deletion vectors of the files it
//!   reads, and their footers, parsed once and held in that same cache
//!   ([`table`]);
//! - building the next version on top of one ([`commit`]);
//! - which snapshots and previous metadata files have expired
//!   ([`maintenance::retention`]), and what an expiry leaves to delete,
//!   found from the snapshots it dropped and the oldest one it kept
//!   ([`maintenance::expired_objects`]);
//! - the terms a table's partitioning is spelled in, and the partition keys
//!   an INSERT's files are cut by, computed from the spec's transforms
//!   ([`types::partitioning`]), and the dataflow that writes the files
//!   ([`io::sink`]).
//!
//! What differs is only where a table's metadata versions are kept and how
//! a new version is made current: that is the [`MetadataStore`] seam, which
//! each datastore implements over what keeps its tables' versions, a
//! directory of numbered metadata files or a catalog service. This crate
//! names no store: a table is created, loaded, refreshed and committed to
//! through the one it is handed, and whatever only one kind of store needs
//! lives with the datastore that has it.

pub mod commit;
mod describe;
mod error;
pub mod files;
pub mod io;
pub mod maintenance;
pub mod metadata;
pub mod table;
pub mod table_metadata_cache;
/// Helpers for tests of this crate and of the crates built on it. Gated
/// behind the `test-support` feature so it never enters a production build.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod types;

pub use error::{Error, Result};
pub use files::partition::Partition;
pub use io::runtime::block_on;
pub use maintenance::compaction_input::CompactionInput;
pub use metadata::layout::{TableLocation, split_location};
pub use metadata::metadata_store::{MetadataFile, MetadataStore, build_table_ident};
pub use table::{ManifestPartitionFiles, Table};
pub use types::partitioning::{PartitionKeys, check_partition_terms};

/// The spec crate this crate's types come from, so a dependant names one
/// version of it.
pub use iceberg;
