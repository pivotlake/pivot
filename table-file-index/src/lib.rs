//! The live data files of a table's partitions, held once, whatever format
//! keeps the table.
//!
//! A table is indexed a partition at a time. Every format describes a file
//! of a partition by the same few facts ([`FileFacts`]): where it is and how
//! large, how many rows it holds, and when it was committed. A format keeps
//! one record per file, which holds those facts beside whatever else the
//! format reads of the file, the file's bounds on the table's sort key
//! included ([`IndexedRecord`]), and a partition's records live in one
//! ordered, persistent [`FileIndex`].
//!
//! - [`file_index`]: the facts, the order files stand in, and the index.
//! - [`sort_key`]: the order a table's sort key puts the bounds of its
//!   files in, each read where its file's record holds it.
//! - [`file_metadata_cache`]: the one cache, within one budget of memory,
//!   that holds the partition indexes and the footers of every table of the
//!   process.
//! - [`weighted_lru`]: the map of groups within a budget of bytes that the
//!   cache keeps its tables' entries in, whatever they are.

pub mod file_index;
pub mod file_metadata_cache;
pub mod sort_key;
pub mod weighted_lru;

pub use file_index::{
    FileFacts, FileIndex, FilesByPath, IndexedRecord, PartitionFiles, VisitFacts, VisitRecord,
};
pub use file_metadata_cache::{
    CacheableIndex, FileMetadataCache, LookupUse, TableKey, TableLock, VersionOrder,
    measure_footer_entry, measure_index_entry,
};
pub use sort_key::{BoundSide, SortValue, compare_sort_bounds, has_sort_range};
pub use weighted_lru::WeightedLru;
