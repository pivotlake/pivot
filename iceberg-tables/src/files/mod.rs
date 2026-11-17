//! A version's files: its data files, the manifests that list them, the
//! deletes over them, the index a scan selects them from, and what a DELETE
//! covers of their partitions.

pub mod data_files;
pub(crate) mod delete_coverage;
pub(crate) mod deletes;
pub mod file_index;
pub mod manifests;
pub(crate) mod partitions_from_filter;
pub(crate) mod pruning;
