//! A version's files: its data files, the manifests that list them, the
//! deletes over them, the partition each sits in, the parts of a manifest
//! every reader finds them through, and the record maintenance takes of
//! each.

pub mod data_files;
pub(crate) mod deletes;
pub(crate) mod manifest_parts;
pub(crate) mod manifests;
pub(crate) mod partition;
pub(crate) mod partitions_from_filter;
pub(crate) mod pruning;
pub(crate) mod sort_bounded_file;
