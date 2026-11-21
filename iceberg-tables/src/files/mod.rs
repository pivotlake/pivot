//! A version's files: its data files, the manifests that list them, the
//! deletes over them, the partition each sits in, and the parts of a
//! manifest every reader finds them through.

pub mod data_files;
pub(crate) mod deletes;
pub(crate) mod manifest_list;
pub(crate) mod manifest_parts;
pub(crate) mod partition;
pub(crate) mod pruning;
