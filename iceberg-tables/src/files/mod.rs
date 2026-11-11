//! A version's files: its data files, the manifests that list them, the
//! deletes over them, and the index a scan selects them from.

pub mod data_files;
pub(crate) mod deletes;
pub mod file_index;
pub(crate) mod file_metadata_cache;
pub mod manifests;
mod pruning;
