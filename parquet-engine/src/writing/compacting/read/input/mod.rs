//! The input files of a compaction: each file with the reads of its columns,
//! what the files share to request and release their rows, and the decoded
//! values as they come back.

pub(crate) mod decoded_values;
pub(crate) mod input_file;
pub(crate) mod read_requests;
