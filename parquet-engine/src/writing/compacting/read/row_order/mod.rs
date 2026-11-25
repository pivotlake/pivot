//! The order in which the rows of the input files go to the output: the
//! merge that decides it from the files' sort columns, and the order of one
//! output row group as decided.

pub(crate) mod key_merge;
pub(crate) mod output_row_group;
