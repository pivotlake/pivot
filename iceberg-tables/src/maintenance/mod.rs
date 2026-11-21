//! What a table's history has expired and what that leaves to delete, and
//! the record maintenance takes of each live file.

pub(crate) mod compaction_input;
pub mod expired_objects;
pub mod retention;
