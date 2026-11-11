//! Reaching a table's objects: the object store its files live in, the spec
//! crate's file IO over it, the dataflow that writes an INSERT's data
//! files, and driving the spec crate's async calls from blocking code.

pub(crate) mod runtime;
pub mod sink;
pub mod storage;
pub mod store;
