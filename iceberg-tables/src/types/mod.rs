//! A table's declared shape as Iceberg metadata and as the planner's
//! columns, the partition keys its spec cuts files by, and Arrow values as
//! Iceberg values.

pub mod columns;
pub(crate) mod datum;
pub mod partitioning;
pub mod schema;
