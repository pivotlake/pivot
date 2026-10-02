//! Conservative predicate evaluation over Arrow metadata, one row per manifest,
//! file, or row group. A false result proves that no data row can match; missing
//! or incompatible statistics retain the object.
//!
//! The planner translates SQL filters into [`ColumnPredicate`]s. Storage
//! adapters supply [`ColumnStatistics`] keyed by table column and separate
//! [`PartitionStatistics`].
//! [`StatisticsBatch::prune`] matches logical column indexes, VARIANT paths and
//! cast types within each column, and projects predicates through the shared
//! [`PartitionTransform`]s.
//! Adapters own the objects in statistics-row order and use the returned mask
//! to select them. The statistics batch contains only bounds and their proofs.
//! Format adapters establish bound validity while loading metadata; this crate
//! has no planner or storage-format dependencies.
//!
//! File adapters omit VARIANT columns and paths. Row-group adapters may expose
//! safe typed path bounds; the evaluator compares them like ordinary columns.

mod bounds;
mod partition;
mod predicate;
mod scalar;
mod statistics;

pub use bounds::ColumnBounds;
pub use partition::PartitionTransform;
pub use predicate::{
    ColumnPredicate, ColumnReference, Comparison, PartitionExpression, PruningPredicate,
};
pub use statistics::{
    ColumnStatistics, PartitionStatistics, StatisticsBatch, VariantPathStatistics,
};

#[cfg(test)]
mod tests;
