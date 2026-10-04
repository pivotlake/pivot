//! Skip the objects of a table (manifests, files, row groups) whose metadata
//! proves that no row can pass the filters pushed into a scan.
//!
//! - A [`Statistic`] is what a storage format knows about one column: the
//!   [`Bounds`] of its values in every object, one slot per object. A plain
//!   column, a field inside a VARIANT column and a partition value are all
//!   the same thing here: bounds on a [`Transform`] of a column path.
//! - [`Statistics::prune`] reads the planner's filters as they were pushed.
//!   Where one compares a column with a constant, alone or under `AND`, `OR`,
//!   `BETWEEN` and `IN`, it checks the column's statistics, and returns which
//!   objects may hold a row that passes.
//!
//! The filters stay above the scan, so pruning only ever skips work, and
//! everything unknown keeps the object: an expression statistics cannot
//! answer, a column without statistics, a null bound, or bounds of a different
//! type than the constant.
//!
//! Storage adapters keep their own metadata and describe it as [`Statistics`]
//! when a query prunes, usually for just the columns its filters compare
//! ([`comparisons`], [`columns`]).

mod bounds;
mod constant_comparison;
mod scalar;
mod statistics;
mod transform;

pub(crate) use bounds::Bounds;
pub(crate) use constant_comparison::{ColumnPath, ConstantComparison};
pub(crate) use statistics::{Statistic, Statistics, columns, comparisons};
pub(crate) use transform::Transform;

#[cfg(test)]
mod tests;
