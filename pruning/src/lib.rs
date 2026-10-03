//! Skip the objects of a table (manifests, files, row groups) whose metadata
//! proves that no row can pass a filter.
//!
//! The model has three parts:
//!
//! - A [`Predicate`] is one `column <comparison> constant` the planner pulled
//!   out of a SQL filter. The filter itself stays above the scan, so pruning
//!   only ever skips work.
//! - A [`Statistic`] is what a storage format knows about one column: the
//!   [`Bounds`] of its values in every object, one slot per object. A plain
//!   column, a field inside a VARIANT column and a partition value are all
//!   the same thing here: bounds on a [`Transform`] of a [`ColumnPath`].
//! - [`Statistics::prune`] evaluates the predicates against the statistics of
//!   the columns they read, and returns which objects may hold a matching row.
//!
//! Everything unknown keeps the object: a column without statistics, a null
//! bound, or bounds of a different type than the constant.
//!
//! Storage adapters keep their own metadata and describe it as [`Statistics`]
//! when a query prunes, usually for just the columns its predicates read
//! ([`Predicate::columns`]).

mod bounds;
mod predicate;
mod scalar;
mod statistics;
mod transform;

pub use bounds::Bounds;
pub use predicate::{ColumnPath, Comparison, Predicate};
pub use statistics::{Statistic, Statistics};
pub use transform::Transform;

#[cfg(test)]
mod tests;
