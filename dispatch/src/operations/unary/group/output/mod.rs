//! The output (merge) phase of a GROUP BY.
//!
//! When every worker has consumed its input, the last one to reach the gather
//! barrier turns the per-worker tables into one [`PartitionJob`] per hash
//! partition. Every worker then steals and runs jobs: each merges one
//! partition's sources ([`merge`]), skipping what a pushed top-k can never use
//! ([`topk_pruning`]), and hands the merged groups to a per-worker
//! [`accumulator`] that packs them into output batches.

mod accumulator;
mod merge;
mod outputter;
mod partition_job;
pub(super) mod topk_pruning;

pub use outputter::GroupOutputter;
pub use partition_job::PartitionJob;
