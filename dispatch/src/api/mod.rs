//! Pipeline specification and builder API.
//!
//! This module provides a fluent API for constructing and executing parallel
//! data processing pipelines. Pipelines are built declaratively and executed
//! across all worker threads.
//!
//! # Examples
//!
//! ## Count with filter
//! ```ignore
//! // SELECT URL FROM hits WHERE URL LIKE '%google%
//! let feed = MemoryFeed::new();
//!
//! PipelineSpec::new()
//!     .table_input(table, Some(Projection::columns([13])))
//!     .filter_builder(|| {
//!         let mut contains = Contains::new("google");
//!         Box::new(move |batch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             contains.run(col)
//!         })
//!     })
//!     .count(feed.output())
//!     .execute();
//!
//! let results = feed.collect();
//! ```
//!
//! ## Order by with limit
//! ```ignore
//! let feed = MemoryFeed::new();
//!
//! PipelineSpec::new()
//!     .table_input(table, Some(Projection::columns([4, 13])))
//!     .filter_builder(|| { /* filter logic */ })
//!     .order_by_limit([OrderBy::new(0, false, false)], 10, feed.output())
//!     .execute();
//! ```
//!
//! ## Group by with count
//! ```ignore
//! let feed = MemoryFeed::new();
//!
//! PipelineSpec::new()
//!     .table_input(table, Some(Projection::columns([13])))
//!     .group_by_count(0, feed.output())
//!     .execute();
//! ```
//!
//! ## Multi-stage pipelines
//! ```ignore
//! // Stage 1: Group by
//! let stage1 = MemoryFeed::new();
//! PipelineSpec::new()
//!     .table_input(table, Some(Projection::columns([13])))
//!     .group_by_count(0, stage1.output())
//!     .execute();
//!
//! // Stage 2: Order results
//! let output = MemoryFeed::new();
//! PipelineSpec::new()
//!     .memory_input(stage1.source())
//!     .order_by_limit([OrderBy::new(1, true, false)], 10, output.output())
//!     .execute();
//!
//! let results = output.collect();
//! ```

mod input_spec;
mod memory_feed;
mod node;
mod operation_spec;
mod pipeline_breaker_spec;
mod pipeline_spec;

use crate::operations::Output;
pub use input_spec::*;
pub use memory_feed::*;
pub use node::*;
pub use pipeline_spec::*;

/// Factory trait for creating output sinks.
///
/// Implemented by [`MemoryOutputSpec`] to create outputs for each worker.
pub trait OutputSpec: 'static {
    /// Create a new output instance. Called once per worker.
    fn build_output(&self) -> Box<dyn Output>;
}
