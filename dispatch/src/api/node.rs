use crate::OrderBy;
use crate::api::OutputSpec;
use crate::api::operation_spec::OperationSpec;
use crate::api::pipeline_breaker_spec::PipelineBreakerSpec;
use crate::api::pipeline_spec::PipelineSpec;
use crate::table::Table;
use arrow_array::{BooleanArray, RecordBatch};
use parquetd::Projection;
use std::sync::Arc;

/// A node in the pipeline graph.
///
/// Represents a point where operations can be chained. Created by input methods
/// on [`PipelineSpec`] or by intermediate operations like `filter_builder()`.
///
/// Chain operations to transform data, then terminate with pipeline breakers liek `count()`,
/// `order_by_limit()`, or `group_by_count()`.
pub struct Node {
    id: usize,
    pipeline: PipelineSpec,
}

impl Node {
    pub fn new(id: usize, pipeline_spec: PipelineSpec) -> Self {
        Self {
            id,
            pipeline: pipeline_spec,
        }
    }

    /// Filter rows using a predicate builder.
    ///
    /// The builder is called once per worker to create the predicate function.
    /// This allows each worker to have its own state (e.g., `Contains`).
    ///
    /// # Examples
    /// ```ignore
    /// node.filter_builder(|| {
    ///     let mut contains = Contains::new("google");
    ///     Box::new(move |batch| {
    ///         let col = batch.column(0).as_any()
    ///             .downcast_ref::<StringViewArray>().unwrap();
    ///         contains.run(col)
    ///     })
    /// })
    /// ```
    pub fn filter_builder(
        &self,
        f: impl Fn() -> Box<dyn FnMut(&RecordBatch) -> BooleanArray + Send + Sync + 'static> + 'static,
    ) -> Node {
        self.pipeline
            .push_operation(self.id, OperationSpec::FilterBuilder(Box::new(f)))
    }

    /// Project columns using a projection builder.
    ///
    /// The builder is called once per worker to create the projection function.
    ///
    /// # Examples
    /// ```ignore
    /// node.project(|| {
    ///     Box::new(|batch| {
    ///         batch.project(&[0, 2]).unwrap()  // keep columns 0 and 2
    ///     })
    /// })
    /// ```
    pub fn project(
        &self,
        f: impl Fn() -> Box<dyn FnMut(&RecordBatch) -> RecordBatch + Send + 'static> + 'static,
    ) -> Node {
        self.pipeline
            .push_operation(self.id, OperationSpec::ProjectBuilder(Box::new(f)))
    }

    /// Count all rows.
    ///
    /// Outputs a single `RecordBatch` with one `UInt64` column containing the count.
    pub fn count(&self, output: impl OutputSpec) -> PipelineSpec {
        self.pipeline
            .push_breaker(self.id, PipelineBreakerSpec::Count(Box::new(output)));
        self.pipeline.clone()
    }

    /// Materialize deferred columns from the table.
    ///
    /// Can be used to minimize IO and row processing if there's a step in the pipeline that limits
    /// the amount of incoming rows
    pub fn materialize(&self, table: Arc<Table>, projection: Option<Projection>) -> Node {
        self.pipeline
            .push_operation(self.id, OperationSpec::Materializer { projection, table })
    }

    /// Terminal: Sort and limit results.
    ///
    /// Performs distributed top-K: each worker computes local top-K,
    /// then results are merged globally.
    ///
    /// ```ignore
    /// node.order_by_limit(
    ///     [OrderBy::new(0, true, false)],  // col 0 descending
    ///     10,                               // limit
    ///     feed.output()
    /// )
    /// ```
    pub fn order_by_limit(
        &self,
        order_bys: impl IntoIterator<Item = OrderBy>,
        limit: usize,
        output: impl OutputSpec,
    ) -> PipelineSpec {
        self.pipeline.push_breaker(
            self.id,
            PipelineBreakerSpec::OrderByLimit {
                order_by: order_bys.into_iter().collect(),
                output: Box::new(output),
                limit,
            },
        );
        self.pipeline.clone()
    }

    /// Group by a column and count occurrences.
    ///
    /// Outputs batches with schema `(key: String, value: UInt64)`.
    pub fn group_by_count(&self, group_column: usize, output: impl OutputSpec) -> PipelineSpec {
        self.pipeline.push_breaker(
            self.id,
            PipelineBreakerSpec::GroupByCount {
                group_by_column: group_column,
                output: Box::new(output),
            },
        );
        self.pipeline.clone()
    }

    /// Get a reference to the parent pipeline spec.
    pub fn pipeline(&self) -> &PipelineSpec {
        &self.pipeline
    }
}
