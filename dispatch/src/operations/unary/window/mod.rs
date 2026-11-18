//! SQL window functions: `function() OVER (PARTITION BY ... ORDER BY ...)`.
//!
//! The operator appends one column per window function to its input. A
//! function's value for a row depends on the other rows of the same
//! partition, so the operator must see all its input before it emits
//! anything.
//!
//! Rows with equal partition keys have to end up next to each other, ordered
//! by the order keys. Sorting the whole input as one sequence would do that,
//! but planning one merge over every row is the part of a sort that does not
//! parallelize. Instead the rows are first split into hash buckets on the
//! partition keys. Equal keys always share a bucket, so every bucket can be
//! sorted, evaluated and emitted on its own.
//!
//! ```text
//!                │  RecordBatch (channel from the operator below)
//!                ▼
//!    ┌───────────────────────┐   CONSUME, on every worker:
//!    │ WindowConsumer        │   1. hash partition keys, copy each batch so a
//!    │                       │      bucket's rows are next to each other
//!    │                       │   2. once enough rows are pending, sort each
//!    │                       │      bucket's rows into a run
//!    │                       │   3. with a row limit, drop rows that cannot pass
//!    └───────────────────────┘
//!                │  per worker: Vec<Vec<SortedRun>>, runs per bucket
//!                │  (handed over through a barrier, not a channel)
//!                ▼
//!    ┌───────────────────────┐   OUTPUT, on every worker:
//!    │ WindowOutputter       │   1. merge each bucket's runs from all workers
//!    │                       │   2. evaluate the functions over the sorted bucket
//!    │                       │   3. apply the row limit
//!    └───────────────────────┘
//!                │  RecordBatch (channel to the operator above)
//!                ▼
//! ```
//!
//! [`WindowConsumer`](consume::WindowConsumer) and
//! [`WindowOutputter`](output::WindowOutputter) each show their own
//! steps.
//!
//! # The sort key
//!
//! A bucket is sorted by (partition keys, order keys), read straight from the
//! input columns by the same comparison code `ORDER BY` uses. The partition
//! keys come first so that each partition is one contiguous stretch of the
//! sorted bucket. Their direction does not matter, only that equal keys end
//! up adjacent.
//!
//! # The row limit
//!
//! A query such as `QUALIFY row_number() OVER (...) <= 1000` only wants the
//! first rows of every partition. The planner hands that to the operator as a
//! [`WindowRowLimit`], and the consume phase uses it to drop rows while consuming,
//! so a worker holds about (distinct keys × limit) rows instead of its whole
//! input. For a limit on `row_number()`, the consume phase goes further
//! and rejects most such rows as they arrive, before they are sorted.
//!
//! # Without PARTITION BY
//!
//! There is one bucket and no hashing. The single bucket's merge is still
//! split into slices that run in parallel.

mod consume;
mod evaluate;
mod factory;
mod output;

pub use evaluate::WindowFunction;
pub use factory::WindowFactory;

use std::sync::Arc;

use arrow_schema::{ArrowError, Field, Schema, SchemaRef};

use crate::RECORD_BATCH_SIZE;
use crate::operations::unary::order_by_limit::OrderBy;

/// What one window operator computes: the functions, and the partitioning and
/// ordering they share.
#[derive(Clone)]
pub struct WindowSpec {
    /// Input columns of the `PARTITION BY` keys. Empty when the whole input is
    /// one partition.
    pub partition_by: Vec<usize>,
    /// The `ORDER BY` keys inside `OVER`. Empty when the order within a
    /// partition is unspecified.
    pub order_by: Vec<OrderBy>,
    /// The functions to compute. Each adds one output column after the input
    /// columns, in this order.
    pub functions: Vec<WindowFunction>,
    /// Rows to drop inside the operator instead of in a filter above it.
    pub row_limit: Option<WindowRowLimit>,
}

/// Keeps only the rows for which one of the window's functions is at most
/// `limit`.
///
/// Every supported function starts at 1 in each partition and never
/// decreases, so the kept rows are the first rows of each partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRowLimit {
    /// Which of [`WindowSpec::functions`] the limit applies to.
    pub function_index: usize,
    pub limit: u64,
}

/// Rows a worker buffers per bucket, on average, before it sorts them into
/// runs. Sorting a full batch at a time keeps runs long, and long runs keep
/// the number of runs a bucket's merge has to combine small.
const PENDING_ROWS_PER_BUCKET: usize = RECORD_BATCH_SIZE;

/// The fewest batches of rows a worker buffers before sorting them, however
/// few buckets there are. An unpartitioned window has one bucket, and this
/// keeps its runs several batches long.
const MIN_PENDING_BATCHES: usize = 16;

/// How the operator lays out its work, derived once from the [`WindowSpec`]
/// and the worker count and shared by every worker.
pub(crate) struct WindowLayout {
    pub(super) spec: WindowSpec,
    /// Number of hash buckets rows are split into. A power of two, and 1 when
    /// there are no partition keys.
    pub(super) bucket_count: usize,
    /// Rows a worker buffers before sorting them into runs.
    pub(super) pending_row_capacity: usize,
    /// The order a bucket's rows are sorted in: the partition keys, then the
    /// order keys. Empty for a window with neither, whose rows stay in
    /// arrival order.
    pub(super) sort_key: Arc<[OrderBy]>,
    /// How many ring buffers an arena of partition keys may grow to.
    pub(super) arena_buffers: usize,
}

impl WindowLayout {
    pub(super) fn new(spec: WindowSpec, total_workers: usize, arena_buffers: usize) -> Self {
        let bucket_count = if spec.partition_by.is_empty() {
            1
        } else {
            (total_workers * BUCKETS_PER_WORKER)
                .next_power_of_two()
                .clamp(MIN_BUCKET_COUNT, MAX_BUCKET_COUNT)
        };
        let pending_row_capacity = bucket_count.max(MIN_PENDING_BATCHES) * PENDING_ROWS_PER_BUCKET;
        let sort_key: Vec<OrderBy> = spec
            .partition_by
            .iter()
            .map(|&column| OrderBy::new(column, false, false))
            .chain(spec.order_by.iter().cloned())
            .collect();
        Self {
            spec,
            bucket_count,
            pending_row_capacity,
            sort_key: sort_key.into(),
            arena_buffers,
        }
    }

    /// The schema of the operator's output: the fields of `input_schema`
    /// followed by one field per function.
    pub(super) fn build_output_schema(
        &self,
        input_schema: &Schema,
    ) -> Result<SchemaRef, ArrowError> {
        let mut fields: Vec<Arc<Field>> = input_schema.fields().iter().cloned().collect();
        for function in &self.spec.functions {
            fields.push(Arc::new(function.output_field(input_schema)?));
        }
        Ok(Arc::new(Schema::new(fields)))
    }
}

/// Buckets per worker a partitioned window aims for. Several buckets per
/// worker keep every worker busy during output even when some buckets hold
/// far more rows than others.
const BUCKETS_PER_WORKER: usize = 4;

/// The most buckets a window uses. Every worker keeps a list of runs per
/// bucket, and each bucket's sort handles its rows separately, so the
/// per-bucket overhead has to stay small against a batch of rows. The limit
/// is high enough to keep [`BUCKETS_PER_WORKER`] on a machine with a couple
/// of hundred workers.
const MAX_BUCKET_COUNT: usize = 1024;

/// The fewest buckets a partitioned window uses. The cost of planning a merge
/// grows with the square of the rows in it, so even a small worker pool
/// splits its rows this many ways.
const MIN_BUCKET_COUNT: usize = 16;
