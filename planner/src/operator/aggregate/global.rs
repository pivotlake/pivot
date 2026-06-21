//! Global aggregates (no GROUP BY).

use super::{Aggregate, aggregation_slots, sum_reads_wide_column};
use crate::compile::Error;
use dispatch::RecordBatchOperatorSpec;

impl Aggregate {
    /// A bare `COUNT(*)` keeps the dedicated row-counter; anything else
    /// (SUM/COUNT/MIN/MAX, possibly several) compiles to the multi-aggregate
    /// operator, one output column per expression. `AVG` never appears here:
    /// DuckDB lowers it to a `sum`+`count` pair with a downstream divide, so
    /// only count/sum/min/max slots reach this point.
    pub(super) fn compile_global(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if self.is_lone_count_star() {
            return Ok(input.count());
        }

        let slots = aggregation_slots(&self.expressions)?;
        // Pick the accumulator width by column type (see `sum_reads_wide_column`).
        Ok(if sum_reads_wide_column(&self.expressions) {
            input.aggregate::<i128>(slots)
        } else {
            input.aggregate::<i64>(slots)
        })
    }
}
