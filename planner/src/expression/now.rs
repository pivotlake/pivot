//! [`Now`]: `now()`, the statement's wall-clock instant.

use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::UTC_TIMEZONE;
use arrow_array::{RecordBatch, TimestampMicrosecondArray};
use std::fmt::{self, Display};
use std::sync::Arc;

/// `now()` (also `CURRENT_TIMESTAMP`) — the current time as a
/// `TIMESTAMP WITH TIME ZONE` (UTC epoch microseconds), Postgres's type for it.
///
/// Carries no arguments: the instant it reports is captured when the call is
/// compiled, not when the expression is built, so the value lives in the
/// compiled closure rather than in this node.
#[derive(Debug, Clone)]
pub struct Now;

impl Now {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Capture the instant once, here at compile time, so every worker and
        // every row of the statement observes the same `now()`. Negative
        // (pre-epoch) clocks are clamped to 0, which can't happen on a sane
        // host.
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        Ok(stateless_expr(move |batch: &RecordBatch| {
            let values = vec![now_micros; batch.num_rows()];
            ExprResult::Array(Arc::new(
                TimestampMicrosecondArray::from(values).with_timezone(UTC_TIMEZONE),
            ))
        }))
    }
}

impl Display for Now {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("now()")
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampMicrosecondType;
    use rstest::rstest;

    #[rstest]
    fn now_returns_current_time_as_a_timestamptz(mut testing_planner: TestingPlanner) {
        let before = micros_since_epoch();

        let batches = run_batches(&mut testing_planner, "SELECT now()");

        let after = micros_since_epoch();
        let col = batches[0].column(0);
        // `now()` surfaces as a TIMESTAMP WITH TIME ZONE carrying the captured
        // UTC instant at the microsecond resolution a timestamp counts in.
        assert_eq!(col.data_type(), &crate::types::timestamp_tz_arrow_type());
        let now = col.as_primitive::<TimestampMicrosecondType>().value(0);
        assert!(
            (before..=after).contains(&now),
            "{now} not in [{before}, {after}]"
        );
    }

    #[rstest]
    fn every_row_of_a_statement_sees_the_same_instant(mut testing_planner: TestingPlanner) {
        // The instant is captured once at compile time, so a multi-row scan
        // cannot see the clock advance part way through.
        let batches = run_batches(&mut testing_planner, "SELECT now() FROM example_table");

        let instants = batches[0]
            .column(0)
            .as_primitive::<TimestampMicrosecondType>();
        assert!(instants.len() > 1, "needs several rows to be meaningful");
        assert!(instants.values().windows(2).all(|pair| pair[0] == pair[1]));
    }

    fn micros_since_epoch() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }
}
