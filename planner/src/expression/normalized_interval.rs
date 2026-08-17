//! [`NormalizedInterval`]: `normalized_interval`, the canonical form an
//! interval must take before it can be compared.
//!
//! An interval's three fields are independent, so one span has several
//! spellings: `25:00:00` and `1 day 01:00:00` describe the same amount of time.
//! Comparing the fields as written would order those two differently, so DuckDB
//! carries first, pushing the sub-day field up into whole days and days up into
//! 30-day months. Its binder wraps every interval it has to order, compare or
//! hash in a `normalized_interval` call
//! (`CollationBinding::PushIntervalCollation`), which is what this implements.
//!
//! Once carried, the fields compare lexicographically from the most significant
//! down, which is exactly the order arrow derives on `IntervalMonthDayNano`, so
//! the comparison and sort kernels are correct on a normalized value and only
//! on a normalized value.
//!
//! The carry only ever moves value upward, so it is not a full canonical form:
//! `1 mon -29 days` keeps its mixed signs rather than collapsing to `1 day`, and
//! still compares as the larger of the two. That is DuckDB's own behaviour, not
//! a shortcut taken here, and it is why an interval is not totally ordered by
//! the duration it stands for.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{DAYS_PER_MONTH, NANOS_PER_DAY};
use arrow_array::cast::AsArray;
use arrow_array::types::IntervalMonthDayNanoType;
use arrow_array::{IntervalMonthDayNanoArray, RecordBatch};
use arrow_buffer::IntervalMonthDayNano;
use std::fmt::{self, Display};
use std::sync::Arc;

/// `normalized_interval(span)` — an interval carried into its canonical form.
#[derive(Debug, Clone)]
pub struct NormalizedInterval {
    pub input: Box<Expression>,
}

impl NormalizedInterval {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input = self.input.compile()?;
        Ok(Box::new(move || {
            let mut eval = input();
            Box::new(move |batch: &RecordBatch| {
                let value = eval(batch);
                let (array, _) = value.as_datum().get();
                let spans = array.as_primitive::<IntervalMonthDayNanoType>();
                let normalized: IntervalMonthDayNanoArray = spans.unary(normalize);
                ExprResult::Array(Arc::new(normalized))
            }) as ExprEvalFn
        }))
    }
}

/// Carry a span's sub-day field into whole days and its days into 30-day
/// months, then hand back whatever no longer fits: DuckDB's
/// `interval_t::Normalize`, both halves.
///
/// Rust and C++ both truncate integer division toward zero, so a negative span
/// keeps every field negative and still reads as the same total.
fn normalize(span: IntervalMonthDayNano) -> IntervalMonthDayNano {
    // Carry left.
    let mut nanoseconds = span.nanoseconds;
    let carried_days = nanoseconds / NANOS_PER_DAY;
    nanoseconds -= carried_days * NANOS_PER_DAY;

    let mut days = i64::from(span.days) + carried_days;
    let carried_months = days / i64::from(DAYS_PER_MONTH);
    days -= carried_months * i64::from(DAYS_PER_MONTH);

    let months = i64::from(span.months) + carried_months;

    // Fit each field, handing what it cannot hold back down to the one below.
    let (months, returned_days) = fit_into_field(months, i64::from(DAYS_PER_MONTH));
    let (days, returned_nanoseconds) = fit_into_field(days + returned_days, NANOS_PER_DAY);
    IntervalMonthDayNano {
        months,
        days,
        nanoseconds: nanoseconds + returned_nanoseconds,
    }
}

/// Fit `value` into its `i32` field, returning it with whatever did not fit,
/// converted to the units of the field below by `scale`. Overflowing a field is
/// the only way a carry can lose value, and handing the excess down keeps the
/// span worth the same rather than pinning it at the limit.
///
/// The returned amount cannot itself overflow: both fields started as `i32`s,
/// which bounds the excess well inside what an `i64` holds.
fn fit_into_field(value: i64, scale: i64) -> (i32, i64) {
    let fitted = value.clamp(i32::MIN.into(), i32::MAX.into());
    (fitted as i32, (value - fitted) * scale)
}

impl Display for NormalizedInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "normalized_interval({})", self.input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(months: i32, days: i32, nanoseconds: i64) -> IntervalMonthDayNano {
        IntervalMonthDayNano {
            months,
            days,
            nanoseconds,
        }
    }

    #[test]
    fn a_span_carries_into_whole_days_and_months() {
        let hours_25 = span(0, 0, 25 * 60 * 60 * 1_000_000_000);
        let days_100 = span(0, 100, 0);

        let normalized = [normalize(hours_25), normalize(days_100)];

        assert_eq!(normalized[0], span(0, 1, 60 * 60 * 1_000_000_000));
        assert_eq!(normalized[1], span(3, 10, 0));
    }

    #[test]
    fn two_spellings_of_one_duration_carry_to_the_same_fields() {
        let spelled_in_hours = span(0, 0, 25 * 60 * 60 * 1_000_000_000);
        let spelled_in_days = span(0, 1, 60 * 60 * 1_000_000_000);

        assert_eq!(normalize(spelled_in_hours), normalize(spelled_in_days));
    }

    #[test]
    fn the_carry_leaves_mixed_signs_alone() {
        // `1 mon -29 days` is a day's worth of time, but the carry only moves
        // value upward, so DuckDB leaves it spelled as written and ranks it
        // above a plain day. Matching that is the point.
        let mixed = span(1, -29, 0);

        let normalized = normalize(mixed);

        assert_eq!(normalized, mixed);
        assert!(normalized > normalize(span(0, 1, 0)));
    }

    #[test]
    fn normalizing_makes_a_hundred_days_outrank_a_month() {
        // Arrow orders the raw fields most-significant first, which without
        // this carry would rank `1 month` above `100 days`.
        let month = span(0, 1, 0);
        let hundred_days = span(0, 100, 0);

        assert!(normalize(hundred_days) > normalize(span(1, 0, 0)));
        assert!(normalize(hundred_days) > normalize(month));
    }

    #[test]
    fn a_carry_that_overflows_a_field_hands_it_back_down() {
        // 60 days carries to 2 months, which the months field has no room for,
        // so those 2 months go back to being 60 days rather than being lost.
        let full = span(i32::MAX, 60, 0);

        let normalized = normalize(full);

        assert_eq!(normalized, span(i32::MAX, 60, 0));
    }

    #[test]
    fn a_negative_span_keeps_every_field_negative() {
        let minus_25_hours = span(0, 0, -25 * 60 * 60 * 1_000_000_000);

        let normalized = normalize(minus_25_hours);

        assert_eq!(normalized, span(0, -1, -60 * 60 * 1_000_000_000));
        assert!(normalized < span(0, 0, 0));
    }
}
