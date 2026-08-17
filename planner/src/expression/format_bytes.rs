//! [`FormatBytes`] — SQL `format_bytes(bytes)`, a byte count rendered for a
//! human reader.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use std::fmt::{self, Display};
use std::sync::Arc;

/// The units a byte count is reported in, from the bare count upwards. Each
/// step is [`MULTIPLIER`] times the one below it.
const UNITS: [&str; 6] = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB"];

/// Binary units: a KiB is 1024 bytes, not 1000.
const MULTIPLIER: u64 = 1024;

/// SQL `format_bytes(bytes)` — an integer byte count as a string in the largest
/// binary unit that leaves a whole part, with one fractional digit
/// (`1536` → `1.5 KiB`). A count below one KiB is reported as itself.
#[derive(Debug, Clone)]
pub struct FormatBytes {
    /// The byte count, a `BIGINT`.
    pub input: Box<Expression>,
}

impl Display for FormatBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "format_bytes({})", self.input)
    }
}

/// One byte count as its human-readable string. A negative count keeps its sign
/// and is rendered from its magnitude.
fn render_bytes(bytes: i64) -> String {
    let sign = if bytes < 0 { "-" } else { "" };
    // How much of the count each unit holds, once the units above it have taken
    // their share: `parts[i]` counts whole `UNITS[i]`s beyond `UNITS[i + 1]`.
    let mut parts = [0u64; UNITS.len()];
    parts[0] = bytes.unsigned_abs();
    // The unit the count is reported in. Dividing stops at the first unit the
    // count doesn't fill, since every unit above that one is empty too. The last
    // unit is never divided out of, so a count past a PiB reports thousands of
    // them rather than saturating.
    let mut unit = 0;
    while unit + 1 < UNITS.len() && parts[unit] >= MULTIPLIER {
        parts[unit + 1] = parts[unit] / MULTIPLIER;
        parts[unit] %= MULTIPLIER;
        unit += 1;
    }

    if unit == 0 {
        let plural = if bytes == 1 { "byte" } else { "bytes" };
        return format!("{sign}{} {plural}", parts[0]);
    }
    // The part below the reported unit, as a single digit: a full unit short of
    // the next one reads `.9`, not `.10`.
    let fraction = parts[unit - 1] * 10 / MULTIPLIER;
    format!("{sign}{}.{fraction} {}", parts[unit], UNITS[unit])
}

impl FormatBytes {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let counts = arr.as_primitive::<Int64Type>();
                let rendered: StringViewArray =
                    counts.iter().map(|count| count.map(render_bytes)).collect();
                ExprResult::Array(Arc::new(rendered) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int64Array};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn renders_each_binary_unit(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "sizes",
            &[(
                "bytes",
                Type::Int64,
                Arc::new(Int64Array::from(vec![
                    0,
                    1,
                    1023,
                    1024,
                    1536,
                    2047,
                    1024 * 1024 - 1,
                    1024 * 1024,
                    3 * 1024 * 1024 * 1024,
                    -2048,
                    -32,
                    i64::MAX,
                    i64::MIN,
                ])) as ArrayRef,
            )],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT format_bytes(bytes) FROM sizes ORDER BY bytes",
        );

        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_str().unwrap().to_string())
                .collect::<Vec<_>>(),
            vec![
                "-8192.0 PiB",
                "-2.0 KiB",
                "-32 bytes",
                "0 bytes",
                "1 byte",
                "1023 bytes",
                "1.0 KiB",
                "1.5 KiB",
                "1.9 KiB",
                "1023.9 KiB",
                "1.0 MiB",
                "3.0 GiB",
                "8191.9 PiB",
            ]
        );
    }

    #[rstest]
    fn renders_a_null_count_as_null(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "nullable_sizes",
            &[(
                "bytes",
                Type::Int64,
                Arc::new(Int64Array::from(vec![None, Some(4096)])) as ArrayRef,
            )],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT format_bytes(bytes) AS size FROM nullable_sizes ORDER BY bytes",
        );

        // The JSON writer omits null fields, so the NULL row has no "size" key.
        let sizes: Vec<Option<&str>> = rows.iter().map(|r| r["size"].as_str()).collect();
        assert_eq!(sizes, vec![Some("4.0 KiB"), None]);
    }
}
