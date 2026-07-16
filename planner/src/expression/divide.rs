//! [`Divide`] — SQL `lhs / rhs` (float division).

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::RecordBatch;
use std::fmt::{self, Display};

/// SQL `lhs / rhs`. Used by `AVG`, which DuckDB lowers to `sum(x) / count(x)`.
///
/// DuckDB binds `/` to float division only: `REAL / REAL` yields `REAL` and
/// every other numeric pair binds to the `DOUBLE` overload, with the operands
/// cast to match. `return_type` carries whichever DuckDB picked, so a `REAL`
/// quotient stays single-precision and still lines up with an enclosing
/// expression bound against `REAL`.
#[derive(Debug, Clone)]
pub struct Divide {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub return_type: Type,
}

impl Display for Divide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} / {})", self.left, self.right)
    }
}

impl Divide {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                // Passing the datums through keeps a constant operand marked as
                // a scalar, which is what lets the kernel broadcast it across
                // the batch rather than reject it as a length-one array. DuckDB
                // binds `/` to its float overloads, so both operands already
                // share one float type.
                let quotient =
                    arrow::compute::kernels::numeric::div(left.as_datum(), right.as_datum())
                        .expect("divide operands share one float type");
                ExprResult::Array(quotient)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Float32Array};
    use rstest::rstest;
    use std::sync::Arc;

    /// Registers `real_table`, whose `f`/`g` are REAL (single-precision), the
    /// one pair of operands DuckDB divides without casting to DOUBLE.
    fn add_real_table(testing_planner: &TestingPlanner) {
        let f: ArrayRef = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0]));
        let g: ArrayRef = Arc::new(Float32Array::from(vec![2.0f32, 2.0, 2.0]));
        testing_planner.add_table(
            "real_table",
            &[("f", Type::Float32, f), ("g", Type::Float32, g)],
        );
    }

    #[rstest]
    fn divides_real_columns(mut testing_planner: TestingPlanner) {
        add_real_table(&testing_planner);

        let rows = run(&mut testing_planner, "SELECT f / g FROM real_table");

        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_f64().unwrap())
                .collect::<Vec<_>>(),
            vec![0.5, 1.0, 1.5]
        );
    }

    #[rstest]
    fn divides_real_column_by_constant(mut testing_planner: TestingPlanner) {
        add_real_table(&testing_planner);

        let rows = run(&mut testing_planner, "SELECT f / 2 FROM real_table");

        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_f64().unwrap())
                .collect::<Vec<_>>(),
            vec![0.5, 1.0, 1.5]
        );
    }

    #[rstest]
    fn adds_a_real_quotient_to_a_real_column(mut testing_planner: TestingPlanner) {
        // The quotient of two REALs stays REAL, so the enclosing `+`, which
        // DuckDB bound against REAL, gets the operand type it expects.
        add_real_table(&testing_planner);

        let rows = run(&mut testing_planner, "SELECT (f / g) + f FROM real_table");

        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_f64().unwrap())
                .collect::<Vec<_>>(),
            vec![1.5, 3.0, 4.5]
        );
    }

    #[rstest]
    fn divides_column_by_constant(mut testing_planner: TestingPlanner) {
        let mut rows = run(&mut testing_planner, "SELECT a / 2 FROM example_table");

        let mut quotients = rows
            .iter_mut()
            .map(|r| only_column(r).as_f64().unwrap())
            .collect::<Vec<_>>();
        quotients.sort_by(f64::total_cmp);

        assert_eq!(quotients, vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[rstest]
    fn divides_aggregate_by_constant(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT MAX(c) / 1000 FROM example_table",
        );

        assert_eq!(only_column(&rows[0]).as_f64().unwrap(), 0.5);
    }

    #[rstest]
    fn divides_grouped_aggregate_by_constant(mut testing_planner: TestingPlanner) {
        // The grouped aggregate yields one row per group, so the constant is the
        // only operand of length one and has to broadcast across the groups.
        let mut rows = run(
            &mut testing_planner,
            "SELECT MAX(c) / 1000 FROM example_table GROUP BY name",
        );

        let mut quotients = rows
            .iter_mut()
            .map(|r| only_column(r).as_f64().unwrap())
            .collect::<Vec<_>>();
        quotients.sort_by(f64::total_cmp);

        assert_eq!(quotients, vec![0.2, 0.3, 0.4, 0.5]);
    }

    #[rstest]
    fn aggregates_a_quotient(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT MAX(c / 1000) FROM example_table",
        );

        assert_eq!(only_column(&rows[0]).as_f64().unwrap(), 0.5);
    }

    #[rstest]
    fn divides_column_by_column(mut testing_planner: TestingPlanner) {
        let mut rows = run(&mut testing_planner, "SELECT b / a FROM example_table");

        let quotients = rows
            .iter_mut()
            .map(|r| only_column(r).as_f64().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(quotients, vec![10.0; 5]);
    }
}
