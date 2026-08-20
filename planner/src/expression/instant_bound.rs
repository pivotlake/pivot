//! Folding `now()` out of a pushed-down range bound.
//!
//! DuckDB folds constant arithmetic itself, but not a call to pivot's `now()`:
//! every pivot-registered scalar function binds to a stub that returns NULL, so
//! the signature marks them volatile to keep DuckDB's constant folding from
//! evaluating one. The cost is that `Timestamp > now() - interval '1 hour'`
//! reaches a datastore's filter pushdown with no constant to prune by, and the
//! scan reads every row group the table has for the `Filter` above it to
//! discard. Resolving the instant here hands the pushdown a constant, while the
//! `Filter` keeps evaluating `now()` itself, so nothing about the answer
//! changes.

use super::{Compare, CompareType, Expression, Function};
use crate::compile::ExprResult;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, Scalar};
use arrow_schema::Schema;
use std::sync::Arc;

/// Resolve a `now()` bound in a pushed-down comparison into the instant it
/// reports, reporting whether it did.
///
/// Only a range bound (`<`, `<=`, `>`, `>=`) folds. A range predicate is
/// advisory: a datastore may use it to skip data that cannot match, and the
/// `Filter` re-applies the comparison to everything that survives, so a
/// constant that differs slightly from the one the `Filter` evaluates costs
/// work and no rows. An equality is not advisory — the Delta scan installs it
/// on the decoder, which drops rows that fail it — so its constant has to be
/// the very one the `Filter` compares against, and the instant this resolves is
/// already microseconds behind it.
///
/// A caller that records the folded bound owns an instant belonging to the
/// statement that planned it, which is what the reported `true` is for: the
/// plan must not be cached, or a later statement replays a prune against an
/// instant that has since moved.
pub fn fold_instant_bounds(expression: Expression) -> (Expression, bool) {
    let Expression::Compare(mut compare) = expression else {
        return (expression, false);
    };
    if !matches!(
        compare.compare_type,
        CompareType::Less
            | CompareType::LessEqual
            | CompareType::Greater
            | CompareType::GreaterEqual
    ) {
        return (Expression::Compare(compare), false);
    }
    // Either side can hold the bound: `ts > now()` and `now() < ts` are the
    // same predicate, and which side the column landed on is DuckDB's business.
    let Compare { left, right, .. } = &mut compare;
    let mut folded = false;
    for side in [left, right] {
        if let Some(instant) = evaluate_instant(side) {
            **side = Expression::Constant(instant);
            folded = true;
        }
    }
    (Expression::Compare(compare), folded)
}

/// The value an instant expression reports, or `None` when `expression` is not
/// one or cannot be evaluated on its own.
fn evaluate_instant(expression: &Expression) -> Option<Scalar<ArrayRef>> {
    if !is_instant(expression) {
        return None;
    }
    // An instant reads no columns, so a batch of one row and no columns is all
    // its evaluation needs.
    let one_row = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        Vec::new(),
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .ok()?;
    let mut evaluate = expression.compile().ok()?();
    match evaluate(&one_row) {
        ExprResult::Scalar(scalar) => Some(scalar),
        ExprResult::Array(array) if array.len() == 1 => Some(Scalar::new(array)),
        ExprResult::Array(_) => None,
    }
}

/// Whether `expression` reads `now()`, alone or shifted by constants.
///
/// A subtree of constants alone is not one: DuckDB already folded it, and
/// reporting it as an instant would cost the plan its place in the cache for
/// nothing.
fn is_instant(expression: &Expression) -> bool {
    match expression {
        Expression::Function(Function::Now(_)) => true,
        Expression::Cast(cast) => is_instant(&cast.source),
        Expression::Function(Function::IntervalArithmetic(shift)) => is_instant(&shift.operand),
        Expression::Function(Function::Arithmetic(arithmetic)) => {
            let (left, right) = (arithmetic.left.as_ref(), arithmetic.right.as_ref());
            is_instant(left) && is_constant(right) || is_constant(left) && is_instant(right)
        }
        _ => false,
    }
}

fn is_constant(expression: &Expression) -> bool {
    matches!(expression, Expression::Constant(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmeticOp, IntervalArithmetic, Now, Ref};
    use crate::types::Type;
    use arrow_array::Datum;
    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampMicrosecondType;
    use rstest::rstest;

    const ONE_HOUR_MICROS: i64 = 3_600_000_000;

    fn timestamp_column() -> Expression {
        Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Timestamp,
            name: None,
        })
    }

    fn an_hour_ago() -> Expression {
        Expression::Function(Function::IntervalArithmetic(IntervalArithmetic {
            op: ArithmeticOp::Sub,
            operand: Box::new(Expression::Function(Function::Now(Now))),
            offset: ONE_HOUR_MICROS,
            result: Type::Timestamp,
        }))
    }

    fn a_fixed_instant() -> Expression {
        Expression::Constant(Scalar::new(
            Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
                1_700_000_000_000_000i64,
            ])) as ArrayRef,
        ))
    }

    fn compare(left: Expression, compare_type: CompareType, right: Expression) -> Expression {
        Expression::Compare(Compare {
            left: Box::new(left),
            right: Box::new(right),
            compare_type,
            return_type: Type::Boolean,
        })
    }

    fn sides(expression: &Expression) -> (&Expression, &Expression) {
        let Expression::Compare(compare) = expression else {
            panic!("a comparison stays a comparison");
        };
        (compare.left.as_ref(), compare.right.as_ref())
    }

    fn constant_micros(expression: &Expression) -> i64 {
        let Expression::Constant(constant) = expression else {
            panic!("expected a folded constant, got {expression}");
        };
        constant
            .get()
            .0
            .as_primitive::<TimestampMicrosecondType>()
            .value(0)
    }

    /// Every range direction folds: each only ever prunes by statistics, which
    /// the `Filter` above the scan re-checks.
    #[rstest]
    #[case(CompareType::Greater)]
    #[case(CompareType::GreaterEqual)]
    #[case(CompareType::Less)]
    #[case(CompareType::LessEqual)]
    fn folds_a_now_bound_in_either_direction(#[case] compare_type: CompareType) {
        let before = micros_since_epoch();

        let (folded, reported) =
            fold_instant_bounds(compare(timestamp_column(), compare_type, an_hour_ago()));

        assert!(reported);
        let micros = constant_micros(sides(&folded).1);
        assert!((before - ONE_HOUR_MICROS..=before).contains(&micros));
    }

    /// An equality is installed on the decoder as a row filter, so its constant
    /// has to be the one the `Filter` evaluates, not one resolved earlier.
    #[rstest]
    #[case(CompareType::Equal)]
    #[case(CompareType::NotEqual)]
    fn leaves_an_equality_unfolded(#[case] compare_type: CompareType) {
        let (folded, reported) =
            fold_instant_bounds(compare(timestamp_column(), compare_type, an_hour_ago()));

        assert!(!reported);
        assert!(matches!(sides(&folded).1, Expression::Function(_)));
    }

    #[test]
    fn folds_the_left_side_when_the_column_is_on_the_right() {
        let (folded, reported) = fold_instant_bounds(compare(
            an_hour_ago(),
            CompareType::Less,
            timestamp_column(),
        ));

        assert!(reported);
        assert!(matches!(sides(&folded).0, Expression::Constant(_)));
    }

    #[test]
    fn leaves_a_bound_reading_a_column_alone() {
        let (folded, reported) = fold_instant_bounds(compare(
            timestamp_column(),
            CompareType::Greater,
            timestamp_column(),
        ));

        assert!(!reported);
        assert!(matches!(sides(&folded).1, Expression::Ref(_)));
    }

    /// A bound DuckDB already folded reports nothing: it holds no instant, so
    /// the plan it belongs to stays cacheable.
    #[test]
    fn reports_nothing_for_a_bound_that_is_already_constant() {
        let (folded, reported) = fold_instant_bounds(compare(
            timestamp_column(),
            CompareType::Greater,
            a_fixed_instant(),
        ));

        assert!(!reported);
        assert_eq!(constant_micros(sides(&folded).1), 1_700_000_000_000_000);
    }

    fn micros_since_epoch() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
    }
}
