//! [`ConditionKey`] — a canonical, hashable identity for a filter conjunction.
//!
//! A condition cache maps (row group, filter conjunction) to the row positions
//! the conjunction keeps, so it needs an identity for "the same conjunction"
//! that holds across queries: column references resolved to schema names (a
//! positional index shifts with the projection), constants encoded with their
//! type, and order-insensitive AND/OR/IN terms. Only a whitelisted,
//! deterministic subset of [`Expression`] is representable; anything else
//! (e.g. `now()`, regex replaces, CASE) yields [`ConditionKeyOutcome::NotCacheable`]
//! and the scan simply runs uncached.

use crate::expression::{CompareType, Expression, Function};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::{ArrayRef, Datum, Scalar};

/// Canonical identity of a filter AND-conjunction over one table scan.
/// Terms are sorted and deduplicated, so `a AND b` and `b AND a` key equal.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConditionKey {
    terms: Vec<TermKey>,
}

/// Why a conjunction could not be keyed: it contains an expression outside the
/// whitelisted deterministic subset. The scan then runs without the condition
/// cache; this is an expected outcome, not an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConditionKeyOutcome {
    Cacheable(ConditionKey),
    NotCacheable,
}

/// A comparison operator, mirroring [`CompareType`] with the ordering and
/// hashing derives the key needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CompareOp {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

impl From<CompareType> for CompareOp {
    fn from(compare_type: CompareType) -> Self {
        match compare_type {
            CompareType::Equal => CompareOp::Equal,
            CompareType::NotEqual => CompareOp::NotEqual,
            CompareType::Less => CompareOp::Less,
            CompareType::Greater => CompareOp::Greater,
            CompareType::LessEqual => CompareOp::LessEqual,
            CompareType::GreaterEqual => CompareOp::GreaterEqual,
        }
    }
}

impl CompareOp {
    /// The operator with its operands swapped (`5 < col` == `col > 5`).
    fn flipped(self) -> Self {
        match self {
            CompareOp::Equal => CompareOp::Equal,
            CompareOp::NotEqual => CompareOp::NotEqual,
            CompareOp::Less => CompareOp::Greater,
            CompareOp::Greater => CompareOp::Less,
            CompareOp::LessEqual => CompareOp::GreaterEqual,
            CompareOp::GreaterEqual => CompareOp::LessEqual,
        }
    }
}

/// One canonicalized expression node. Structured (not a display string) so two
/// different conditions can never collide on formatting quirks.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum TermKey {
    /// A column reference by its table schema name.
    Column(String),
    /// A constant: its arrow type plus the formatted value (`None` for a null
    /// constant). The type disambiguates equal renderings across types.
    Constant {
        data_type: String,
        value: Option<String>,
    },
    Compare {
        op: CompareOp,
        left: Box<TermKey>,
        right: Box<TermKey>,
    },
    /// AND/OR with sorted operands, so operand order does not fragment the key.
    And(Vec<TermKey>),
    Or(Vec<TermKey>),
    Not(Box<TermKey>),
    Contains {
        haystack: Box<TermKey>,
        needle: Box<TermKey>,
    },
    Between {
        input: Box<TermKey>,
        lower: Box<TermKey>,
        upper: Box<TermKey>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
    InList {
        input: Box<TermKey>,
        values: Vec<TermKey>,
    },
    Cast {
        target: String,
        source: Box<TermKey>,
    },
}

/// Build the canonical key for the AND of `conditions`, with column references
/// resolved through `scan_output_names` (the scan's projected column names, in
/// batch position order). Returns [`ConditionKeyOutcome::NotCacheable`] when any
/// condition falls outside the whitelisted deterministic subset.
pub fn build_condition_key(
    conditions: &[&Expression],
    scan_output_names: &[String],
) -> ConditionKeyOutcome {
    let mut terms = Vec::with_capacity(conditions.len());
    for condition in conditions {
        match build_term(condition, scan_output_names) {
            Some(term) => terms.push(term),
            None => return ConditionKeyOutcome::NotCacheable,
        }
    }
    terms.sort_unstable();
    terms.dedup();
    ConditionKeyOutcome::Cacheable(ConditionKey { terms })
}

fn build_term(expression: &Expression, scan_output_names: &[String]) -> Option<TermKey> {
    match expression {
        Expression::Ref(r) => scan_output_names
            .get(r.column_idx)
            .map(|name| TermKey::Column(name.clone())),
        Expression::Constant(scalar) => Some(encode_constant(scalar)),
        Expression::Compare(c) => {
            let op = CompareOp::from(c.compare_type);
            let left = build_term(&c.left, scan_output_names)?;
            let right = build_term(&c.right, scan_output_names)?;
            // Put the constant on the right, so `5 < col` and `col > 5` key equal.
            let (op, left, right) = match (&left, &right) {
                (TermKey::Constant { .. }, r) if !matches!(r, TermKey::Constant { .. }) => {
                    (op.flipped(), right.clone(), left)
                }
                _ => (op, left, right),
            };
            Some(TermKey::Compare {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        Expression::Conjunction(c) => {
            let mut children = c
                .children
                .iter()
                .map(|child| build_term(child, scan_output_names))
                .collect::<Option<Vec<_>>>()?;
            children.sort_unstable();
            Some(match c.op {
                crate::expression::ConjunctionOp::And => TermKey::And(children),
                crate::expression::ConjunctionOp::Or => TermKey::Or(children),
            })
        }
        Expression::Not(n) => Some(TermKey::Not(Box::new(build_term(
            &n.input,
            scan_output_names,
        )?))),
        Expression::Function(Function::Contains(c)) => Some(TermKey::Contains {
            haystack: Box::new(build_term(&c.haystack, scan_output_names)?),
            needle: Box::new(build_term(&c.needle, scan_output_names)?),
        }),
        Expression::Between(b) => Some(TermKey::Between {
            input: Box::new(build_term(&b.input, scan_output_names)?),
            lower: Box::new(build_term(&b.lower, scan_output_names)?),
            upper: Box::new(build_term(&b.upper, scan_output_names)?),
            lower_inclusive: b.lower_inclusive,
            upper_inclusive: b.upper_inclusive,
        }),
        Expression::InList(i) => {
            let input = build_term(&i.input, scan_output_names)?;
            let mut values = i
                .values
                .iter()
                .map(|v| build_term(v, scan_output_names))
                .collect::<Option<Vec<_>>>()?;
            values.sort_unstable();
            Some(TermKey::InList {
                input: Box::new(input),
                values,
            })
        }
        Expression::Cast(c) => Some(TermKey::Cast {
            target: c.target.to_string(),
            source: Box::new(build_term(&c.source, scan_output_names)?),
        }),
        // Everything else (other functions, CASE, aggregates) is either
        // non-deterministic or has no canonical form; the conjunction is
        // simply not cacheable.
        _ => None,
    }
}

fn encode_constant(scalar: &Scalar<ArrayRef>) -> TermKey {
    let (array, _) = scalar.get();
    let data_type = format!("{:?}", array.data_type());
    let value = if array.is_null(0) {
        None
    } else {
        // Formatting a supported constant type is infallible; an unformattable
        // type would silently merge distinct constants, so fail loudly instead.
        let formatter = ArrayFormatter::try_new(array, &FormatOptions::default())
            .expect("constant type must be formattable for the condition key");
        Some(formatter.value(0).to_string())
    };
    TermKey::Constant { data_type, value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{Compare, Conjunction, ConjunctionOp, Contains, Not, Ref};
    use crate::types::Type;
    use arrow_array::{Int32Array, Int64Array, StringViewArray};
    use std::sync::Arc;

    fn column(idx: usize) -> Expression {
        Expression::Ref(Ref {
            column_idx: idx,
            return_type: Type::Utf8,
            name: None,
        })
    }

    fn int32_constant(value: i32) -> Expression {
        Expression::Constant(Scalar::new(
            Arc::new(Int32Array::from(vec![value])) as ArrayRef
        ))
    }

    fn compare(compare_type: CompareType, left: Expression, right: Expression) -> Expression {
        Expression::Compare(Compare {
            left: Box::new(left),
            right: Box::new(right),
            compare_type,
            return_type: Type::Int8,
        })
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn same_condition_keys_equal_across_projections() {
        let narrow = compare(CompareType::Equal, column(0), int32_constant(7));
        let wide = compare(CompareType::Equal, column(2), int32_constant(7));

        let narrow_key = build_condition_key(&[&narrow], &names(&["url"]));
        let wide_key = build_condition_key(&[&wide], &names(&["a", "b", "url"]));

        assert_eq!(narrow_key, wide_key);
    }

    #[test]
    fn condition_order_and_operand_side_do_not_fragment_the_key() {
        let a = compare(CompareType::Greater, column(0), int32_constant(5));
        let a_flipped = compare(CompareType::Less, int32_constant(5), column(0));
        let b = compare(CompareType::Equal, column(1), int32_constant(9));

        let forward = build_condition_key(&[&a, &b], &names(&["x", "y"]));
        let backward = build_condition_key(&[&b, &a_flipped], &names(&["x", "y"]));

        assert_eq!(forward, backward);
    }

    #[test]
    fn equal_renderings_of_distinct_types_key_differently() {
        let as_int32 = compare(CompareType::Equal, column(0), int32_constant(7));
        let as_int64 = compare(
            CompareType::Equal,
            column(0),
            Expression::Constant(Scalar::new(Arc::new(Int64Array::from(vec![7])) as ArrayRef)),
        );

        let int32_key = build_condition_key(&[&as_int32], &names(&["x"]));
        let int64_key = build_condition_key(&[&as_int64], &names(&["x"]));

        assert_ne!(int32_key, int64_key);
    }

    #[test]
    fn negated_contains_is_cacheable() {
        let needle =
            Expression::Constant(Scalar::new(
                Arc::new(StringViewArray::from(vec!["google"])) as ArrayRef
            ));
        let condition = Expression::Not(Not {
            input: Box::new(Expression::Function(Function::Contains(Contains {
                needle: Box::new(needle),
                haystack: Box::new(column(0)),
            }))),
        });

        let key = build_condition_key(&[&condition], &names(&["url"]));

        assert!(matches!(key, ConditionKeyOutcome::Cacheable(_)));
    }

    #[test]
    fn conjunction_operand_order_does_not_fragment_the_key() {
        let a = compare(CompareType::Equal, column(0), int32_constant(1));
        let b = compare(CompareType::Equal, column(1), int32_constant(2));
        let ab = Expression::Conjunction(Conjunction {
            op: ConjunctionOp::And,
            children: vec![a.clone(), b.clone()],
        });
        let ba = Expression::Conjunction(Conjunction {
            op: ConjunctionOp::Or,
            children: vec![b, a],
        });

        let and_key = build_condition_key(&[&ab], &names(&["x", "y"]));
        let or_key = build_condition_key(&[&ba], &names(&["x", "y"]));

        assert_ne!(and_key, or_key);
    }

    #[test]
    fn volatile_function_is_not_cacheable() {
        let condition = compare(
            CompareType::Greater,
            Expression::Function(Function::Now),
            int32_constant(0),
        );

        let key = build_condition_key(&[&condition], &names(&["x"]));

        assert_eq!(key, ConditionKeyOutcome::NotCacheable);
    }

    #[test]
    fn out_of_range_column_reference_is_not_cacheable() {
        let condition = compare(CompareType::Equal, column(usize::MAX), int32_constant(1));

        let key = build_condition_key(&[&condition], &names(&["x"]));

        assert_eq!(key, ConditionKeyOutcome::NotCacheable);
    }
}
