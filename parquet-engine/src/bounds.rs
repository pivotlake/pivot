//! Pruning by ranges. A layer of a table's metadata (the manifests of a
//! snapshot, the files of a manifest) has a range per item for some keys: a
//! column's lower and upper bound, a partition field's value. The layer lays
//! them out as arrays, one slot per item, and one evaluator checks every item
//! against a predicate in a single comparison. A key the layer does not have
//! prunes nothing.

use std::collections::HashMap;

use arrow_arith::boolean::{and, and_not};
use arrow_array::{Array, ArrayRef, BooleanArray, Scalar};
use planner::expression::CompareType;

use crate::row_group_stats::{bounds_exclude, nan_satisfies};

/// What a range is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundKey {
    /// A declared column, by position.
    Column(usize),
    /// A field of a partition spec, by the spec's id and the field's position
    /// in it.
    PartitionField { spec_id: i32, position: usize },
}

/// One key's range in every item of a layer.
pub struct Range {
    /// The least value in each item, in the key's physical Arrow type. Null
    /// where unknown; `max` is null in the same slots.
    pub min: ArrayRef,
    /// The greatest value in each item.
    pub max: ArrayRef,
    /// Items known to hold only NULL for the key. They match no comparison,
    /// whatever their bounds say.
    pub all_null: BooleanArray,
    /// Items that may hold a NaN: a float key whose statistics do not prove
    /// it NaN-free. Bounds leave NaN out, and NaN sorts above every value, so
    /// such an item may match a comparison its bounds exclude.
    pub may_hold_nan: BooleanArray,
}

/// A comparison items are pruned by: `key <compare> c` for any `c` in
/// `constants`. One constant is a plain comparison; several make an `IN`,
/// which is what a transform projects some comparisons to.
pub struct RangePredicate {
    pub key: BoundKey,
    pub compare: CompareType,
    pub constants: Vec<Scalar<ArrayRef>>,
}

/// What one layer knows about its items, by key.
pub struct Bounds {
    items: usize,
    ranges: HashMap<BoundKey, Range>,
}

impl Bounds {
    /// A layer of `items` items that knows nothing yet.
    pub fn new(items: usize) -> Self {
        Self {
            items,
            ranges: HashMap::new(),
        }
    }

    /// Record `range`, one slot per item, as the layer's range for `key`.
    pub fn insert(&mut self, key: BoundKey, range: Range) {
        assert!(
            range.min.len() == self.items
                && range.max.len() == self.items
                && range.all_null.len() == self.items
                && range.may_hold_nan.len() == self.items,
            "a range has one slot per item"
        );
        self.ranges.insert(key, range);
    }

    pub fn len(&self) -> usize {
        self.items
    }

    pub fn is_empty(&self) -> bool {
        self.items == 0
    }

    pub fn range(&self, key: BoundKey) -> Option<&Range> {
        self.ranges.get(&key)
    }

    /// Which items `predicates` cannot rule out: `true` per item to keep.
    /// Each predicate is one comparison over every item. A key the layer does
    /// not have, bounds of another type than the constant, and an item with
    /// no bound all keep the item.
    pub fn kept(&self, predicates: &[RangePredicate]) -> Vec<bool> {
        let mut pruned = vec![false; self.items];
        for predicate in predicates {
            let Some(range) = self.ranges.get(&predicate.key) else {
                continue;
            };
            let excluded = range.excluded_by(predicate);
            for (item, is_pruned) in pruned.iter_mut().enumerate() {
                *is_pruned |= range.all_null.value(item)
                    || excluded
                        .as_ref()
                        .is_some_and(|mask| mask.is_valid(item) && mask.value(item));
            }
        }
        pruned.into_iter().map(|pruned| !pruned).collect()
    }
}

impl Range {
    /// Which items' bounds prove `predicate` false: `true` where they do,
    /// null where an item has no bound, `None` when the bounds are not the
    /// constants' type. With several constants an item is excluded only if
    /// every constant excludes it. An item that may hold a NaN is never
    /// excluded where a NaN would match.
    fn excluded_by(&self, predicate: &RangePredicate) -> Option<BooleanArray> {
        let mut excluded: Option<BooleanArray> = None;
        for constant in &predicate.constants {
            let mut by_constant = bounds_exclude(&self.min, &self.max, predicate.compare, constant)
                .ok()
                .flatten()?;
            if nan_satisfies(predicate.compare, constant) {
                by_constant = and_not(&by_constant, &self.may_hold_nan).ok()?;
            }
            excluded = Some(match excluded {
                Some(so_far) => and(&so_far, &by_constant).ok()?,
                None => by_constant,
            });
        }
        excluded
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int32Array, Int64Array};

    use super::*;

    /// A layer of three items whose column 0 has the given `[min, max]`
    /// bounds and is entirely null where `all_null` says.
    fn layer(bounds: &[Option<(i64, i64)>], all_null: &[bool]) -> Bounds {
        let mut layer = Bounds::new(bounds.len());
        layer.insert(
            BoundKey::Column(0),
            Range {
                min: Arc::new(Int64Array::from_iter(
                    bounds.iter().map(|bound| bound.map(|(min, _)| min)),
                )),
                max: Arc::new(Int64Array::from_iter(
                    bounds.iter().map(|bound| bound.map(|(_, max)| max)),
                )),
                all_null: BooleanArray::from(all_null.to_vec()),
                may_hold_nan: BooleanArray::from(vec![false; bounds.len()]),
            },
        );
        layer
    }

    /// A layer of floating-point items with the given `[min, max]` bounds,
    /// each of which may hold a NaN where `may_hold_nan` says.
    fn floating_layer(bounds: &[(f64, f64)], may_hold_nan: &[bool]) -> Bounds {
        let mut layer = Bounds::new(bounds.len());
        layer.insert(
            BoundKey::Column(0),
            Range {
                min: Arc::new(Float64Array::from_iter_values(
                    bounds.iter().map(|(min, _)| *min),
                )),
                max: Arc::new(Float64Array::from_iter_values(
                    bounds.iter().map(|(_, max)| *max),
                )),
                all_null: BooleanArray::from(vec![false; bounds.len()]),
                may_hold_nan: BooleanArray::from(may_hold_nan.to_vec()),
            },
        );
        layer
    }

    fn floating_predicate(compare: CompareType, constant: f64) -> RangePredicate {
        RangePredicate {
            key: BoundKey::Column(0),
            compare,
            constants: vec![Scalar::new(
                Arc::new(Float64Array::from(vec![constant])) as ArrayRef
            )],
        }
    }

    #[test]
    fn an_item_that_may_hold_a_nan_satisfies_what_a_nan_does() {
        let layer = floating_layer(&[(1.0, 1.0), (1.0, 1.0)], &[true, false]);

        // NaN is greater than 2, so only the NaN-free item is ruled out; NaN
        // is less than nothing, so both are.
        assert_eq!(
            layer.kept(&[floating_predicate(CompareType::Greater, 2.0)]),
            [true, false]
        );
        assert_eq!(
            layer.kept(&[floating_predicate(CompareType::Less, 0.0)]),
            [false, false]
        );
    }

    fn predicate(key: BoundKey, compare: CompareType, constants: &[i64]) -> RangePredicate {
        RangePredicate {
            key,
            compare,
            constants: constants
                .iter()
                .map(
                    |constant| Scalar::new(Arc::new(Int64Array::from(vec![*constant])) as ArrayRef),
                )
                .collect(),
        }
    }

    #[test]
    fn an_item_is_kept_unless_its_range_proves_no_match() {
        let layer = layer(
            &[Some((3, 6)), Some((10, 20)), None],
            &[false, false, false],
        );
        let column = BoundKey::Column(0);

        assert_eq!(
            layer.kept(&[predicate(column, CompareType::Greater, &[6])]),
            [false, true, true]
        );
        assert_eq!(
            layer.kept(&[predicate(column, CompareType::Equal, &[2])]),
            [false, false, true]
        );
        assert_eq!(
            layer.kept(&[predicate(column, CompareType::Equal, &[4])]),
            [true, false, true]
        );
    }

    #[test]
    fn an_all_null_item_satisfies_no_comparison() {
        let layer = layer(&[Some((3, 6)), Some((3, 6))], &[true, false]);

        let kept = layer.kept(&[predicate(BoundKey::Column(0), CompareType::Equal, &[4])]);

        assert_eq!(kept, [false, true]);
    }

    #[test]
    fn several_constants_exclude_an_item_only_where_each_does() {
        let layer = layer(&[Some((1, 1)), Some((5, 5)), Some((9, 9))], &[false; 3]);

        let kept = layer.kept(&[predicate(BoundKey::Column(0), CompareType::Equal, &[5, 9])]);

        assert_eq!(kept, [false, true, true]);
    }

    #[test]
    fn a_key_the_layer_does_not_know_prunes_nothing() {
        let layer = layer(&[Some((3, 6))], &[false]);
        let unknown = BoundKey::PartitionField {
            spec_id: 0,
            position: 0,
        };

        let kept = layer.kept(&[predicate(unknown, CompareType::Equal, &[99])]);

        assert_eq!(kept, [true]);
    }

    #[test]
    fn bounds_of_another_type_prune_nothing() {
        let mut layer = Bounds::new(1);
        layer.insert(
            BoundKey::Column(0),
            Range {
                min: Arc::new(Int32Array::from(vec![3])),
                max: Arc::new(Int32Array::from(vec![6])),
                all_null: BooleanArray::from(vec![false]),
                may_hold_nan: BooleanArray::from(vec![false]),
            },
        );

        let kept = layer.kept(&[predicate(BoundKey::Column(0), CompareType::Equal, &[99])]);

        assert_eq!(kept, [true]);
    }
}
