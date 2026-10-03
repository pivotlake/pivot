use arrow_array::{Array, BooleanArray, Datum};
use arrow_buffer::BooleanBuffer;

use crate::{Bounds, ColumnPath, Predicate, Transform};

/// Bounds on `transform(column)` in each object. Column bounds use
/// [`Transform::Identity`]; a partition field bounds its own transform of the
/// column, with equal lower and upper where the value is exact.
#[derive(Clone, Debug)]
pub struct Statistic {
    pub column: ColumnPath,
    pub transform: Transform,
    pub bounds: Bounds,
}

/// The statistics of a list of objects (manifests, files or row groups), one
/// slot per object in the adapter's own order. Several statistics may describe
/// one column; each can exclude objects on its own.
#[derive(Clone, Debug)]
pub struct Statistics {
    len: usize,
    statistics: Vec<Statistic>,
}

impl Statistics {
    /// Panics unless every statistic has `len` slots and one type for both
    /// of its bounds.
    pub fn new(len: usize, statistics: Vec<Statistic>) -> Self {
        for Statistic { column, bounds, .. } in &statistics {
            assert!(
                bounds.lower.len() == len && bounds.upper.len() == len && bounds.len() == len,
                "bounds of {column:?} do not have one slot per object"
            );
            assert_eq!(
                bounds.lower.data_type(),
                bounds.upper.data_type(),
                "bounds of {column:?} have two types"
            );
        }
        Self { len, statistics }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// One non-null slot per object: false proves that no row of the object
    /// satisfies every predicate.
    pub fn prune(&self, predicates: &[Predicate]) -> BooleanArray {
        let mut keep = BooleanBuffer::new_set(self.len);
        for predicate in predicates {
            // A comparison with NULL is never true, whatever the object holds.
            if predicate.value.get().0.is_null(0) {
                return BooleanArray::new(BooleanBuffer::new_unset(self.len), None);
            }
            for statistic in &self.statistics {
                if statistic.column != predicate.column {
                    continue;
                }
                for (comparison, constant) in statistic
                    .transform
                    .project(predicate.comparison, &predicate.value)
                {
                    keep = &keep & &statistic.bounds.may_match(comparison, &constant);
                }
            }
        }
        BooleanArray::new(keep, None)
    }

    /// The `objects` that survive [`Self::prune`], given in statistics order.
    pub fn select<'a, T>(
        &self,
        objects: &'a [T],
        predicates: &[Predicate],
    ) -> impl Iterator<Item = &'a T> + use<'a, T> {
        assert_eq!(objects.len(), self.len, "one object per statistics slot");
        let keep = self.prune(predicates);
        objects
            .iter()
            .enumerate()
            .filter(move |(row, _)| keep.value(*row))
            .map(|(_, object)| object)
    }
}
