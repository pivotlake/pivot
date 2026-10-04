use arrow_array::{Array, BooleanArray, Datum};
use arrow_buffer::BooleanBuffer;
use planner::expression::{CompareType, ConjunctionOp, Expression};

use super::{Bounds, ColumnPath, ConstantComparison, Transform};

/// Each value of an `IN` list costs a pass over the statistics, so a longer
/// list is left to the filter above the scan.
const MAX_IN_LIST_VALUES: usize = 32;

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

    /// One non-null slot per object: false proves that no row of the object
    /// passes every filter.
    pub fn prune(&self, filters: &[Expression]) -> BooleanArray {
        let keep = filters
            .iter()
            .map(|filter| self.may_be_true(filter))
            .fold(self.all(), |keep, mask| &keep & &mask);
        BooleanArray::new(keep, None)
    }

    /// The `objects` that survive [`Self::prune`], given in statistics order.
    pub fn select<'a, T>(
        &self,
        objects: &'a [T],
        filters: &[Expression],
    ) -> impl Iterator<Item = &'a T> + use<'a, T> {
        assert_eq!(objects.len(), self.len, "one object per statistics slot");
        let keep = self.prune(filters);
        objects
            .iter()
            .enumerate()
            .filter(move |(row, _)| keep.value(*row))
            .map(|(_, object)| object)
    }

    /// The objects that may hold a row for which `expression` is true. An
    /// expression that statistics say nothing about may be true anywhere.
    fn may_be_true(&self, expression: &Expression) -> BooleanBuffer {
        match expression {
            Expression::Compare(compare) => match ConstantComparison::from_compare(compare) {
                Some(comparison) => self.may_satisfy(&comparison),
                None => self.all(),
            },
            Expression::Between(between) => match ConstantComparison::from_between(between) {
                Some([lower, upper]) => &self.may_satisfy(&lower) & &self.may_satisfy(&upper),
                None => self.all(),
            },
            Expression::Conjunction(conjunction) => {
                let children = conjunction
                    .children
                    .iter()
                    .map(|child| self.may_be_true(child));
                match conjunction.op {
                    ConjunctionOp::And => children.fold(self.all(), |keep, mask| &keep & &mask),
                    ConjunctionOp::Or => children.fold(self.none(), |keep, mask| &keep | &mask),
                }
            }
            // A row is in the list when it equals one of its values.
            Expression::InList(list) if list.values.len() <= MAX_IN_LIST_VALUES => list
                .values
                .iter()
                .map(|value| match equality(&list.input, value) {
                    Some(comparison) => self.may_satisfy(&comparison),
                    None => self.all(),
                })
                .fold(self.none(), |keep, mask| &keep | &mask),
            _ => self.all(),
        }
    }

    /// The objects that may hold a row satisfying `comparison`, by every
    /// statistic of its column.
    fn may_satisfy(&self, comparison: &ConstantComparison) -> BooleanBuffer {
        // A comparison with NULL is never true, whatever the object holds.
        if comparison.constant.get().0.is_null(0) {
            return self.none();
        }
        let mut keep = self.all();
        for statistic in &self.statistics {
            if statistic.column != comparison.column {
                continue;
            }
            for (compare_type, constant) in statistic
                .transform
                .project(comparison.compare_type, &comparison.constant)
            {
                keep = &keep & &statistic.bounds.may_match(compare_type, &constant);
            }
        }
        keep
    }

    fn all(&self) -> BooleanBuffer {
        BooleanBuffer::new_set(self.len)
    }

    fn none(&self) -> BooleanBuffer {
        BooleanBuffer::new_unset(self.len)
    }
}

/// `input = value` as a comparison of a column with a constant.
fn equality(input: &Expression, value: &Expression) -> Option<ConstantComparison> {
    let Expression::Constant(constant) = value else {
        return None;
    };
    ConstantComparison::of(input, CompareType::Equal, constant)
}

/// Every comparison of a column with a constant that pruning by `filters`
/// consults statistics for, wherever it sits in them. Its columns are the
/// only ones whose statistics a query needs.
pub fn comparisons(filters: &[Expression]) -> Vec<ConstantComparison> {
    let mut comparisons = Vec::new();
    for filter in filters {
        collect_comparisons(filter, &mut comparisons);
    }
    comparisons
}

fn collect_comparisons(expression: &Expression, comparisons: &mut Vec<ConstantComparison>) {
    match expression {
        Expression::Compare(compare) => {
            comparisons.extend(ConstantComparison::from_compare(compare))
        }
        Expression::Between(between) => comparisons.extend(
            ConstantComparison::from_between(between)
                .into_iter()
                .flatten(),
        ),
        Expression::Conjunction(conjunction) => conjunction
            .children
            .iter()
            .for_each(|child| collect_comparisons(child, comparisons)),
        Expression::InList(list) if list.values.len() <= MAX_IN_LIST_VALUES => comparisons.extend(
            list.values
                .iter()
                .filter_map(|value| equality(&list.input, value)),
        ),
        _ => {}
    }
}

/// The distinct columns among `comparisons`, in first-use order.
pub fn columns(comparisons: &[ConstantComparison]) -> Vec<&ColumnPath> {
    let mut columns: Vec<&ColumnPath> = Vec::new();
    for comparison in comparisons {
        if !columns.contains(&&comparison.column) {
            columns.push(&comparison.column);
        }
    }
    columns
}
