//! Pushed-down filter conjuncts a scan evaluates itself, so which rows of a
//! row group pass one is remembered across queries (see
//! [`super::selection_cache`]).
//!
//! The filter above the scan still applies every conjunct: a scan only ever
//! narrows what it emits, so a predicate it cannot evaluate or remember costs
//! correctness nothing.

use crate::types::leaves::first_leaf;
use crate::types::metadata::RowGroupMetadata;
use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_buffer::BooleanBuffer;
use dispatch::Projection;
use planner::compile::{ExprEvalFn, ExprFn, ExprResult};
use planner::expression::Expression;
use std::hash::{DefaultHasher, Hash, Hasher};

/// A filter conjunct over a scan's own columns whose result per row group
/// the scan can take from the selection cache or gather into it. It reads
/// columns only, through deterministic operations, so the rows it keeps in a
/// row group are the same whenever it is evaluated.
#[derive(Clone, Debug)]
pub struct ScanPredicate {
    /// The conjunct, naming the table's columns by storage position.
    expression: Expression,
    /// The conjunct's printed form, with every column named by its storage
    /// position; what tells one conjunct from another.
    printed: String,
    /// The storage columns the conjunct reads, in the order its printed form
    /// names them.
    columns: Vec<usize>,
}

impl ScanPredicate {
    /// The conjunct as a scan predicate, when it reads at least one column
    /// and nothing but columns through deterministic operations.
    pub fn from_filter(filter: &Expression) -> Option<Self> {
        if !filter.is_deterministic_over_columns() {
            return None;
        }
        let mut columns = Vec::new();
        filter
            .clone()
            .visit_column_refs_mut(&mut |reference| columns.push(reference.column_idx));
        if columns.is_empty() {
            return None;
        }
        Some(Self {
            expression: filter.clone(),
            printed: format!("{filter:?}"),
            columns,
        })
    }

    /// Identifies the conjunct within `row_group`'s file: its printed form
    /// together with the file leaf each column it reads starts at there, so
    /// two tables laying the same file out differently never share an
    /// entry, while two queries spelling the same condition do.
    pub fn fingerprint(&self, row_group: &RowGroupMetadata) -> u64 {
        let fields = row_group.schema.fields();
        let mut hasher = DefaultHasher::new();
        self.printed.hash(&mut hasher);
        for &column in &self.columns {
            first_leaf(fields, column).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The conjunct compiled against the batches a scan with `projection`
    /// emits, or `None` when a column it reads is not emitted as itself (not
    /// projected, or projected as an extract of the column).
    pub fn compile_for(&self, projection: &Projection) -> Option<CompiledScanPredicate> {
        let mut projected = self.expression.clone();
        let mut every_column_emitted = true;
        projected.visit_column_refs_mut(&mut |reference| {
            let position = projection
                .column_indices
                .iter()
                .position(|&column| column == reference.column_idx);
            match position {
                Some(position) if projection.extract_at(position).is_none() => {
                    reference.column_idx = position;
                }
                _ => every_column_emitted = false,
            }
        });
        if !every_column_emitted {
            return None;
        }
        let build_evaluator = projected.compile().ok()?;
        Some(CompiledScanPredicate { build_evaluator })
    }
}

/// A [`ScanPredicate`] compiled against a scan's emitted batches.
pub struct CompiledScanPredicate {
    build_evaluator: ExprFn,
}

impl CompiledScanPredicate {
    /// A worker's own evaluator of the predicate.
    pub fn evaluator(&self) -> ExprEvalFn {
        (self.build_evaluator)()
    }
}

/// The rows of a `rows`-row batch a predicate's `result` holds for: a bit per
/// row, set where the result is true, clear where it is false or null.
pub fn passing_rows(result: ExprResult, rows: usize) -> BooleanBuffer {
    match result {
        ExprResult::Array(array) => {
            let booleans = array
                .as_boolean_opt()
                .expect("a filter conjunct evaluates to booleans");
            match booleans.nulls() {
                Some(nulls) => booleans.values() & nulls.inner(),
                None => booleans.values().clone(),
            }
        }
        ExprResult::Scalar(scalar) => {
            let array = scalar.into_inner();
            let booleans = array
                .as_boolean_opt()
                .expect("a filter conjunct evaluates to booleans");
            if booleans.len() == 1 && booleans.is_valid(0) && booleans.value(0) {
                BooleanBuffer::new_set(rows)
            } else {
                BooleanBuffer::new_unset(rows)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use arrow_array::{ArrayRef, BooleanArray, Int32Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use planner::expression::{Compare, CompareType, Ref};
    use planner::types::Type;
    use std::sync::Arc;

    /// The filter `column #column_idx > value` over an `Int32` column.
    pub(crate) fn column_greater_than(column_idx: usize, value: i32) -> Expression {
        Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx,
                return_type: Type::Int32,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(
                Arc::new(Int32Array::from(vec![value])) as ArrayRef,
            ))),
            compare_type: CompareType::Greater,
            return_type: Type::Boolean,
        })
    }

    #[test]
    fn a_comparison_of_a_column_is_a_scan_predicate_and_a_constant_alone_is_not() {
        let comparison = column_greater_than(0, 2);
        let constant = Expression::Constant(Scalar::new(
            Arc::new(BooleanArray::from(vec![true])) as ArrayRef
        ));

        let from_comparison = ScanPredicate::from_filter(&comparison);
        let from_constant = ScanPredicate::from_filter(&constant);

        assert!(from_comparison.is_some());
        assert!(from_constant.is_none());
    }

    #[test]
    fn a_predicate_evaluates_on_the_projected_batch_with_nulls_failing() {
        let predicate = ScanPredicate::from_filter(&column_greater_than(1, 2)).unwrap();
        let projection = Projection::columns([1]);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("b", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![Some(1), Some(3), None, Some(5)])) as ArrayRef],
        )
        .unwrap();

        let mut evaluator = predicate.compile_for(&projection).unwrap().evaluator();
        let passing = passing_rows(evaluator(&batch), 4);

        assert_eq!(passing.set_indices().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn a_predicate_reading_an_unprojected_column_does_not_compile() {
        let predicate = ScanPredicate::from_filter(&column_greater_than(3, 2)).unwrap();

        let compiled = predicate.compile_for(&Projection::columns([0, 1]));

        assert!(compiled.is_none());
    }
}
