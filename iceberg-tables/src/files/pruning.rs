//! Which manifests and data files a scan can skip: the filters pushed into it,
//! as Iceberg predicates, checked against what the snapshot's metadata says
//! about each one.
//!
//! The filters stay above the scan, so pruning only ever skips work, and
//! everything unknown keeps the object: an expression with no Iceberg form, a
//! column without bounds, a bound that does not compare with the constant.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::Datum as _;
use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
};
use iceberg::spec::{
    Datum, Literal, NestedField, PartitionField, PartitionSpec, PrimitiveLiteral, PrimitiveType,
    Schema, SchemaRef, Struct, Transform, Type as IcebergType,
};
use iceberg::{ErrorKind, Result};
use planner::expression::{CompareType, ConjunctionOp, Expression};
use planner::types::Type;

use crate::types::datum::constant_to_datum;

/// Each value of an `IN` list is compared with every object's bounds, so a
/// longer list is left to the filter above the scan.
const MAX_IN_LIST_VALUES: usize = 200;

/// What `filters` require of a row of `schema`, one predicate per filter that
/// has an Iceberg form the schema accepts. A row passes when all are true.
pub(crate) fn bind_filters(schema: &SchemaRef, filters: &[Expression]) -> Vec<BoundPredicate> {
    filters
        .iter()
        .filter_map(|filter| {
            build_predicate(schema, filter)
                .bind(schema.clone(), true)
                .ok()
        })
        .collect()
}

/// `expression` as an Iceberg predicate: true wherever the expression may be.
fn build_predicate(schema: &Schema, expression: &Expression) -> Predicate {
    match expression {
        Expression::Compare(compare) => compare_with_constant(
            schema,
            &compare.left,
            map_operator(compare.compare_type),
            &compare.right,
        ),
        Expression::Between(between) => {
            let (lower, upper) = match (between.lower_inclusive, between.upper_inclusive) {
                (true, true) => (CompareType::GreaterEqual, CompareType::LessEqual),
                (true, false) => (CompareType::GreaterEqual, CompareType::Less),
                (false, true) => (CompareType::Greater, CompareType::LessEqual),
                (false, false) => (CompareType::Greater, CompareType::Less),
            };
            let input = &between.input;
            compare_with_constant(schema, input, map_operator(lower), &between.lower).and(
                compare_with_constant(schema, input, map_operator(upper), &between.upper),
            )
        }
        Expression::Conjunction(conjunction) => {
            let children = conjunction
                .children
                .iter()
                .map(|child| build_predicate(schema, child))
                .collect();
            match conjunction.op {
                ConjunctionOp::And => combine(children, Predicate::AlwaysTrue, Predicate::and),
                ConjunctionOp::Or => combine(children, Predicate::AlwaysFalse, Predicate::or),
            }
        }
        // A row is in the list when it equals one of its values.
        Expression::InList(list) if list.values.len() <= MAX_IN_LIST_VALUES => {
            let equalities = list
                .values
                .iter()
                .map(|value| {
                    compare_with_constant(schema, &list.input, PredicateOperator::Eq, value)
                })
                .collect();
            combine(equalities, Predicate::AlwaysFalse, Predicate::or)
        }
        _ => Predicate::AlwaysTrue,
    }
}

/// `predicates` joined pairwise by `join` into one tree, `of_none` when there
/// are none. The tree is as shallow as their number allows: a chain as long as
/// a wide `OR` would overflow the stack of whatever walks it.
fn combine(
    mut predicates: Vec<Predicate>,
    of_none: Predicate,
    join: fn(Predicate, Predicate) -> Predicate,
) -> Predicate {
    while predicates.len() > 1 {
        let mut joined = Vec::with_capacity(predicates.len().div_ceil(2));
        let mut remaining = predicates.into_iter();
        while let Some(left) = remaining.next() {
            joined.push(match remaining.next() {
                Some(right) => join(left, right),
                None => left,
            });
        }
        predicates = joined;
    }
    predicates.pop().unwrap_or(of_none)
}

/// `column <operator> constant`, when `column` reads a column of the table and
/// `constant` is one. DuckDB normally puts the column on the left; a constant
/// on the left is not one.
fn compare_with_constant(
    schema: &Schema,
    column: &Expression,
    operator: PredicateOperator,
    constant: &Expression,
) -> Predicate {
    // DuckDB casts a TIMESTAMP column to TIMESTAMPTZ to compare it with one.
    // In UTC the cast keeps the microsecond value, so the comparison is the
    // column's own, and binding reads the constant as a TIMESTAMP.
    let column = match column {
        Expression::Cast(cast)
            if cast.target == Type::TimestampTz
                && matches!(
                    cast.source(),
                    Expression::Ref(reference) if reference.return_type == Type::Timestamp
                ) =>
        {
            cast.source()
        }
        column => column,
    };
    let (Expression::Ref(reference), Expression::Constant(constant)) = (column, constant) else {
        return Predicate::AlwaysTrue;
    };
    // A comparison with NULL is never true.
    if constant.get().0.is_null(0) {
        return Predicate::AlwaysFalse;
    }
    // The constant is read as its own type, not the column's: the planner
    // widens a constant beyond the column's range rather than refuse it, and
    // the evaluator compares the bounds across that widening.
    let Some(constant) = constant_to_datum(constant.get().0) else {
        return Predicate::AlwaysTrue;
    };
    let name = &schema.as_struct().fields()[reference.column_idx].name;
    Predicate::Binary(BinaryExpression::new(
        operator,
        Reference::new(name),
        constant,
    ))
}

fn map_operator(compare_type: CompareType) -> PredicateOperator {
    match compare_type {
        CompareType::Equal => PredicateOperator::Eq,
        CompareType::NotEqual => PredicateOperator::NotEq,
        CompareType::Less => PredicateOperator::LessThan,
        CompareType::LessEqual => PredicateOperator::LessThanOrEq,
        CompareType::Greater => PredicateOperator::GreaterThan,
        CompareType::GreaterEqual => PredicateOperator::GreaterThanOrEq,
    }
}

/// `literal` as a value of `data_type`: the type it was written as, or the one
/// that type has since been promoted to. `None` for a pair with no such
/// reading.
pub(crate) fn literal_to_datum(
    literal: &PrimitiveLiteral,
    data_type: &PrimitiveType,
) -> Option<Datum> {
    Some(match (literal, data_type) {
        (PrimitiveLiteral::Boolean(value), PrimitiveType::Boolean) => Datum::bool(*value),
        (PrimitiveLiteral::Int(value), PrimitiveType::Int) => Datum::int(*value),
        (PrimitiveLiteral::Int(value), PrimitiveType::Date) => Datum::date(*value),
        (PrimitiveLiteral::Int(value), PrimitiveType::Long) => Datum::long(*value),
        (PrimitiveLiteral::Long(value), PrimitiveType::Long) => Datum::long(*value),
        (PrimitiveLiteral::Long(value), PrimitiveType::Timestamp) => {
            Datum::timestamp_micros(*value)
        }
        (PrimitiveLiteral::Long(value), PrimitiveType::Timestamptz) => {
            Datum::timestamptz_micros(*value)
        }
        (PrimitiveLiteral::Float(value), PrimitiveType::Float) => Datum::float(value.0),
        (PrimitiveLiteral::Float(value), PrimitiveType::Double) => Datum::double(value.0),
        (PrimitiveLiteral::Double(value), PrimitiveType::Double) => Datum::double(value.0),
        (PrimitiveLiteral::String(value), PrimitiveType::String) => Datum::string(value),
        (PrimitiveLiteral::Int128(unscaled), PrimitiveType::Decimal { .. }) => {
            Datum::try_from_bytes(&unscaled.to_be_bytes(), data_type.clone()).ok()?
        }
        _ => return None,
    })
}

/// The type of the values of the partition `field`, when they can prune: the
/// field has a transform with known values, of a column `schema` still has.
pub(crate) fn find_partition_value_type(
    schema: &Schema,
    field: &PartitionField,
) -> Option<PrimitiveType> {
    if matches!(field.transform, Transform::Void | Transform::Unknown) {
        return None;
    }
    let source = schema.field_by_id(field.source_id)?;
    let value_type = field.transform.result_type(&source.field_type).ok()?;
    value_type.as_primitive_type().cloned()
}

/// What `row_filters`, over the columns of `schema`, require of a partition
/// tuple written under `partition_spec`: a row that passes them has a tuple
/// that passes this. It is over the fields of the spec whose values can prune.
fn project_filters(
    schema: &Schema,
    partition_spec: &PartitionSpec,
    row_filters: &[BoundPredicate],
) -> Result<BoundPredicate> {
    // Binding needs the partition fields as a schema, each with the type of
    // its values.
    let partition_fields = partition_spec.fields().iter().filter_map(|field| {
        let value_type = find_partition_value_type(schema, field)?;
        Some(Arc::new(NestedField::optional(
            field.field_id,
            &field.name,
            IcebergType::Primitive(value_type),
        )))
    });
    let partition_schema = Arc::new(Schema::builder().with_fields(partition_fields).build()?);
    let mut projected = Predicate::AlwaysTrue;
    for row_filter in row_filters {
        projected = projected.and(project(partition_spec, &partition_schema, row_filter));
    }
    projected.bind(partition_schema, true)
}

/// What `predicate`, over the table's columns, requires of the fields of
/// `partition_schema`.
fn project(
    partition_spec: &PartitionSpec,
    partition_schema: &SchemaRef,
    predicate: &BoundPredicate,
) -> Predicate {
    let source_id = match predicate {
        BoundPredicate::AlwaysFalse => return Predicate::AlwaysFalse,
        BoundPredicate::And(children) => {
            let [left, right] = children.inputs();
            return project(partition_spec, partition_schema, left).and(project(
                partition_spec,
                partition_schema,
                right,
            ));
        }
        BoundPredicate::Or(children) => {
            let [left, right] = children.inputs();
            return project(partition_spec, partition_schema, left).or(project(
                partition_spec,
                partition_schema,
                right,
            ));
        }
        BoundPredicate::Binary(comparison) => comparison.term().field().id,
        _ => return Predicate::AlwaysTrue,
    };
    // Every field derived from the column constrains the tuple on its own. A
    // requirement that cannot be derived, or whose constant the field's type
    // does not hold, constrains nothing.
    let mut projected = Predicate::AlwaysTrue;
    for field in partition_spec.fields() {
        if field.source_id != source_id || partition_schema.field_by_id(field.field_id).is_none() {
            continue;
        }
        let requirement = field
            .transform
            .project(&field.name, predicate)
            .ok()
            .flatten();
        if let Some(requirement) = requirement
            && requirement.bind(partition_schema.clone(), true).is_ok()
        {
            projected = projected.and(requirement);
        }
    }
    projected
}

/// The fields `predicates` compare with constants, each once.
pub(crate) fn list_compared_fields(predicates: &[BoundPredicate]) -> Vec<&NestedField> {
    let mut fields = Vec::new();
    for predicate in predicates {
        collect_compared_fields(predicate, &mut fields);
    }
    fields
}

fn collect_compared_fields<'a>(predicate: &'a BoundPredicate, fields: &mut Vec<&'a NestedField>) {
    let field = match predicate {
        BoundPredicate::And(children) | BoundPredicate::Or(children) => {
            for child in children.inputs() {
                collect_compared_fields(child, fields);
            }
            return;
        }
        BoundPredicate::Binary(comparison) => comparison.term().field(),
        BoundPredicate::Set(set) => set.term().field(),
        _ => return,
    };
    if fields.iter().all(|compared| compared.id != field.id) {
        fields.push(field);
    }
}

/// What metadata says about the values of one field in one object, in a form
/// that tells whether one of them may satisfy a comparison.
pub(crate) trait ValueRange {
    fn field_id(&self) -> i32;

    /// Whether a value in the range may satisfy `<operator> constant`.
    fn may_match_comparison(&self, operator: PredicateOperator, constant: &Datum) -> bool;
}

/// Whether a value between two bounds may satisfy `<operator> constant`.
/// `compare_lower` and `compare_upper` order the bound of their side against
/// the constant; `None` for a bound that is unknown or does not compare with
/// it.
pub(crate) fn may_bounds_match_comparison(
    operator: PredicateOperator,
    compare_lower: impl Fn() -> Option<Ordering>,
    compare_upper: impl Fn() -> Option<Ordering>,
) -> bool {
    let is_lower = |ordered: fn(Ordering) -> bool| compare_lower().is_some_and(ordered);
    let is_upper = |ordered: fn(Ordering) -> bool| compare_upper().is_some_and(ordered);
    let excluded = match operator {
        PredicateOperator::Eq => is_lower(Ordering::is_gt) || is_upper(Ordering::is_lt),
        // Every value equals the constant only when both bounds do.
        PredicateOperator::NotEq => is_lower(Ordering::is_eq) && is_upper(Ordering::is_eq),
        PredicateOperator::LessThan => is_lower(Ordering::is_ge),
        PredicateOperator::LessThanOrEq => is_lower(Ordering::is_gt),
        PredicateOperator::GreaterThan => is_upper(Ordering::is_le),
        PredicateOperator::GreaterThanOrEq => is_upper(Ordering::is_lt),
        _ => false,
    };
    !excluded
}

/// A range whose bounds are held as values.
#[derive(Clone)]
pub(crate) struct FieldRange {
    pub field_id: i32,
    /// Every non-null value is at least this. `None` where unknown.
    pub lower: Option<Datum>,
    /// Every non-null value is at most this. `None` where unknown.
    pub upper: Option<Datum>,
    /// Whether every value is NULL.
    pub all_null: bool,
}

impl ValueRange for FieldRange {
    fn field_id(&self) -> i32 {
        self.field_id
    }

    fn may_match_comparison(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        if self.all_null {
            return false;
        }
        let compare = |bound: &Option<Datum>| bound.as_ref()?.partial_cmp(constant);
        may_bounds_match_comparison(operator, || compare(&self.lower), || compare(&self.upper))
    }
}

/// How the files written under one schema record a column that filters
/// compare, resolved once for all of them.
#[derive(Clone)]
pub(crate) struct ComparedColumn {
    field_id: i32,
    /// The column's type in the table's current schema, which the filters'
    /// constants are of.
    column_type: PrimitiveType,
    /// Whether the files have the column.
    is_written: bool,
    /// Whether the files' bounds for the column are of a known type: the
    /// table lists the schema they were written under, and the column is
    /// primitive in it.
    is_bounded: bool,
}

impl ComparedColumn {
    /// `column`, a column of the table's current schema that a filter
    /// compares with a constant, as the files written under `file_schema`
    /// record it; `None` for a schema the table no longer lists, whose files
    /// are taken to have every column, in a type that is not known.
    pub(crate) fn resolve(column: &NestedField, file_schema: Option<&Schema>) -> Self {
        let column_type = column
            .field_type
            .as_primitive_type()
            .expect("a column compared with a constant is primitive")
            .clone();
        let (is_written, is_bounded) = match file_schema {
            None => (true, false),
            Some(file_schema) => match file_schema.field_by_id(column.id) {
                None => (false, false),
                Some(stored_field) => (true, stored_field.field_type.is_primitive()),
            },
        };
        Self {
            field_id: column.id,
            column_type,
            is_written,
            is_bounded,
        }
    }

    pub(crate) fn field_id(&self) -> i32 {
        self.field_id
    }

    /// Whether the files have the column. Initial defaults are rejected on
    /// load: a column the files' schema lacks is NULL in every row.
    pub(crate) fn is_written(&self) -> bool {
        self.is_written
    }

    /// How every bound of `bound_type` recorded for the column orders
    /// against `constant`, found once for all of them. The bounds order
    /// against it when the constant is of the column's type and they were
    /// recorded in that type, or in one it is a widening of, and values of
    /// the type order as the arrays that hold the bounds do; any other
    /// bounds are [`BoundComparison::Unknown`].
    pub(crate) fn plan_comparison<'k>(
        &self,
        bound_type: &PrimitiveType,
        constant: &'k Datum,
    ) -> BoundComparison<'k> {
        if !self.is_bounded || constant.data_type() != &self.column_type {
            return BoundComparison::Unknown;
        }
        match (bound_type, &self.column_type, constant.literal()) {
            (
                PrimitiveType::Boolean,
                PrimitiveType::Boolean,
                PrimitiveLiteral::Boolean(constant),
            ) => BoundComparison::Boolean(*constant),
            (PrimitiveType::Int, PrimitiveType::Int, PrimitiveLiteral::Int(constant))
            | (PrimitiveType::Date, PrimitiveType::Date, PrimitiveLiteral::Int(constant)) => {
                BoundComparison::Integer(i64::from(*constant))
            }
            (PrimitiveType::Long, PrimitiveType::Long, PrimitiveLiteral::Long(constant))
            | (
                PrimitiveType::Timestamp,
                PrimitiveType::Timestamp,
                PrimitiveLiteral::Long(constant),
            )
            | (
                PrimitiveType::Timestamptz,
                PrimitiveType::Timestamptz,
                PrimitiveLiteral::Long(constant),
            )
            // A bound recorded before the column was widened.
            | (PrimitiveType::Int, PrimitiveType::Long, PrimitiveLiteral::Long(constant)) => {
                BoundComparison::Integer(*constant)
            }
            (PrimitiveType::Float, PrimitiveType::Float, PrimitiveLiteral::Float(constant)) => {
                BoundComparison::Float(constant.0)
            }
            (PrimitiveType::Float, PrimitiveType::Double, PrimitiveLiteral::Double(constant)) => {
                BoundComparison::FloatAsDouble(constant.0)
            }
            (PrimitiveType::Double, PrimitiveType::Double, PrimitiveLiteral::Double(constant)) => {
                BoundComparison::Double(constant.0)
            }
            (PrimitiveType::String, PrimitiveType::String, PrimitiveLiteral::String(constant)) => {
                BoundComparison::String(constant)
            }
            // The constant is of the column's scale, and a decimal bound
            // holds its unscaled value whatever precision it was recorded
            // under, so bounds recorded at that scale order as their
            // unscaled values do.
            (
                PrimitiveType::Decimal {
                    scale: bound_scale, ..
                },
                PrimitiveType::Decimal { scale, .. },
                PrimitiveLiteral::Int128(constant),
            ) if bound_scale == scale => BoundComparison::Decimal(*constant),
            // Bounds of a `time`, a nanosecond timestamp, a `uuid`, a
            // `fixed` or a `binary` column, and bounds recorded in a type
            // the column's is not a widening of.
            _ => BoundComparison::Unknown,
        }
    }
}

/// How the bounds a manifest records for a column, all of one type, order
/// against one constant.
pub(crate) enum BoundComparison<'k> {
    /// The bounds say nothing of how the column's values order against
    /// the constant: they rule no file out.
    Unknown,
    /// A boolean bound orders against this one, `false` before `true`.
    Boolean(bool),
    /// A bound that is an integer, a date or a timestamp orders as its
    /// value does against this one.
    Integer(i64),
    /// A `float` bound orders against this one as IEEE 754's total order
    /// has them.
    Float(f32),
    /// A `float` bound, widened, orders against this `double` as IEEE 754's
    /// total order has them.
    FloatAsDouble(f64),
    /// A `double` bound orders against this one as IEEE 754's total order
    /// has them.
    Double(f64),
    /// A decimal bound orders as its unscaled value does against this one,
    /// the unscaled value of a decimal of the bounds' scale.
    Decimal(i128),
    /// A string bound orders against this one by its bytes.
    String(&'k str),
}

/// Whether an object may hold something `predicate` is true for. `ranges` is
/// what the object's metadata knows about the fields the predicate compares; a
/// field without a range is unknown.
pub(crate) fn may_match_predicate<Range: ValueRange>(
    predicate: &BoundPredicate,
    ranges: &[Range],
) -> bool {
    let find_range = |field: &NestedField| ranges.iter().find(|range| range.field_id() == field.id);
    match predicate {
        BoundPredicate::AlwaysFalse => false,
        BoundPredicate::And(children) => {
            let [left, right] = children.inputs();
            may_match_predicate(left, ranges) && may_match_predicate(right, ranges)
        }
        BoundPredicate::Or(children) => {
            let [left, right] = children.inputs();
            may_match_predicate(left, ranges) || may_match_predicate(right, ranges)
        }
        BoundPredicate::Binary(comparison) => find_range(comparison.term().field())
            .is_none_or(|range| range.may_match_comparison(comparison.op(), comparison.literal())),
        // A field is in the set when it equals one of its values.
        BoundPredicate::Set(set) if set.op() == PredicateOperator::In => {
            find_range(set.term().field()).is_none_or(|range| {
                set.literals()
                    .iter()
                    .any(|value| range.may_match_comparison(PredicateOperator::Eq, value))
            })
        }
        _ => true,
    }
}

/// A field of a partition spec whose values can prune.
struct PartitionValueField {
    /// Where a tuple of the spec holds the field's value.
    position: usize,
    field_id: i32,
    value_type: PrimitiveType,
}

/// What a scan's filters require of the files written under one partition
/// spec, resolved once for all of them.
pub(crate) struct PartitionPruner {
    /// What the filters require of a tuple of the spec.
    filter: BoundPredicate,
    /// The fields `filter` compares.
    compared_fields: Vec<PartitionValueField>,
    /// How many values a tuple of the spec has at least: one for every field
    /// up to the last whose values can prune.
    tuple_length: usize,
}

impl PartitionPruner {
    /// What `row_filters`, over the columns of `schema`, require of the files
    /// written under `partition_spec`.
    pub(crate) fn new(
        schema: &Schema,
        partition_spec: &PartitionSpec,
        row_filters: &[BoundPredicate],
    ) -> Result<Self> {
        let filter = project_filters(schema, partition_spec, row_filters)?;
        let mut prunable_fields = Vec::new();
        for (position, field) in partition_spec.fields().iter().enumerate() {
            if let Some(value_type) = find_partition_value_type(schema, field) {
                prunable_fields.push(PartitionValueField {
                    position,
                    field_id: field.field_id,
                    value_type,
                });
            }
        }
        let tuple_length = prunable_fields.last().map_or(0, |field| field.position + 1);
        let compared = list_compared_fields(std::slice::from_ref(&filter));
        prunable_fields.retain(|field| {
            compared
                .iter()
                .any(|compared| compared.id == field.field_id)
        });
        Ok(Self {
            filter,
            compared_fields: prunable_fields,
            tuple_length,
        })
    }

    /// What the filters require of a tuple of the spec.
    pub(crate) fn partition_filter(&self) -> &BoundPredicate {
        &self.filter
    }

    /// Whether a file of the partition whose tuple is `partition` may hold
    /// a row the filters keep. Every row of such a file shares a field's
    /// value, so the value is both of the field's bounds. `values` is
    /// scratch space the caller keeps from one partition to the next.
    pub(crate) fn may_partition_match(
        &self,
        partition: &Struct,
        values: &mut Vec<FieldRange>,
    ) -> Result<bool> {
        let partition = partition.fields();
        if partition.len() < self.tuple_length {
            return Err(iceberg::Error::new(
                ErrorKind::DataInvalid,
                "file partition does not match its spec",
            ));
        }
        values.clear();
        for field in &self.compared_fields {
            match &partition[field.position] {
                Some(Literal::Primitive(value)) => {
                    let value = literal_to_datum(value, &field.value_type);
                    values.push(FieldRange {
                        field_id: field.field_id,
                        lower: value.clone(),
                        upper: value,
                        all_null: false,
                    });
                }
                Some(_) => {}
                // A NULL partition value means the column is NULL in every row.
                None => values.push(FieldRange {
                    field_id: field.field_id,
                    lower: None,
                    upper: None,
                    all_null: true,
                }),
            }
        }
        Ok(may_match_predicate(&self.filter, values))
    }
}
