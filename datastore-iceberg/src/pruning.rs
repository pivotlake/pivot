//! Which manifests and data files a scan can skip: the filters pushed into it,
//! as Iceberg predicates, checked against what the snapshot's metadata says
//! about each one.
//!
//! The filters stay above the scan, so pruning only ever skips work, and
//! everything unknown keeps the object: an expression with no Iceberg form, a
//! column without bounds, a bound that does not compare with the constant.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{ArrayRef, Datum as _, Scalar};
use arrow_schema::{DataType, TimeUnit};
use iceberg::Result;
use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
};
use iceberg::spec::{
    Datum, NestedField, PartitionField, PartitionSpec, PartitionSpecRef, PrimitiveLiteral,
    PrimitiveType, Schema, SchemaRef, Transform, Type as IcebergType,
};
use planner::expression::{CompareType, ConjunctionOp, Expression};
use planner::types::Type;

/// Each value of an `IN` list is compared with every object's bounds, so a
/// longer list is left to the filter above the scan.
const MAX_IN_LIST_VALUES: usize = 200;

/// What `filters` require of a row of `schema`, one predicate per filter that
/// has an Iceberg form the schema accepts. A row passes when all are true.
pub(crate) fn bind_filters(schema: &SchemaRef, filters: &[Expression]) -> Vec<BoundPredicate> {
    filters
        .iter()
        .filter_map(|filter| to_predicate(schema, filter).bind(schema.clone(), true).ok())
        .collect()
}

/// `expression` as an Iceberg predicate: true wherever the expression may be.
fn to_predicate(schema: &Schema, expression: &Expression) -> Predicate {
    match expression {
        Expression::Compare(compare) => compare_with_constant(
            schema,
            &compare.left,
            to_operator(compare.compare_type),
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
            compare_with_constant(schema, input, to_operator(lower), &between.lower).and(
                compare_with_constant(schema, input, to_operator(upper), &between.upper),
            )
        }
        Expression::Conjunction(conjunction) => {
            let children = conjunction
                .children
                .iter()
                .map(|child| to_predicate(schema, child))
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
    let Some(constant) = constant_to_datum(constant) else {
        return Predicate::AlwaysTrue;
    };
    let name = &schema.as_struct().fields()[reference.column_idx].name;
    Predicate::Binary(BinaryExpression::new(
        operator,
        Reference::new(name),
        constant,
    ))
}

fn to_operator(compare_type: CompareType) -> PredicateOperator {
    match compare_type {
        CompareType::Equal => PredicateOperator::Eq,
        CompareType::NotEqual => PredicateOperator::NotEq,
        CompareType::Less => PredicateOperator::LessThan,
        CompareType::LessEqual => PredicateOperator::LessThanOrEq,
        CompareType::Greater => PredicateOperator::GreaterThan,
        CompareType::GreaterEqual => PredicateOperator::GreaterThanOrEq,
    }
}

/// A non-null constant as an Iceberg value of the type it has, or `None` for a
/// type Iceberg columns do not hold.
fn constant_to_datum(constant: &Scalar<ArrayRef>) -> Option<Datum> {
    let array = constant.get().0;
    Some(match array.data_type() {
        DataType::Boolean => Datum::bool(array.as_boolean().value(0)),
        DataType::Int32 => Datum::int(array.as_primitive::<Int32Type>().value(0)),
        DataType::Int64 => Datum::long(array.as_primitive::<Int64Type>().value(0)),
        DataType::Float32 => Datum::float(array.as_primitive::<Float32Type>().value(0)),
        DataType::Float64 => Datum::double(array.as_primitive::<Float64Type>().value(0)),
        DataType::Date32 => Datum::date(array.as_primitive::<Date32Type>().value(0)),
        DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
            let micros = array.as_primitive::<TimestampMicrosecondType>().value(0);
            match timezone {
                Some(_) => Datum::timestamptz_micros(micros),
                None => Datum::timestamp_micros(micros),
            }
        }
        DataType::Decimal64(precision, scale) => {
            let unscaled = array.as_primitive::<Decimal64Type>().value(0);
            decimal_to_datum(i128::from(unscaled), *precision, *scale)?
        }
        DataType::Decimal128(precision, scale) => {
            let unscaled = array.as_primitive::<Decimal128Type>().value(0);
            decimal_to_datum(unscaled, *precision, *scale)?
        }
        DataType::Utf8View => Datum::string(array.as_string_view().value(0)),
        _ => return None,
    })
}

fn decimal_to_datum(unscaled: i128, precision: u8, scale: i8) -> Option<Datum> {
    let decimal = PrimitiveType::Decimal {
        precision: u32::from(precision),
        scale: u32::try_from(scale).ok()?,
    };
    Datum::try_from_bytes(&unscaled.to_be_bytes(), decimal).ok()
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
/// tuple written under each of `partition_specs`, by spec id.
pub(crate) fn project_filters_per_spec(
    schema: &Schema,
    partition_specs: &[PartitionSpecRef],
    row_filters: &[BoundPredicate],
) -> Result<HashMap<i32, BoundPredicate>> {
    let mut partition_filters = HashMap::new();
    for partition_spec in partition_specs {
        let partition_filter = project_filters(schema, partition_spec, row_filters)?;
        partition_filters.insert(partition_spec.spec_id(), partition_filter);
    }
    Ok(partition_filters)
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

/// What metadata says about the values of one field in one object.
pub(crate) struct FieldRange {
    pub field_id: i32,
    /// Every non-null value is at least this. `None` where unknown.
    pub lower: Option<Datum>,
    /// Every non-null value is at most this. `None` where unknown.
    pub upper: Option<Datum>,
    /// Whether every value is NULL.
    pub all_null: bool,
}

impl FieldRange {
    /// Whether a value in the range may satisfy `<operator> constant`.
    fn may_satisfy(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        if self.all_null {
            return false;
        }
        let proves = |bound: &Option<Datum>, holds: fn(Ordering) -> bool| {
            bound
                .as_ref()
                .and_then(|bound| bound.partial_cmp(constant))
                .is_some_and(holds)
        };
        let excluded = match operator {
            PredicateOperator::Eq => {
                proves(&self.lower, Ordering::is_gt) || proves(&self.upper, Ordering::is_lt)
            }
            // Every value equals the constant only when both bounds do.
            PredicateOperator::NotEq => {
                proves(&self.lower, Ordering::is_eq) && proves(&self.upper, Ordering::is_eq)
            }
            PredicateOperator::LessThan => proves(&self.lower, Ordering::is_ge),
            PredicateOperator::LessThanOrEq => proves(&self.lower, Ordering::is_gt),
            PredicateOperator::GreaterThan => proves(&self.upper, Ordering::is_le),
            PredicateOperator::GreaterThanOrEq => proves(&self.upper, Ordering::is_lt),
            _ => false,
        };
        !excluded
    }
}

/// Whether an object may hold something `predicate` is true for. `ranges` is
/// what the object's metadata knows about the fields the predicate compares; a
/// field without a range is unknown.
pub(crate) fn may_match(predicate: &BoundPredicate, ranges: &[FieldRange]) -> bool {
    let find_range = |field: &NestedField| ranges.iter().find(|range| range.field_id == field.id);
    match predicate {
        BoundPredicate::AlwaysFalse => false,
        BoundPredicate::And(children) => {
            let [left, right] = children.inputs();
            may_match(left, ranges) && may_match(right, ranges)
        }
        BoundPredicate::Or(children) => {
            let [left, right] = children.inputs();
            may_match(left, ranges) || may_match(right, ranges)
        }
        BoundPredicate::Binary(comparison) => find_range(comparison.term().field())
            .is_none_or(|range| range.may_satisfy(comparison.op(), comparison.literal())),
        // A field is in the set when it equals one of its values.
        BoundPredicate::Set(set) if set.op() == PredicateOperator::In => {
            find_range(set.term().field()).is_none_or(|range| {
                set.literals()
                    .iter()
                    .any(|value| range.may_satisfy(PredicateOperator::Eq, value))
            })
        }
        _ => true,
    }
}
