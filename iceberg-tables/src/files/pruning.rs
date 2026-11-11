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

use arrow_array::Datum as _;
use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
};
use iceberg::spec::{
    DataFile, Datum, Literal, NestedField, PartitionField, PartitionSpec, PartitionSpecRef,
    PrimitiveLiteral, PrimitiveType, Schema, SchemaRef, Struct, Transform, Type as IcebergType,
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

/// What metadata says about the values of one field in one object, in a form
/// that tells whether one of them may satisfy a comparison.
pub(crate) trait ValueRange {
    fn field_id(&self) -> i32;

    /// Whether a value in the range may satisfy `<operator> constant`.
    fn may_satisfy(&self, operator: PredicateOperator, constant: &Datum) -> bool;
}

/// Whether a value between two bounds may satisfy `<operator> constant`.
/// `compare_lower` and `compare_upper` order the bound of their side against
/// the constant; `None` for a bound that is unknown or does not compare with
/// it.
fn may_bounds_satisfy(
    operator: PredicateOperator,
    compare_lower: impl Fn() -> Option<Ordering>,
    compare_upper: impl Fn() -> Option<Ordering>,
) -> bool {
    let lower = |holds: fn(Ordering) -> bool| compare_lower().is_some_and(holds);
    let upper = |holds: fn(Ordering) -> bool| compare_upper().is_some_and(holds);
    let excluded = match operator {
        PredicateOperator::Eq => lower(Ordering::is_gt) || upper(Ordering::is_lt),
        // Every value equals the constant only when both bounds do.
        PredicateOperator::NotEq => lower(Ordering::is_eq) && upper(Ordering::is_eq),
        PredicateOperator::LessThan => lower(Ordering::is_ge),
        PredicateOperator::LessThanOrEq => lower(Ordering::is_gt),
        PredicateOperator::GreaterThan => upper(Ordering::is_le),
        PredicateOperator::GreaterThanOrEq => upper(Ordering::is_lt),
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

    fn may_satisfy(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        if self.all_null {
            return false;
        }
        let compare = |bound: &Option<Datum>| bound.as_ref()?.partial_cmp(constant);
        may_bounds_satisfy(operator, || compare(&self.lower), || compare(&self.upper))
    }
}

/// How the files written under one schema record a column that filters
/// compare, resolved once for all of them.
#[derive(Clone)]
pub(crate) struct StoredColumn {
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

impl StoredColumn {
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

    /// What `data`, the manifest entry of a file, says about the column's
    /// values in the file.
    pub(crate) fn read_range<'a>(&'a self, data: &'a DataFile) -> StoredRange<'a> {
        let null_count = data.null_value_counts().get(&self.field_id);
        StoredRange {
            column: self,
            lower: data.lower_bounds().get(&self.field_id),
            upper: data.upper_bounds().get(&self.field_id),
            // Initial defaults are rejected on load: a column the file's
            // schema lacks is NULL in every row.
            all_null: !self.is_written || null_count == Some(&data.record_count()),
        }
    }

    /// How `bound`, recorded for the column, orders against `constant`;
    /// `None` when the bound's type is not known, or its value read as the
    /// column's type does not compare with the constant.
    fn compare_bound(&self, bound: &Datum, constant: &Datum) -> Option<Ordering> {
        if !self.is_bounded {
            return None;
        }
        // A bound recorded in the column's own type compares as it is; one
        // recorded before the column was widened is read as the column's.
        if bound.data_type() == &self.column_type && is_read_unchanged(&self.column_type) {
            return bound.partial_cmp(constant);
        }
        literal_to_datum(bound.literal(), &self.column_type)?.partial_cmp(constant)
    }
}

/// Whether [`literal_to_datum`] reads a value of `data_type` as that type,
/// giving the value back.
fn is_read_unchanged(data_type: &PrimitiveType) -> bool {
    matches!(
        data_type,
        PrimitiveType::Boolean
            | PrimitiveType::Int
            | PrimitiveType::Date
            | PrimitiveType::Long
            | PrimitiveType::Timestamp
            | PrimitiveType::Timestamptz
            | PrimitiveType::Float
            | PrimitiveType::Double
            | PrimitiveType::String
            | PrimitiveType::Decimal { .. }
    )
}

/// A range whose bounds are the values a file's manifest entry holds.
pub(crate) struct StoredRange<'a> {
    column: &'a StoredColumn,
    /// Every non-null value is at least this. `None` where unknown.
    lower: Option<&'a Datum>,
    /// Every non-null value is at most this. `None` where unknown.
    upper: Option<&'a Datum>,
    /// Whether every value is NULL.
    all_null: bool,
}

impl ValueRange for StoredRange<'_> {
    fn field_id(&self) -> i32 {
        self.column.field_id
    }

    fn may_satisfy(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        if self.all_null {
            return false;
        }
        let compare = |bound: Option<&Datum>| self.column.compare_bound(bound?, constant);
        may_bounds_satisfy(operator, || compare(self.lower), || compare(self.upper))
    }
}

/// Whether an object may hold something `predicate` is true for. `ranges` is
/// what the object's metadata knows about the fields the predicate compares; a
/// field without a range is unknown.
pub(crate) fn may_match<Range: ValueRange>(predicate: &BoundPredicate, ranges: &[Range]) -> bool {
    let find_range = |field: &NestedField| ranges.iter().find(|range| range.field_id() == field.id);
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

    /// Whether a file whose tuple is `partition` may hold a row the filters
    /// keep. Every row of the file shares a field's value, so the value is
    /// both of the field's bounds. `values` is scratch space the caller keeps
    /// from one file to the next.
    pub(crate) fn may_match(
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
        Ok(may_match(&self.filter, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat};

    /// Every primitive type a bound may be recorded in or a column may be of.
    fn list_primitive_types() -> Vec<PrimitiveType> {
        vec![
            PrimitiveType::Boolean,
            PrimitiveType::Int,
            PrimitiveType::Long,
            PrimitiveType::Float,
            PrimitiveType::Double,
            PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            },
            PrimitiveType::Date,
            PrimitiveType::Time,
            PrimitiveType::Timestamp,
            PrimitiveType::Timestamptz,
            PrimitiveType::TimestampNs,
            PrimitiveType::TimestamptzNs,
            PrimitiveType::String,
            PrimitiveType::Uuid,
            PrimitiveType::Fixed(4),
            PrimitiveType::Binary,
        ]
    }

    /// Values at the edges of their types, as bounds recorded in every type
    /// and as constants a filter compares a column with.
    fn list_values() -> Vec<Datum> {
        let mut values = vec![Datum::bool(false), Datum::bool(true)];
        for value in [0, 1, -1, 7, i32::MIN, i32::MAX] {
            values.extend([Datum::int(value), Datum::date(value)]);
        }
        for value in [0, 1, -1, 7, i64::from(i32::MAX) + 1, i64::MIN, i64::MAX] {
            values.extend([
                Datum::long(value),
                Datum::timestamp_micros(value),
                Datum::timestamptz_micros(value),
                Datum::timestamp_nanos(value),
            ]);
        }
        for value in [0.0, -0.0, 1.5, -1.5, f32::NAN, f32::MAX] {
            values.extend([Datum::float(value), Datum::double(f64::from(value))]);
        }
        values.push(Datum::double(1.00000005));
        for text in ["", "a", "ab", "abd", "b", "é", "zzzzzzzz"] {
            values.extend([Datum::string(text), Datum::binary(text.bytes())]);
        }
        for decimal in ["0.00", "1.50", "-1.50", "0.007", "184467440737095516.16"] {
            values.push(Datum::decimal_from_str(decimal).unwrap());
        }
        values
    }

    /// The column 1 of `column_type` as files written while it was of
    /// `stored_type` record it.
    fn resolve_column(stored_type: &PrimitiveType, column_type: &PrimitiveType) -> StoredColumn {
        let schema_of = |primitive: &PrimitiveType| {
            let field =
                NestedField::optional(1, "value", IcebergType::Primitive(primitive.clone()));
            Schema::builder()
                .with_fields(vec![Arc::new(field)])
                .build()
                .unwrap()
        };
        let current = schema_of(column_type);
        StoredColumn::resolve(
            current.field_by_id(1).unwrap(),
            Some(&schema_of(stored_type)),
        )
    }

    /// A nine-row file whose manifest entry records, for field 1, no NULL
    /// and `bound` on both sides.
    fn file_bounded_by(bound: Datum) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("file.parquet".into())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(9)
            .null_value_counts(HashMap::from([(1, 0)]))
            .lower_bounds(HashMap::from([(1, bound.clone())]))
            .upper_bounds(HashMap::from([(1, bound)]))
            .build()
            .unwrap()
    }

    #[test]
    fn a_bound_orders_against_a_constant_as_its_value_read_in_the_columns_type_does() {
        let types = list_primitive_types();
        let values = list_values();

        let mut compared = 0;
        for column_type in &types {
            for bound in &values {
                let column = resolve_column(bound.data_type(), column_type);
                for constant in &values {
                    let ordering = column.compare_bound(bound, constant);

                    let read = literal_to_datum(bound.literal(), column_type);
                    assert_eq!(
                        ordering,
                        read.and_then(|read| read.partial_cmp(constant)),
                        "{bound:?} read as {column_type}, against {constant:?}"
                    );
                    if ordering.is_some() {
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000, "{compared}");
    }

    #[test]
    fn a_bound_orders_as_values_of_its_type_do() {
        // A bound as it was recorded, against a constant of the column's type.
        let micros = Datum::timestamp_micros;
        let cases = [
            (Datum::bool(false), Datum::bool(true), Ordering::Less),
            (Datum::int(-1), Datum::int(1), Ordering::Less),
            (Datum::date(-400), Datum::date(-401), Ordering::Greater),
            (Datum::long(i64::MIN), Datum::long(0), Ordering::Less),
            (micros(-1), micros(-1), Ordering::Equal),
            (micros(256), micros(1), Ordering::Greater),
            (
                Datum::float(-0.0_f32),
                Datum::float(0.0_f32),
                Ordering::Less,
            ),
            (
                Datum::float(f32::NAN),
                Datum::float(f32::MAX),
                Ordering::Greater,
            ),
            (Datum::double(-1.5), Datum::double(-1.0), Ordering::Less),
            (Datum::string("a"), Datum::string("ab"), Ordering::Less),
            (Datum::string("é"), Datum::string("z"), Ordering::Greater),
            // An upper bound cut short and rounded up is above what it was
            // cut from.
            (
                Datum::string("abd"),
                Datum::string("abcde"),
                Ordering::Greater,
            ),
            // Bounds recorded before the column was widened.
            (Datum::int(-5), Datum::long(1_i64 << 40), Ordering::Less),
            (
                Datum::float(1.0_f32),
                Datum::double(1.00000005),
                Ordering::Less,
            ),
        ];

        for (bound, constant, expected) in cases {
            let column = resolve_column(bound.data_type(), constant.data_type());

            let ordering = column.compare_bound(&bound, &constant);

            assert_eq!(ordering, Some(expected), "{bound} against {constant}");
        }
    }

    #[test]
    fn a_range_of_stored_bounds_satisfies_what_the_same_range_of_values_does() {
        let operators = [
            PredicateOperator::Eq,
            PredicateOperator::NotEq,
            PredicateOperator::LessThan,
            PredicateOperator::LessThanOrEq,
            PredicateOperator::GreaterThan,
            PredicateOperator::GreaterThanOrEq,
        ];
        let column = resolve_column(&PrimitiveType::Long, &PrimitiveType::Long);
        let bounds = [None, Some(-3_i64), Some(4)];

        for (lower, upper) in bounds
            .iter()
            .flat_map(|lower| bounds.map(|upper| (*lower, upper)))
        {
            let stored = (lower.map(Datum::long), upper.map(Datum::long));
            let stored_range = StoredRange {
                column: &column,
                lower: stored.0.as_ref(),
                upper: stored.1.as_ref(),
                all_null: false,
            };
            let value_range = FieldRange {
                field_id: 1,
                lower: lower.map(Datum::long),
                upper: upper.map(Datum::long),
                all_null: false,
            };
            for operator in operators {
                for constant in (-5..=5).map(Datum::long) {
                    let may_satisfy = stored_range.may_satisfy(operator, &constant);

                    assert_eq!(
                        may_satisfy,
                        value_range.may_satisfy(operator, &constant),
                        "[{lower:?}, {upper:?}] {operator} {constant}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_column_the_files_schema_lacks_is_null_and_one_of_an_unlisted_schema_unknown() {
        let current = resolve_column(&PrimitiveType::Long, &PrimitiveType::Long);
        let other_column =
            NestedField::optional(2, "other", IcebergType::Primitive(PrimitiveType::Long));
        let lacking = Schema::builder()
            .with_fields(vec![Arc::new(other_column)])
            .build()
            .unwrap();
        let column = NestedField::optional(1, "value", IcebergType::Primitive(PrimitiveType::Long));
        let holding_five = file_bounded_by(Datum::long(5));
        let recording_nothing = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("file.parquet".into())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(9)
            .build()
            .unwrap();

        let written = current.read_range(&holding_five);
        let lacked = StoredColumn::resolve(&column, Some(&lacking));
        let lacked = lacked.read_range(&recording_nothing);
        let unlisted = StoredColumn::resolve(&column, None);
        let unlisted = unlisted.read_range(&holding_five);

        let equals_six =
            |range: &StoredRange| range.may_satisfy(PredicateOperator::Eq, &Datum::long(6));
        let differs_from_six =
            |range: &StoredRange| range.may_satisfy(PredicateOperator::NotEq, &Datum::long(6));
        assert!(!equals_six(&written) && differs_from_six(&written));
        assert!(!equals_six(&lacked) && !differs_from_six(&lacked));
        assert!(equals_six(&unlisted) && differs_from_six(&unlisted));
    }
}
