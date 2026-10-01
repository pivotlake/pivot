//! Pruning by partition. A pushed predicate on a column the table partitions
//! by is projected through the partition field's transform onto the field's
//! values: `id = 5` under `bucket(id, 16)` becomes `id_bucket = bucket(5)`,
//! `at < '2020-01-02'` under `day(at)` becomes `at_day <= 2020-01-01`. A
//! manifest whose summary of the field cannot satisfy the projection is not
//! read; a file whose value of it cannot is not opened.

use arrow_arith::boolean::and;
use arrow_array::{Array, ArrayRef, BooleanArray, Datum as ArrowDatum, Scalar};
use iceberg::expr::{BinaryExpression, Bind, Predicate, PredicateOperator, Reference};
use iceberg::spec::{
    Datum, FieldSummary, PartitionField, PartitionSpec, PrimitiveType, SchemaRef,
    Type as IcebergType,
};
use parquet_engine::{PushedPredicate, bounds_exclude};
use planner::expression::CompareType;
use planner::types::Type;

use crate::columns::to_pivot_type;
use crate::values::{build_scalar, to_datum};
use crate::{Error, Result};

/// A pushed predicate projected onto one field of a partition spec: what a
/// row's value of the field must satisfy for the row to satisfy the
/// predicate. Inclusive, as a transform's projection is: a value that
/// satisfies this may hold matching rows, one that does not cannot.
pub(crate) struct PartitionPredicate {
    /// The field's position in the spec.
    pub(crate) field: usize,
    /// The field's Iceberg type, which a manifest summary's bounds of it are
    /// encoded in.
    field_type: PrimitiveType,
    /// The field's Pivot type, which its values compare in.
    value_type: Type,
    test: PartitionTest,
}

/// What the projection asks of the field's value.
enum PartitionTest {
    Compare(CompareType, Scalar<ArrayRef>),
    In(Vec<Scalar<ArrayRef>>),
}

impl PartitionPredicate {
    /// Whether a manifest whose summary of the field is `summary` can hold a
    /// matching row. A summary without bounds has no non-NULL value, and
    /// NULL satisfies nothing; unless the field is floating point and the
    /// summary does not rule out NaN, which is what such a summary leaves.
    pub(crate) fn manifest_may_match(&self, summary: &FieldSummary) -> bool {
        let (Some(lower), Some(upper)) = (&summary.lower_bound, &summary.upper_bound) else {
            let floating = matches!(
                self.field_type,
                PrimitiveType::Float | PrimitiveType::Double
            );
            return floating && summary.contains_nan != Some(false);
        };
        let (Some(lower), Some(upper)) = (self.decode(lower), self.decode(upper)) else {
            return true;
        };
        !self
            .test
            .excludes(&lower, &upper)
            .is_some_and(|excluded| excluded.is_valid(0) && excluded.value(0))
    }

    /// Which files `values`, one file's value of the field per slot, prove
    /// hold no matching row: `true` per file to prune. A NULL value satisfies
    /// nothing.
    pub(crate) fn files_excluded(&self, values: &ArrayRef) -> BooleanArray {
        let excluded = self.test.excludes(values, values);
        (0..values.len())
            .map(|file| {
                values.is_null(file)
                    || excluded
                        .as_ref()
                        .is_some_and(|excluded| excluded.is_valid(file) && excluded.value(file))
            })
            .collect()
    }

    /// A summary bound of the field as a scalar its values compare in, or
    /// `None` for bytes that do not decode to one.
    fn decode(&self, bound: &[u8]) -> Option<Scalar<ArrayRef>> {
        let datum = Datum::try_from_bytes(bound, self.field_type.clone()).ok()?;
        build_scalar(&self.value_type, datum.literal())
    }
}

impl PartitionTest {
    /// Which of the ranges `[min, max]` prove the test unsatisfied: `true`
    /// per range that does, null where a range has no bound, `None` when the
    /// bounds are not of the constant's type. An `IN` is unsatisfied only
    /// where every member is.
    fn excludes(&self, min: &dyn ArrowDatum, max: &dyn ArrowDatum) -> Option<BooleanArray> {
        match self {
            Self::Compare(compare, constant) => {
                bounds_exclude(min, max, *compare, constant).ok().flatten()
            }
            Self::In(members) => {
                let mut excluded: Option<BooleanArray> = None;
                for member in members {
                    let by_member = bounds_exclude(min, max, CompareType::Equal, member)
                        .ok()
                        .flatten()?;
                    excluded = Some(match excluded {
                        Some(so_far) => and(&so_far, &by_member).ok()?,
                        None => by_member,
                    });
                }
                excluded
            }
        }
    }
}

/// `predicates` projected onto `spec`: for each predicate on a column the
/// spec partitions by, what the partition field's value must satisfy. A
/// predicate the field's transform cannot carry (`id > 5` through a bucket),
/// or a variant-path predicate, projects to nothing. `field_ids` maps a
/// predicate's column index to its Iceberg field id.
pub(crate) fn project_predicates(
    table: &str,
    schema: &SchemaRef,
    spec: &PartitionSpec,
    field_ids: &[i32],
    predicates: &[PushedPredicate],
) -> Result<Vec<PartitionPredicate>> {
    let mut projected = Vec::new();
    for predicate in predicates
        .iter()
        .filter(|predicate| predicate.path.is_empty())
    {
        let source_id = field_ids[predicate.column_idx];
        let source = schema
            .field_by_id(source_id)
            .expect("a declared column is a field of the schema");
        let Some(constant) = to_datum(&predicate.value) else {
            continue;
        };
        let projection_error = |error: iceberg::Error| Error::PartitionProjection {
            table: table.to_string(),
            column: source.name.clone(),
            source: Box::new(error),
        };
        // Binding converts the constant to the column's type, and turns one
        // beyond the type's range into an always-true or always-false
        // predicate, which no transform projects.
        let bound = Predicate::Binary(BinaryExpression::new(
            predicate_operator(predicate.compare_type),
            Reference::new(&source.name),
            constant,
        ))
        .bind(schema.clone(), true)
        .map_err(projection_error)?;
        for (position, field) in spec
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.source_id == source_id)
        {
            let Some((field_type, value_type)) = partition_value_type(schema, field) else {
                continue;
            };
            let Some(projection) = field
                .transform
                .project(&field.name, &bound)
                .map_err(projection_error)?
            else {
                continue;
            };
            let Some(test) = partition_test(&value_type, projection) else {
                continue;
            };
            projected.push(PartitionPredicate {
                field: position,
                field_type,
                value_type,
                test,
            });
        }
    }
    Ok(projected)
}

/// The type of partition field `field`'s values: what its transform yields
/// of its source column, as Iceberg and as Pivot types. `None` when the
/// source is not in `schema` or the values are of a type Pivot does not
/// compare; such a field prunes nothing.
pub(crate) fn partition_value_type(
    schema: &SchemaRef,
    field: &PartitionField,
) -> Option<(PrimitiveType, Type)> {
    let source = schema.field_by_id(field.source_id)?;
    let IcebergType::Primitive(field_type) =
        field.transform.result_type(&source.field_type).ok()?
    else {
        return None;
    };
    let value_type = to_pivot_type(&IcebergType::Primitive(field_type.clone()))?;
    Some((field_type, value_type))
}

/// `projection`, a transform's projection of a predicate, as a test over the
/// field's values of `value_type`. `None` for a shape no test evaluates.
fn partition_test(value_type: &Type, projection: Predicate) -> Option<PartitionTest> {
    match projection {
        Predicate::Binary(expression) => {
            let compare = match expression.op() {
                PredicateOperator::Eq => CompareType::Equal,
                PredicateOperator::NotEq => CompareType::NotEqual,
                PredicateOperator::LessThan => CompareType::Less,
                PredicateOperator::LessThanOrEq => CompareType::LessEqual,
                PredicateOperator::GreaterThan => CompareType::Greater,
                PredicateOperator::GreaterThanOrEq => CompareType::GreaterEqual,
                _ => return None,
            };
            let constant = build_scalar(value_type, expression.literal().literal())?;
            Some(PartitionTest::Compare(compare, constant))
        }
        Predicate::Set(expression) if expression.op() == PredicateOperator::In => {
            let members = expression
                .literals()
                .iter()
                .map(|member| build_scalar(value_type, member.literal()))
                .collect::<Option<Vec<_>>>()?;
            Some(PartitionTest::In(members))
        }
        _ => None,
    }
}

fn predicate_operator(compare: CompareType) -> PredicateOperator {
    match compare {
        CompareType::Equal => PredicateOperator::Eq,
        CompareType::NotEqual => PredicateOperator::NotEq,
        CompareType::Less => PredicateOperator::LessThan,
        CompareType::LessEqual => PredicateOperator::LessThanOrEq,
        CompareType::Greater => PredicateOperator::GreaterThan,
        CompareType::GreaterEqual => PredicateOperator::GreaterThanOrEq,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use arrow_array::{Date32Array, Int32Array, Int64Array, StringViewArray};
    use iceberg::spec::{NestedField, PrimitiveLiteral, Schema, Transform, UnboundPartitionSpec};
    use iceberg::transform::create_transform_function;
    use planner::expression::{Compare, Expression, Ref, TableFilter};

    use super::*;

    /// The one pushed predicate `column <compare> constant` yields.
    pub(crate) fn pushed_predicate(
        column: usize,
        column_type: Type,
        compare: CompareType,
        constant: ArrayRef,
    ) -> PushedPredicate {
        let filter = TableFilter::Expression(Box::new(Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx: column,
                return_type: column_type,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(constant))),
            compare_type: compare,
            return_type: Type::Boolean,
        })));
        let mut predicates = PushedPredicate::from_filter(filter);
        assert_eq!(predicates.len(), 1, "one bound yields one predicate");
        predicates.remove(0)
    }

    /// `id BIGINT, name VARCHAR, at TIMESTAMP`, with field ids 1..3.
    fn schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", IcebergType::Primitive(PrimitiveType::Long))
                        .into(),
                    NestedField::optional(2, "name", IcebergType::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::optional(
                        3,
                        "at",
                        IcebergType::Primitive(PrimitiveType::Timestamp),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        )
    }

    /// A spec of one field: `transform` of source field `source_id`, named
    /// `name`.
    fn spec(source_id: i32, name: &str, transform: Transform) -> PartitionSpec {
        UnboundPartitionSpec::builder()
            .add_partition_field(source_id, name, transform)
            .unwrap()
            .build()
            .bind(schema())
            .unwrap()
    }

    /// `predicates` projected onto `spec` over [`schema`]'s columns.
    fn project(spec: &PartitionSpec, predicates: &[PushedPredicate]) -> Vec<PartitionPredicate> {
        project_predicates("db.t", &schema(), spec, &[1, 2, 3], predicates).unwrap()
    }

    fn id_is(compare: CompareType, constant: i64) -> PushedPredicate {
        pushed_predicate(
            0,
            Type::Int64,
            compare,
            Arc::new(Int64Array::from(vec![constant])),
        )
    }

    fn excluded(predicate: &PartitionPredicate, values: ArrayRef) -> Vec<bool> {
        predicate.files_excluded(&values).iter().flatten().collect()
    }

    #[test]
    fn an_identity_field_takes_the_predicate_as_is() {
        let spec = spec(1, "id", Transform::Identity);

        let projected = project(&spec, &[id_is(CompareType::Greater, 5)]);

        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].field, 0);
        assert_eq!(
            excluded(
                &projected[0],
                Arc::new(Int64Array::from(vec![Some(3), Some(7), None]))
            ),
            [true, false, true]
        );
    }

    #[test]
    fn a_bucket_field_carries_equality_and_nothing_else() {
        let spec = spec(1, "id_bucket", Transform::Bucket(16));
        let bucket_of_5 = create_transform_function(&Transform::Bucket(16))
            .unwrap()
            .transform_literal(&Datum::long(5))
            .unwrap()
            .expect("a bucket of 5");
        let PrimitiveLiteral::Int(bucket_of_5) = *bucket_of_5.literal() else {
            panic!("a bucket is an int");
        };

        let equal = project(&spec, &[id_is(CompareType::Equal, 5)]);
        let greater = project(&spec, &[id_is(CompareType::Greater, 5)]);

        assert_eq!(
            excluded(
                &equal[0],
                Arc::new(Int32Array::from(vec![bucket_of_5, (bucket_of_5 + 1) % 16]))
            ),
            [false, true]
        );
        assert!(greater.is_empty());
    }

    #[test]
    fn a_day_field_projects_a_range_onto_its_days() {
        let spec = spec(3, "at_day", Transform::Day);
        let micros_per_day = 86_400 * 1_000_000;
        let (day_18262, day_18263): (i32, i32) = (18_262, 18_263);
        let before_day_18263 = pushed_predicate(
            2,
            Type::Timestamp,
            CompareType::Less,
            Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
                i64::from(day_18263) * micros_per_day,
            ])),
        );

        let projected = project(&spec, &[before_day_18263]);

        assert_eq!(projected.len(), 1);
        assert_eq!(
            excluded(
                &projected[0],
                Arc::new(Date32Array::from(vec![day_18262, day_18263]))
            ),
            [false, true]
        );
    }

    #[test]
    fn a_truncate_field_matches_the_constants_prefix() {
        let spec = spec(2, "name_prefix", Transform::Truncate(2));
        let name_is_annie = pushed_predicate(
            1,
            Type::Utf8,
            CompareType::Equal,
            Arc::new(StringViewArray::from(vec!["annie"])),
        );

        let projected = project(&spec, &[name_is_annie]);

        assert_eq!(
            excluded(
                &projected[0],
                Arc::new(StringViewArray::from(vec![Some("an"), Some("bo"), None]))
            ),
            [false, true, true]
        );
    }

    #[test]
    fn a_manifest_is_read_unless_its_summary_rules_the_field_out() {
        let spec = spec(1, "id", Transform::Identity);
        let projected = project(&spec, &[id_is(CompareType::Equal, 9)]);
        let bounded = |lower: i64, upper: i64| FieldSummary {
            lower_bound: Some(Datum::long(lower).to_bytes().unwrap()),
            upper_bound: Some(Datum::long(upper).to_bytes().unwrap()),
            ..FieldSummary::default()
        };
        let all_null = FieldSummary {
            contains_null: true,
            ..FieldSummary::default()
        };

        assert!(projected[0].manifest_may_match(&bounded(3, 12)));
        assert!(!projected[0].manifest_may_match(&bounded(3, 7)));
        assert!(!projected[0].manifest_may_match(&all_null));
    }
}
