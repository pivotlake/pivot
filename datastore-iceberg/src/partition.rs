//! Pruning by partition. A predicate on a column the table partitions by is
//! projected through the partition field's transform: `id = 5` under
//! `bucket(id, 16)` becomes `id_bucket = bucket(5)`, and `at < '2020-01-02'`
//! under `day(at)` becomes `at_day <= 2020-01-01`. The result is a predicate
//! on the partition field, checked against what each layer has of it: a
//! manifest list's summary per manifest, a manifest's value per file.

use std::collections::BTreeSet;

use arrow_array::{ArrayRef, BooleanArray, Scalar};
use iceberg::expr::{BinaryExpression, Bind, Predicate, PredicateOperator, Reference};
use iceberg::spec::{
    DataFile, Datum, FieldSummary, Literal, NestedField, PartitionField, PartitionSpec,
    PrimitiveType, SchemaRef, Type as IcebergType,
};
use parquet_engine::{BoundKey, PushedPredicate, Range, RangePredicate};
use planner::expression::CompareType;
use planner::types::Type;

use crate::columns::to_pivot_type;
use crate::values::{build_array, build_scalar, to_datum};
use crate::{Error, Result};

/// Turn the query's predicates into predicates on partition fields.
///
/// `id = 5` under `bucket(id, 16)` means every matching row is in the bucket
/// of 5. `at < '2020-01-02'` under `day(at)` means every matching row is on
/// 2020-01-01 or earlier. This works that out for each predicate against
/// each partition field built from its column, in every spec the table has
/// used. A predicate a transform cannot carry, like `id > 5` through a
/// bucket, is left out. Each predicate comes with its column's schema field,
/// since a spec names its source column by field id.
pub(crate) fn project_predicates<'a>(
    table: &str,
    schema: &SchemaRef,
    specs: impl Iterator<Item = &'a PartitionSpec>,
    predicates: &[(&NestedField, &PushedPredicate)],
) -> Result<Vec<RangePredicate>> {
    let specs: Vec<&PartitionSpec> = specs.collect();
    let mut projected = Vec::new();
    for (source, predicate) in predicates {
        let Some(constant) = to_datum(&predicate.value) else {
            continue;
        };
        // Binding converts the constant to the column's type. A constant
        // outside the type's range binds to always-true or always-false,
        // which no transform projects.
        let bound = Predicate::Binary(BinaryExpression::new(
            predicate_operator(predicate.compare_type),
            Reference::new(&source.name),
            constant,
        ))
        .bind(schema.clone(), true)
        .map_err(|error| Error::PartitionProjection {
            table: table.to_string(),
            column: source.name.clone(),
            source: Box::new(error),
        })?;
        for spec in &specs {
            for (position, field) in spec
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| field.source_id == source.id)
            {
                let Some((_, value_type)) = partition_value_type(schema, field) else {
                    continue;
                };
                let Some(projection) =
                    field
                        .transform
                        .project(&field.name, &bound)
                        .map_err(|error| Error::PartitionProjection {
                            table: table.to_string(),
                            column: source.name.clone(),
                            source: Box::new(error),
                        })?
                else {
                    continue;
                };
                let Some((compare, constants)) = comparison_of(&value_type, projection) else {
                    continue;
                };
                projected.push(RangePredicate {
                    key: BoundKey::PartitionField {
                        spec_id: spec.spec_id(),
                        position,
                    },
                    compare,
                    constants,
                });
            }
        }
    }
    Ok(projected)
}

/// A manifest list's partition summaries as a layer: for each field of each
/// spec in use, the field's lower and upper bound in every manifest.
///
/// `manifests` gives each manifest's spec id and its summaries, one per field
/// of that spec; `spec_of` looks a spec up by id. A manifest of another spec,
/// or without summaries, has no bound for the field. A summary with no
/// bounds means the field is NULL in every file of the manifest, which no
/// comparison matches. The exception is a float field whose summary does not
/// rule out NaN, since bounds leave NaN out.
pub(crate) fn manifest_list_ranges<'a>(
    schema: &SchemaRef,
    spec_of: impl Fn(i32) -> Option<&'a PartitionSpec>,
    manifests: &[(i32, Option<&[FieldSummary]>)],
) -> Vec<(BoundKey, Range)> {
    let spec_ids: BTreeSet<i32> = manifests.iter().map(|(spec_id, _)| *spec_id).collect();
    let mut ranges = Vec::new();
    for spec_id in spec_ids {
        let Some(spec) = spec_of(spec_id) else {
            continue;
        };
        for (position, field) in spec.fields().iter().enumerate() {
            let Some((field_type, value_type)) = partition_value_type(schema, field) else {
                continue;
            };
            let floating = matches!(field_type, PrimitiveType::Float | PrimitiveType::Double);
            let summaries: Vec<Option<&FieldSummary>> = manifests
                .iter()
                .map(|(manifest_spec, summaries)| {
                    (*manifest_spec == spec_id)
                        .then_some(summaries.as_ref()?.get(position))
                        .flatten()
                })
                .collect();
            let decode = |bound: Option<&[u8]>| -> Option<Datum> {
                Datum::try_from_bytes(bound?, field_type.clone()).ok()
            };
            let lower: Vec<Option<Datum>> = summaries
                .iter()
                .map(|summary| decode((*summary)?.lower_bound.as_deref().map(Vec::as_slice)))
                .collect();
            let upper: Vec<Option<Datum>> = summaries
                .iter()
                .map(|summary| decode((*summary)?.upper_bound.as_deref().map(Vec::as_slice)))
                .collect();
            let all_null = BooleanArray::from_iter(summaries.iter().map(|summary| {
                Some(summary.is_some_and(|summary| {
                    summary.lower_bound.is_none()
                        && summary.upper_bound.is_none()
                        && !(floating && summary.contains_nan != Some(false))
                }))
            }));
            let may_hold_nan = BooleanArray::from_iter(summaries.iter().map(|summary| {
                Some(summary.is_some_and(|summary| floating && summary.contains_nan != Some(false)))
            }));
            ranges.push((
                BoundKey::PartitionField { spec_id, position },
                Range {
                    min: build_array(
                        &value_type,
                        lower.iter().map(|bound| bound.as_ref().map(Datum::literal)),
                    ),
                    max: build_array(
                        &value_type,
                        upper.iter().map(|bound| bound.as_ref().map(Datum::literal)),
                    ),
                    all_null,
                    may_hold_nan,
                },
            ));
        }
    }
    ranges
}

/// A manifest's partition values as a layer: for each field of `spec`, every
/// file's value, as a range of one value. A NULL value matches no
/// comparison. A NaN value is the range itself and compares as NaN does.
pub(crate) fn partition_value_ranges(
    schema: &SchemaRef,
    spec: &PartitionSpec,
    data_files: &[&DataFile],
) -> Vec<(BoundKey, Range)> {
    spec.fields()
        .iter()
        .enumerate()
        .filter_map(|(position, field)| {
            let (_, value_type) = partition_value_type(schema, field)?;
            let values = build_array(
                &value_type,
                data_files
                    .iter()
                    .map(|file| match file.partition().fields().get(position) {
                        Some(Some(Literal::Primitive(value))) => Some(value),
                        _ => None,
                    }),
            );
            let all_null = BooleanArray::from_iter(data_files.iter().map(|file| {
                Some(matches!(
                    file.partition().fields().get(position),
                    Some(None)
                ))
            }));
            Some((
                BoundKey::PartitionField {
                    spec_id: spec.spec_id(),
                    position,
                },
                Range {
                    min: values.clone(),
                    max: values,
                    all_null,
                    may_hold_nan: BooleanArray::from(vec![false; data_files.len()]),
                },
            ))
        })
        .collect()
}

/// The type of `field`'s values, as Iceberg and Pivot types: what its
/// transform makes of its source column. `None` when the source is not in
/// `schema` or Pivot cannot compare the type. Such a field prunes nothing.
fn partition_value_type(
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

/// `projection`, what a transform made of a predicate, as a comparison over
/// values of `value_type`: the operator and the constants, any of which may
/// match. `None` for a shape no range can judge.
fn comparison_of(
    value_type: &Type,
    projection: Predicate,
) -> Option<(CompareType, Vec<Scalar<ArrayRef>>)> {
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
            Some((compare, vec![constant]))
        }
        Predicate::Set(expression) if expression.op() == PredicateOperator::In => {
            let members = expression
                .literals()
                .iter()
                .map(|member| build_scalar(value_type, member.literal()))
                .collect::<Option<Vec<_>>>()?;
            Some((CompareType::Equal, members))
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

    use arrow_array::{
        Date32Array, Int32Array, Int64Array, StringViewArray, TimestampMicrosecondArray,
    };
    use iceberg::spec::{NestedField, PrimitiveLiteral, Schema, Transform, UnboundPartitionSpec};
    use iceberg::transform::create_transform_function;
    use parquet_engine::Bounds;
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

    /// `predicates` projected onto `spec` over [`schema`]'s columns, whose
    /// field ids are their positions plus one.
    fn project(spec: &PartitionSpec, predicates: &[PushedPredicate]) -> Vec<RangePredicate> {
        let schema = schema();
        let sourced: Vec<(&NestedField, &PushedPredicate)> = predicates
            .iter()
            .map(|predicate| {
                let source = schema
                    .field_by_id(predicate.column_idx as i32 + 1)
                    .unwrap()
                    .as_ref();
                (source, predicate)
            })
            .collect();
        project_predicates("db.t", &schema, [spec].into_iter(), &sourced).unwrap()
    }

    fn id_is(compare: CompareType, constant: i64) -> PushedPredicate {
        pushed_predicate(
            0,
            Type::Int64,
            compare,
            Arc::new(Int64Array::from(vec![constant])),
        )
    }

    /// Which of `values`, files' values of the one field of a spec, the
    /// projection `predicate` keeps.
    fn kept(predicate: &RangePredicate, values: ArrayRef) -> Vec<bool> {
        let mut layer = Bounds::new(values.len());
        let all_null =
            BooleanArray::from_iter((0..values.len()).map(|file| Some(values.is_null(file))));
        layer.insert(
            predicate.key,
            Range {
                may_hold_nan: BooleanArray::from(vec![false; values.len()]),
                min: values.clone(),
                max: values,
                all_null,
            },
        );
        layer.kept(std::slice::from_ref(predicate))
    }

    #[test]
    fn an_identity_field_takes_the_predicate_as_is() {
        let spec = spec(1, "id", Transform::Identity);

        let projected = project(&spec, &[id_is(CompareType::Greater, 5)]);

        assert_eq!(projected.len(), 1);
        assert_eq!(
            projected[0].key,
            BoundKey::PartitionField {
                spec_id: 0,
                position: 0
            }
        );
        assert_eq!(
            kept(
                &projected[0],
                Arc::new(Int64Array::from(vec![Some(3), Some(7), None]))
            ),
            [false, true, false]
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
            kept(
                &equal[0],
                Arc::new(Int32Array::from(vec![bucket_of_5, (bucket_of_5 + 1) % 16]))
            ),
            [true, false]
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
            Arc::new(TimestampMicrosecondArray::from(vec![
                i64::from(day_18263) * micros_per_day,
            ])),
        );

        let projected = project(&spec, &[before_day_18263]);

        assert_eq!(projected.len(), 1);
        assert_eq!(
            kept(
                &projected[0],
                Arc::new(Date32Array::from(vec![day_18262, day_18263]))
            ),
            [true, false]
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
            kept(
                &projected[0],
                Arc::new(StringViewArray::from(vec![Some("an"), Some("bo"), None]))
            ),
            [true, false, false]
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
        let summaries = [bounded(3, 12), bounded(3, 7), all_null];
        let manifests: Vec<(i32, Option<&[FieldSummary]>)> = summaries
            .iter()
            .map(|summary| (0, Some(std::slice::from_ref(summary))))
            .chain([(0, None), (7, Some(&summaries[1..2]))])
            .collect();

        let mut layer = Bounds::new(manifests.len());
        for (key, range) in manifest_list_ranges(&schema(), |_| Some(&spec), &manifests) {
            layer.insert(key, range);
        }

        // Bounds that admit 9, bounds that do not, no non-null value, no
        // summaries, and a manifest of a spec the predicate was not projected
        // onto.
        assert_eq!(layer.kept(&projected), [true, false, false, true, true]);
    }
}
