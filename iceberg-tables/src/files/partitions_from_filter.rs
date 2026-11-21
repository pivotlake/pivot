//! The partitions a scan can touch, computed from its filters alone: when
//! what the filters require of a partition spec's tuples leaves every field
//! of the spec only a few values, the tuples a kept row may sit under are
//! the combinations of those values. When the filters leave some field more
//! than that there is no answer here, and the scan's partitions are listed
//! from the manifests.
//!
//! The requirement read here is the one manifests and files are pruned by
//! ([`pruning`](crate::files::pruning)): the filters projected through each
//! field's transform by the spec crate, so a field's values are the
//! transform's own. The filters determine a field's values by comparisons
//! of it that hold only for values of a finite set: equalities and `IN`
//! lists, and, for a field whose values are whole numbers that stand next to
//! one another (a year, month, day or hour, or an integer or date kept as it
//! is), a range bounded on both sides. A comparison never holds for NULL,
//! so NULL is never among the values the filters leave a field.

use iceberg::expr::{BoundPredicate, PredicateOperator};
use iceberg::spec::{
    Datum, Literal, PartitionField, PartitionSpec, PrimitiveLiteral, PrimitiveType, Struct,
    TableMetadata, Transform,
};

use crate::files::pruning::find_partition_value_type;

/// How many partitions a scan computes from its filters at most. Each one
/// is checked against every manifest-list entry and, where an entry's bounds
/// admit it, asked of the manifests whose entries do not rule it out,
/// whether or not it holds a file, so the limit bounds what a plan spends
/// on partitions that turn out empty: a filter that spans more is cheaper
/// to answer from the partitions that exist.
pub(crate) const MAX_PARTITIONS_FROM_FILTER: usize = 256;

/// The values a filter leaves one partition field in the tuples it may hold
/// for, NULL never among them.
enum FieldValues {
    /// The filter does not narrow the field to a set this can name.
    Unrestricted,
    /// One of these, each of the field's type.
    Among(Vec<Datum>),
    /// A whole number from `lowest` to `highest`, both included; `None` on
    /// a side the filter leaves open.
    Between {
        lowest: Option<i64>,
        highest: Option<i64>,
    },
}

impl FieldValues {
    /// The values both `self` and `other` hold.
    fn intersect(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unrestricted, restricted) | (restricted, Self::Unrestricted) => restricted,
            (Self::Among(values), Self::Among(others)) => Self::Among(
                values
                    .into_iter()
                    .filter(|value| others.contains(value))
                    .collect(),
            ),
            (Self::Among(values), Self::Between { lowest, highest })
            | (Self::Between { lowest, highest }, Self::Among(values)) => Self::Among(
                values
                    .into_iter()
                    .filter(|value| {
                        // A value that is not a whole number is not known
                        // to be outside the range.
                        read_whole_number(value).is_none_or(|number| {
                            lowest.is_none_or(|lowest| lowest <= number)
                                && highest.is_none_or(|highest| number <= highest)
                        })
                    })
                    .collect(),
            ),
            (
                Self::Between { lowest, highest },
                Self::Between {
                    lowest: other_lowest,
                    highest: other_highest,
                },
            ) => Self::Between {
                lowest: match (lowest, other_lowest) {
                    (Some(lowest), Some(other)) => Some(lowest.max(other)),
                    (lowest, other) => lowest.or(other),
                },
                highest: match (highest, other_highest) {
                    (Some(highest), Some(other)) => Some(highest.min(other)),
                    (highest, other) => highest.or(other),
                },
            },
        }
    }

    /// Values that include every one `self` or `other` holds: a range joined
    /// with something else is the range that spans both.
    fn unite(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unrestricted, _) | (_, Self::Unrestricted) => Self::Unrestricted,
            (Self::Among(mut values), Self::Among(others)) => {
                for value in others {
                    if !values.contains(&value) {
                        values.push(value);
                    }
                }
                Self::Among(values)
            }
            (Self::Among(values), Self::Between { lowest, highest })
            | (Self::Between { lowest, highest }, Self::Among(values)) => {
                let Some(numbers) = values
                    .iter()
                    .map(read_whole_number)
                    .collect::<Option<Vec<i64>>>()
                else {
                    return Self::Unrestricted;
                };
                let (Some(least), Some(greatest)) =
                    (numbers.iter().min().copied(), numbers.iter().max().copied())
                else {
                    return Self::Between { lowest, highest };
                };
                Self::Between {
                    lowest: lowest.map(|lowest| lowest.min(least)),
                    highest: highest.map(|highest| highest.max(greatest)),
                }
            }
            (
                Self::Between { lowest, highest },
                Self::Between {
                    lowest: other_lowest,
                    highest: other_highest,
                },
            ) => Self::Between {
                lowest: lowest.zip(other_lowest).map(|(one, other)| one.min(other)),
                highest: highest
                    .zip(other_highest)
                    .map(|(one, other)| one.max(other)),
            },
        }
    }
}

/// The whole number `value` holds, for a value of a type whose values are
/// whole numbers a tuple keeps as they are.
fn read_whole_number(value: &Datum) -> Option<i64> {
    match (value.data_type(), value.literal()) {
        (PrimitiveType::Int | PrimitiveType::Date, PrimitiveLiteral::Int(number)) => {
            Some(i64::from(*number))
        }
        (PrimitiveType::Long, PrimitiveLiteral::Long(number)) => Some(*number),
        _ => None,
    }
}

/// `number` as a tuple holds a value of `value_type`, a type
/// [`read_whole_number`] reads; `None` for a number the type does not hold.
fn build_whole_number(number: i64, value_type: &PrimitiveType) -> Option<Literal> {
    match value_type {
        PrimitiveType::Int | PrimitiveType::Date => Some(Literal::Primitive(
            PrimitiveLiteral::Int(i32::try_from(number).ok()?),
        )),
        PrimitiveType::Long => Some(Literal::Primitive(PrimitiveLiteral::Long(number))),
        _ => None,
    }
}

/// Whether two values of `value_type` are equal exactly when a tuple holds
/// them as the same literal, so a value computed from a filter names the
/// partition of the files that hold it. A float is not: zero has two
/// encodings that compare equal.
fn is_equal_as_keyed(value_type: &PrimitiveType) -> bool {
    matches!(
        value_type,
        PrimitiveType::Boolean
            | PrimitiveType::Int
            | PrimitiveType::Long
            | PrimitiveType::Date
            | PrimitiveType::Time
            | PrimitiveType::Timestamp
            | PrimitiveType::Timestamptz
            | PrimitiveType::TimestampNs
            | PrimitiveType::TimestamptzNs
            | PrimitiveType::String
            | PrimitiveType::Decimal { .. }
    )
}

/// Whether every whole number between two values of a field with
/// `transform` and values of `value_type` may itself be a value of the
/// field, so a bounded range of them is worth naming one by one.
fn is_range_dense(transform: Transform, value_type: &PrimitiveType) -> bool {
    match transform {
        Transform::Year | Transform::Month | Transform::Day | Transform::Hour => true,
        Transform::Identity => matches!(
            value_type,
            PrimitiveType::Int | PrimitiveType::Long | PrimitiveType::Date
        ),
        _ => false,
    }
}

/// The values `filter`, a requirement of a partition tuple, leaves the field
/// `field_id`, whose values are of `value_type`: every tuple the filter
/// holds for has one of them in the field.
fn read_field_values(
    filter: &BoundPredicate,
    field_id: i32,
    value_type: &PrimitiveType,
) -> FieldValues {
    match filter {
        BoundPredicate::AlwaysFalse => FieldValues::Among(Vec::new()),
        BoundPredicate::And(children) => {
            let [left, right] = children.inputs();
            read_field_values(left, field_id, value_type)
                .intersect(read_field_values(right, field_id, value_type))
        }
        BoundPredicate::Or(children) => {
            let [left, right] = children.inputs();
            read_field_values(left, field_id, value_type)
                .unite(read_field_values(right, field_id, value_type))
        }
        BoundPredicate::Binary(comparison) if comparison.term().field().id == field_id => {
            read_comparison_values(comparison.op(), comparison.literal(), value_type)
        }
        BoundPredicate::Set(set)
            if set.term().field().id == field_id && set.op() == PredicateOperator::In =>
        {
            if set
                .literals()
                .iter()
                .all(|value| value.data_type() == value_type)
            {
                FieldValues::Among(set.literals().iter().cloned().collect())
            } else {
                FieldValues::Unrestricted
            }
        }
        _ => FieldValues::Unrestricted,
    }
}

/// The values of a field of `value_type` that `<operator> constant` holds
/// for. A constant of another type than the field's is not read as one of
/// its values.
fn read_comparison_values(
    operator: PredicateOperator,
    constant: &Datum,
    value_type: &PrimitiveType,
) -> FieldValues {
    if constant.data_type() != value_type {
        return FieldValues::Unrestricted;
    }
    if operator == PredicateOperator::Eq {
        return FieldValues::Among(vec![constant.clone()]);
    }
    let Some(number) = read_whole_number(constant) else {
        return FieldValues::Unrestricted;
    };
    // No whole number is below the least or above the greatest.
    let nothing = FieldValues::Among(Vec::new());
    match operator {
        PredicateOperator::LessThan => match number.checked_sub(1) {
            Some(highest) => FieldValues::Between {
                lowest: None,
                highest: Some(highest),
            },
            None => nothing,
        },
        PredicateOperator::LessThanOrEq => FieldValues::Between {
            lowest: None,
            highest: Some(number),
        },
        PredicateOperator::GreaterThan => match number.checked_add(1) {
            Some(lowest) => FieldValues::Between {
                lowest: Some(lowest),
                highest: None,
            },
            None => nothing,
        },
        PredicateOperator::GreaterThanOrEq => FieldValues::Between {
            lowest: Some(number),
            highest: None,
        },
        _ => FieldValues::Unrestricted,
    }
}

/// Whether every schema `metadata` lists that has the source column of
/// `field` gives the field's values the type `value_type`: a manifest keeps
/// a tuple's values in the type its own schema gives them, so only then
/// does a value of `value_type` name the partition of every file that holds
/// it.
fn is_typed_alike_in_every_schema(
    metadata: &TableMetadata,
    field: &PartitionField,
    value_type: &PrimitiveType,
) -> bool {
    metadata.schemas_iter().all(|schema| {
        schema.field_by_id(field.source_id).is_none()
            || find_partition_value_type(schema, field).as_ref() == Some(value_type)
    })
}

/// The values `partition_filter` leaves `field`, as a tuple holds them, or
/// `None` when it does not narrow the field to values that can be named, or
/// leaves it more than `limit` of them.
fn enumerate_field_values(
    metadata: &TableMetadata,
    field: &PartitionField,
    partition_filter: &BoundPredicate,
    limit: usize,
) -> Option<Vec<Literal>> {
    let value_type = find_partition_value_type(metadata.current_schema(), field)?;
    if !is_equal_as_keyed(&value_type)
        || !is_typed_alike_in_every_schema(metadata, field, &value_type)
    {
        return None;
    }
    match read_field_values(partition_filter, field.field_id, &value_type) {
        FieldValues::Unrestricted => None,
        FieldValues::Among(values) => (values.len() <= limit).then(|| {
            values
                .iter()
                .map(|value| Literal::Primitive(value.literal().clone()))
                .collect()
        }),
        FieldValues::Between {
            lowest: Some(lowest),
            highest: Some(highest),
        } if is_range_dense(field.transform, &value_type) => {
            let count = i128::from(highest) - i128::from(lowest) + 1;
            if count > limit as i128 {
                return None;
            }
            (lowest..=highest)
                .map(|number| build_whole_number(number, &value_type))
                .collect()
        }
        FieldValues::Between { .. } => None,
    }
}

/// The tuples of `partition_spec` that `partition_filter`, what a scan's
/// filters require of a tuple of the spec, may hold for: every combination
/// of the values it leaves each field. `None` when it leaves a field's
/// values unrestricted, or the combinations number more than `limit`. A
/// spec without fields has the one empty tuple.
pub(crate) fn enumerate_tuples_from_filter(
    metadata: &TableMetadata,
    partition_spec: &PartitionSpec,
    partition_filter: &BoundPredicate,
    limit: usize,
) -> Option<Vec<Struct>> {
    let mut tuples: Vec<Vec<Option<Literal>>> = vec![Vec::new()];
    for field in partition_spec.fields() {
        // With no tuple left, any number of values combines into none.
        let field_limit = limit.checked_div(tuples.len()).unwrap_or(limit);
        let values = enumerate_field_values(metadata, field, partition_filter, field_limit)?;
        tuples = tuples
            .iter()
            .flat_map(|tuple| {
                values.iter().map(move |value| {
                    let mut extended = tuple.clone();
                    extended.push(Some(value.clone()));
                    extended
                })
            })
            .collect();
    }
    (tuples.len() <= limit).then(|| tuples.into_iter().map(Struct::from_iter).collect())
}
