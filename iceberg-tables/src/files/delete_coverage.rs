//! How much of each partition a DELETE covers, read off the partition's
//! tuple alone.
//!
//! A DELETE is answered by taking whole partitions out of the table, with no
//! data file read or written. Its conditions are projected through each
//! partition spec twice, as Iceberg defines the two projections:
//!
//! - the *inclusive* projection is true of the tuple of every partition
//!   that holds a row the conditions are true of;
//! - the *strict* projection is true only of the tuple of a partition whose
//!   every row the conditions are true of.
//!
//! A partition the inclusive projection is false of holds no row to delete.
//! One the strict projection is true of holds nothing else, and is dropped.
//! Any other is cut by the conditions: it may hold rows to delete beside
//! rows to keep, which only a rewrite of its files could tell apart.
//!
//! `ts < TIMESTAMP '2026-11-15 00:00:00'` covers the whole of every day
//! before the 15th of a table partitioned by `day(ts)`, and none of the
//! days from it on; with `10:00:00` for the time it cuts the 15th.
//!
//! A comparison is true of no NULL, and a tuple's field is NULL exactly
//! when the column it is derived from is, so a partition with a NULL in a
//! field holds no row a comparison of that field's column is true of.
//!
//! Whatever is not known of a tuple counts against the DELETE: a value that
//! does not compare with the constant makes the inclusive projection true
//! and the strict one false.

use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
    UnaryExpression,
};
use iceberg::spec::{
    Datum, Literal, PartitionSpec, PrimitiveLiteral, PrimitiveType, SchemaRef, TableMetadata,
    Transform,
};
use iceberg::{ErrorKind, Result};

use crate::files::file_index::Partition;
use crate::files::pruning::{
    PartitionValueField, build_partition_schema, list_partition_value_fields, literal_to_datum,
};

/// How much of a partition's rows a DELETE's conditions are true of, as far
/// as the partition's tuple tells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionCoverage {
    /// No row of the partition is deleted.
    Nothing,
    /// Every row of the partition is deleted.
    Whole,
    /// The partition may hold a row to delete, and is not known to hold
    /// only such rows.
    Partial,
}

/// Which of a spec's two projections of a condition is taken.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Projection {
    Inclusive,
    Strict,
}

/// What a DELETE's conditions say of the partitions of one spec.
struct SpecCoverage {
    /// True of the tuple of every partition that holds a row to delete.
    inclusive: BoundPredicate,
    /// True only of the tuple of a partition whose every row is deleted.
    strict: BoundPredicate,
    /// The fields of the spec whose values the projections can compare.
    fields: Vec<PartitionValueField>,
}

/// What a DELETE's conditions cover of a table's partitions, resolved once
/// for all of them.
pub(crate) struct DeleteCoverage {
    /// By partition spec id, in id order.
    specs: Vec<(i32, SpecCoverage)>,
}

impl DeleteCoverage {
    /// What `row_filters`, each true of exactly the rows one condition of a
    /// DELETE is true of, cover of the partitions of the table `metadata`
    /// describes. A row is deleted when every one of them is true of it;
    /// with none, every row is.
    pub(crate) fn new(metadata: &TableMetadata, row_filters: &[BoundPredicate]) -> Result<Self> {
        let schema = metadata.current_schema();
        let mut specs = Vec::new();
        for partition_spec in metadata.partition_specs_iter() {
            let partition_schema = build_partition_schema(schema, partition_spec)?;
            let project_all = |projection: Projection| -> Result<BoundPredicate> {
                let mut projected = Predicate::AlwaysTrue;
                for row_filter in row_filters {
                    projected = projected.and(project(
                        partition_spec,
                        &partition_schema,
                        row_filter,
                        projection,
                    ));
                }
                projected.bind(partition_schema.clone(), true)
            };
            specs.push((
                partition_spec.spec_id(),
                SpecCoverage {
                    inclusive: project_all(Projection::Inclusive)?,
                    strict: project_all(Projection::Strict)?,
                    fields: list_partition_value_fields(schema, partition_spec),
                },
            ));
        }
        specs.sort_unstable_by_key(|(spec_id, _)| *spec_id);
        Ok(Self { specs })
    }

    /// How much of `partition` the conditions cover. A partition of a spec
    /// the table does not list is taken to be cut: nothing is known of its
    /// tuple.
    pub(crate) fn classify(&self, partition: &Partition) -> Result<PartitionCoverage> {
        let Ok(position) = self
            .specs
            .binary_search_by_key(&partition.spec_id(), |(spec_id, _)| *spec_id)
        else {
            return Ok(PartitionCoverage::Partial);
        };
        let spec = &self.specs[position].1;
        let values = read_tuple_values(&spec.fields, partition)?;
        if evaluate(&spec.inclusive, &values) == Some(false) {
            return Ok(PartitionCoverage::Nothing);
        }
        if evaluate(&spec.strict, &values) == Some(true) {
            return Ok(PartitionCoverage::Whole);
        }
        Ok(PartitionCoverage::Partial)
    }
}

/// `predicate`, over the table's columns, projected onto the fields of
/// `partition_spec` that `partition_schema` holds.
fn project(
    partition_spec: &PartitionSpec,
    partition_schema: &SchemaRef,
    predicate: &BoundPredicate,
    projection: Projection,
) -> Predicate {
    // What a projection that says nothing of a predicate stands for: the
    // inclusive one rules no partition out, the strict one none in.
    let unknown = match projection {
        Projection::Inclusive => Predicate::AlwaysTrue,
        Projection::Strict => Predicate::AlwaysFalse,
    };
    let (source_id, is_comparison) = match predicate {
        BoundPredicate::AlwaysTrue => return Predicate::AlwaysTrue,
        BoundPredicate::AlwaysFalse => return Predicate::AlwaysFalse,
        BoundPredicate::And(children) => {
            let [left, right] = children.inputs();
            return project(partition_spec, partition_schema, left, projection).and(project(
                partition_spec,
                partition_schema,
                right,
                projection,
            ));
        }
        BoundPredicate::Or(children) => {
            let [left, right] = children.inputs();
            return project(partition_spec, partition_schema, left, projection).or(project(
                partition_spec,
                partition_schema,
                right,
                projection,
            ));
        }
        BoundPredicate::Not(_) => return unknown,
        BoundPredicate::Unary(unary) => (unary.term().field().id, false),
        BoundPredicate::Binary(comparison) => (comparison.term().field().id, true),
        BoundPredicate::Set(set) => (set.term().field().id, true),
    };
    // Every field derived from the column bounds the partition on its own:
    // a partition holds a row the predicate is true of only when each
    // field's inclusive projection is true of its tuple, and holds only
    // such rows as soon as one field's strict projection is.
    let mut projected = unknown;
    for field in partition_spec.fields() {
        if field.source_id != source_id || partition_schema.field_by_id(field.field_id).is_none() {
            continue;
        }
        let of_field = match projection {
            Projection::Inclusive => field.transform.project(&field.name, predicate),
            Projection::Strict => {
                field
                    .transform
                    .strict_project(&field.name, predicate)
                    .map(|strict| {
                        strict.and_then(|strict| correct_strict_time_bound(field.transform, strict))
                    })
            }
        };
        let of_field = of_field.ok().flatten().filter(|of_field| {
            of_field
                .clone()
                .bind(partition_schema.clone(), true)
                .is_ok()
        });
        projected = match projection {
            Projection::Inclusive => {
                let mut required = of_field.unwrap_or(Predicate::AlwaysTrue);
                if is_comparison {
                    required = required.and(Predicate::Unary(UnaryExpression::new(
                        PredicateOperator::NotNull,
                        Reference::new(&field.name),
                    )));
                }
                projected.and(required)
            }
            Projection::Strict => projected.or(of_field.unwrap_or(Predicate::AlwaysFalse)),
        };
    }
    projected
}

/// `strict`, the strict projection of a comparison through `transform`,
/// made to hold for files of writers that put a time before 1970 one
/// partition late: Iceberg's own writers did so for the year, month, day
/// and hour of such a time, before its release 0.14. A partition's value
/// may then be one above the value its rows have, so a lower bound below
/// zero moves up by one, and an inequality with a value below zero is taken
/// to say nothing. `None` when nothing is left of the projection.
fn correct_strict_time_bound(transform: Transform, strict: Predicate) -> Option<Predicate> {
    if !matches!(
        transform,
        Transform::Year | Transform::Month | Transform::Day | Transform::Hour
    ) {
        return Some(strict);
    }
    let Predicate::Binary(comparison) = &strict else {
        // The projection of a `NOT IN` list is not corrected, so it is not
        // taken.
        return match strict {
            Predicate::Unary(_) => Some(strict),
            _ => None,
        };
    };
    let PrimitiveLiteral::Int(value) = comparison.literal().literal() else {
        return None;
    };
    if *value >= 0 {
        return Some(strict);
    }
    match comparison.op() {
        PredicateOperator::LessThan | PredicateOperator::LessThanOrEq => Some(strict),
        PredicateOperator::GreaterThan | PredicateOperator::GreaterThanOrEq => {
            let moved_up = match comparison.literal().data_type() {
                PrimitiveType::Date => Datum::date(value + 1),
                PrimitiveType::Int => Datum::int(value + 1),
                _ => return None,
            };
            Some(Predicate::Binary(BinaryExpression::new(
                comparison.op(),
                comparison.term().clone(),
                moved_up,
            )))
        }
        _ => None,
    }
}

/// What a tuple holds in one field.
enum TupleValue {
    Null,
    Value(Datum),
    /// A value that is not NULL and is not read: of a type whose values do
    /// not compare as a query's do, or one the field's type does not hold.
    Unread,
}

/// The value `partition`'s tuple holds in each of `fields`, by field id.
fn read_tuple_values(
    fields: &[PartitionValueField],
    partition: &Partition,
) -> Result<Vec<(i32, TupleValue)>> {
    let tuple = partition.tuple().fields();
    fields
        .iter()
        .map(|field| {
            let value = tuple.get(field.position).ok_or_else(|| {
                iceberg::Error::new(
                    ErrorKind::DataInvalid,
                    "file partition does not match its spec",
                )
            })?;
            let value = match value {
                None => TupleValue::Null,
                // Zero has two float encodings that compare equal, and NaN
                // orders by its encoding.
                Some(Literal::Primitive(_))
                    if matches!(
                        field.value_type,
                        PrimitiveType::Float | PrimitiveType::Double
                    ) =>
                {
                    TupleValue::Unread
                }
                Some(Literal::Primitive(value)) => literal_to_datum(value, &field.value_type)
                    .map_or(TupleValue::Unread, TupleValue::Value),
                Some(_) => TupleValue::Unread,
            };
            Ok((field.field_id, value))
        })
        .collect()
}

/// Whether `predicate`, over the fields of a partition spec, is true of the
/// tuple holding `values`; `None` when the values read do not tell. A
/// comparison is true of no NULL.
fn evaluate(predicate: &BoundPredicate, values: &[(i32, TupleValue)]) -> Option<bool> {
    let find_value = |field_id: i32| {
        values
            .iter()
            .find(|(value_field_id, _)| *value_field_id == field_id)
            .map(|(_, value)| value)
    };
    match predicate {
        BoundPredicate::AlwaysTrue => Some(true),
        BoundPredicate::AlwaysFalse => Some(false),
        BoundPredicate::And(children) => {
            let [left, right] = children.inputs();
            match (evaluate(left, values), evaluate(right, values)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        }
        BoundPredicate::Or(children) => {
            let [left, right] = children.inputs();
            match (evaluate(left, values), evaluate(right, values)) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        }
        BoundPredicate::Not(_) => None,
        BoundPredicate::Unary(unary) => {
            let is_null = matches!(find_value(unary.term().field().id)?, TupleValue::Null);
            match unary.op() {
                PredicateOperator::IsNull => Some(is_null),
                PredicateOperator::NotNull => Some(!is_null),
                _ => None,
            }
        }
        BoundPredicate::Binary(comparison) => match find_value(comparison.term().field().id)? {
            TupleValue::Null => Some(false),
            TupleValue::Unread => None,
            TupleValue::Value(value) => compare(value, comparison.op(), comparison.literal()),
        },
        BoundPredicate::Set(set) => {
            let value = match find_value(set.term().field().id)? {
                TupleValue::Null => return Some(false),
                TupleValue::Unread => return None,
                TupleValue::Value(value) => value,
            };
            let mut is_every_constant_compared = true;
            let mut is_among = false;
            for constant in set.literals() {
                match compare(value, PredicateOperator::Eq, constant) {
                    Some(true) => is_among = true,
                    Some(false) => {}
                    None => is_every_constant_compared = false,
                }
            }
            match set.op() {
                PredicateOperator::In if is_among => Some(true),
                PredicateOperator::NotIn if is_among => Some(false),
                PredicateOperator::In if is_every_constant_compared => Some(false),
                PredicateOperator::NotIn if is_every_constant_compared => Some(true),
                _ => None,
            }
        }
    }
}

/// Whether `value <operator> constant` holds; `None` when the two do not
/// compare, or the operator is not a comparison of order.
fn compare(value: &Datum, operator: PredicateOperator, constant: &Datum) -> Option<bool> {
    let ordering = value.partial_cmp(constant)?;
    Some(match operator {
        PredicateOperator::Eq => ordering.is_eq(),
        PredicateOperator::NotEq => ordering.is_ne(),
        PredicateOperator::LessThan => ordering.is_lt(),
        PredicateOperator::LessThanOrEq => ordering.is_le(),
        PredicateOperator::GreaterThan => ordering.is_gt(),
        PredicateOperator::GreaterThanOrEq => ordering.is_ge(),
        _ => return None,
    })
}
