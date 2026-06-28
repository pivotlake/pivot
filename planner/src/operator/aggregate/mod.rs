//! [`Aggregate`] — GROUP BY + aggregate functions.
//!
//! [`Aggregate::compile`] is a router: it inspects the groups and aggregate
//! expressions and dispatches to one of a handful of lowering strategies, each
//! in its own submodule:
//!
//! | query shape | strategy |
//! |---|---|
//! | `COUNT(DISTINCT x)` alone, no GROUP BY | [`global_distinct`] |
//! | grouped, one `COUNT(DISTINCT)` (alone or with other aggregates) | [`grouped_distinct`] |
//! | no GROUP BY | [`global`] |
//! | any other GROUP BY (incl. a computed key) | [`grouped`] |
//!
//! The shared expression→slot lowering and accumulator-width rule live here,
//! since the global and grouped paths both use them.

mod global;
mod global_distinct;
mod grouped;
mod grouped_distinct;
mod reinterpret;

use self::reinterpret::{reinterpret_columns, temporal_to_int};
use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression};
use crate::types::Type;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use dispatch::{
    AggregationKind, AggregationSlot, GroupLimit, RecordBatchOperatorSpec, RowKeySchema,
};
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::Arc;

/// GROUP BY + aggregate functions.
#[derive(Debug)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
    /// A LIMIT pushed into this grouped aggregate by the plan-rewrite passes:
    /// [`GroupLimit::TopK`] for an `ORDER BY <slot> DESC LIMIT k` feeding it (see
    /// `PlanNode::annotate_group_topn`), [`GroupLimit::First`] for a plain
    /// `LIMIT k` (`PlanNode::annotate_group_limit`). The
    /// group operator then emits only each partition's kept rows instead of
    /// every group.
    pub output_limit: Option<GroupLimit>,
}

impl TryFrom<duckdb_operator::Aggregate> for Aggregate {
    type Error = super::Error;
    fn try_from(a: duckdb_operator::Aggregate) -> Result<Self, Self::Error> {
        Ok(Aggregate {
            output_limit: None,
            groups: a
                .groups
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: a
                .expressions
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let groups = render(&self.groups);
        let exprs = render(&self.expressions);
        write!(f, "Aggregate(groups: [{groups}], exprs: [{exprs}])")
    }
}

fn render(exprs: &[Expression]) -> String {
    exprs
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // `COUNT(DISTINCT x)` as the sole aggregate with no GROUP BY: a dedicated
        // single-level global fast path.
        if let [Expression::AggregateFunc(AggregateFunc::CountDistinct(a))] =
            self.expressions.as_slice()
            && self.groups.is_empty()
        {
            return self.compile_global_distinct(input, a);
        }

        // Aggregates compute on the int a temporal column stores (a DATE is Int32
        // days, a TIMESTAMP is Int64 seconds), so reinterpret any temporal value
        // column to that int before the readers see it, and restore the temporal
        // type on the output (group keys, and MIN/MAX of a date) afterwards. The
        // int-ify is gated to a MIN/MAX of a date/timestamp, so ordinary
        // aggregates and a plain `GROUP BY date` (whose key the row reader already
        // casts) pay nothing.
        let input = self.int_ify_temporal_values(input);

        // One `COUNT(DISTINCT)` over a GROUP BY, alone or mixed with non-distinct
        // aggregates, lowers to a two-level GROUP BY.
        let n_distinct = self
            .expressions
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Expression::AggregateFunc(AggregateFunc::CountDistinct(_))
                )
            })
            .count();
        let grouped = if !self.groups.is_empty() && n_distinct == 1 {
            self.compile_grouped_distinct(input)?
        } else if self.groups.is_empty() {
            self.compile_global(input)?
        } else {
            // Every GROUP BY (plain column keys, two-int-key, the row fallback, or
            // a single computed key) lowers through the general grouped path.
            self.compile_grouped(input)?
        };
        Ok(self.restore_temporal_output(grouped))
    }

    /// Reinterpret every temporal column of the aggregate input to the int it
    /// stores (`Date32 -> Int32`, `Timestamp -> Int64`), zero-copy, so the value
    /// readers (which only know int columns) can read a `MIN`/`MAX` of a date.
    /// A no-op unless the aggregate actually takes a `MIN`/`MAX` of a date/
    /// timestamp, so the common paths are untouched.
    fn int_ify_temporal_values(&self, input: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
        let has_temporal_value = self
            .expressions
            .iter()
            .any(|e| aggregate_value_temporal(e).is_some());
        if !has_temporal_value {
            return input;
        }
        input.project(|| {
            |batch: RecordBatch| reinterpret_columns(batch, |_, dt| temporal_to_int(dt))
        })
    }

    /// Restore the temporal arrow type on the grouped/global output's columns:
    /// the group-key columns lead (one per `self.groups`, typed by its
    /// `result_type`), followed by one value column per `self.expressions` (typed
    /// only for a `MIN`/`MAX` of a date). Zero-copy where the widths match.
    fn restore_temporal_output(&self, op: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
        let mut targets: Vec<Option<DataType>> = self
            .groups
            .iter()
            .map(|g| g.result_type().ok().and_then(|t| temporal_output_type(&t)))
            .collect();
        targets.extend(self.expressions.iter().map(aggregate_value_temporal));
        if !targets.iter().any(Option::is_some) {
            return op;
        }
        let targets = Arc::new(targets);
        op.project(move || {
            let targets = targets.clone();
            move |batch: RecordBatch| {
                reinterpret_columns(batch, |i, _| targets.get(i).and_then(Option::clone))
            }
        })
    }

    fn is_lone_count_star(&self) -> bool {
        matches!(
            self.expressions.as_slice(),
            [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
        )
    }
}

/// Lower each aggregate expression to its [`AggregationSlot`]. Shared by the
/// global and grouped multi-aggregate paths, which build slots identically. A
/// `MIN`/`MAX` over a `Utf8` column folds the byte extreme (`StrMin`/`StrMax`);
/// over an integer column the numeric one. `COUNT(*)` ignores its column, so the
/// index is a placeholder.
///
/// Each slot's output type is DuckDB's declared result type for the call (its
/// physical arrow type: `BIGINT`→`Int64` for a count, `HUGEINT`→`Decimal128` for
/// a sum, the input type for a `MIN`/`MAX`). The accumulator may store it at a
/// different width; the output phase casts to this. A `MIN`/`MAX` of a date keeps
/// its physical int here and is restored to the temporal type by
/// [`Aggregate::restore_temporal_output`], the one place dates are coerced.
fn aggregation_slots(exprs: &[Expression]) -> Result<Vec<AggregationSlot>, Error> {
    exprs
        .iter()
        .map(|e| {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            let (kind, column) = match func {
                AggregateFunc::CountStar(_) => (AggregationKind::CountStar, 0),
                AggregateFunc::Count(a) => (AggregationKind::Count, a.column.column_idx),
                AggregateFunc::Sum(a) => (AggregationKind::Sum, a.column.column_idx),
                AggregateFunc::Min(a) => (
                    extreme_kind(
                        &a.column.return_type,
                        AggregationKind::StrMin,
                        AggregationKind::Min,
                    )
                    .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                    a.column.column_idx,
                ),
                AggregateFunc::Max(a) => (
                    extreme_kind(
                        &a.column.return_type,
                        AggregationKind::StrMax,
                        AggregationKind::Max,
                    )
                    .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                    a.column.column_idx,
                ),
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            };
            Ok(AggregationSlot::new(
                kind,
                column,
                crate::types::physical_arrow_type(func.return_type()),
            ))
        })
        .collect()
}

/// The MIN/MAX kind for a column of type `ty`: the byte-wise `string` extreme for
/// a `Utf8` column (folded through the value container's arena path), the
/// `numeric` extreme for the integer widths the executor can read
/// (`Int16`/`Int32`/`Int64`) and for `Date`/`Timestamp` (which the reader sees as
/// the int they store, after [`Aggregate::int_ify_temporal_values`]). `None` for
/// any other type, so the caller reports a clean `UnsupportedAggregateExpression`
/// rather than a worker panic in the reader.
fn extreme_kind(
    ty: &Type,
    string: AggregationKind,
    numeric: AggregationKind,
) -> Option<AggregationKind> {
    match ty {
        Type::Utf8 => Some(string),
        // UInt8/16/32 fold losslessly through the reader's i64 accumulator;
        // UInt64 is excluded (it would wrap above i64::MAX) so it reports a clean
        // unsupported error rather than a silently wrong extreme.
        Type::Int8
        | Type::Int16
        | Type::Int32
        | Type::Int64
        | Type::UInt8
        | Type::UInt16
        | Type::UInt32
        | Type::Date
        | Type::Timestamp => Some(numeric),
        _ => None,
    }
}

/// The arrow output type of a temporal `Type`, or `None` for a non-temporal one.
/// A temporal type is one whose canonical arrow type has a backing int form
/// ([`temporal_to_int`]); that int is what the group-by computes on, and this
/// canonical type is restored on its output. The one place dates are coerced.
pub(super) fn temporal_output_type(t: &Type) -> Option<DataType> {
    let arrow = crate::types::physical_arrow_type(t);
    temporal_to_int(&arrow).is_some().then_some(arrow)
}

/// The temporal arrow type a `MIN`/`MAX` aggregate emits, or `None` for any other
/// expression (or a `MIN`/`MAX` over a non-temporal column). Used to gate the
/// int-ify of value columns and to restore the temporal type on the output.
fn aggregate_value_temporal(e: &Expression) -> Option<DataType> {
    match e {
        Expression::AggregateFunc(AggregateFunc::Min(a) | AggregateFunc::Max(a)) => {
            temporal_output_type(&a.column.return_type)
        }
        _ => None,
    }
}

/// Map group-key types to the arrow types the [`RowKeyExtractor`](dispatch::RowKeyExtractor)
/// encodes, in key order. Shared by the general grouped path
/// ([`grouped`]) and the `COUNT(DISTINCT)` two-level lowering
/// ([`grouped_distinct`]). Returns `None` if any key has a
/// type the row encoding doesn't support, so the caller reports it unsupported
/// rather than panicking in `RowKeySchema::new`.
pub(super) fn row_key_schema<'a>(
    types: impl IntoIterator<Item = &'a Type>,
) -> Option<RowKeySchema> {
    let arrow = types
        .into_iter()
        .map(row_key_arrow_type)
        .collect::<Option<Vec<_>>>()?;
    Some(RowKeySchema::new(arrow))
}

/// The arrow type the row encoder uses for one group-key column, or `None` for a
/// type it can't encode.
///
/// Only the integer widths and `Utf8View` are byte-packable into a row key,
/// exactly the rule [`RowKeySchema::new`](dispatch::RowKeySchema) enforces. A
/// `DATE`/`TIMESTAMP` key drops to its backing int ([`temporal_to_int`]) and
/// rides along for free; the row reader casts each key column to this type, so
/// grouping on the integer day/second count is lossless (the temporal type is
/// restored on the output). `Boolean`/`Float64`/`Decimal`/`Int128` can't pack,
/// so they return `None` for a clean "unsupported" rather than a panic in
/// `RowKeySchema::new`.
fn row_key_arrow_type(t: &Type) -> Option<DataType> {
    let dt = crate::types::physical_arrow_type(t);
    let dt = temporal_to_int(&dt).unwrap_or(dt);
    (dt.is_integer() || dt == DataType::Utf8View).then_some(dt)
}

/// The accumulator-width rule shared by the global and grouped paths: `i128`
/// only when a `SUM` reads a 64-bit column (whose total can overflow `i64`),
/// else `i64`.
fn sum_reads_wide_column(exprs: &[Expression]) -> bool {
    exprs.iter().any(|e| {
        matches!(
            e,
            Expression::AggregateFunc(AggregateFunc::Sum(a)) if a.column.return_type == Type::Int64
        )
    })
}
