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

use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression};
use crate::types::Type;
use arrow_schema::DataType;
use dispatch::{
    AggregationKind, AggregationSlot, GroupLimit, RecordBatchOperatorSpec, RowKeySchema,
};
use duckdb_planner::operator as duckdb_operator;
use std::fmt;

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
        if !self.groups.is_empty() && n_distinct == 1 {
            return self.compile_grouped_distinct(input);
        }

        if self.groups.is_empty() {
            return self.compile_global(input);
        }

        // Every GROUP BY (plain column keys, two-int-key, the row fallback, or a
        // single computed key) lowers through the general grouped path.
        self.compile_grouped(input)
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
fn aggregation_slots(exprs: &[Expression]) -> Result<Vec<AggregationSlot>, Error> {
    exprs
        .iter()
        .map(|e| {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            Ok(match func {
                AggregateFunc::CountStar(_) => AggregationSlot::new(AggregationKind::CountStar, 0),
                AggregateFunc::Count(a) => {
                    AggregationSlot::new(AggregationKind::Count, a.column.column_idx)
                }
                AggregateFunc::Sum(a) => {
                    let kind = if a.column.return_type == Type::Float64 {
                        AggregationKind::FloatSum
                    } else {
                        AggregationKind::Sum
                    };
                    AggregationSlot::new(kind, a.column.column_idx)
                }
                AggregateFunc::Min(a) => extreme_kind(
                    &a.column.return_type,
                    AggregationKind::StrMin,
                    AggregationKind::Min,
                    AggregationKind::FloatMin,
                )
                .map(|kind| AggregationSlot::new(kind, a.column.column_idx))
                .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                AggregateFunc::Max(a) => extreme_kind(
                    &a.column.return_type,
                    AggregationKind::StrMax,
                    AggregationKind::Max,
                    AggregationKind::FloatMax,
                )
                .map(|kind| AggregationSlot::new(kind, a.column.column_idx))
                .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            })
        })
        .collect()
}

/// The MIN/MAX kind for a column of type `ty`: the byte-wise `string` extreme for
/// a `Utf8` column (folded through the value container's arena path), the
/// `numeric` extreme for the integer widths the executor can read
/// (`Int16`/`Int32`/`Int64`), or the `float` extreme for a `Float64` column.
/// `None` for any other type, so the caller reports a clean
/// `UnsupportedAggregateExpression` rather than a worker panic in the reader.
fn extreme_kind(
    ty: &Type,
    string: AggregationKind,
    numeric: AggregationKind,
    float: AggregationKind,
) -> Option<AggregationKind> {
    match ty {
        Type::Utf8 => Some(string),
        Type::Int16 | Type::Int32 | Type::Int64 => Some(numeric),
        Type::Float64 => Some(float),
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
fn row_key_arrow_type(t: &Type) -> Option<DataType> {
    Some(match t {
        Type::Int8 => DataType::Int8,
        Type::Int16 => DataType::Int16,
        Type::Int32 => DataType::Int32,
        Type::Int64 => DataType::Int64,
        Type::Utf8 => DataType::Utf8View,
        // DATE is days-since-epoch (arrives as Date32 or the parquet-physical
        // integer); TIMESTAMP is Int64 epoch seconds. The row reader casts each
        // key column to the schema type, so encoding them as their integer
        // day/second count is lossless and groups identically. This is what lets
        // a wide key tuple like `(Int64, Date)` group instead of erroring.
        Type::Date => DataType::Int32,
        Type::Timestamp => DataType::Int64,
        _ => return None,
    })
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
