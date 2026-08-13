//! [`OrderBy`] — sorts its input by one or more keys.
//!
//! Also holds the [`OrderByDirection`] / [`OrderByNode`] key types shared with
//! [`TopN`](crate::operator::TopN).

use crate::compile::Error;
use crate::expression::Expression;
use dispatch::{OrderBy as DispatchOrderBy, RecordBatchOperatorSpec};
use duckdb_planner::duckdb_bridge::duckdb_types::OrderType;
use std::fmt;

/// Sort direction specification for an ORDER BY clause.
#[derive(Debug)]
pub enum OrderByDirection {
    Default,
    Asc,
    Desc,
}

impl From<OrderType> for OrderByDirection {
    /// Map a DuckDB `OrderType` to a direction, treating anything other than an
    /// explicit `ASCENDING`/`DESCENDING` (i.e. `INVALID`/`ORDER_DEFAULT`) as
    /// [`OrderByDirection::Default`].
    fn from(d: OrderType) -> Self {
        if d == OrderType::ASCENDING {
            OrderByDirection::Asc
        } else if d == OrderType::DESCENDING {
            OrderByDirection::Desc
        } else {
            OrderByDirection::Default
        }
    }
}

impl fmt::Display for OrderByDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OrderByDirection::Asc | OrderByDirection::Default => "ASC",
            OrderByDirection::Desc => "DESC",
        };
        f.write_str(s)
    }
}

/// A single sort key within an ORDER BY or TopN operator.
#[derive(Debug)]
pub struct OrderByNode {
    pub direction: OrderByDirection,
    pub expression: Expression,
}

impl fmt::Display for OrderByNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.expression, self.direction)
    }
}

/// Sorts its input by one or more keys.
#[derive(Debug)]
pub struct OrderBy {
    pub order_bys: Vec<OrderByNode>,
}

impl fmt::Display for OrderBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let orders = self
            .order_bys
            .iter()
            .map(|o| o.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "OrderBy({orders})")
    }
}

impl OrderBy {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let orders = self
            .order_bys
            .iter()
            .map(|node| {
                let col = match &node.expression {
                    Expression::Ref(r) => Ok(r.column_idx),
                    expr => Err(Error::UnsupportedOrderByExpression(expr.clone())),
                }?;
                let descending = matches!(node.direction, OrderByDirection::Desc);
                Ok(DispatchOrderBy::new(col, descending, false))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(input.order_by(orders))
    }
}
