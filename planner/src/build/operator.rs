//! Constructs Pivot [`Operator`](crate::operator::Operator)s by walking DuckDB's
//! borrowed logical-operator handles. Each operator's `from_handle` is implemented
//! here (the walk in [`super`] dispatches to them), so the `crate::operator`
//! modules hold only Pivot IR (the operator types, their `Display`, and their
//! `compile`). Expression construction is the sibling
//! [`expression`](super::expression).

use std::any::Any;

use dispatch::RowDelivery;
use duckdb_planner::DuckDBTable;
use duckdb_planner::catalog_provider::OptionalTableWrapper;
use duckdb_planner::duckdb_bridge::duckdb_types::LimitNodeType;
use duckdb_planner::handle::{
    Aggregate as AggregateView, CreateTable as CreateTableView, Filter as FilterView,
    Insert as InsertView, Limit as LimitView, OrderBy as OrderByView, OrderKey,
    Projection as ProjectionView, Reset as ResetView, Set as SetView,
    TableFunctionScan as TableFunctionScanView, TableScan as TableScanView, TopN as TopNView,
    Values as ValuesView,
};

use super::{BuildCtx, build_scan_columns};
use crate::catalog::{BoundTable, Column, CreateTableRequest, DuckDBTableAdapter};
use crate::expression::{Error as ExpressionError, Expression};
use crate::operator::{
    Aggregate, CreateTable, Error as OperatorError, Filter, Input, Insert, Limit, OrderBy,
    OrderByNode, Projection, SetVariable, TableFunctionScan, TopN, Values,
};
use crate::types::type_from_logical;

impl Projection {
    pub(crate) fn from_handle(view: ProjectionView<'_>) -> Result<Projection, OperatorError> {
        Ok(Projection {
            projections: view
                .exprs()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Values {
    pub(crate) fn from_handle(view: ValuesView<'_>) -> Result<Values, OperatorError> {
        let rows = (0..view.row_count())
            .map(|row| {
                (0..view.column_count())
                    .map(|column| Expression::from_handle(view.expression(row, column)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Values { rows })
    }
}

impl Insert {
    pub(crate) fn from_handle(view: InsertView<'_>) -> Result<Insert, OperatorError> {
        if view.has_column_map() {
            return Err(OperatorError::Unsupported(
                "INSERT with an explicit target-column list is not supported".to_string(),
            ));
        }
        if view.returns_rows() {
            return Err(OperatorError::Unsupported(
                "INSERT ... RETURNING is not supported".to_string(),
            ));
        }
        Ok(Insert {
            table: bind_table(*view.take_table()),
        })
    }
}

impl Filter {
    pub(crate) fn from_handle(view: FilterView<'_>) -> Result<Filter, OperatorError> {
        Ok(Filter {
            conditions: view
                .exprs()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
            // `annotate_filter_delivery` revisits this once the tree is built
            // and the operator below this filter is known.
            delivery: RowDelivery::Coalesced,
        })
    }
}

impl Aggregate {
    pub(crate) fn from_handle(view: AggregateView<'_>) -> Result<Aggregate, OperatorError> {
        Ok(Aggregate {
            output_limit: None,
            groups: view
                .groups()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: view
                .expressions()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl OrderBy {
    pub(crate) fn from_handle(view: OrderByView<'_>) -> Result<OrderBy, OperatorError> {
        Ok(OrderBy {
            order_bys: build_orders(view.keys())?,
        })
    }
}

impl TopN {
    pub(crate) fn from_handle(
        view: TopNView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<TopN, OperatorError> {
        let produces_dynamic_filter = match view.dynamic_filter() {
            Some(df) => Some(ctx.dynamic_filter(df)?),
            None => None,
        };
        Ok(TopN {
            order_bys: build_orders(view.keys())?,
            limit: view.limit(),
            offset: view.offset(),
            produces_dynamic_filter,
        })
    }
}

impl Limit {
    pub(crate) fn from_handle(view: LimitView<'_>) -> Result<Limit, OperatorError> {
        Ok(Limit {
            limit: limit_bound(view.value_kind(), || view.value(), "value")?,
            offset: limit_bound(view.offset_kind(), || view.offset(), "offset")?.unwrap_or(0),
        })
    }
}

impl Input {
    pub(crate) fn from_handle(
        scan: TableScanView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<Input, OperatorError> {
        Ok(Input {
            table: bind_table(*scan.take_table()),
            columns: build_scan_columns(scan.output_columns())?,
            dynamic_filters: scan
                .dynamic_filters()
                .map(|df| ctx.dynamic_filter(df))
                .collect::<Result<Vec<_>, _>>()?,
            emit_row_group_metadata: false,
        })
    }
}

impl TableFunctionScan {
    pub(crate) fn from_handle(
        view: TableFunctionScanView<'_>,
    ) -> Result<TableFunctionScan, OperatorError> {
        // Only positional parameters are supported; reject named parameters rather
        // than silently drop a bound argument.
        if view.has_named_params() {
            return Err(OperatorError::Unsupported(format!(
                "table function {} with named parameters is not supported",
                view.function_name()
            )));
        }
        Ok(TableFunctionScan::new(
            view.function_name(),
            view.params().collect(),
            build_scan_columns(view.output_columns())?,
        ))
    }
}

impl CreateTable {
    pub(crate) fn from_handle(view: CreateTableView<'_>) -> Result<CreateTable, OperatorError> {
        Ok(CreateTable {
            request: CreateTableRequest {
                datastore_name: view.datastore(),
                name: view.name(),
                columns: view
                    .columns()
                    .map(|(name, col_type)| {
                        Ok(Column {
                            name,
                            col_type: type_from_logical(col_type)?,
                        })
                    })
                    .collect::<Result<Vec<_>, OperatorError>>()?,
                options: view.options().collect(),
                if_not_exists: view.if_not_exists(),
            },
            or_replace: view.or_replace(),
            temporary: view.temporary(),
            has_query: view.has_query(),
            constraint_count: view.constraint_count(),
        })
    }
}

impl SetVariable {
    pub(crate) fn from_set(view: SetView<'_>) -> SetVariable {
        SetVariable {
            name: view.name(),
            value: Some(view.value()),
        }
    }

    /// `RESET name` is modelled as a `SET` with no value (the consumer reads "no
    /// value" as "off / default").
    pub(crate) fn from_reset(view: ResetView<'_>) -> SetVariable {
        SetVariable {
            name: view.name(),
            value: None,
        }
    }
}

/// Lower a sequence of handle sort keys into Pivot [`OrderByNode`]s. Shared by the
/// `OrderBy` and `TopN` constructors.
fn build_orders<'a>(
    keys: impl Iterator<Item = OrderKey<'a>>,
) -> Result<Vec<OrderByNode>, ExpressionError> {
    keys.map(|key| {
        Ok(OrderByNode {
            direction: key.direction.into(),
            expression: Expression::from_handle(key.expression)?,
        })
    })
    .collect()
}

/// Resolve a bound catalog entry into the Pivot [`BoundTable`] it wraps: the DuckDB
/// table is a [`DuckDBTableAdapter`] holding the `Box<dyn BoundTable>`.
fn bind_table(wrapper: OptionalTableWrapper) -> Box<dyn BoundTable> {
    let duck: Box<dyn DuckDBTable> = wrapper.table.expect("planner returned an unbound table");
    let any: Box<dyn Any> = duck;
    let adapter: Box<DuckDBTableAdapter> = any
        .downcast::<DuckDBTableAdapter>()
        .expect("scan table should be a DuckDBTableAdapter");
    adapter.table
}

/// Interpret a `LIMIT`/`OFFSET` bound's [`LimitNodeType`] into an `Option<usize>`:
/// an unset bound is unbounded (`None`), a constant is the row count, and a
/// percentage/expression bound is rejected (pivot only handles a fixed row count).
///
/// `value` is read lazily: DuckDB's constant accessor throws for a non-constant
/// bound, so it must only be called once the kind is known to be a constant.
fn limit_bound(
    kind: LimitNodeType,
    value: impl FnOnce() -> usize,
    what: &str,
) -> Result<Option<usize>, OperatorError> {
    match kind {
        LimitNodeType::UNSET => Ok(None),
        LimitNodeType::CONSTANT_VALUE => Ok(Some(value())),
        _ => Err(OperatorError::Unsupported(format!(
            "Unsupported non-constant LIMIT {what}"
        ))),
    }
}
