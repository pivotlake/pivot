//! Constructs Pivot [`Operator`](crate::operator::Operator)s by walking DuckDB's
//! borrowed logical-operator handles. Each operator's `from_handle` is implemented
//! here (the walk in [`super`] dispatches to them), so the `crate::operator`
//! modules hold only Pivot IR (the operator types, their `Display`, and their
//! `compile`). Expression construction is the sibling
//! [`expression`](super::expression).

use std::any::Any;

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
use crate::catalog::{Column, CreateTableRequest, DuckDBTableAdapter, Table};
use crate::expression::{Error as ExpressionError, Expression};
use crate::operator::{
    Aggregate, CreateTable, Error as OperatorError, Filter, Input, Insert, Limit, OrderBy,
    OrderByNode, PreparedColumn, PreparedValues, Projection, SetVariable, TableFunctionScan, TopN,
    Values,
};
use crate::types::{Type, physical_arrow_type, type_from_logical};
use arrow::compute::concat;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

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
        let rows: Vec<Vec<Expression>> = (0..view.row_count())
            .map(|row| {
                (0..view.column_count())
                    .map(|column| Expression::from_handle(view.expression(row, column)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // A prepared VALUES (some cell is a bound parameter) can't be
        // materialized until Execute, so keep its parameter "pointer" columns; a
        // literal VALUES is turned into a ready batch here, at plan time.
        if references_parameters(&rows) {
            Ok(Values::Prepared(build_prepared_values(rows)?))
        } else if rows.iter().flatten().all(is_plan_literal) {
            Ok(Values::Literal(materialize_values(rows)?))
        } else {
            Ok(Values::Dynamic(rows))
        }
    }
}

/// Whether any cell references a bound parameter (a prepared VALUES).
fn references_parameters(rows: &[Vec<Expression>]) -> bool {
    rows.iter().flatten().any(|cell| {
        let mut params = Vec::new();
        cell.collect_params(&mut params);
        !params.is_empty()
    })
}

/// Whether a cell is fully known while the plan is built. DuckDB normally folds
/// deterministic literal expressions to constants; casts can remain explicit.
fn is_plan_literal(cell: &Expression) -> bool {
    match cell {
        Expression::Constant(_) => true,
        Expression::Cast(cast) => is_plan_literal(&cast.source),
        _ => false,
    }
}

/// Evaluate a literal VALUES into one `RecordBatch` at plan time. Each cell is a
/// column-free constant expression (DuckDB folds `1+2`, `upper('x')`, … to
/// constants during planning), so it evaluates against a single empty row.
fn materialize_values(rows: Vec<Vec<Expression>>) -> Result<RecordBatch, OperatorError> {
    let column_count = rows.first().map_or(0, Vec::len);
    let empty_row = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .expect("one empty row is a valid batch");

    let mut fields = Vec::with_capacity(column_count);
    let mut columns = Vec::with_capacity(column_count);
    for column in 0..column_count {
        let cells: Vec<ArrayRef> = rows
            .iter()
            .map(|row| {
                let builder = row[column]
                    .compile()
                    .map_err(|e| OperatorError::Unsupported(e.to_string()))?;
                Ok(builder()(&empty_row).into_array(1))
            })
            .collect::<Result<_, OperatorError>>()?;
        let refs: Vec<&dyn Array> = cells.iter().map(AsRef::as_ref).collect();
        let array = concat(&refs)
            .map_err(|e| OperatorError::Unsupported(format!("building VALUES batch: {e}")))?;
        fields.push(Field::new("", array.data_type().clone(), true));
        columns.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| OperatorError::Unsupported(format!("building VALUES batch: {e}")))
}

/// Build the parameter "pointer" columns for a prepared VALUES: each cell must
/// be a bound parameter (optionally cast to the column type).
fn build_prepared_values(rows: Vec<Vec<Expression>>) -> Result<PreparedValues, OperatorError> {
    let column_count = rows.first().map_or(0, Vec::len);
    let row_count = rows.len();
    let mut columns = Vec::with_capacity(column_count);
    for column in 0..column_count {
        let mut param_indexes = Vec::with_capacity(row_count);
        let mut param_types = Vec::with_capacity(row_count);
        let mut output_type: Option<DataType> = None;
        for row in &rows {
            let (index, param_type, cell_output_type) = parameter_cell(&row[column])?;
            param_indexes.push(index);
            param_types.push(param_type);
            if output_type
                .as_ref()
                .is_some_and(|output_type| output_type != &cell_output_type)
            {
                return Err(OperatorError::Unsupported(
                    "VALUES rows in one column have different output types".to_string(),
                ));
            }
            output_type = Some(cell_output_type);
        }
        let output_type = output_type.ok_or_else(|| {
            OperatorError::Unsupported("VALUES with no rows is not supported".to_string())
        })?;
        columns.push(PreparedColumn::new(param_indexes, param_types, output_type));
    }
    Ok(PreparedValues::new(columns, row_count))
}

/// Resolve a prepared VALUES cell to `(parameter index, parameter type, emitted
/// Arrow type)`. The cell is a bound parameter, or a cast of one to the column
/// type.
fn parameter_cell(cell: &Expression) -> Result<(usize, Type, DataType), OperatorError> {
    match cell {
        Expression::Parameter(p) => {
            let ty = parameter_type(p)?;
            let arrow = physical_arrow_type(&ty);
            Ok((p.index, ty, arrow))
        }
        Expression::Cast(c) => match c.source.as_ref() {
            Expression::Parameter(p) => Ok((p.index, parameter_type(p)?, c.target_arrow.clone())),
            _ => Err(unsupported_values_cell()),
        },
        _ => Err(unsupported_values_cell()),
    }
}

/// A prepared `VALUES` parameter is bound to its target column, so DuckDB always
/// resolves its type; a missing one is unexpected.
fn parameter_type(p: &crate::expression::ParameterRef) -> Result<Type, OperatorError> {
    p.ty.clone().ok_or_else(|| {
        OperatorError::Unsupported(format!(
            "could not determine the type of parameter ${}",
            p.index + 1
        ))
    })
}

fn unsupported_values_cell() -> OperatorError {
    OperatorError::Unsupported(
        "VALUES mixing literals and parameters, or a computed parameter cell, is not supported"
            .to_string(),
    )
}

impl Insert {
    pub(crate) fn from_handle(view: InsertView<'_>) -> Result<Insert, OperatorError> {
        // pivot writes the input columns to the table positionally, so a column
        // list is fine only when it's the full set in table order (identity map).
        // A reordered or partial list would misroute columns.
        if !view.has_positional_column_map() {
            return Err(OperatorError::Unsupported(
                "INSERT with a reordered or partial target-column list is not supported"
                    .to_string(),
            ));
        }
        if view.returns_rows() {
            return Err(OperatorError::Unsupported(
                "INSERT ... RETURNING is not supported".to_string(),
            ));
        }
        Ok(Insert {
            table: resolve_table(*view.take_table()),
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
            table: resolve_table(*scan.take_table()),
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

/// Resolve a bound catalog entry into the Pivot [`Table`] it wraps: the DuckDB
/// table is a [`DuckDBTableAdapter`] holding the `Box<dyn Table>`.
fn resolve_table(wrapper: OptionalTableWrapper) -> Box<dyn Table> {
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
