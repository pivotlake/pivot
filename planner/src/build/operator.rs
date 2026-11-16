//! Constructs Pivot [`Operator`](crate::operator::Operator)s by walking DuckDB's
//! borrowed logical-operator handles. Each operator's `from_handle` is implemented
//! here (the walk in [`super`] dispatches to them), so the `crate::operator`
//! modules hold only Pivot IR (the operator types, their `Display`, and their
//! `compile`). Expression construction is the sibling
//! [`expression`](super::expression).

use std::any::Any;
use std::collections::HashMap;

use arrow_array::{Datum, Scalar, new_null_array};
use dispatch::{OrderBy as DispatchOrderBy, RowDelivery};
use duckdb_planner::DuckDBTable;
use duckdb_planner::ScalarValue;
use duckdb_planner::catalog_provider::OptionalTableWrapper;
use duckdb_planner::duckdb_bridge::duckdb_types::{ExpressionType, LimitNodeType, OrderByNullType};
use duckdb_planner::handle::{
    Aggregate as AggregateView, BridgeError, ChunkGet as ChunkGetView, Compact as CompactView,
    CopyFromStdin as CopyFromStdinView, CreateSchema as CreateSchemaView,
    CreateTable as CreateTableView, CreateUser as CreateUserView, Distinct as DistinctView,
    Drop as DropView, Expr, Expression as DuckExpression, Filter as FilterView,
    Insert as InsertView, Limit as LimitView, OrderBy as OrderByView, OrderKey,
    Projection as ProjectionView, Reset as ResetView, RunningFrame, Set as SetView, SortedByKey,
    TableFunctionScan as TableFunctionScanView, TableScan as TableScanView, TopN as TopNView,
    Unnest as UnnestView, UnnestedList as UnnestedListView, Values as ValuesView,
    Window as WindowView,
};

use super::{BuildCtx, build_scan_columns};
use crate::catalog::{
    BoundTable, Column, CreateSchemaRequest, CreateTableRequest, DropTableRequest,
    DuckDBTableAdapter, PARTITION_BY_OPTION, SORT_BY_OPTION, split_partition_terms,
};
use crate::dynamic_filter::JoinFilterScanInfo;
use crate::expression::{Error as ExpressionError, Expression, Ref};
use crate::operator::{
    Aggregate, Compact, CopyFormat, CopyFromStdin, CreateSchema, CreateTable, CreateUser, Distinct,
    DropTable, Error as OperatorError, Filter, Input, Insert, Limit, Operator, OrderBy,
    OrderByDirection, OrderByNode, Projection, SetVariable, TableFunctionScan, TopN, Unnest,
    UnnestedList, Values, Window, WindowFunction,
};
use crate::plan::PlanNode;
use crate::types::{Type, build_scalar_value, physical_arrow_type, type_from_logical};

impl Projection {
    pub(crate) fn from_handle(view: ProjectionView<'_>) -> Result<Projection, OperatorError> {
        Ok(Projection {
            projections: view
                .exprs()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Values {
    pub(crate) fn from_handle(view: ValuesView<'_>) -> Result<Values, OperatorError> {
        let rows = (0..view.row_count()?)
            .map(|row| {
                (0..view.column_count()?)
                    .map(|column| {
                        Expression::from_handle(view.expression(row, column)?)
                            .map_err(OperatorError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Values { rows })
    }

    /// A CHUNK_GET, DuckDB's scan of an in-memory constant chunk (what its
    /// optimizer rewrites a long `IN` list into), becomes a VALUES source
    /// whose cells are the chunk's constants.
    pub(crate) fn from_chunk_get(view: ChunkGetView<'_>) -> Result<Values, OperatorError> {
        let rows = (0..view.row_count()?)
            .map(|row| {
                (0..view.column_count()?)
                    .map(|column| {
                        let cell = view.value(column, row)?;
                        Ok(Expression::Constant(build_scalar_value(cell)?))
                    })
                    .collect::<Result<Vec<_>, OperatorError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Values { rows })
    }
}

impl Insert {
    /// Build the insert, along with the projection that reshapes its child when
    /// the statement named its target columns (see
    /// [`build_column_list_projection`]). The walk in [`super`] puts that
    /// projection between the two.
    pub(crate) fn from_handle(
        view: InsertView<'_>,
    ) -> Result<(Insert, Option<Projection>), OperatorError> {
        if view.returns_rows()? {
            return Err(OperatorError::Unsupported(
                "INSERT ... RETURNING is not supported".to_string(),
            ));
        }
        let column_map = view.column_map()?;
        let table = bind_table(*view.take_table()?);
        let column_list = build_column_list_projection(&column_map, &table.columns())?;
        Ok((Insert { table }, column_list))
    }
}

/// Reshape the child of an `INSERT INTO t (b, a) ...`, which emits only the
/// listed columns in the order they were written, into the table's full column
/// list: every table column reads the child column that fills it, and a column
/// the statement leaves out becomes a NULL of that column's type. Unfilled can
/// only mean NULL here, since the catalog carries no DEFAULT values.
///
/// `None` when the statement named no columns: its child already emits the
/// table's columns in order, which is what the insert path expects.
fn build_column_list_projection(
    column_map: &[Option<usize>],
    columns: &[Column],
) -> Result<Option<Projection>, OperatorError> {
    if column_map.is_empty() {
        return Ok(None);
    }
    if column_map.len() != columns.len() {
        return Err(OperatorError::InvalidStatement(format!(
            "INSERT names {} columns of a table that has {}",
            column_map.len(),
            columns.len()
        )));
    }
    let listed = column_map.iter().flatten().count();
    if listed == 0 {
        // The column map is all-unfilled only for DEFAULT VALUES: a written
        // column list always fills at least one column.
        return Err(OperatorError::Unsupported(
            "INSERT ... DEFAULT VALUES is not supported".to_string(),
        ));
    }
    // The binder resolved the names, so each source is a distinct child column;
    // check anyway, so a disagreement fails loudly instead of scrambling
    // columns or reading past the child's width.
    let mut filled = vec![false; listed];
    let projections = columns
        .iter()
        .zip(column_map)
        .map(|(column, source)| {
            let Some(&column_idx) = source.as_ref() else {
                let null = new_null_array(&physical_arrow_type(&column.col_type), 1);
                return Ok(Expression::Constant(Scalar::new(null)));
            };
            let taken = filled.get_mut(column_idx).ok_or_else(|| {
                OperatorError::InvalidStatement(format!(
                    "INSERT column \"{}\" reads value {column_idx} of {listed}",
                    column.name
                ))
            })?;
            if std::mem::replace(taken, true) {
                return Err(OperatorError::InvalidStatement(format!(
                    "INSERT fills more than one column from value {column_idx}"
                )));
            }
            Ok(Expression::Ref(Ref {
                column_idx,
                return_type: column.col_type.clone(),
                name: Some(column.name.clone()),
            }))
        })
        .collect::<Result<Vec<_>, OperatorError>>()?;
    Ok(Some(Projection { projections }))
}

impl Filter {
    pub(crate) fn from_handle(view: FilterView<'_>) -> Result<Filter, OperatorError> {
        Ok(Filter {
            conditions: view
                .exprs()?
                .into_iter()
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
                .groups()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: view
                .expressions()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Distinct {
    /// Build a [`Distinct`] from a `SELECT DISTINCT` node. The node's declared
    /// output is the child's columns unchanged, while the dedup operator emits
    /// its key columns in key order; the two agree only when the targets are
    /// exactly the child's columns, so anything else (a `DISTINCT ON`, whose
    /// non-target columns take an arbitrary row's values) is rejected. A key
    /// tuple dedups the same in any order, so sorting the keys into column
    /// order makes the output the child's layout.
    pub(crate) fn from_handle(
        view: DistinctView<'_>,
        child_width: usize,
    ) -> Result<Distinct, OperatorError> {
        if view.is_distinct_on()? {
            return Err(OperatorError::Unsupported(
                "DISTINCT ON is not supported".to_string(),
            ));
        }
        let mut keys = view
            .targets()?
            .into_iter()
            .map(|target| match Expression::from_handle(target)? {
                Expression::Ref(r) => Ok((r.column_idx, r.return_type)),
                other => Err(OperatorError::Unsupported(format!(
                    "DISTINCT over a computed expression is not supported: {other}"
                ))),
            })
            .collect::<Result<Vec<_>, OperatorError>>()?;
        keys.sort_unstable_by_key(|(column, _)| *column);
        let covers_child = keys.len() == child_width
            && keys.iter().enumerate().all(|(i, (column, _))| i == *column);
        if !covers_child {
            return Err(OperatorError::Unsupported(
                "DISTINCT over a subset of the output columns is not supported".to_string(),
            ));
        }
        Ok(Distinct { keys })
    }
}

impl Unnest {
    pub(crate) fn from_handle(view: UnnestView<'_>) -> Result<Unnest, OperatorError> {
        Ok(Unnest {
            lists: view
                .lists()?
                .into_iter()
                .map(build_unnested_list)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// Read one unnested list as an expression per element. The list has to be
/// written out in the query, which reaches here in one of two shapes: a
/// `list_value(...)` call with one argument per element, or, when every
/// element is a constant, the single list constant DuckDB folds that call into.
fn build_unnested_list(unnested: UnnestedListView<'_>) -> Result<UnnestedList, OperatorError> {
    let element_type = type_from_logical(unnested.element_type)?;
    // A variant column is told apart from a plain struct by a tag on its
    // field, which the unnest's output schema does not carry.
    if element_type == Type::Variant {
        return Err(OperatorError::Unsupported(
            "UNNEST of a list of VARIANT values is not supported".to_string(),
        ));
    }
    let elements = match unnested.list.expression()? {
        DuckExpression::Constant(list) => match list.value()? {
            // A NULL list unnests to no rows, the same as an empty one.
            ScalarValue::Null(_) => Vec::new(),
            _ => list
                .list_elements()?
                .into_iter()
                .map(|element| Ok(Expression::Constant(build_scalar_value(element)?)))
                .collect::<Result<Vec<_>, OperatorError>>()?,
        },
        DuckExpression::Function(list) if list.name()? == "list_value" => list
            .children()?
            .into_iter()
            .map(|element| Ok(Expression::from_handle(element)?))
            .collect::<Result<Vec<_>, OperatorError>>()?,
        _ => {
            return Err(OperatorError::Unsupported(
                "UNNEST is only supported over a list written out in the query, such as [a, b]"
                    .to_string(),
            ));
        }
    };
    Ok(UnnestedList {
        element_type,
        elements,
    })
}

impl OrderBy {
    pub(crate) fn from_handle(view: OrderByView<'_>) -> Result<OrderBy, OperatorError> {
        Ok(OrderBy {
            order_bys: build_orders(view.keys()?)?,
        })
    }
}

impl TopN {
    pub(super) fn from_handle(
        view: TopNView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<TopN, OperatorError> {
        let produces_dynamic_filter = match view.dynamic_filter()? {
            Some(df) => Some(ctx.dynamic_filter(df)?),
            None => None,
        };
        Ok(TopN {
            order_bys: build_orders(view.keys()?)?,
            limit: view.limit()?,
            offset: view.offset()?,
            produces_dynamic_filter,
        })
    }
}

impl Window {
    /// Translate a `LogicalWindow` over `child` into a subtree that emits what
    /// DuckDB's node does: the child's columns, then one column per window
    /// expression, in expression order.
    ///
    /// DuckDB puts every window expression of a `SELECT` into one
    /// `LogicalWindow`, whatever their `OVER` clauses. Pivot's window operator
    /// computes functions that share one `OVER` clause, so the result is a
    /// chain of [`Window`] nodes, one per distinct clause:
    ///
    /// ```text
    ///   DuckDB                         Pivot
    ///
    ///   LogicalWindow                  Projection   (only when the columns below
    ///     row_number() OVER (A)            │         are not already in DuckDB's
    ///     rank()       OVER (B)            │         order)
    ///     dense_rank() OVER (A)        Window (B): rank
    ///        │                             │
    ///      child                       Window (A): row_number, dense_rank
    ///                                      │
    ///                                  Projection   (only when a key is a
    ///                                      │         computed expression: it is
    ///                                    child       appended as a column)
    /// ```
    pub(super) fn from_handle(
        view: WindowView<'_>,
        child: PlanNode,
    ) -> Result<PlanNode, OperatorError> {
        // A scan that reads no column declares one placeholder column and
        // emits batches with none (see `Input::reads_no_column`). The columns
        // above are numbered after the placeholder, and a window appends its
        // functions to the columns that are really there, so the placeholder
        // gets a real column of NULLs to keep the numbering DuckDB resolved.
        let placeholder_type = match &child.operator {
            Operator::Input(scan) if scan.reads_no_column() => Some(scan.columns[0].result_type()?),
            _ => None,
        };
        let child = match placeholder_type {
            None => child,
            Some(placeholder_type) => {
                let nulls = new_null_array(&physical_arrow_type(&placeholder_type), 1);
                PlanNode {
                    name: "WINDOW_PLACEHOLDER_COLUMN".to_string(),
                    inputs: vec![child],
                    operator: Operator::Projection(Projection {
                        projections: vec![Expression::Constant(Scalar::new(nulls))],
                    }),
                }
            }
        };
        let child_types = child.output_types()?;
        let child_width = child_types.len();

        // One window per distinct `OVER` clause, in order of each clause's
        // first appearance, and for every function of each window, which
        // window expression it is.
        let mut windows: Vec<Window> = Vec::new();
        let mut expressions_of_window: Vec<Vec<usize>> = Vec::new();
        let mut computed_keys = ComputedWindowKeys::new(child_width);
        let expressions = view.expressions()?;
        for (expression_index, expression) in expressions.iter().enumerate() {
            let function = match expression.function_type()? {
                ExpressionType::WINDOW_ROW_NUMBER => WindowFunction::RowNumber,
                ExpressionType::WINDOW_RANK => WindowFunction::Rank,
                ExpressionType::WINDOW_RANK_DENSE => WindowFunction::DenseRank,
                ExpressionType::WINDOW_PERCENT_RANK => WindowFunction::PercentRank,
                kind @ (ExpressionType::WINDOW_LAG | ExpressionType::WINDOW_LEAD) => {
                    if expression.ignores_nulls()? {
                        return Err(OperatorError::Unsupported(format!(
                            "{kind} with IGNORE NULLS is not supported"
                        )));
                    }
                    let arguments = expression.arguments()?;
                    let column = computed_keys.find_key_column(arguments[0])?;
                    let offset = match arguments.get(1) {
                        None => 1,
                        Some(&offset) => Window::read_integer_constant(&Expression::from_handle(
                            offset,
                        )?)
                        .and_then(|offset| u64::try_from(offset).ok())
                        .ok_or_else(|| {
                            OperatorError::Unsupported(format!(
                                "the offset of {kind} must be a constant that is not negative"
                            ))
                        })?,
                    };
                    if let Some(&default) = arguments.get(2) {
                        let default_is_null = match Expression::from_handle(default)? {
                            Expression::Constant(default) => default.get().0.is_null(0),
                            _ => false,
                        };
                        if !default_is_null {
                            return Err(OperatorError::Unsupported(format!(
                                "a default value for {kind} is not supported"
                            )));
                        }
                    }
                    if kind == ExpressionType::WINDOW_LAG {
                        WindowFunction::Lag { column, offset }
                    } else {
                        WindowFunction::Lead { column, offset }
                    }
                }
                ExpressionType::WINDOW_AGGREGATE => {
                    let aggregate = expression.aggregate_name()?.unwrap_or_default();
                    if !matches!(aggregate.as_str(), "sum" | "sum_no_overflow")
                        || expression.is_distinct_or_filtered()?
                    {
                        return Err(OperatorError::Unsupported(format!(
                            "window aggregate {aggregate} is not supported; only a plain sum is"
                        )));
                    }
                    let Some(frame) = expression.running_frame()? else {
                        return Err(OperatorError::Unsupported(
                            "sum over a window is supported for a frame from the start of the \
                             partition to the current row"
                                .to_string(),
                        ));
                    };
                    let arguments = expression.arguments()?;
                    WindowFunction::RunningSum {
                        column: computed_keys.find_key_column(arguments[0])?,
                        through_tie_group: frame == RunningFrame::ThroughTies,
                    }
                }
                other => {
                    return Err(OperatorError::Unsupported(format!(
                        "window function {other} is not supported"
                    )));
                }
            };
            // An ordering inside the call, as in `rank(ORDER BY x) OVER (...)`,
            // replaces the OVER clause's ordering for that one function.
            if expression.argument_order_count()? != 0 {
                return Err(OperatorError::Unsupported(
                    "an ORDER BY inside a window function call is not supported; \
                     write the ordering in the OVER clause"
                        .to_string(),
                ));
            }
            let partition_by = expression
                .partitions()?
                .into_iter()
                .map(|key| computed_keys.find_key_column(key))
                .collect::<Result<Vec<_>, _>>()?;
            let order_keys = expression.order_keys()?;
            let order_key_handles: Vec<Expr<'_>> =
                order_keys.iter().map(|key| key.expression).collect();
            let order_by = build_orders(order_keys)?
                .into_iter()
                .zip(order_key_handles)
                .map(|(key, handle)| {
                    Ok(DispatchOrderBy::new(
                        computed_keys.find_key_column(handle)?,
                        matches!(key.direction, OrderByDirection::Desc),
                        key.nulls_first,
                    ))
                })
                .collect::<Result<Vec<_>, OperatorError>>()?;
            let same_clause = windows.iter().position(|window| {
                window.partition_by == partition_by && window.order_by == order_by
            });
            match same_clause {
                Some(window_index) => {
                    windows[window_index].functions.push(function);
                    expressions_of_window[window_index].push(expression_index);
                }
                None => {
                    windows.push(Window {
                        partition_by,
                        order_by,
                        functions: vec![function],
                        row_limit: None,
                    });
                    expressions_of_window.push(vec![expression_index]);
                }
            }
        }

        let mut column_types = child_types;
        let mut node = child;
        if !computed_keys.expressions.is_empty() {
            node = computed_keys.append_to(node, &mut column_types)?;
        }

        // Stack one node per clause. Each appends its functions' columns to
        // whatever the node below emits.
        let mut column_of_expression = vec![0; expressions.len()];
        for (window, expression_indices) in windows.into_iter().zip(expressions_of_window) {
            let function_types = window.function_types(&column_types)?;
            for (expression_index, function_type) in
                expression_indices.into_iter().zip(function_types)
            {
                column_of_expression[expression_index] = column_types.len();
                column_types.push(function_type);
            }
            node = PlanNode {
                name: "WINDOW".to_string(),
                inputs: vec![node],
                operator: Operator::Window(window),
            };
        }

        // DuckDB's layout is the child's columns followed by the expressions
        // in order. The chain emits that already unless it appended key
        // columns or grouped the expressions out of order.
        let duckdb_layout: Vec<usize> = (0..child_width).chain(column_of_expression).collect();
        let already_in_duckdb_layout = duckdb_layout.len() == column_types.len()
            && duckdb_layout
                .iter()
                .enumerate()
                .all(|(position, &column)| position == column);
        if already_in_duckdb_layout {
            return Ok(node);
        }
        let output_columns = duckdb_layout
            .into_iter()
            .map(|column_idx| (column_idx, column_types[column_idx].clone()));
        Ok(PlanNode {
            name: "WINDOW_OUTPUT_ORDER".to_string(),
            inputs: vec![node],
            operator: Operator::Projection(Projection::of_columns(output_columns)),
        })
    }
}

/// The partition keys, order keys and function arguments of a `LogicalWindow`
/// that are computed expressions instead of plain columns. Each is projected onto the end of
/// the child's columns once, and every key that computes the same expression
/// reads that one column. Two functions written with the same `OVER` clause
/// then compare as the same clause and share one [`Window`] node.
struct ComputedWindowKeys<'plan> {
    child_width: usize,
    /// DuckDB's expression of each computed key, for telling whether a later
    /// key computes the same thing.
    handles: Vec<Expr<'plan>>,
    /// The same keys as Pivot expressions, in the order their columns follow
    /// the child's columns.
    expressions: Vec<Expression>,
}

impl<'plan> ComputedWindowKeys<'plan> {
    fn new(child_width: usize) -> Self {
        Self {
            child_width,
            handles: Vec::new(),
            expressions: Vec::new(),
        }
    }

    /// The column a partition or order key is read from. A plain column is
    /// read where it is. A computed expression is read from the column
    /// appended for it, which is added on its first use.
    fn find_key_column(&mut self, key: Expr<'plan>) -> Result<usize, OperatorError> {
        let expression = Expression::from_handle(key)?;
        if let Expression::Ref(column) = &expression {
            return Ok(column.column_idx);
        }
        for (position, existing_key) in self.handles.iter().enumerate() {
            if existing_key.equals(key)? {
                return Ok(self.child_width + position);
            }
        }
        self.handles.push(key);
        self.expressions.push(expression);
        Ok(self.child_width + self.expressions.len() - 1)
    }

    /// Wrap `input` in a projection that passes its columns through and
    /// appends the computed keys after them. `column_types` holds the input's
    /// column types and gains the keys' types.
    fn append_to(
        self,
        input: PlanNode,
        column_types: &mut Vec<Type>,
    ) -> Result<PlanNode, OperatorError> {
        let mut projection = Projection::of_columns(column_types.iter().cloned().enumerate());
        for key in &self.expressions {
            column_types.push(key.result_type()?);
        }
        projection.projections.extend(self.expressions);
        Ok(PlanNode {
            name: "WINDOW_KEY_COMPUTE".to_string(),
            inputs: vec![input],
            operator: Operator::Projection(projection),
        })
    }
}

impl Limit {
    pub(crate) fn from_handle(view: LimitView<'_>) -> Result<Limit, OperatorError> {
        Ok(Limit {
            limit: limit_bound(view.value_kind()?, || view.value(), "value")?,
            offset: limit_bound(view.offset_kind()?, || view.offset(), "offset")?.unwrap_or(0),
        })
    }
}

impl Input {
    pub(super) fn from_handle(
        scan: TableScanView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<Input, OperatorError> {
        let join_filter_info = match scan.join_filter_set_id()? {
            Some(filter_set_id) => Some(JoinFilterScanInfo {
                filter_set_id,
                proj_to_storage: scan.storage_columns()?,
            }),
            None => None,
        };
        Ok(Input {
            table: bind_table(*scan.take_table()?),
            columns: build_scan_columns(scan.output_columns()?)?,
            dynamic_filters: scan
                .dynamic_filters()?
                .into_iter()
                .map(|df| ctx.dynamic_filter(df))
                .collect::<Result<Vec<_>, _>>()?,
            key_bitset_filters: Vec::new(),
            duckdb_table_binding_index: Some(scan.table_index()?),
            join_filter_info,
            emit_row_group_metadata: false,
        })
    }
}

impl TableFunctionScan {
    pub(super) fn from_handle(
        view: TableFunctionScanView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<TableFunctionScan, OperatorError> {
        // Only positional parameters are supported; reject named parameters rather
        // than silently drop a bound argument.
        if view.has_named_params()? {
            return Err(OperatorError::Unsupported(format!(
                "table function {} with named parameters is not supported",
                view.function_name()?
            )));
        }
        if !view.has_bound_table()? {
            return Err(OperatorError::Unsupported(format!(
                "table function {} was not registered by Pivot",
                view.function_name()?
            )));
        }
        let bound_table = bind_table(*view.bound_table()?);
        Ok(TableFunctionScan::new(
            view.function_name()?,
            view.params()?,
            build_scan_columns(view.output_columns()?)?,
            bound_table,
            view.dynamic_filters()?
                .into_iter()
                .map(|df| ctx.dynamic_filter(df))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl CreateSchema {
    pub(crate) fn from_handle(view: CreateSchemaView<'_>) -> Result<CreateSchema, OperatorError> {
        Ok(CreateSchema {
            request: CreateSchemaRequest {
                datastore_name: view.datastore()?,
                name: view.name()?,
                if_not_exists: view.if_not_exists()?,
            },
            or_replace: view.or_replace()?,
        })
    }
}

impl DropTable {
    pub(crate) fn from_handle(view: DropView<'_>) -> Result<DropTable, OperatorError> {
        Ok(DropTable {
            request: DropTableRequest {
                datastore_name: view.datastore()?,
                schema_name: view.schema()?,
                name: view.name()?,
                if_exists: view.if_exists()?,
            },
            cascade: view.cascade()?,
        })
    }
}

impl CreateTable {
    pub(crate) fn from_handle(view: CreateTableView<'_>) -> Result<CreateTable, OperatorError> {
        let mut options: HashMap<String, String> = view.options()?.into_iter().collect();
        // A clause names what a table option names, so it reaches the
        // datastore as that option, and a statement gives one of the two.
        let refuse_both = |named: &str, clause: &str, option: &str| {
            OperatorError::InvalidStatement(format!(
                "CREATE TABLE names its {named} twice, by {clause} and by the `{option}` option; give one of them"
            ))
        };
        let partitioned_by = view.partitioned_by()?;
        if !partitioned_by.is_empty() {
            if options.contains_key(PARTITION_BY_OPTION) {
                return Err(refuse_both(
                    "partitioning",
                    "PARTITIONED BY",
                    PARTITION_BY_OPTION,
                ));
            }
            let terms = partitioned_by.join(", ");
            // The option separates its terms by the commas outside
            // parentheses, so a key that reads back as other terms has no
            // spelling in it.
            if split_partition_terms(&terms) != partitioned_by {
                return Err(OperatorError::InvalidStatement(format!(
                    "PARTITIONED BY ({terms}) names a key the `{PARTITION_BY_OPTION}` option has no spelling for; a key is a column or a transform of one"
                )));
            }
            options.insert(PARTITION_BY_OPTION.to_string(), terms);
        }
        // The clause's grammar takes no direction and no NULL ordering, so a
        // key that is a column is what the option names: the column,
        // ascending with NULLs first.
        let sorted_by = view.sorted_by()?;
        if !sorted_by.is_empty() {
            if options.contains_key(SORT_BY_OPTION) {
                return Err(refuse_both("sorting", "SORTED BY", SORT_BY_OPTION));
            }
            let columns = sorted_by
                .into_iter()
                .map(|key| match key {
                    // The option separates its columns by commas.
                    SortedByKey::Column(column) if !column.contains(',') => Ok(column),
                    SortedByKey::Column(key) | SortedByKey::Expression(key) => {
                        Err(OperatorError::InvalidStatement(format!(
                            "SORTED BY key `{key}` is not a column the `{SORT_BY_OPTION}` option can name; a table is sorted by columns, each ascending with NULLs first"
                        )))
                    }
                })
                .collect::<Result<Vec<String>, OperatorError>>()?;
            options.insert(SORT_BY_OPTION.to_string(), columns.join(", "));
        }
        Ok(CreateTable {
            request: CreateTableRequest {
                datastore_name: view.datastore()?,
                schema_name: view.schema()?,
                name: view.name()?,
                columns: view
                    .columns()?
                    .into_iter()
                    .map(|(name, col_type)| {
                        Ok(Column {
                            name,
                            col_type: type_from_logical(col_type)?,
                        })
                    })
                    .collect::<Result<Vec<_>, OperatorError>>()?,
                options,
                if_not_exists: view.if_not_exists()?,
            },
            or_replace: view.or_replace()?,
            temporary: view.temporary()?,
            has_query: view.has_query()?,
            constraint_count: view.constraint_count()?,
        })
    }
}

impl SetVariable {
    pub(crate) fn from_set(view: SetView<'_>) -> Result<SetVariable, BridgeError> {
        Ok(SetVariable {
            name: view.name()?,
            value: Some(view.value()?),
        })
    }

    /// `RESET name` is modelled as a `SET` with no value (the consumer reads "no
    /// value" as "off / default").
    pub(crate) fn from_reset(view: ResetView<'_>) -> Result<SetVariable, BridgeError> {
        Ok(SetVariable {
            name: view.name()?,
            value: None,
        })
    }
}

impl Compact {
    pub(crate) fn from_handle(view: CompactView<'_>) -> Result<Compact, BridgeError> {
        Ok(Compact {
            datastore: view.datastore()?,
            schema: view.schema()?,
            table: view.table()?,
            final_sweep: view.final_sweep()?,
        })
    }
}

impl CopyFromStdin {
    pub(crate) fn from_handle(view: CopyFromStdinView<'_>) -> Result<CopyFromStdin, OperatorError> {
        let table = bind_table(*view.take_table()?);
        let columns = view.column_indexes()?;
        // Formats are case-insensitive; canonicalize before interpreting.
        let format = view.format()?.map(|format| format.to_lowercase());
        let format = CopyFormat::resolve(format.as_deref(), &view.options()?)
            .map_err(OperatorError::InvalidStatement)?;

        Ok(CopyFromStdin {
            table,
            columns,
            format,
        })
    }
}

impl CreateUser {
    pub(crate) fn from_handle(view: CreateUserView<'_>) -> Result<CreateUser, BridgeError> {
        Ok(CreateUser {
            name: view.name()?,
            password: view.password()?,
        })
    }
}

/// Lower a sequence of handle sort keys into Pivot [`OrderByNode`]s. Shared by the
/// `OrderBy` and `TopN` constructors and the window translation.
fn build_orders(keys: Vec<OrderKey<'_>>) -> Result<Vec<OrderByNode>, ExpressionError> {
    keys.into_iter()
        .map(|key| {
            let nulls_first = match key.null_order {
                OrderByNullType::NULLS_FIRST => true,
                OrderByNullType::NULLS_LAST => false,
                unresolved => return Err(ExpressionError::UnsupportedNullOrder(unresolved)),
            };
            Ok(OrderByNode {
                direction: key.direction.into(),
                nulls_first,
                expression: Expression::from_handle(key.expression)?,
            })
        })
        .collect()
}

/// Resolve a bound catalog entry into the Pivot [`BoundTable`] it wraps.
fn bind_table(wrapper: OptionalTableWrapper) -> Box<dyn BoundTable> {
    let duck: Box<dyn DuckDBTable> = wrapper.table.expect("planner returned an unbound table");
    let any: Box<dyn Any> = duck;
    let adapter: Box<DuckDBTableAdapter> = any
        .downcast::<DuckDBTableAdapter>()
        .expect("scan table should be the planner's table adapter");
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
    value: impl FnOnce() -> Result<usize, BridgeError>,
    what: &str,
) -> Result<Option<usize>, OperatorError> {
    match kind {
        LimitNodeType::UNSET => Ok(None),
        LimitNodeType::CONSTANT_VALUE => Ok(Some(value()?)),
        _ => Err(OperatorError::Unsupported(format!(
            "Unsupported non-constant LIMIT {what}"
        ))),
    }
}
