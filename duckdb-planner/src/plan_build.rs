//! Builds the Rust [`PlanNode`] tree by walking DuckDB's own plan objects.
//!
//! `extract_plan` hands back a [`PlanHandle`](crate::duckdb_bridge::ffi::PlanHandle)
//! that owns DuckDB's resolved `LogicalOperator` tree. This module walks that
//! live tree through the FFI accessors (`lo_*` for operators, `expr_*` for
//! expressions) and assembles the typed [`Operator`]/[`Expression`] values.
//! Nothing is serialized: the discriminants the accessors report are DuckDB's
//! own `LogicalOperatorType` / `ExpressionType` bytes, matched here against the
//! [`duckdb_types`](crate::duckdb_bridge::duckdb_types) mirrors.
//!
//! The DuckDB-specific shaping the bridge used to do in C++ now lives here:
//! collapsing the late-materialization SEMI join into a [`Materialize`], lifting
//! a scan's pushed-down `table_filters` back into a `Filter`, replaying a
//! filter's `projection_map`, and stripping the threaded-up row-id column.
//!
//! [`Materialize`]: crate::operator::Materialize

use std::collections::HashMap;

use crate::catalog_provider::OptionalTableWrapper;
use crate::duckdb_bridge::duckdb_types::{
    ExpressionType, LimitNodeType, LogicalOperatorType, LogicalTypeId,
};
use crate::duckdb_bridge::ffi;
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{
    AggregateFunc, Between, Case, CaseCheck, Cast, Compare, Conjunction, Expression, Function,
    InList, Not, Ref,
};
use crate::operator::{
    Aggregate, CreateTable, CreateTableColumn, DummyScan, Explain, Filter, Limit, Operator,
    OrderBy, OrderByDirection, OrderByNode, Projection, RawInput, RawMaterialize, SetVariable,
    TableFunctionScan, TopN,
};
use crate::plan::PlanNode;
use crate::types::ScalarValue;

/// Error raised when the bridge reports a node or expression shape the Rust
/// builder doesn't support (e.g. a non-constant `LIMIT`, or an expression type
/// with no mapping). Surfaces as a `Bridge` error from `plan`.
#[derive(Debug, thiserror::Error)]
#[error("plan build error: {0}")]
pub struct BuildError(pub String);

/// The tables collected from each base-table scan during the walk, in scan
/// order; `RawInput::table_id` indexes this list and `resolve_inputs` later
/// binds them.
type Tables = Vec<Box<OptionalTableWrapper>>;
/// Maps a shared `DynamicFilterData` cell (by pointer identity) to its slot id,
/// so a Top-N producer and the scans that consume its filter agree on a slot.
type DynamicFilterSlots = HashMap<usize, usize>;

/// Build the whole plan tree from DuckDB's root operator, collecting the scanned
/// tables (still unresolved) into `tables`.
pub(crate) fn build_plan(
    root: &ffi::LogicalOperator,
    tables: &mut Tables,
) -> Result<PlanNode, BuildError> {
    let mut dynamic_filter_slots = DynamicFilterSlots::new();
    build_plan_node(root, tables, &mut dynamic_filter_slots)
}

fn build_plan_node(
    op: &ffi::LogicalOperator,
    tables: &mut Tables,
    dynamic_filter_slots: &mut DynamicFilterSlots,
) -> Result<PlanNode, BuildError> {
    use LogicalOperatorType as L;
    let lo = L::from_u8(ffi::lo_type(op));

    // DuckDB's late_materialization optimizer rewrites a wide Top-N/Limit scan
    // into a row-id SEMI join. Collapse that into pivot's Materialize rather than
    // executing a join (only the late-mat shape, not a user IN/EXISTS semi-join).
    if matches!(lo, L::LOGICAL_COMPARISON_JOIN) && ffi::lo_is_late_materialization_join(op) {
        return build_late_materialization(op, tables, dynamic_filter_slots);
    }

    let mut inputs = Vec::with_capacity(ffi::lo_child_count(op));
    for i in 0..ffi::lo_child_count(op) {
        inputs.push(build_plan_node(
            ffi::lo_child(op, i),
            tables,
            dynamic_filter_slots,
        )?);
    }

    // Drop the row-id ORDER BY DuckDB synthesizes above a late-materialized plain
    // LIMIT: pivot can't run it and doesn't need it. The ORDER BY is a
    // pass-through, so returning its child keeps column positions unchanged.
    if matches!(lo, L::LOGICAL_ORDER_BY)
        && inputs.first().is_some_and(is_materialize_over_empty_scan)
    {
        return Ok(inputs.into_iter().next().unwrap());
    }

    // Static `col op const` filters DuckDB pushed into a scan's table_filters,
    // lifted back into a synthetic Filter above the scan below.
    let mut pushed_conditions: Vec<Expression> = Vec::new();

    let operator = match lo {
        L::LOGICAL_PROJECTION => Operator::Projection(Projection {
            projections: build_expr_seq(ffi::lo_projection_expr_count(op), |i| {
                ffi::lo_projection_expr(op, i)
            })?,
        }),
        L::LOGICAL_GET if !ffi::lo_get_has_table(op) => build_table_function_scan(op)?,
        L::LOGICAL_GET => {
            pushed_conditions = build_pushed_conditions(op)?;
            let table_id = tables.len();
            tables.push(ffi::lo_get_take_table(op));
            Operator::RawInput(RawInput {
                table_id,
                columns: build_get_output_columns(op)?,
                dynamic_filters: build_get_dynamic_filters(op, dynamic_filter_slots),
                emit_row_group_metadata: false,
            })
        }
        L::LOGICAL_ORDER_BY => Operator::OrderBy(OrderBy {
            order_bys: build_orders(
                ffi::lo_orderby_count(op),
                |i| ffi::lo_orderby_direction(op, i),
                |i| ffi::lo_orderby_expr(op, i),
            )?,
        }),
        L::LOGICAL_AGGREGATE_AND_GROUP_BY => Operator::Aggregate(Aggregate {
            groups: build_expr_seq(ffi::lo_aggregate_group_count(op), |i| {
                ffi::lo_aggregate_group(op, i)
            })?,
            expressions: build_expr_seq(ffi::lo_aggregate_expr_count(op), |i| {
                ffi::lo_aggregate_expr(op, i)
            })?,
        }),
        L::LOGICAL_FILTER => Operator::Filter(Filter {
            conditions: build_expr_seq(ffi::lo_filter_expr_count(op), |i| {
                ffi::lo_filter_expr(op, i)
            })?,
        }),
        L::LOGICAL_TOP_N => Operator::TopN(TopN {
            order_bys: build_orders(
                ffi::lo_topn_order_count(op),
                |i| ffi::lo_topn_order_direction(op, i),
                |i| ffi::lo_topn_order_expr(op, i),
            )?,
            limit: ffi::lo_topn_limit(op),
            offset: ffi::lo_topn_offset(op),
            produces_dynamic_filter: ffi::lo_topn_has_dynamic_filter(op).then(|| DynamicFilter {
                slot_id: get_or_assign_slot(
                    dynamic_filter_slots,
                    ffi::lo_topn_dynamic_filter_data_id(op),
                ),
                column_idx: ffi::lo_topn_dynamic_filter_column(op),
                compare_type: ExpressionType::from_u8(ffi::lo_topn_dynamic_filter_comparison(op)),
            }),
        }),
        L::LOGICAL_LIMIT => Operator::Limit(Limit {
            limit: build_limit_bound(
                ffi::lo_limit_value_kind(op),
                || ffi::lo_limit_value(op),
                "value",
            )?,
            offset: build_limit_bound(
                ffi::lo_limit_offset_kind(op),
                || ffi::lo_limit_offset(op),
                "offset",
            )?,
        }),
        L::LOGICAL_CREATE_TABLE => build_create_table(op),
        L::LOGICAL_DUMMY_SCAN => Operator::DummyScan(DummyScan {}),
        L::LOGICAL_EXPLAIN => Operator::Explain(Explain {}),
        L::LOGICAL_SET => Operator::Set(SetVariable {
            name: ffi::lo_set_name(op),
            value: Some(ffi::lo_set_value(op)),
        }),
        // Model RESET as a SET with no value (the consumer reads "no value" as
        // "off / default").
        L::LOGICAL_RESET => Operator::Set(SetVariable {
            name: ffi::lo_reset_name(op),
            value: None,
        }),
        _ => {
            return Err(BuildError(format!(
                "Unsupported operator type: {}",
                ffi::lo_name(op)
            )));
        }
    };

    let node = PlanNode {
        name: ffi::lo_name(op),
        inputs,
        operator,
    };

    // Reattach the scan's static pushed-down filters as a Filter above it, so the
    // Rust side keeps seeing `Filter -> Input` exactly as with filter_pushdown off.
    if matches!(lo, L::LOGICAL_GET) && !pushed_conditions.is_empty() {
        return Ok(PlanNode {
            name: "PUSHDOWN_FILTER".to_string(),
            inputs: vec![node],
            operator: Operator::Filter(Filter {
                conditions: pushed_conditions,
            }),
        });
    }

    // A LogicalFilter may carry a projection_map: it outputs only the listed
    // subset/reordering of its child's columns. Replay it by wrapping the filter
    // in a Projection that selects exactly projection_map, positionally, so refs
    // above the filter line up.
    if matches!(lo, L::LOGICAL_FILTER) && ffi::lo_filter_projection_map_count(op) > 0 {
        let projections = (0..ffi::lo_filter_projection_map_count(op))
            .map(|i| {
                Expression::Ref(Ref {
                    column_idx: ffi::lo_filter_projection_map_index(op, i),
                    return_type: LogicalTypeId::from_u8(ffi::lo_filter_type_id(op, i)),
                    name: None,
                })
            })
            .collect();
        return Ok(PlanNode {
            name: "FILTER_PROJECTION".to_string(),
            inputs: vec![node],
            operator: Operator::Projection(Projection { projections }),
        });
    }

    Ok(node)
}

/// Collapse DuckDB's late-materialization SEMI join into a pivot Materialize: the
/// narrow pipeline (RHS) runs as-is with its row-id column stripped, and the
/// materializer re-reads the LHS columns for the surviving rows.
fn build_late_materialization(
    op: &ffi::LogicalOperator,
    tables: &mut Tables,
    dynamic_filter_slots: &mut DynamicFilterSlots,
) -> Result<PlanNode, BuildError> {
    let columns: Vec<usize> = (0..ffi::lo_late_materialization_column_count(op))
        .map(|i| ffi::lo_late_materialization_column(op, i))
        .collect();

    // The narrow pipeline is the RHS; translate it normally, then drop the row-id
    // column DuckDB threaded through it for the join we're discarding.
    let mut child = build_plan_node(ffi::lo_child(op, 1), tables, dynamic_filter_slots)?;
    strip_trailing_rowid(&mut child);
    // Tag the narrow scan to emit row-group metadata, and reuse its table_id for
    // the Materialize so both resolve to (a clone of) the same table.
    let table_id = prepare_narrow_scan(&mut child).ok_or_else(|| {
        BuildError("late materialization without a base-table scan is not supported".to_string())
    })?;

    Ok(PlanNode {
        name: "Materialize".to_string(),
        inputs: vec![child],
        operator: Operator::RawMaterialize(RawMaterialize { table_id, columns }),
    })
}

fn build_table_function_scan(op: &ffi::LogicalOperator) -> Result<Operator, BuildError> {
    // Only positional parameters are supported; reject named parameters rather
    // than silently drop a bound argument.
    if ffi::lo_get_has_named_params(op) {
        return Err(BuildError(format!(
            "table function {} with named parameters is not supported",
            ffi::lo_get_function_name(op)
        )));
    }
    Ok(Operator::TableFunctionScan(TableFunctionScan {
        function_name: ffi::lo_get_function_name(op),
        args: (0..ffi::lo_get_param_count(op))
            .map(|i| ScalarValue {
                logical_type: LogicalTypeId::from_u8(ffi::lo_get_param_type(op, i)),
                raw_value: ffi::lo_get_param_value(op, i),
            })
            .collect(),
        columns: build_get_output_columns(op)?,
    }))
}

/// A scan's projected output columns, each a positional `BOUND_REF` over storage
/// column indices.
fn build_get_output_columns(op: &ffi::LogicalOperator) -> Result<Vec<Expression>, BuildError> {
    Ok((0..ffi::lo_get_output_count(op))
        .map(|i| {
            Expression::Ref(Ref {
                column_idx: ffi::lo_get_output_column(op, i),
                return_type: LogicalTypeId::from_u8(ffi::lo_get_output_type(op, i)),
                name: None,
            })
        })
        .collect())
}

fn build_get_dynamic_filters(
    op: &ffi::LogicalOperator,
    dynamic_filter_slots: &mut DynamicFilterSlots,
) -> Vec<DynamicFilter> {
    (0..ffi::lo_get_dynamic_filter_count(op))
        .map(|i| DynamicFilter {
            slot_id: get_or_assign_slot(
                dynamic_filter_slots,
                ffi::lo_get_dynamic_filter_data_id(op, i),
            ),
            column_idx: ffi::lo_get_dynamic_filter_column(op, i),
            compare_type: ExpressionType::from_u8(ffi::lo_get_dynamic_filter_comparison(op, i)),
        })
        .collect()
}

fn build_pushed_conditions(op: &ffi::LogicalOperator) -> Result<Vec<Expression>, BuildError> {
    let list = ffi::lo_get_pushed_conditions(op).map_err(|e| BuildError(e.to_string()))?;
    (0..ffi::expr_list_count(&list))
        .map(|i| build_expression(ffi::expr_list_get(&list, i)))
        .collect()
}

fn build_create_table(op: &ffi::LogicalOperator) -> Operator {
    Operator::CreateTable(CreateTable {
        name: ffi::lo_create_table_name(op),
        columns: (0..ffi::lo_create_column_count(op))
            .map(|i| CreateTableColumn {
                name: ffi::lo_create_column_name(op, i),
                col_type: LogicalTypeId::from_u8(ffi::lo_create_column_type(op, i)),
            })
            .collect(),
        options: (0..ffi::lo_create_option_count(op))
            .map(|i| {
                (
                    ffi::lo_create_option_key(op, i),
                    ffi::lo_create_option_value(op, i),
                )
            })
            .collect(),
        if_not_exists: ffi::lo_create_if_not_exists(op),
        or_replace: ffi::lo_create_or_replace(op),
        temporary: ffi::lo_create_temporary(op),
        has_query: ffi::lo_create_has_query(op),
        constraint_count: ffi::lo_create_constraint_count(op),
    })
}

/// Interpret a `LIMIT`/`OFFSET` bound's [`LimitNodeType`] into an
/// `Option<usize>`: an unset bound is unbounded (`None`), a constant is the row
/// count, and a percentage/expression bound is rejected (pivot only handles a
/// fixed row count).
fn build_limit_bound(
    kind: u8,
    value: impl Fn() -> usize,
    what: &str,
) -> Result<Option<usize>, BuildError> {
    match LimitNodeType::from_u8(kind) {
        LimitNodeType::UNSET => Ok(None),
        LimitNodeType::CONSTANT_VALUE => Ok(Some(value())),
        _ => Err(BuildError(format!("Unsupported non-constant LIMIT {what}"))),
    }
}

fn build_orders<'a>(
    count: usize,
    direction: impl Fn(usize) -> u8,
    expr: impl Fn(usize) -> &'a ffi::Expression,
) -> Result<Vec<OrderByNode>, BuildError> {
    (0..count)
        .map(|i| {
            Ok(OrderByNode {
                direction: OrderByDirection::from_u8(direction(i)),
                expression: build_expression(expr(i))?,
            })
        })
        .collect()
}

/// Get the dense slot id for a shared dynamic-filter cell, assigning a new one
/// on first sight.
///
/// A Top-N produces a dynamic filter into a `DynamicFilterData` cell that one or
/// more scans consume; producer and consumers reference the *same* cell. We
/// can't pass that C++ pointer to Rust, so each cell is identified by its address
/// (`data_id`). This assigns a stable small integer per distinct address (first
/// seen `0`, next `1`, ...), returning the same slot for the same cell, so a
/// producer and its consumers share a `slot_id` the compile step can wire up.
fn get_or_assign_slot(dynamic_filter_slots: &mut DynamicFilterSlots, data_id: usize) -> usize {
    let next_slot = dynamic_filter_slots.len();
    *dynamic_filter_slots.entry(data_id).or_insert(next_slot)
}

// ---- Late-materialization tree surgery (operates on the built Rust tree) ----

/// Whether `node` is a late-mat Materialize whose narrow scan reads no data
/// columns (the shape of a plain LIMIT, once the row-id column is stripped).
fn is_materialize_over_empty_scan(node: &PlanNode) -> bool {
    if node.name != "Materialize" {
        return false;
    }
    let mut cur = node;
    while let Some(child) = cur.inputs.first() {
        cur = child;
        if let Operator::RawInput(raw) = &cur.operator {
            return raw.columns.is_empty();
        }
    }
    false
}

/// Remove DuckDB's row-id column from an already-built late-mat narrow subtree.
/// DuckDB appends the row-id last at every level, so dropping it never shifts
/// another column's position. Returns the output position that held the row-id.
fn strip_trailing_rowid(node: &mut PlanNode) -> Option<usize> {
    let rowid = ffi::rowid_column_id();
    if let Operator::RawInput(raw) = &mut node.operator {
        let pos = raw
            .columns
            .iter()
            .position(|e| matches!(e, Expression::Ref(r) if r.column_idx == rowid))?;
        raw.columns.remove(pos);
        return Some(pos);
    }

    let child_rowid = strip_trailing_rowid(node.inputs.first_mut()?);

    // A projection that carried the row-id up references it positionally in its
    // child's output; drop that one entry. Other operators pass columns through.
    if let (Operator::Projection(proj), Some(rowid_pos)) = (&mut node.operator, child_rowid) {
        if let Some(pos) = proj
            .projections
            .iter()
            .position(|e| matches!(e, Expression::Ref(r) if r.column_idx == rowid_pos))
        {
            proj.projections.remove(pos);
            return Some(pos);
        }
    }
    child_rowid
}

/// Walk a late-mat narrow subtree to its scan, flag it to emit row-group metadata,
/// and return its `table_id` (which the Materialize reuses).
fn prepare_narrow_scan(node: &mut PlanNode) -> Option<usize> {
    if let Operator::RawInput(raw) = &mut node.operator {
        raw.emit_row_group_metadata = true;
        return Some(raw.table_id);
    }
    prepare_narrow_scan(node.inputs.first_mut()?)
}

// ---- Expressions ----

fn build_expr_seq<'a>(
    count: usize,
    child: impl Fn(usize) -> &'a ffi::Expression,
) -> Result<Vec<Expression>, BuildError> {
    (0..count).map(|i| build_expression(child(i))).collect()
}

/// Build an [`Expression`] (and its whole subtree) from a DuckDB bound expression.
/// Shared by plan extraction and the table-filter pushdown hook.
pub(crate) fn build_expression(expr: &ffi::Expression) -> Result<Expression, BuildError> {
    use ExpressionType as E;
    let return_type = || LogicalTypeId::from_u8(ffi::expr_return_type(expr));
    let ref_name = || ffi::expr_has_alias(expr).then(|| ffi::expr_alias(expr));

    match E::from_u8(ffi::expr_type(expr)) {
        E::BOUND_REF => Ok(Expression::Ref(Ref {
            column_idx: ffi::expr_ref_index(expr),
            return_type: return_type(),
            name: ref_name(),
        })),
        E::BOUND_COLUMN_REF => Ok(Expression::Ref(Ref {
            column_idx: ffi::expr_columnref_index(expr),
            return_type: return_type(),
            name: ref_name(),
        })),
        compare_type @ (E::COMPARE_EQUAL
        | E::COMPARE_NOTEQUAL
        | E::COMPARE_LESSTHAN
        | E::COMPARE_GREATERTHAN
        | E::COMPARE_LESSTHANOREQUALTO
        | E::COMPARE_GREATERTHANOREQUALTO) => Ok(Expression::Compare(Compare {
            left: Box::new(build_expression(ffi::expr_comparison_left(expr))?),
            right: Box::new(build_expression(ffi::expr_comparison_right(expr))?),
            compare_type,
            return_type: return_type(),
        })),
        E::COMPARE_BETWEEN => Ok(Expression::Between(Between {
            input: Box::new(build_expression(ffi::expr_between_input(expr))?),
            lower: Box::new(build_expression(ffi::expr_between_lower(expr))?),
            upper: Box::new(build_expression(ffi::expr_between_upper(expr))?),
            lower_inclusive: ffi::expr_between_lower_inclusive(expr),
            upper_inclusive: ffi::expr_between_upper_inclusive(expr),
        })),
        E::VALUE_CONSTANT => Ok(Expression::Constant(ScalarValue {
            logical_type: LogicalTypeId::from_u8(ffi::expr_constant_type(expr)),
            raw_value: ffi::expr_constant_value(expr),
        })),
        E::BOUND_AGGREGATE => Ok(Expression::AggregateFunc(AggregateFunc {
            aggregate_function: ffi::expr_aggregate_name(expr),
            // An aggregate folds its input column directly, so strip any cast
            // DuckDB wrapped the argument in (AVG/SUM keep their narrow input).
            params: (0..ffi::expr_aggregate_child_count(expr))
                .map(|i| build_aggregate_param(ffi::expr_aggregate_child(expr, i)))
                .collect::<Result<Vec<_>, BuildError>>()?,
            distinct: ffi::expr_aggregate_distinct(expr),
            return_type: return_type(),
        })),
        E::BOUND_FUNCTION => Ok(Expression::Function(Function {
            function: ffi::expr_function_name(expr),
            params: build_expr_seq(ffi::expr_function_child_count(expr), |i| {
                ffi::expr_function_child(expr, i)
            })?,
            return_type: return_type(),
        })),
        E::COMPARE_IN => Ok(Expression::InList(InList {
            // child 0 is the tested expression; children 1.. are the list values.
            input: Box::new(build_expression(ffi::expr_operator_child(expr, 0))?),
            values: build_expr_seq(
                ffi::expr_operator_child_count(expr).saturating_sub(1),
                |i| ffi::expr_operator_child(expr, i + 1),
            )?,
        })),
        conjunction_type @ (E::CONJUNCTION_AND | E::CONJUNCTION_OR) => {
            Ok(Expression::Conjunction(Conjunction {
                conjunction_type,
                children: build_expr_seq(ffi::expr_conjunction_child_count(expr), |i| {
                    ffi::expr_conjunction_child(expr, i)
                })?,
            }))
        }
        E::CASE_EXPR => {
            let checks = (0..ffi::expr_case_check_count(expr))
                .map(|i| {
                    Ok(CaseCheck {
                        when: Box::new(build_expression(ffi::expr_case_when(expr, i))?),
                        then: Box::new(build_expression(ffi::expr_case_then(expr, i))?),
                    })
                })
                .collect::<Result<Vec<_>, BuildError>>()?;
            Ok(Expression::Case(Case {
                checks,
                else_expr: Box::new(build_expression(ffi::expr_case_else(expr))?),
            }))
        }
        E::OPERATOR_NOT => Ok(Expression::Not(Not {
            input: Box::new(build_expression(ffi::expr_operator_child(expr, 0))?),
        })),
        // Honor the cast: build a `Cast` over the child. The cast's own return
        // type is its target (a `BoundCastExpression`'s `return_type`).
        E::OPERATOR_CAST => Ok(Expression::Cast(Cast {
            child: Box::new(build_expression(ffi::expr_cast_child(expr))?),
            target_type: LogicalTypeId::from_u8(ffi::expr_return_type(expr)),
        })),
        _ => Err(BuildError(format!(
            "Unsupported expression of type {}",
            ffi::expr_type(expr)
        ))),
    }
}

/// Build an aggregate's argument, stripping any leading cast(s) DuckDB inserted:
/// the reducers fold the raw input column (an `AVG`/`SUM` keeps its narrow
/// accumulator), so the cast would only force a widening copy.
fn build_aggregate_param(expr: &ffi::Expression) -> Result<Expression, BuildError> {
    let mut expr = expr;
    while ExpressionType::from_u8(ffi::expr_type(expr)) as u8 == ExpressionType::OPERATOR_CAST as u8
    {
        expr = ffi::expr_cast_child(expr);
    }
    build_expression(expr)
}
