//! Rebuilds the owned [`PlanNode`]/[`Operator`]/[`Expression`] tree from the
//! FlatBuffers `PlanResult` the C++ bridge produced (see `schema/plan.fbs`).
//!
//! The wire format only ever carries the *raw* (pre-resolution) form of a scan,
//! so a FlatBuffers `Input` becomes an [`Operator::RawInput`] and a `Materialize`
//! becomes an [`Operator::RawMaterialize`]; `PlanNode::resolve_inputs` later
//! attaches the bound `DuckDBTable` to each.

use std::collections::HashMap;

use crate::Error;
use crate::duckdb_bridge::duckdb_types::{ExpressionType, LogicalTypeId, OrderType};
use crate::duckdb_bridge::plan_fb as fb;
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{
    AggregateFunc, Between, Case, CaseCheck, Compare, Conjunction, ConstantComparison, Expression,
    Function, InList, Not, Ref, TableFilter,
};
use crate::operator::{
    Aggregate, CreateTable, CreateTableColumn, DummyScan, Explain, Filter, Limit, Operator, OrderBy,
    OrderByDirection, OrderByNode, Projection, RawInput, RawMaterialize, SetVariable,
    TableFunctionScan, TopN,
};
use crate::plan::PlanNode;
use crate::types::ScalarValue;

fn decode_error(msg: impl Into<String>) -> Error {
    Error::Decode(msg.into())
}

/// Unwrap a FlatBuffers union member that should be present. A `None` here means
/// the bridge wrote a buffer that disagrees with the schema, which is a bug.
fn req<T>(value: Option<T>, what: &str) -> Result<T, Error> {
    value.ok_or_else(|| decode_error(format!("malformed plan: missing {what}")))
}

/// Map each item of an optional FlatBuffers vector through `f`, collecting into a
/// `Vec` (empty when the vector is absent).
fn try_collect<T, U>(
    items: Option<impl IntoIterator<Item = T>>,
    f: impl FnMut(T) -> Result<U, Error>,
) -> Result<Vec<U>, Error> {
    match items {
        Some(items) => items.into_iter().map(f).collect(),
        None => Ok(Vec::new()),
    }
}

// DuckDB type discriminants are 1-byte enums on the C++ side; the FlatBuffers
// schema carries them as raw `ubyte`, so we reinterpret them back here exactly as
// they were stamped.
fn decode_logical_type(value: u8) -> LogicalTypeId {
    unsafe { std::mem::transmute::<u8, LogicalTypeId>(value) }
}

fn decode_expression_type(value: u8) -> ExpressionType {
    unsafe { std::mem::transmute::<u8, ExpressionType>(value) }
}

fn decode_order_direction(value: u8) -> OrderByDirection {
    if value == OrderType::ASCENDING as u8 {
        OrderByDirection::Asc
    } else if value == OrderType::DESCENDING as u8 {
        OrderByDirection::Desc
    } else {
        OrderByDirection::Default
    }
}

fn decode_scalar_value(value: fb::ScalarValue) -> ScalarValue {
    ScalarValue {
        logical_type: decode_logical_type(value.logical_type()),
        raw_value: value.raw_value().unwrap_or_default().to_string(),
    }
}

fn decode_dynamic_filter(filter: fb::DynamicFilter) -> DynamicFilter {
    DynamicFilter {
        slot_id: filter.slot_id() as usize,
        column_idx: filter.column_idx() as usize,
        compare_type: decode_expression_type(filter.compare_type()),
    }
}

fn decode_expression(node: fb::Expression) -> Result<Expression, Error> {
    use fb::ExpressionKind as Kind;
    Ok(match node.kind_type() {
        Kind::Ref => {
            let r = req(node.kind_as_ref(), "Ref")?;
            Expression::Ref(Ref {
                column_idx: r.column_idx() as usize,
                return_type: decode_logical_type(r.return_type()),
                name: r.name().map(str::to_string),
            })
        }
        Kind::Compare => {
            let c = req(node.kind_as_compare(), "Compare")?;
            Expression::Compare(Compare {
                left: Box::new(decode_expression(c.left())?),
                right: Box::new(decode_expression(c.right())?),
                compare_type: decode_expression_type(c.compare_type()),
                return_type: decode_logical_type(c.return_type()),
            })
        }
        Kind::Between => {
            let b = req(node.kind_as_between(), "Between")?;
            Expression::Between(Between {
                input: Box::new(decode_expression(b.input())?),
                lower: Box::new(decode_expression(b.lower())?),
                upper: Box::new(decode_expression(b.upper())?),
                lower_inclusive: b.lower_inclusive(),
                upper_inclusive: b.upper_inclusive(),
            })
        }
        Kind::ScalarValue => Expression::Constant(decode_scalar_value(req(
            node.kind_as_scalar_value(),
            "ScalarValue",
        )?)),
        Kind::AggregateFunc => {
            let a = req(node.kind_as_aggregate_func(), "AggregateFunc")?;
            Expression::AggregateFunc(AggregateFunc {
                aggregate_function: a.aggregate_function().unwrap_or_default().to_string(),
                params: try_collect(a.params().map(|v| v.iter()), decode_expression)?,
                distinct: a.distinct(),
                return_type: decode_logical_type(a.return_type()),
            })
        }
        Kind::Function => {
            let func = req(node.kind_as_function(), "Function")?;
            Expression::Function(Function {
                function: func.function().unwrap_or_default().to_string(),
                params: try_collect(func.params().map(|v| v.iter()), decode_expression)?,
                return_type: decode_logical_type(func.return_type()),
            })
        }
        Kind::InList => {
            let in_list = req(node.kind_as_in_list(), "InList")?;
            Expression::InList(InList {
                input: Box::new(decode_expression(in_list.input())?),
                values: try_collect(in_list.values().map(|v| v.iter()), decode_expression)?,
            })
        }
        Kind::Conjunction => {
            let conj = req(node.kind_as_conjunction(), "Conjunction")?;
            Expression::Conjunction(Conjunction {
                conjunction_type: decode_expression_type(conj.conjunction_type()),
                children: try_collect(conj.children().map(|v| v.iter()), decode_expression)?,
            })
        }
        Kind::Case => {
            let case = req(node.kind_as_case(), "Case")?;
            Expression::Case(Case {
                checks: try_collect(case.checks().map(|v| v.iter()), decode_case_check)?,
                else_expr: Box::new(decode_expression(case.else_expr())?),
            })
        }
        Kind::Not => {
            let not = req(node.kind_as_not(), "Not")?;
            Expression::Not(Not {
                input: Box::new(decode_expression(not.input())?),
            })
        }
        other => return Err(decode_error(format!("unknown expression kind {}", other.0))),
    })
}

fn decode_case_check(check: fb::CaseCheck) -> Result<CaseCheck, Error> {
    Ok(CaseCheck {
        when: Box::new(decode_expression(check.when_expr())?),
        then: Box::new(decode_expression(check.then_expr())?),
    })
}

fn decode_order_by_node(node: fb::OrderByNode) -> Result<OrderByNode, Error> {
    Ok(OrderByNode {
        direction: decode_order_direction(node.direction()),
        expression: decode_expression(node.expression())?,
    })
}

/// Convert a [`fb::TableFilter`] (the filter-pushdown callback payload) into the
/// owned [`TableFilter`].
pub(crate) fn decode_table_filter(filter: fb::TableFilter) -> Result<TableFilter, Error> {
    use fb::TableFilterKind as Kind;
    Ok(match filter.kind_type() {
        Kind::Expression => TableFilter::Expression(Box::new(decode_expression(req(
            filter.kind_as_expression(),
            "Expression",
        )?)?)),
        Kind::ConstantComparison => {
            let c = req(filter.kind_as_constant_comparison(), "ConstantComparison")?;
            TableFilter::ConstantComparison(ConstantComparison {
                column_ref: Box::new(decode_expression(c.column_ref())?),
                compare_type: decode_expression_type(c.compare_type()),
                constant: decode_scalar_value(c.constant()),
            })
        }
        other => return Err(decode_error(format!("unknown table-filter kind {}", other.0))),
    })
}

fn decode_operator(node: fb::PlanNode) -> Result<Operator, Error> {
    use fb::OperatorKind as Kind;
    Ok(match node.op_type() {
        Kind::Input => {
            let input = req(node.op_as_input(), "Input")?;
            Operator::RawInput(RawInput {
                table_id: input.table_id() as usize,
                columns: try_collect(input.columns().map(|v| v.iter()), decode_expression)?,
                dynamic_filters: try_collect(input.dynamic_filters().map(|v| v.iter()), |f| {
                    Ok(decode_dynamic_filter(f))
                })?,
                emit_row_group_metadata: input.emit_row_group_metadata(),
            })
        }
        Kind::TableFunctionScan => {
            let scan = req(node.op_as_table_function_scan(), "TableFunctionScan")?;
            Operator::TableFunctionScan(TableFunctionScan {
                function_name: scan.function_name().unwrap_or_default().to_string(),
                args: try_collect(scan.args().map(|v| v.iter()), |a| Ok(decode_scalar_value(a)))?,
                columns: try_collect(scan.columns().map(|v| v.iter()), decode_expression)?,
            })
        }
        Kind::Projection => {
            let projection = req(node.op_as_projection(), "Projection")?;
            Operator::Projection(Projection {
                projections: try_collect(projection.projections().map(|v| v.iter()), decode_expression)?,
            })
        }
        Kind::OrderBy => {
            let order_by = req(node.op_as_order_by(), "OrderBy")?;
            Operator::OrderBy(OrderBy {
                order_bys: try_collect(order_by.order_bys().map(|v| v.iter()), decode_order_by_node)?,
            })
        }
        Kind::Aggregate => {
            let aggregate = req(node.op_as_aggregate(), "Aggregate")?;
            Operator::Aggregate(Aggregate {
                groups: try_collect(aggregate.groups().map(|v| v.iter()), decode_expression)?,
                expressions: try_collect(aggregate.expressions().map(|v| v.iter()), decode_expression)?,
            })
        }
        Kind::Filter => {
            let filter = req(node.op_as_filter(), "Filter")?;
            Operator::Filter(Filter {
                conditions: try_collect(filter.conditions().map(|v| v.iter()), decode_expression)?,
            })
        }
        Kind::TopN => {
            let top_n = req(node.op_as_top_n(), "TopN")?;
            Operator::TopN(TopN {
                order_bys: try_collect(top_n.order_bys().map(|v| v.iter()), decode_order_by_node)?,
                limit: top_n.limit() as usize,
                offset: top_n.offset() as usize,
                produces_dynamic_filter: top_n.produces_dynamic_filter().map(decode_dynamic_filter),
            })
        }
        Kind::Limit => {
            let limit = req(node.op_as_limit(), "Limit")?;
            Operator::Limit(Limit {
                limit: limit.limit().map(|v| v as usize),
                offset: limit.offset().map(|v| v as usize),
            })
        }
        Kind::CreateTable => {
            let create = req(node.op_as_create_table(), "CreateTable")?;
            let mut options = HashMap::new();
            if let Some(entries) = create.options() {
                for entry in entries.iter() {
                    options.insert(
                        entry.key().unwrap_or_default().to_string(),
                        entry.value().unwrap_or_default().to_string(),
                    );
                }
            }
            Operator::CreateTable(CreateTable {
                name: create.name().unwrap_or_default().to_string(),
                columns: try_collect(create.columns().map(|v| v.iter()), |c| {
                    Ok(CreateTableColumn {
                        name: c.name().unwrap_or_default().to_string(),
                        col_type: decode_logical_type(c.col_type()),
                    })
                })?,
                options,
                if_not_exists: create.if_not_exists(),
                or_replace: create.or_replace(),
                temporary: create.temporary(),
                has_query: create.has_query(),
                constraint_count: create.constraint_count() as usize,
            })
        }
        Kind::DummyScan => Operator::DummyScan(DummyScan {}),
        Kind::Explain => Operator::Explain(Explain {}),
        Kind::SetVariable => {
            let set = req(node.op_as_set_variable(), "SetVariable")?;
            Operator::Set(SetVariable {
                name: set.name().unwrap_or_default().to_string(),
                // `has_value` is what separates `SET x = ''` (Some("")) from a
                // RESET (None); an absent/empty value string alone cannot.
                value: set
                    .has_value()
                    .then(|| set.value().unwrap_or_default().to_string()),
            })
        }
        Kind::Materialize => {
            let materialize = req(node.op_as_materialize(), "Materialize")?;
            Operator::RawMaterialize(RawMaterialize {
                table_id: materialize.table_id() as usize,
                columns: try_collect(materialize.columns().map(|v| v.iter()), |c| Ok(c as usize))?,
            })
        }
        other => return Err(decode_error(format!("unknown operator kind {}", other.0))),
    })
}

/// Convert one FlatBuffers plan node (and its subtree) into an owned [`PlanNode`].
pub(crate) fn decode_plan_node(node: fb::PlanNode) -> Result<PlanNode, Error> {
    let inputs = try_collect(node.inputs().map(|v| v.iter()), decode_plan_node)?;
    Ok(PlanNode {
        name: node.name().unwrap_or_default().to_string(),
        inputs,
        operator: decode_operator(node)?,
    })
}
