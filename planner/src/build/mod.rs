//! Builds the Pivot [`PlanNode`] tree by walking DuckDB's plan handles.
//!
//! [`duckdb_planner::Plan`] hands back the root [`LogicalOp`]; this module walks
//! that live tree through the handle accessors ([`LogicalOp::operator`] dispatches
//! each node to a typed [`Operator`](duckdb_planner::handle::Operator) view) and
//! assembles the typed Pivot [`Operator`]/[`Expression`] values directly. There is
//! no intermediate IR: a handle is one borrowed reference and reading a field is
//! one FFI call.
//!
//! Each node's construction is a `from_handle` on its Pivot type, implemented in
//! the [`operator`] and [`expression`] submodules; this module drives the walk and
//! the DuckDB-specific tree shaping that no single node owns: collapsing the
//! late-materialization SEMI join into a [`Materialize`], lifting a scan's
//! pushed-down `table_filters` back into a [`Filter`], replaying a filter's
//! `projection_map`, and stripping the threaded-up row-id column.

use std::collections::HashMap;

use duckdb_planner::LogicalOp;
use duckdb_planner::duckdb_bridge::duckdb_types::LogicalTypeId;
use duckdb_planner::handle::{
    ComparisonJoin as ComparisonJoinView, DynamicFilterRef, Operator as DuckOperator,
    TableScan as TableScanView, rowid_column_id,
};

use crate::catalog::Table;
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{Error as ExpressionError, Expression, Ref};
use crate::operator::{
    Aggregate, CreateTable, DummyScan, Error as OperatorError, Explain, Filter, Input, Insert,
    Limit, Materialize, Operator, OrderBy, Projection, SetVariable, TableFunctionScan, TopN,
    Values,
};
use crate::plan::{self, PlanNode};
use crate::types::type_from_logical;

mod expression;
mod operator;

/// Per-walk state: the dense dynamic-filter slot ids, assigned by the pointer
/// identity of DuckDB's shared `DynamicFilterData` cell so a Top-N producer and
/// the scans that consume its filter agree on a slot.
pub(crate) struct BuildCtx {
    dynamic_filter_slots: HashMap<usize, usize>,
}

impl BuildCtx {
    /// The dense slot id for a shared dynamic-filter cell, assigning a new one on
    /// first sight. The shared cell can't be passed to Rust, so it is identified
    /// by its address (`data_id`); this maps each distinct address to a stable
    /// small integer (first seen `0`, next `1`, ...) so a producer and its
    /// consumers share a `slot_id` the compile step can wire up.
    fn slot_for(&mut self, data_id: usize) -> usize {
        let next_slot = self.dynamic_filter_slots.len();
        *self
            .dynamic_filter_slots
            .entry(data_id)
            .or_insert(next_slot)
    }

    /// Build a planner [`DynamicFilter`] from a handle reference, assigning its
    /// shared slot. Shared by the scan and Top-N construction (see [`operator`]).
    fn dynamic_filter(&mut self, df: DynamicFilterRef) -> Result<DynamicFilter, ExpressionError> {
        Ok(DynamicFilter {
            slot_id: self.slot_for(df.data_id),
            column_idx: df.column,
            compare_type: df.comparison.try_into()?,
        })
    }
}

/// Build the whole Pivot plan tree from DuckDB's root operator handle.
pub(crate) fn build_plan(root: LogicalOp<'_>) -> Result<PlanNode, plan::Error> {
    let mut ctx = BuildCtx {
        dynamic_filter_slots: HashMap::new(),
    };
    Ok(build_node(root, &mut ctx)?)
}

fn build_node(op: LogicalOp<'_>, ctx: &mut BuildCtx) -> Result<PlanNode, OperatorError> {
    let kind = op.operator();

    // DuckDB's late_materialization optimizer rewrites a wide Top-N/Limit scan
    // into a row-id SEMI join. Collapse that into pivot's Materialize rather than
    // executing a join (only the late-mat shape, not a user IN/EXISTS semi-join).
    if let DuckOperator::ComparisonJoin(join) = kind
        && join.is_late_materialization()
    {
        return build_late_materialization(op, join, ctx);
    }

    let mut inputs = op
        .children()
        .map(|child| build_node(child, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    // Drop the row-id ORDER BY DuckDB synthesizes above a late-materialized plain
    // LIMIT: pivot can't run it and doesn't need it. The ORDER BY is a
    // pass-through, so returning its child keeps column positions unchanged.
    if matches!(kind, DuckOperator::OrderBy(_))
        && inputs.first().is_some_and(is_materialize_over_empty_scan)
    {
        return Ok(inputs.into_iter().next().unwrap());
    }

    // DuckDB's optimizer pushes simple `column <op> constant` predicates down
    // into the scan itself (its `table_filters`), so the `Filter` operator that
    // would sit above the scan disappears. Pivot wants that explicit `Filter`
    // back, so a base-table scan collects the pushed predicates here and, after
    // the node is built (below), wraps it in a synthetic `Filter` — restoring the
    // same `Filter -> Input` shape as if pushdown were off. Empty for every other
    // operator.
    let mut pushed_conditions: Vec<Expression> = Vec::new();

    let operator = match kind {
        DuckOperator::Projection(p) => Operator::Projection(Projection::from_handle(p)?),
        DuckOperator::Filter(f) => Operator::Filter(Filter::from_handle(f)?),
        DuckOperator::Aggregate(a) => Operator::Aggregate(Aggregate::from_handle(a)?),
        DuckOperator::OrderBy(o) => Operator::OrderBy(OrderBy::from_handle(o)?),
        DuckOperator::TopN(t) => Operator::TopN(TopN::from_handle(t, ctx)?),
        DuckOperator::Limit(l) => Operator::Limit(Limit::from_handle(l)?),
        DuckOperator::TableScan(scan) => {
            // The pushdown lift is a walk-level transform (it wraps the scan in a
            // Filter below), so it stays here rather than in `Input::from_handle`.
            pushed_conditions = build_pushed_conditions(scan)?;
            Operator::Input(Input::from_handle(scan, ctx)?)
        }
        DuckOperator::TableFunctionScan(view) => {
            Operator::TableFunctionScan(TableFunctionScan::from_handle(view)?)
        }
        DuckOperator::CreateTable(c) => Operator::CreateTable(CreateTable::from_handle(c)?),
        DuckOperator::Insert(i) => Operator::Insert(Insert::from_handle(i)?),
        DuckOperator::ExpressionGet(e) => Operator::Values(Values::from_handle(e)?),
        DuckOperator::Set(s) => Operator::SetVariable(SetVariable::from_set(s)),
        DuckOperator::Reset(r) => Operator::SetVariable(SetVariable::from_reset(r)),
        // No view to construct from: these carry no kind-specific payload.
        DuckOperator::DummyScan => Operator::DummyScan(DummyScan),
        DuckOperator::Explain => Operator::Explain(Explain),
        DuckOperator::ComparisonJoin(_) | DuckOperator::Unsupported => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported operator type: {}",
                op.name()
            )));
        }
    };

    // DuckDB's logical VALUES shape is ExpressionGet -> DummyScan. Its own
    // physical planner evaluates foldable VALUES expressions eagerly and turns
    // them into a COLUMN_DATA_SCAN. Pivot owns physical execution, so lower the
    // same logical shape to a nullary Values source and discard the trigger-only
    // dummy child. Keep any non-dummy child intact so compilation reports the
    // unsupported shape instead of silently discarding meaningful input.
    if matches!(operator, Operator::Values(_))
        && matches!(
            inputs.as_slice(),
            [PlanNode {
                operator: Operator::DummyScan(_),
                inputs: dummy_inputs,
                ..
            }] if dummy_inputs.is_empty()
        )
    {
        inputs.clear();
    }

    let node = PlanNode {
        name: op.name(),
        inputs,
        operator,
    };

    // Reattach the scan's static pushed-down filters as a Filter above it, so the
    // Rust side keeps seeing `Filter -> Input` exactly as with filter_pushdown off.
    // Only a TableScan ever populates `pushed_conditions`.
    if !pushed_conditions.is_empty() {
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
    if let DuckOperator::Filter(f) = kind {
        let projections = build_scan_columns(f.projection_map())?;
        if !projections.is_empty() {
            return Ok(PlanNode {
                name: "FILTER_PROJECTION".to_string(),
                inputs: vec![node],
                operator: Operator::Projection(Projection { projections }),
            });
        }
    }

    Ok(node)
}

/// A scan's projected output columns (and a filter's `projection_map`), each a
/// positional `BOUND_REF` over storage column indices. Shared by the walk's
/// `projection_map` replay and the scan constructors in [`operator`].
fn build_scan_columns(
    columns: impl Iterator<Item = (usize, LogicalTypeId)>,
) -> Result<Vec<Expression>, ExpressionError> {
    columns
        .map(|(column_idx, type_id)| {
            Ok(Expression::Ref(Ref {
                column_idx,
                return_type: type_from_logical(type_id)?,
                name: None,
            }))
        })
        .collect()
}

fn build_pushed_conditions(scan: TableScanView<'_>) -> Result<Vec<Expression>, OperatorError> {
    let list = scan
        .pushed_conditions()
        .map_err(|e| OperatorError::Unsupported(e.to_string()))?;
    Ok(list
        .iter()
        .map(Expression::from_handle)
        .collect::<Result<Vec<_>, _>>()?)
}

/// Collapse DuckDB's late-materialization SEMI join into a pivot Materialize: the
/// narrow pipeline (RHS) runs as-is with its row-id column stripped, and the
/// materializer re-reads the LHS columns for the surviving rows.
fn build_late_materialization(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    let columns: Vec<usize> = join.columns().collect();

    // The narrow pipeline is the RHS; translate it normally, then drop the row-id
    // column DuckDB threaded through it for the join we're discarding.
    let mut child = build_node(op.child(1), ctx)?;
    strip_trailing_rowid(&mut child);
    // Flag the narrow scan to emit row-group metadata, and clone its table for the
    // Materialize so both read (a clone of) the same table.
    let table = prepare_narrow_scan(&mut child).ok_or_else(|| {
        OperatorError::Unsupported(
            "late materialization without a base-table scan is not supported".to_string(),
        )
    })?;

    Ok(PlanNode {
        name: "Materialize".to_string(),
        inputs: vec![child],
        operator: Operator::Materialize(Materialize { table, columns }),
    })
}

// ---- Late-materialization tree surgery (operates on the built Pivot tree) ----

/// Whether `node` is a late-mat Materialize whose narrow scan reads no data
/// columns (the shape of a plain LIMIT, once the row-id column is stripped).
fn is_materialize_over_empty_scan(node: &PlanNode) -> bool {
    if node.name != "Materialize" {
        return false;
    }
    let mut cur = node;
    while let Some(child) = cur.inputs.first() {
        cur = child;
        if let Operator::Input(input) = &cur.operator {
            return input.columns.is_empty();
        }
    }
    false
}

/// Remove DuckDB's row-id column from an already-built late-mat narrow subtree.
/// DuckDB appends the row-id last at every level, so dropping it never shifts
/// another column's position. Returns the output position that held the row-id.
fn strip_trailing_rowid(node: &mut PlanNode) -> Option<usize> {
    let rowid = rowid_column_id();
    if let Operator::Input(input) = &mut node.operator {
        let pos = input
            .columns
            .iter()
            .position(|e| matches!(e, Expression::Ref(r) if r.column_idx == rowid))?;
        input.columns.remove(pos);
        return Some(pos);
    }

    let child_rowid = strip_trailing_rowid(node.inputs.first_mut()?);

    // A projection that carried the row-id up references it positionally in its
    // child's output; drop that one entry. Other operators pass columns through.
    if let (Operator::Projection(proj), Some(rowid_pos)) = (&mut node.operator, child_rowid)
        && let Some(pos) = proj
            .projections
            .iter()
            .position(|e| matches!(e, Expression::Ref(r) if r.column_idx == rowid_pos))
    {
        proj.projections.remove(pos);
        return Some(pos);
    }
    child_rowid
}

/// Walk a late-mat narrow subtree to its scan, flag it to emit row-group
/// metadata, and return a clone of its table (which the Materialize re-reads).
///
/// HACK: this finds the scan positionally (first input until an `Input` turns
/// up), which silently tags the wrong scan if the narrow subtree ever branches.
/// Should be rewritten to key off the row-id scan `strip_trailing_rowid` touched.
fn prepare_narrow_scan(node: &mut PlanNode) -> Option<Box<dyn Table>> {
    if let Operator::Input(input) = &mut node.operator {
        input.emit_row_group_metadata = true;
        return Some(input.table.clone_box());
    }
    prepare_narrow_scan(node.inputs.first_mut()?)
}
