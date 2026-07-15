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
use duckdb_planner::duckdb_bridge::duckdb_types::{ExpressionType, JoinType, LogicalTypeId};
use duckdb_planner::handle::{
    ComparisonJoin as ComparisonJoinView, DynamicFilterRef, Operator as DuckOperator,
    TableScan as TableScanView, rowid_column_id,
};

use crate::catalog::Table;
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{
    Compare, Error as ExpressionError, Expression, Function, Ref, VariantToJson,
};
use crate::operator::{
    Aggregate, CreateTable, DummyScan, Error as OperatorError, Explain, Filter, Input, Join, Limit,
    Materialize, Operator, OrderBy, Projection, SetVariable, TableFunctionScan, TopN,
};
use crate::plan::{self, PlanNode};
use crate::types::{Type, type_from_logical};

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
    let plan = build_node(root, &mut ctx)?;
    Ok(render_variant_outputs(plan)?)
}

/// Render the query's variant-typed output columns as JSON text.
///
/// A variant's physical layout can differ per file (each file shreds by its own
/// data), so handing the raw struct to the client would mean result batches of
/// varying shape, and an unreadable binary value even when they don't vary.
/// When the plan's output contains a variant column, wrap the whole plan in one
/// more projection that renders those columns and passes the rest through, so
/// the client always sees uniform JSON text. Asking the plan for its
/// [`output_types`](PlanNode::output_types) makes this work for any root
/// operator; below the added projection, everything still operates on the raw
/// variant.
fn render_variant_outputs(plan: PlanNode) -> Result<PlanNode, crate::compile::Error> {
    let types = plan.output_types()?;
    if !types.contains(&Type::Variant) {
        return Ok(plan);
    }

    let projections = types
        .iter()
        .enumerate()
        .map(|(column_idx, column_type)| {
            let column = Expression::Ref(Ref {
                column_idx,
                return_type: column_type.clone(),
                name: None,
            });
            match column_type {
                Type::Variant => Expression::Function(Function::VariantToJson(VariantToJson {
                    input: Box::new(column),
                })),
                _ => column,
            }
        })
        .collect();
    Ok(PlanNode {
        name: "render variant outputs".to_string(),
        inputs: vec![plan],
        operator: Operator::Projection(Projection { projections }),
    })
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

    let inputs = op
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
        DuckOperator::Set(s) => Operator::SetVariable(SetVariable::from_set(s)),
        DuckOperator::Reset(r) => Operator::SetVariable(SetVariable::from_reset(r)),
        // No view to construct from: these carry no kind-specific payload.
        DuckOperator::DummyScan => Operator::DummyScan(DummyScan),
        DuckOperator::Explain => Operator::Explain(Explain),
        DuckOperator::ComparisonJoin(join) => {
            return build_join(op, join, inputs);
        }
        DuckOperator::Unsupported => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported operator type: {}",
                op.name()
            )));
        }
    };

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

/// Translate a general comparison join into pivot's [`Join`]. The supported
/// shape is an INNER join with a single equality condition whose sides are
/// plain column refs of a key type the dispatch join handles (`Int64`);
/// anything else reports the specific gap. The probe is DuckDB's left child
/// and the build its right, matching its hash-join convention (the cost model
/// puts the smaller relation on the right).
///
/// DuckDB's join projection maps (which trim the join's output to the columns
/// actually used above it) are replayed as a `Projection` on top, since the
/// dispatch join always emits every probe column followed by every build
/// column.
fn build_join(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    inputs: Vec<PlanNode>,
) -> Result<PlanNode, OperatorError> {
    if join.join_type() != JoinType::INNER {
        return Err(OperatorError::Unsupported(format!(
            "Unsupported join type: {:?}",
            join.join_type()
        )));
    }
    let conditions: Vec<_> = join.conditions().collect();
    // The hash join keys on one equality condition; every other condition
    // (further equalities, or range comparisons) filters the joined rows
    // below, which is equivalent for an INNER join.
    let hash_condition = conditions
        .iter()
        .position(|c| c.comparison == ExpressionType::COMPARE_EQUAL)
        .ok_or_else(|| {
            OperatorError::Unsupported(
                "joins must have at least one equality condition".to_string(),
            )
        })?;

    let probe_key = join_key_ref(Expression::from_handle(conditions[hash_condition].left)?)?;
    let build_key = join_key_ref(Expression::from_handle(conditions[hash_condition].right)?)?;

    let probe_types = inputs[0].output_types()?;
    let build_types = inputs[1].output_types()?;
    let left_map: Vec<usize> = join.left_projection_map().collect();
    let right_map: Vec<usize> = join.right_projection_map().collect();

    // The residual conditions, rebased onto the join's output: probe columns
    // keep their index, build columns shift past them.
    let residual_conditions = conditions
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != hash_condition)
        .map(|(_, condition)| {
            let Expression::Ref(left) = Expression::from_handle(condition.left)? else {
                return Err(OperatorError::Unsupported(
                    "extra join conditions must compare plain columns".to_string(),
                ));
            };
            let Expression::Ref(right) = Expression::from_handle(condition.right)? else {
                return Err(OperatorError::Unsupported(
                    "extra join conditions must compare plain columns".to_string(),
                ));
            };
            Ok(Expression::Compare(Compare {
                left: Box::new(Expression::Ref(left)),
                right: Box::new(Expression::Ref(Ref {
                    column_idx: probe_types.len() + right.column_idx,
                    ..right
                })),
                compare_type: condition.comparison.clone().try_into().map_err(|_| {
                    OperatorError::Unsupported(format!(
                        "Unsupported join comparison type: {:?}",
                        condition.comparison
                    ))
                })?,
                return_type: Type::Boolean,
            }))
        })
        .collect::<Result<Vec<_>, OperatorError>>()?;

    let mut node = PlanNode {
        name: op.name(),
        inputs,
        operator: Operator::Join(Join {
            probe_key,
            build_key,
        }),
    };
    if !residual_conditions.is_empty() {
        node = PlanNode {
            name: "JOIN_RESIDUAL_FILTER".to_string(),
            inputs: vec![node],
            operator: Operator::Filter(Filter {
                conditions: residual_conditions,
            }),
        };
    }
    if left_map.is_empty() && right_map.is_empty() {
        return Ok(node);
    }

    // Replay the projection maps: refs above the join were resolved against
    // the trimmed output (kept left columns, then kept right columns), so
    // select exactly those positions out of the join's full concatenation.
    let left_kept = if left_map.is_empty() {
        (0..probe_types.len()).collect()
    } else {
        left_map
    };
    let right_kept = if right_map.is_empty() {
        (0..build_types.len()).collect()
    } else {
        right_map
    };
    let projections = left_kept
        .into_iter()
        .map(|i| (i, probe_types[i].clone()))
        .chain(
            right_kept
                .into_iter()
                .map(|i| (probe_types.len() + i, build_types[i].clone())),
        )
        .map(|(column_idx, return_type)| {
            Expression::Ref(Ref {
                column_idx,
                return_type,
                name: None,
            })
        })
        .collect();
    Ok(PlanNode {
        name: "JOIN_PROJECTION".to_string(),
        inputs: vec![node],
        operator: Operator::Projection(Projection { projections }),
    })
}

/// The column ref a join key must be, with the key type restriction the
/// dispatch join imposes.
fn join_key_ref(key: Expression) -> Result<usize, OperatorError> {
    let Expression::Ref(key) = key else {
        return Err(OperatorError::Unsupported(format!(
            "join keys must be plain columns, got: {key:?}"
        )));
    };
    if key.return_type != Type::Int64 {
        return Err(OperatorError::Unsupported(format!(
            "Unsupported join key type: {:?}",
            key.return_type
        )));
    }
    Ok(key.column_idx)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Aggregate;

    fn variant_ref(column_idx: usize) -> Expression {
        Expression::Ref(Ref {
            column_idx,
            return_type: Type::Variant,
            name: None,
        })
    }

    /// A variant output is rendered even when the root isn't a projection:
    /// `output_types` sees through any operator, and the render wraps the
    /// whole plan rather than rewriting inside it.
    #[test]
    fn renders_a_variant_output_under_an_aggregate_root() {
        let plan = PlanNode {
            name: "aggregate".to_string(),
            inputs: Vec::new(),
            operator: Operator::Aggregate(Aggregate {
                groups: vec![variant_ref(0)],
                expressions: Vec::new(),
                output_limit: None,
            }),
        };

        let rendered = render_variant_outputs(plan).unwrap();

        assert_eq!(rendered.output_types().unwrap(), vec![Type::Utf8]);
        let Operator::Projection(projection) = &rendered.operator else {
            panic!("expected a render projection above the aggregate root");
        };
        assert!(matches!(
            &projection.projections[0],
            Expression::Function(Function::VariantToJson(_))
        ));
    }

    /// A plan without variant outputs is returned untouched, with no extra
    /// projection.
    #[test]
    fn leaves_variant_free_outputs_alone() {
        let plan = PlanNode {
            name: "projection".to_string(),
            inputs: Vec::new(),
            operator: Operator::Projection(Projection {
                projections: vec![Expression::Ref(Ref {
                    column_idx: 0,
                    return_type: Type::Int64,
                    name: None,
                })],
            }),
        };

        let rendered = render_variant_outputs(plan).unwrap();

        assert_eq!(rendered.output_types().unwrap(), vec![Type::Int64]);
        assert!(rendered.inputs.is_empty(), "no wrapper was added");
    }
}
