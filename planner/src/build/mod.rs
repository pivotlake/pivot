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

use dispatch::RowDelivery;
use duckdb_planner::BoundLogicalType;
use duckdb_planner::LogicalOp;
use duckdb_planner::duckdb_bridge::duckdb_types::{ExpressionType, JoinType};
use duckdb_planner::handle::{
    ComparisonJoin as ComparisonJoinView, DynamicFilterRef, JoinConditionEntry,
    Operator as DuckOperator, TableScan as TableScanView, rowid_column_id,
};

use crate::catalog::BoundTable;
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{Cast, Error as ExpressionError, Expression, Function, Ref, VariantGet};
use crate::operator::{
    Aggregate, Compact, CreateSchema, CreateTable, Cte, CteScan, DummyScan, EmptyResult,
    Error as OperatorError, Explain, Filter, Input, Insert, Join, JoinKind, Limit, Materialize,
    Operator, OrderBy, Projection, SetVariable, TableFunctionScan, TopN, Values,
};
use crate::plan::{self, PlanNode};
use crate::types::{Type, physical_arrow_type, type_from_logical};

mod expression;
mod operator;

/// Per-walk state: the dense dynamic-filter slot ids, assigned by the pointer
/// identity of DuckDB's shared `DynamicFilterData` cell so a Top-N producer and
/// the scans that consume its filter agree on a slot.
pub(crate) struct BuildCtx {
    dynamic_filter_slots: HashMap<usize, usize>,
    /// What each CTE walked so far produces, keyed by DuckDB's CTE index. A
    /// scan of a CTE is a leaf, so it takes its output shape from here rather
    /// than from a child, which is why a CTE's definition is walked before the
    /// query reading it.
    cte_outputs: HashMap<usize, (Vec<Type>, Vec<bool>)>,
    /// How many scans of each CTE have been walked, so the CTE knows how many
    /// readers to feed.
    cte_sites: HashMap<usize, usize>,
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
        cte_outputs: HashMap::new(),
        cte_sites: HashMap::new(),
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
                // Serialize a variant output column as JSON text: a cast to
                // text, which `Cast::compile` renders each document with.
                Type::Variant => Expression::Cast(Cast {
                    target: Type::Utf8,
                    target_arrow: physical_arrow_type(&Type::Utf8),
                    source: Box::new(column),
                }),
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
    let kind = op.operator()?;

    // DuckDB's late_materialization optimizer rewrites a wide Top-N/Limit scan
    // into a row-id SEMI join. Collapse that into pivot's Materialize rather than
    // executing a join (only the late-mat shape, not a user IN/EXISTS semi-join).
    if let DuckOperator::ComparisonJoin(join) = kind
        && join.is_late_materialization()?
    {
        return build_late_materialization(op, join, ctx);
    }

    // A CTE's two children are walked in order for a reason, so it can't go
    // through the generic loop below: the definition's output shape is what the
    // scans in the body declare as their own.
    if let DuckOperator::MaterializedCte(cte) = kind {
        return build_cte(op, cte.cte_index()?, ctx);
    }

    let inputs = op
        .children()?
        .into_iter()
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
        DuckOperator::Values(v) => Operator::Values(Values::from_handle(v)?),
        DuckOperator::Insert(i) => Operator::Insert(Insert::from_handle(i)?),
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
        DuckOperator::CreateSchema(c) => Operator::CreateSchema(CreateSchema::from_handle(c)?),
        DuckOperator::Set(s) => Operator::SetVariable(SetVariable::from_set(s)?),
        DuckOperator::Reset(r) => Operator::SetVariable(SetVariable::from_reset(r)?),
        DuckOperator::Compact(c) => Operator::Compact(Compact::from_handle(c)?),
        DuckOperator::EmptyResult(empty) => Operator::EmptyResult(EmptyResult {
            types: empty
                .column_types()?
                .into_iter()
                .map(type_from_logical)
                .collect::<Result<_, _>>()?,
        }),
        // No view to construct from: these carry no kind-specific payload.
        DuckOperator::DummyScan => Operator::DummyScan(DummyScan),
        DuckOperator::Explain => Operator::Explain(Explain),
        DuckOperator::ComparisonJoin(join) => {
            return build_join(op, join, inputs);
        }
        DuckOperator::CteRef(cte_ref) => {
            Operator::CteScan(build_cte_scan(cte_ref.cte_index()?, ctx)?)
        }
        // A CTE is walked above, before its children.
        DuckOperator::MaterializedCte(_) => {
            unreachable!("a CTE walks its own children")
        }
        DuckOperator::Unsupported => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported operator type: {}",
                op.name()?
            )));
        }
    };

    let mut node = PlanNode {
        name: op.name()?,
        inputs,
        operator,
    };

    // A filter directly above a base-table scan folds the scan's lifted
    // pushed-down conditions into itself (see `absorb_scan_pushdown_filter`).
    // The peek at the DuckDB child is what grounds the fold: only a scan child
    // means the built input's Filter can be the synthetic wrapper.
    if matches!(kind, DuckOperator::Filter(_))
        && match op.children()?.first() {
            Some(child) => matches!(child.operator()?, DuckOperator::TableScan(_)),
            None => false,
        }
    {
        node = absorb_scan_pushdown_filter(node);
    }

    // Reattach the scan's static pushed-down filters as a Filter above it, so the
    // Rust side keeps seeing `Filter -> Input` exactly as with filter_pushdown off.
    // Only a TableScan ever populates `pushed_conditions`.
    if !pushed_conditions.is_empty() {
        return Ok(PlanNode {
            name: "PUSHDOWN_FILTER".to_string(),
            inputs: vec![node],
            operator: Operator::Filter(Filter {
                conditions: pushed_conditions,
                delivery: RowDelivery::Coalesced,
            }),
        });
    }

    // A LogicalFilter may carry a projection_map: it outputs only the listed
    // subset/reordering of its child's columns. Replay it by wrapping the filter
    // in a Projection that selects exactly projection_map, positionally, so refs
    // above the filter line up.
    if let DuckOperator::Filter(f) = kind {
        // A filter's projection_map is a plain positional reorder of its child's
        // columns, never a variant field extract, so each carries an empty path.
        let projections = build_scan_columns(
            f.projection_map()?
                .into_iter()
                .map(|(idx, ty)| (idx, ty, Vec::new()))
                .collect(),
        )?;
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

/// Fold the synthetic pushdown filter under `node` into `node` itself.
///
/// Called only for a filter whose DuckDB child is a base-table scan. By
/// construction (see the `TableScan` arm) the built input is then either the
/// raw `Input` or the synthetic Filter carrying the scan's lifted pushed-down
/// conditions - never a user filter - so a Filter input is always the
/// wrapper, and it is safe to dissolve. The wrapper's conditions go first:
/// they sit closer to the scan, where DuckDB puts the typically most
/// selective predicates. One filter evaluates its conditions progressively
/// and materializes survivors once, where a stack of filters would
/// re-materialize at every level - the lower one at its own (much weaker)
/// selectivity.
fn absorb_scan_pushdown_filter(node: PlanNode) -> PlanNode {
    if node.inputs.len() != 1 || !matches!(node.inputs[0].operator, Operator::Filter(_)) {
        // The scan had no pushed-down conditions, so there is no wrapper.
        return node;
    }
    let PlanNode {
        name,
        mut inputs,
        operator,
    } = node;
    let Operator::Filter(filter) = operator else {
        unreachable!("the caller matched a filter");
    };
    let wrapper = inputs.remove(0);
    let Operator::Filter(mut merged) = wrapper.operator else {
        unreachable!("matched as a filter above");
    };
    merged.conditions.extend(filter.conditions);
    PlanNode {
        name,
        inputs: wrapper.inputs,
        operator: Operator::Filter(merged),
    }
}

/// Translate a general comparison join into pivot's [`Join`]. The supported
/// shape is an INNER, RIGHT or SEMI join with one or more equality conditions
/// whose sides are plain column refs of key types the dispatch join handles;
/// anything else reports the specific gap. The probe is DuckDB's left child
/// and the build its right, matching its hash-join convention (the cost model
/// puts the smaller relation on the right).
///
/// A join may additionally carry expression-form conditions (a predicate
/// referencing both sides that is not a bare comparison, e.g. an OR of
/// per-side conjunctions). Those become the [`Join`]'s residual: part of the
/// join itself, evaluated on key-matched pairs, so a pair the predicate
/// rejects is not a match. That placement is what keeps outer and semi joins
/// correct, where the predicate takes part in the match decision (an outer
/// build row all of whose pairs are rejected is emitted null-filled). DuckDB
/// resolved such a predicate against the concatenation of both children's
/// full outputs (see its `ColumnBindingResolver`), which is the layout the
/// dispatch join evaluates it in.
///
/// RIGHT is the outer join whose preserved side is that right child, so it maps
/// onto the dispatch join's build-side outer mode. A written `LEFT JOIN` lands
/// here as RIGHT whenever its preserved relation is the smaller one; LEFT
/// itself (preserving the streamed side) is not supported yet.
///
/// SEMI keeps rows of the left child, so it is semi on the *probe* side. The
/// mirrored RIGHT_SEMI, which DuckDB's build-probe-side optimizer produces when
/// it would rather build the hash table from the left child, is not supported
/// yet.
///
/// DuckDB's join projection maps (which trim the join's output to the columns
/// actually used above it) are folded into the join's own output lists, so
/// the dispatch probe never materializes trimmed columns. An empty map means
/// every column of that side is kept. (DuckDB also emits an empty map when NO
/// column of a side is referenced above the join, so a bare `COUNT(*)` join
/// still materializes both keys; distinguishing that case needs information
/// the plan does not carry.)
fn build_join(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    inputs: Vec<PlanNode>,
) -> Result<PlanNode, OperatorError> {
    let mut probe_keys = Vec::new();
    let mut build_keys = Vec::new();
    let mut key_types = Vec::new();
    let mut residual_predicates = Vec::new();
    for entry in join.conditions()? {
        let condition = match entry {
            JoinConditionEntry::Comparison(condition) => condition,
            JoinConditionEntry::Expression(predicate) => {
                residual_predicates.push(predicate);
                continue;
            }
        };
        if condition.comparison != ExpressionType::COMPARE_EQUAL {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported join comparison type: {:?}",
                condition.comparison
            )));
        }
        let (probe_key, probe_key_type) = join_key_ref(Expression::from_handle(condition.left)?)?;
        let (build_key, build_key_type) = join_key_ref(Expression::from_handle(condition.right)?)?;
        if probe_key_type != build_key_type {
            return Err(OperatorError::Unsupported(format!(
                "join key types differ: {probe_key_type:?} vs {build_key_type:?} \
                 (the planner casts both sides of a condition to a common type, so this plan \
                 shape is unexpected)"
            )));
        }
        probe_keys.push(probe_key);
        build_keys.push(build_key);
        key_types.push(probe_key_type);
    }
    if key_types.is_empty() {
        return Err(OperatorError::Unsupported(
            "joins must have at least one equality condition".to_string(),
        ));
    }

    let residual_filters = residual_predicates
        .into_iter()
        .map(Expression::from_handle)
        .collect::<Result<Vec<_>, _>>()?;

    let probe_types = inputs[0].output_types()?;
    let build_types = inputs[1].output_types()?;
    let left_map: Vec<usize> = join.left_projection_map()?;
    let right_map: Vec<usize> = join.right_projection_map()?;

    // Refs above the join were resolved against the trimmed output (kept left
    // columns, then kept right columns), which is exactly the layout the join
    // emits with these lists.
    let probe_output: Vec<usize> = if left_map.is_empty() {
        (0..probe_types.len()).collect()
    } else {
        left_map
    };
    let kept_build_output: Vec<usize> = if right_map.is_empty() {
        (0..build_types.len()).collect()
    } else {
        right_map
    };
    let (kind, build_output) = match join.join_type()? {
        JoinType::INNER => (JoinKind::Inner, kept_build_output),
        JoinType::RIGHT => (JoinKind::BuildOuter, kept_build_output),
        // A semi join emits no build column at all. DuckDB agrees, and says so
        // by returning the left bindings alone for such a join rather than
        // through the right projection map, which it never reads here - so the
        // map's "empty means keep every column" reading must not be applied.
        JoinType::SEMI => (JoinKind::ProbeSemi, Vec::new()),
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported join type: {other:?}"
            )));
        }
    };
    let probe_nullability = inputs[0].output_nullability();
    let build_nullability = inputs[1].output_nullability();
    let probe_column_types = probe_output
        .iter()
        .map(|&i| (probe_types[i].clone(), probe_nullability[i]))
        .collect();
    let build_column_types = build_output
        .iter()
        .map(|&i| (build_types[i].clone(), build_nullability[i]))
        .collect();
    Ok(PlanNode {
        name: op.name()?,
        inputs,
        operator: Operator::Join(Join {
            probe_keys,
            build_keys,
            key_types,
            probe_output,
            build_output,
            probe_column_types,
            build_column_types,
            residual_filters,
            kind,
        }),
    })
}

/// The column ref a join key must be, with the key type restriction the
/// dispatch join imposes: any fixed-width hashable type, or text. Floats
/// hash and compare by SQL semantics, not bit patterns, so they are out.
fn join_key_ref(key: Expression) -> Result<(usize, Type), OperatorError> {
    let Expression::Ref(key) = key else {
        return Err(OperatorError::Unsupported(format!(
            "join keys must be plain columns, got: {key:?}"
        )));
    };
    match key.return_type {
        Type::Int8
        | Type::Int16
        | Type::Int32
        | Type::Int64
        | Type::UInt8
        | Type::UInt16
        | Type::UInt32
        | Type::UInt64
        | Type::Int128
        | Type::Decimal { .. }
        | Type::Date
        | Type::Timestamp
        | Type::Utf8 => Ok((key.column_idx, key.return_type)),
        _ => Err(OperatorError::Unsupported(format!(
            "Unsupported join key type: {:?}",
            key.return_type
        ))),
    }
}

/// A scan's projected output columns (and a filter's `projection_map`), each a
/// positional `BOUND_REF` over storage column indices. Shared by the walk's
/// `projection_map` replay and the scan constructors in [`operator`].
fn build_scan_columns(
    columns: Vec<(usize, BoundLogicalType, Vec<String>)>,
) -> Result<Vec<Expression>, ExpressionError> {
    columns
        .into_iter()
        .map(|(column_idx, col_type, path)| {
            let return_type = type_from_logical(col_type)?;
            if path.is_empty() {
                return Ok(Expression::Ref(Ref {
                    column_idx,
                    return_type,
                    name: None,
                }));
            }
            // DuckDB's projection-pushdown pushed a variant field extract into
            // this scan output. Read it as a `VariantGet` over the variant
            // column so the reader resolves the path to a single shredded leaf
            // instead of materializing the whole variant. A `VARIANT` output is
            // a bare extraction (the sub-variant); any other type is a typed read.
            let as_type = match return_type {
                Type::Variant => None,
                typed => Some(typed),
            };
            Ok(Expression::Function(Function::VariantGet(VariantGet {
                input: Box::new(Expression::Ref(Ref {
                    column_idx,
                    return_type: Type::Variant,
                    name: None,
                })),
                path,
                as_type,
            })))
        })
        .collect()
}

fn build_pushed_conditions(scan: TableScanView<'_>) -> Result<Vec<Expression>, OperatorError> {
    let list = scan.pushed_conditions()?;
    Ok(list
        .exprs()?
        .into_iter()
        .map(Expression::from_handle)
        .collect::<Result<Vec<_>, _>>()?)
}

/// Walk a CTE: its definition first, so the scans reading it in the body know
/// what shape the rows arrive in, then the body.
fn build_cte(
    op: LogicalOp<'_>,
    cte_index: usize,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    let definition = build_node(op.child(0)?, ctx)?;
    let types = definition.output_types()?;
    let nullable = definition.output_nullability();
    ctx.cte_outputs.insert(cte_index, (types, nullable));

    let body = build_node(op.child(1)?, ctx)?;
    let sites = ctx.cte_sites.remove(&cte_index).unwrap_or_default();
    if sites == 0 {
        return Err(OperatorError::Unsupported(format!(
            "CTE #{cte_index} is materialized but never read"
        )));
    }

    Ok(PlanNode {
        name: op.name()?,
        inputs: vec![definition, body],
        operator: Operator::Cte(Cte { cte_index, sites }),
    })
}

/// One scan of a CTE, taking its output shape from the definition walked earlier.
fn build_cte_scan(cte_index: usize, ctx: &mut BuildCtx) -> Result<CteScan, OperatorError> {
    let Some((types, nullable)) = ctx.cte_outputs.get(&cte_index) else {
        return Err(OperatorError::Unsupported(format!(
            "CTE #{cte_index} is read outside the plan that defines it"
        )));
    };
    *ctx.cte_sites.entry(cte_index).or_default() += 1;
    Ok(CteScan {
        cte_index,
        types: types.clone(),
        nullable: nullable.clone(),
    })
}

/// Collapse DuckDB's late-materialization SEMI join into a pivot Materialize: the
/// narrow pipeline (RHS) runs as-is with its row-id column stripped, and the
/// materializer re-reads the LHS columns for the surviving rows.
fn build_late_materialization(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    let columns: Vec<usize> = join.columns()?;

    // The narrow pipeline is the RHS; translate it normally, then drop the row-id
    // column DuckDB threaded through it for the join we're discarding.
    let mut child = build_node(op.child(1)?, ctx)?;
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
fn prepare_narrow_scan(node: &mut PlanNode) -> Option<Box<dyn BoundTable>> {
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
            Expression::Cast(cast) if cast.target == Type::Utf8
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
