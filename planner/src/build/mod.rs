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

use dispatch::RangeCompare;
use dispatch::RowDelivery;
use duckdb_planner::BoundLogicalType;
use duckdb_planner::Expr;
use duckdb_planner::LogicalOp;
use duckdb_planner::duckdb_bridge::duckdb_types::{ExpressionType, JoinType};
use duckdb_planner::handle::{
    ComparisonJoin as ComparisonJoinView, DelimGet as DelimGetView, DelimJoin as DelimJoinView,
    JoinCondition, JoinConditionEntry, Operator as DuckOperator, TableScan as TableScanView,
    rowid_column_id,
};

use crate::catalog::BoundTable;
use crate::expression::{Cast, Error as ExpressionError, Expression, Function, Ref, VariantGet};
use crate::operator::{
    Aggregate, Compact, CopyFromStdin, CreateSchema, CreateTable, CreateUser, Cte, CteScan,
    Distinct, DummyScan, Error as OperatorError, Explain, Filter, Input, Insert, Join, JoinKind,
    Limit, Materialize, Operator, OrderBy, Projection, SetVariable, TableFunctionScan, TopN,
    Values,
};
use crate::plan::{self, PlanNode};
use crate::types::{Type, physical_arrow_type, type_from_logical};

mod context;
mod expression;
mod operator;

use context::BuildCtx;

/// Build the whole Pivot plan tree from DuckDB's root operator handle.
pub(crate) fn build_plan(root: LogicalOp<'_>) -> Result<PlanNode, plan::Error> {
    let mut ctx = BuildCtx::default();
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

    // A delim join walks its children in order too: the subquery side's delim
    // scans declare the de-duplicated columns of the outer side, which must
    // have been walked first.
    if let DuckOperator::DelimJoin(delim) = kind {
        return build_delim_join(op, delim, ctx);
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
        DuckOperator::ChunkGet(c) => Operator::Values(Values::from_chunk_get(c)?),
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
        DuckOperator::CopyFromStdin(c) => Operator::CopyFromStdin(CopyFromStdin::from_handle(c)?),
        DuckOperator::CreateUser(c) => Operator::CreateUser(CreateUser::from_handle(c)?),
        // No view to construct from: these carry no kind-specific payload.
        DuckOperator::DummyScan => Operator::DummyScan(DummyScan),
        DuckOperator::Explain => Operator::Explain(Explain),
        DuckOperator::ComparisonJoin(join) => {
            return build_join(op, join, inputs);
        }
        DuckOperator::CteRef(cte_ref) => {
            Operator::CteScan(ctx.create_cte_scan(cte_ref.cte_index()?)?)
        }
        DuckOperator::DelimGet(get) => Operator::CteScan(build_delim_get_scan(get, ctx)?),
        // A delim join is walked above, before its children.
        DuckOperator::DelimJoin(_) => {
            unreachable!("a delim join walks its own children")
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
/// RIGHT is the outer join whose preserved side is that right child, so it
/// maps onto the dispatch join's build-side outer mode, and LEFT onto its
/// probe-side outer mode. DuckDB writes whichever puts the preserved relation
/// on the side its cost model wants it: a written `LEFT JOIN` lands here as
/// RIGHT whenever its preserved relation is the smaller one, and stays LEFT
/// when it is the larger.
///
/// SEMI keeps rows of the left child, so it is semi on the *probe* side. The
/// mirrored RIGHT_SEMI, which DuckDB's build-probe-side optimizer produces
/// when it would rather build the hash table from the preserved relation (an
/// `IN` whose outer table is far smaller than the subquery's), is semi on the
/// build side.
///
/// ANTI is SEMI's negation, one output row per left-child row with no
/// surviving match, so it is anti on the probe side. Here the mirrored
/// RIGHT_ANTI *is* supported, as anti on the build side: the same optimizer
/// rewrites an ANTI into it whenever the preserved relation is the cheaper
/// side to build the hash table from, which for anti joins (preserving the
/// rows a typically larger side fails to match) is routine rather than rare.
///
/// MARK keeps every left-child row and appends a boolean marker column, the
/// three-valued result of `IN (...)`: what DuckDB plans an uncorrelated `IN`
/// subquery, or the constant chunk its optimizer rewrites a long `IN` list
/// into, as. The marker is TRUE on a match, FALSE on a miss, and NULL where
/// a null key leaves the miss unknown.
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
    let mut inequalities = Vec::new();
    for entry in join.conditions()? {
        let condition = match entry {
            JoinConditionEntry::Comparison(condition) => condition,
            JoinConditionEntry::Expression(predicate) => {
                residual_predicates.push(predicate);
                continue;
            }
        };
        if condition.comparison != ExpressionType::COMPARE_EQUAL {
            inequalities.push(condition);
            continue;
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
    // With no equality to hash on, a single comparison condition runs as a
    // range join instead.
    if key_types.is_empty() {
        return build_range_join(op, join, inputs, inequalities, residual_predicates);
    }
    if !inequalities.is_empty() {
        return Err(OperatorError::Unsupported(format!(
            "Unsupported join comparison type: {:?}",
            inequalities[0].comparison
        )));
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
    let kept_probe_output: Vec<usize> = if left_map.is_empty() {
        (0..probe_types.len()).collect()
    } else {
        left_map
    };
    let kept_build_output: Vec<usize> = if right_map.is_empty() {
        (0..build_types.len()).collect()
    } else {
        right_map
    };
    let (kind, probe_output, build_output) = match join.join_type()? {
        JoinType::INNER => (JoinKind::Inner, kept_probe_output, kept_build_output),
        JoinType::RIGHT => (JoinKind::BuildOuter, kept_probe_output, kept_build_output),
        JoinType::LEFT => (JoinKind::ProbeOuter, kept_probe_output, kept_build_output),
        // A semi or anti join emits no build column at all. DuckDB agrees, and
        // says so by returning the left bindings alone for such a join rather
        // than through the right projection map, which it never reads here -
        // so the map's "empty means keep every column" reading must not be
        // applied.
        JoinType::SEMI => (JoinKind::ProbeSemi, kept_probe_output, Vec::new()),
        JoinType::ANTI => (JoinKind::ProbeAnti, kept_probe_output, Vec::new()),
        // The mirror images: a right semi or anti join emits build columns
        // alone, and DuckDB never reads the left projection map for either.
        JoinType::RIGHT_SEMI => (JoinKind::BuildSemi, Vec::new(), kept_build_output),
        JoinType::RIGHT_ANTI => (JoinKind::BuildAnti, Vec::new(), kept_build_output),
        // A mark join emits the left bindings plus its marker column, which
        // the dispatch join appends after the probe columns. The marker's
        // three-valued semantics are defined per key comparison, so only the
        // single-equality shape (what `IN` compiles to) translates.
        JoinType::MARK => {
            if probe_keys.len() != 1 || !residual_filters.is_empty() {
                return Err(OperatorError::Unsupported(
                    "mark joins are supported on exactly one equality condition".to_string(),
                ));
            }
            (JoinKind::ProbeMark, kept_probe_output, Vec::new())
        }
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

/// Build a [`JoinKind::Range`] join: an inner join whose one condition is a
/// `<`/`<=`/`>`/`>=` comparison, with no equality to hash on.
fn build_range_join(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    mut inputs: Vec<PlanNode>,
    inequalities: Vec<JoinCondition<'_>>,
    residual_predicates: Vec<Expr<'_>>,
) -> Result<PlanNode, OperatorError> {
    let [condition] = inequalities.as_slice() else {
        return Err(OperatorError::Unsupported(format!(
            "joins must have an equality condition or exactly one comparison, got {} comparisons",
            inequalities.len()
        )));
    };
    if !residual_predicates.is_empty() {
        return Err(OperatorError::Unsupported(
            "a range join does not support extra conditions".to_string(),
        ));
    }
    if join.join_type()? != JoinType::INNER {
        return Err(OperatorError::Unsupported(format!(
            "Unsupported range join type: {:?}",
            join.join_type()?
        )));
    }
    let compare = match &condition.comparison {
        &ExpressionType::COMPARE_LESSTHAN => RangeCompare::Less,
        &ExpressionType::COMPARE_LESSTHANOREQUALTO => RangeCompare::LessEq,
        &ExpressionType::COMPARE_GREATERTHAN => RangeCompare::Greater,
        &ExpressionType::COMPARE_GREATERTHANOREQUALTO => RangeCompare::GreaterEq,
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported join comparison type: {other:?}"
            )));
        }
    };

    let probe_key_expr = Expression::from_handle(condition.left)?;
    let build_key_expr = Expression::from_handle(condition.right)?;
    let key_type = probe_key_expr.result_type()?;
    if build_key_expr.result_type()? != key_type {
        return Err(OperatorError::Unsupported(format!(
            "join key types differ: {key_type:?} vs {:?} \
             (the planner casts both sides of a condition to a common type, so this plan \
             shape is unexpected)",
            build_key_expr.result_type()?
        )));
    }
    // The sorted build side compares keys by their native values, which
    // matches SQL ordering only for these types (floats order NaN
    // differently, strings are not fixed-width).
    match key_type {
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
        | Type::Timestamp => {}
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported range join key type: {other:?}"
            )));
        }
    }

    let build_node = inputs.pop().expect("a join has two inputs");
    let probe_node = inputs.pop().expect("a join has two inputs");
    // The output lists below index the sides' original columns, so they are
    // sized before a computed key gets materialized as an extra column.
    let probe_width = probe_node.output_types()?.len();
    let build_width = build_node.output_types()?.len();
    let (probe_node, probe_key) = materialize_key_column(probe_node, probe_key_expr)?;
    let (build_node, build_key) = materialize_key_column(build_node, build_key_expr)?;

    let left_map: Vec<usize> = join.left_projection_map()?;
    let right_map: Vec<usize> = join.right_projection_map()?;
    let probe_output: Vec<usize> = if left_map.is_empty() {
        (0..probe_width).collect()
    } else {
        left_map
    };
    let build_output: Vec<usize> = if right_map.is_empty() {
        (0..build_width).collect()
    } else {
        right_map
    };

    let probe_types = probe_node.output_types()?;
    let build_types = build_node.output_types()?;
    let probe_nullability = probe_node.output_nullability();
    let build_nullability = build_node.output_nullability();
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
        inputs: vec![probe_node, build_node],
        operator: Operator::Join(Join {
            probe_keys: vec![probe_key],
            build_keys: vec![build_key],
            key_types: vec![key_type],
            probe_output,
            build_output,
            probe_column_types,
            build_column_types,
            residual_filters: Vec::new(),
            kind: JoinKind::Range(compare),
        }),
    })
}

/// The column a join key expression is read from. A plain column ref is its
/// own answer; a computed key (e.g. the cast the planner wraps one side in to
/// unify types) is materialized by a projection that passes every existing
/// column through and appends the key as one more.
fn materialize_key_column(
    input: PlanNode,
    key: Expression,
) -> Result<(PlanNode, usize), OperatorError> {
    if let Expression::Ref(key) = &key {
        return Ok((input, key.column_idx));
    }
    let types = input.output_types()?;
    let mut projections: Vec<Expression> = types
        .iter()
        .enumerate()
        .map(|(column_idx, column_type)| {
            Expression::Ref(Ref {
                column_idx,
                return_type: column_type.clone(),
                name: None,
            })
        })
        .collect();
    let key_column = projections.len();
    projections.push(key);
    Ok((
        PlanNode {
            name: "materialize join key".to_string(),
            inputs: vec![input],
            operator: Operator::Projection(Projection { projections }),
        },
        key_column,
    ))
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
    ctx.register_cte_output(cte_index, types, nullable);

    let body = build_node(op.child(1)?, ctx)?;
    let sites = ctx.take_cte_sites(cte_index);
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

/// One scan of the innermost enclosing delim join's de-duplicated values.
fn build_delim_get_scan(
    get: DelimGetView<'_>,
    ctx: &mut BuildCtx,
) -> Result<CteScan, OperatorError> {
    let Some(dedup_keys_cte) = ctx.delim_target() else {
        return Err(OperatorError::Unsupported(
            "DELIM_GET outside a delim join".to_string(),
        ));
    };
    let scan = ctx.create_cte_scan(dedup_keys_cte)?;
    let declared: Vec<Type> = get
        .column_types()?
        .into_iter()
        .map(type_from_logical)
        .collect::<Result<_, _>>()
        .map_err(ExpressionError::from)?;
    if declared != scan.types {
        return Err(OperatorError::Unsupported(format!(
            "DELIM_GET declares {declared:?} but the delim join de-duplicates {:?}",
            scan.types
        )));
    }
    Ok(scan)
}

/// Translate DuckDB's delim join into ordinary Pivot operators.
///
/// A delim join is how DuckDB evaluates many correlated subqueries without
/// running the subquery separately for every outer row. For example:
///
/// ```sql
/// SELECT ...
/// FROM sales, parts
/// WHERE p_key = s_part
///   AND p_flag = 'a'
///   AND s_qty < (
///       SELECT avg(s2.s_qty)
///       FROM sales s2
///       WHERE s2.s_part = p_key
///   )
/// ```
///
/// The inner query is correlated because its `p_key` comes from the outer
/// query. Another way to read it is: "for each relevant `p_key`, calculate an
/// average, then attach that average to every outer row with that key."
///
/// DuckDB represents this as a delim join with two ordinary children and one
/// implicit data dependency between them:
///
/// * Child 0 is the **outer side**. It produces the candidate rows from
///   `sales` and `parts`, before applying the predicate that uses the scalar
///   subquery. These rows can contain the same correlation key many times.
/// * Child 1 is the **subquery side**. It produces one `(key, answer)` row for
///   each relevant correlation key. Child 1 has no normal input edge from
///   child 0, so a `DELIM_GET` leaf supplies it with the distinct keys that
///   child 0 produced. The delim join's `duplicate_eliminated_columns` say
///   which child-0 columns make up those keys. A key may be a tuple of several
///   columns, not just one column.
///
/// The dataflow is therefore:
///
/// 1. Run the outer side once.
/// 2. Keep its complete rows for the final join, and separately take the
///    distinct correlation keys from those rows.
/// 3. Feed the distinct keys to every `DELIM_GET` in the subquery side.
/// 4. Run the subquery once for that set of keys, producing `(key, answer)`
///    rows.
/// 5. Join those answers back to the complete outer rows. Duplicate outer
///    keys deliberately reappear here: every original row receives the answer
///    for its key.
///
/// For a concrete example, suppose the outer side produces:
///
/// ```text
/// p_key  s_qty
///      1      4
///      1      8
///      2      9
/// ```
///
/// `DELIM_DEDUP` reduces the correlation-key column to `{1, 2}`. The
/// subquery's `DELIM_GET` reads those two keys, joins them to `sales s2`, and
/// produces `(1, avg 6)` and `(2, avg 9)`. The final delim join attaches the
/// answers to the original rows:
///
/// ```text
/// p_key  s_qty  avg
///      1      4    6
///      1      8    6
///      2      9    9
/// ```
///
/// The `s_qty < avg` filter above the delim join then keeps only the first
/// row. One batched subquery plan computed the answers for the distinct key
/// set rather than rerunning the subquery for every outer row.
///
/// Pivot has no native delim-join operator, so this function expresses the
/// implicit dependency with two synthetic CTEs. Here a CTE means "run this
/// definition once, make its rows available to the scans in its body, and
/// return the body's output."
///
/// * `outer_rows_cte` publishes all outer rows. It has two readers: the
///   distinct operation and the final join.
/// * `dedup_keys_cte` publishes only the distinct correlation keys. Each
///   translated `DELIM_GET` becomes a scan of this CTE.
///
/// The complete plan produced here is:
///
/// ```text
/// DELIM_JOIN: Cte(outer_rows_cte)
/// ├── definition: (outer side)
/// │                 Produces the complete candidate rows once.
/// └── body: DELIM_KEY_FEED: Cte(dedup_keys_cte)
///     ├── definition: DELIM_DEDUP
///     │   └── DELIM_OUTER_ROWS: CteScan(outer_rows_cte)
///     │                 First reader of the complete outer rows; emits
///     │                 only the distinct correlation-key tuples.
///     └── body: DELIM_SUBQUERY_JOIN
///         ├── DELIM_OUTER_ROWS: CteScan(outer_rows_cte)
///         │                 Second reader of the complete outer rows; probes
///         │                 for its subquery answer.
///         └── (subquery side)
///             └── ... DELIM_GET -> CteScan(dedup_keys_cte)
///                       Reads the keys published by DELIM_DEDUP and
///                       computes one answer per key, the join's build side.
/// ```
///
/// Every kind keeps that orientation: the subquery side's answer table (one
/// row per distinct key) is the small side, so it is the one worth building a
/// hash table from, and the outer rows probe it. LEFT preserves the outer
/// rows through [`JoinKind::ProbeOuter`], which pads an unmatched probe row
/// with NULL build columns; SEMI keeps a probing row only when an answer
/// exists, and ANTI (a decorrelated `NOT EXISTS`) only when none does.
///
/// DuckDB may express the final delim-join condition as `IS NOT DISTINCT FROM`
/// because the distinct key set can contain NULL. For the supported shapes,
/// the subquery reaches this final join through its own ordinary equality
/// predicate against the key set; that equality does not produce a subquery
/// key for NULL. We can therefore use plain equality here: non-NULL keys behave
/// identically, while a NULL-keyed outer row remains unmatched and is padded
/// for LEFT, discarded for SEMI/INNER, and kept for ANTI, matching DuckDB's
/// result (a `NOT EXISTS` whose correlated equality compares against NULL
/// finds no row, so it holds).
fn build_delim_join(
    op: LogicalOp<'_>,
    delim: DelimJoinView<'_>,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    if delim.is_flipped()? {
        return Err(OperatorError::Unsupported(
            "flipped delim join (de-duplicating the RHS) is not supported".to_string(),
        ));
    }
    let join = delim.join();

    let outer_side = build_node(op.child(0)?, ctx)?;
    let outer_types = outer_side.output_types()?;
    let outer_nullability = outer_side.output_nullability();

    let mut correlation_columns: Vec<(usize, Type)> = Vec::new();
    for column in delim.delim_columns()? {
        let column = Expression::from_handle(column)?;
        let Expression::Ref(reference) = column else {
            return Err(OperatorError::Unsupported(format!(
                "delim columns must be plain columns, got: {column:?}"
            )));
        };
        correlation_columns.push((reference.column_idx, reference.return_type));
    }
    if correlation_columns.is_empty() {
        return Err(OperatorError::Unsupported(
            "a delim join must de-duplicate at least one column".to_string(),
        ));
    }

    // The two shared streams, as synthetic CTE channels: the outer side's
    // rows, and the dedup'd correlation keys the subquery side scans.
    let outer_rows_cte = ctx.create_synthetic_cte(outer_types.clone(), outer_nullability.clone());
    let dedup_keys_cte = ctx.create_synthetic_cte(
        correlation_columns
            .iter()
            .map(|(_, ty)| ty.clone())
            .collect(),
        correlation_columns
            .iter()
            .map(|&(idx, _)| outer_nullability[idx])
            .collect(),
    );

    let subquery_child = op.child(1)?;
    let subquery_side =
        ctx.with_delim_target(dedup_keys_cte, |ctx| build_node(subquery_child, ctx))?;
    let delim_scan_sites = ctx.take_cte_sites(dedup_keys_cte);
    if delim_scan_sites == 0 {
        return Err(OperatorError::Unsupported(
            "a delim join whose subquery side reads no DELIM_GET".to_string(),
        ));
    }

    let mut outer_keys = Vec::new();
    let mut subquery_keys = Vec::new();
    let mut key_types = Vec::new();
    for entry in join.conditions()? {
        let condition = match entry {
            JoinConditionEntry::Comparison(condition) => condition,
            JoinConditionEntry::Expression(_) => {
                return Err(OperatorError::Unsupported(
                    "delim joins with expression-form conditions are not supported".to_string(),
                ));
            }
        };
        if !matches!(
            condition.comparison,
            ExpressionType::COMPARE_EQUAL | ExpressionType::COMPARE_NOT_DISTINCT_FROM
        ) {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported delim join comparison type: {:?}",
                condition.comparison
            )));
        }
        let (outer_key, outer_key_type) = join_key_ref(Expression::from_handle(condition.left)?)?;
        let (subquery_key, subquery_key_type) =
            join_key_ref(Expression::from_handle(condition.right)?)?;
        if outer_key_type != subquery_key_type {
            return Err(OperatorError::Unsupported(format!(
                "join key types differ: {outer_key_type:?} vs {subquery_key_type:?} \
                 (the planner casts both sides of a condition to a common type, so this plan \
                 shape is unexpected)"
            )));
        }
        outer_keys.push(outer_key);
        subquery_keys.push(subquery_key);
        key_types.push(outer_key_type);
    }
    if key_types.is_empty() {
        return Err(OperatorError::Unsupported(
            "joins must have at least one equality condition".to_string(),
        ));
    }

    let subquery_types = subquery_side.output_types()?;
    let subquery_nullability = subquery_side.output_nullability();
    let left_map: Vec<usize> = join.left_projection_map()?;
    let right_map: Vec<usize> = join.right_projection_map()?;
    let outer_output: Vec<usize> = if left_map.is_empty() {
        (0..outer_types.len()).collect()
    } else {
        left_map
    };
    let subquery_output: Vec<usize> = if right_map.is_empty() {
        (0..subquery_types.len()).collect()
    } else {
        right_map
    };

    let read_outer_rows = || PlanNode {
        name: "DELIM_OUTER_ROWS".to_string(),
        inputs: Vec::new(),
        operator: Operator::CteScan(CteScan {
            cte_index: outer_rows_cte,
            types: outer_types.clone(),
            nullable: outer_nullability.clone(),
        }),
    };
    let column_types = |output: &[usize], types: &[Type], nullability: &[bool]| {
        output
            .iter()
            .map(|&i| (types[i].clone(), nullability[i]))
            .collect::<Vec<_>>()
    };

    let outer_column_types = column_types(&outer_output, &outer_types, &outer_nullability);
    let subquery_column_types =
        column_types(&subquery_output, &subquery_types, &subquery_nullability);
    // Every kind keeps the natural orientation: the outer side probes the
    // subquery side's (small, one row per distinct key) answer table. LEFT
    // preserves the outer side through the probe-side outer mode, and a semi
    // or anti join emits no build column at all (see `build_join`).
    let (kind, build_output, build_column_types) = match join.join_type()? {
        JoinType::LEFT => (JoinKind::ProbeOuter, subquery_output, subquery_column_types),
        JoinType::INNER => (JoinKind::Inner, subquery_output, subquery_column_types),
        JoinType::SEMI => (JoinKind::ProbeSemi, Vec::new(), Vec::new()),
        JoinType::ANTI => (JoinKind::ProbeAnti, Vec::new(), Vec::new()),
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported delim join type: {other:?}"
            )));
        }
    };
    let join_node = PlanNode {
        name: "DELIM_SUBQUERY_JOIN".to_string(),
        inputs: vec![read_outer_rows(), subquery_side],
        operator: Operator::Join(Join {
            probe_keys: outer_keys,
            build_keys: subquery_keys,
            key_types,
            probe_output: outer_output,
            build_output,
            probe_column_types: outer_column_types,
            build_column_types,
            residual_filters: Vec::new(),
            kind,
        }),
    };

    let dedup = PlanNode {
        name: "DELIM_DEDUP".to_string(),
        inputs: vec![read_outer_rows()],
        operator: Operator::Distinct(Distinct {
            keys: correlation_columns,
        }),
    };
    let key_feed = PlanNode {
        name: "DELIM_KEY_FEED".to_string(),
        inputs: vec![dedup, join_node],
        operator: Operator::Cte(Cte {
            cte_index: dedup_keys_cte,
            sites: delim_scan_sites,
        }),
    };
    Ok(PlanNode {
        name: op.name()?,
        inputs: vec![outer_side, key_feed],
        operator: Operator::Cte(Cte {
            cte_index: outer_rows_cte,
            sites: 2,
        }),
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
