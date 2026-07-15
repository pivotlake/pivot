//! Safe borrowed handles over DuckDB's live plan objects.
//!
//! `extract_plan` hands back a [`Plan`] that owns DuckDB's resolved
//! `LogicalOperator` tree. Consumers walk that tree through [`LogicalOp`] and
//! [`Expr`], borrowed handles tethered by their lifetime to the owning [`Plan`]
//! (like a `rusqlite::Row` borrowed from its `Statement`). Every `unsafe` enum
//! decode and raw FFI accessor stays behind these methods, so callers read the
//! plan in safe, typed Rust without ever touching a discriminant byte or an
//! opaque pointer.
//!
//! [`LogicalOp`] carries the accessors shared by every operator (`op_type`,
//! `name`, `children`); the kind-specific ones live on the typed views of
//! [`Operator`], which [`LogicalOp::operator`] dispatches to. The views are
//! zero-cost borrowed references, not owned data: there is no intermediate
//! representation here, and materializing the tree into owned structs is the
//! consumer's job.

use cxx::UniquePtr;

use crate::catalog_provider::OptionalTableWrapper;
use crate::duckdb_bridge::duckdb_types::{
    ExpressionType, JoinType, LimitNodeType, LogicalOperatorType, LogicalTypeId, OrderType,
};
use crate::duckdb_bridge::ffi;
use crate::types::ScalarValue;

/// DuckDB's virtual row-id column identifier, used by late-materialization
/// row-id stripping to recognise the threaded-up row-id column.
pub fn rowid_column_id() -> usize {
    ffi::rowid_column_id()
}

/// Decode a DuckDB [`Value`](ffi::Value) into a typed [`ScalarValue`]: read its
/// type, then the matching typed accessor. Shared by query constants and
/// table-function arguments. A type the bridge doesn't decode lands in
/// [`ScalarValue::Other`].
fn scalar_from_value(v: &ffi::Value) -> ScalarValue {
    use LogicalTypeId as L;
    match LogicalTypeId::from_u8(ffi::value_type(v)) {
        L::BOOLEAN => ScalarValue::Boolean(ffi::value_bool(v)),
        L::TINYINT => ScalarValue::Int8(ffi::value_i8(v)),
        L::SMALLINT => ScalarValue::Int16(ffi::value_i16(v)),
        L::INTEGER => ScalarValue::Int32(ffi::value_i32(v)),
        L::BIGINT => ScalarValue::Int64(ffi::value_i64(v)),
        L::UTINYINT => ScalarValue::UInt8(ffi::value_u8(v)),
        L::USMALLINT => ScalarValue::UInt16(ffi::value_u16(v)),
        L::UINTEGER => ScalarValue::UInt32(ffi::value_u32(v)),
        L::UBIGINT => ScalarValue::UInt64(ffi::value_u64(v)),
        L::HUGEINT => ScalarValue::Int128(
            ((ffi::value_hugeint_hi(v) as i128) << 64) | ffi::value_hugeint_lo(v) as i128,
        ),
        L::FLOAT => ScalarValue::Float32(ffi::value_f32(v)),
        L::DOUBLE => ScalarValue::Float64(ffi::value_f64(v)),
        L::VARCHAR => ScalarValue::Utf8(ffi::value_string(v)),
        L::DATE => ScalarValue::Date(ffi::value_date(v)),
        L::TIMESTAMP => ScalarValue::Timestamp(ffi::value_timestamp(v)),
        L::INTERVAL => ScalarValue::Interval {
            months: ffi::value_interval_months(v),
            days: ffi::value_interval_days(v),
            micros: ffi::value_interval_micros(v),
        },
        other => ScalarValue::Other(other),
    }
}

/// Owns DuckDB's resolved plan tree (kept alive while it is walked) plus the
/// binder-resolved client column names. Hand out the root with [`Plan::root`].
pub struct Plan {
    handle: UniquePtr<ffi::PlanHandle>,
    output_names: Vec<String>,
}

impl Plan {
    pub(crate) fn new(handle: UniquePtr<ffi::PlanHandle>, output_names: Vec<String>) -> Self {
        Self {
            handle,
            output_names,
        }
    }

    /// The root operator of the plan, borrowed for the lifetime of this [`Plan`].
    pub fn root(&self) -> LogicalOp<'_> {
        LogicalOp {
            raw: ffi::plan_root(&self.handle),
        }
    }

    /// The result column names DuckDB resolved for the client, in output order
    /// (e.g. `["hour", "count_star()"]`). Empty when the bridge could not recover
    /// them.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Consume the plan, returning the resolved output names.
    pub fn into_output_names(self) -> Vec<String> {
        self.output_names
    }
}

/// A borrowed DuckDB `LogicalOperator`, tethered to the [`Plan`] that owns it.
/// Cheap to copy (it is one reference). Carries the accessors common to every
/// operator; reach the kind-specific ones through [`operator`](Self::operator).
#[derive(Clone, Copy)]
pub struct LogicalOp<'plan> {
    raw: &'plan ffi::LogicalOperator,
}

impl<'plan> LogicalOp<'plan> {
    /// The operator's [`LogicalOperatorType`].
    pub fn op_type(self) -> LogicalOperatorType {
        LogicalOperatorType::from_u8(ffi::lo_type(self.raw))
    }

    /// The operator's display name (`LogicalOperator::GetName`).
    pub fn name(self) -> String {
        ffi::lo_name(self.raw)
    }

    pub fn child_count(self) -> usize {
        ffi::lo_child_count(self.raw)
    }

    /// The optimizer's row estimate for this operator, when it computed one.
    pub fn estimated_cardinality(self) -> Option<u64> {
        ffi::lo_has_estimated_cardinality(self.raw).then(|| ffi::lo_estimated_cardinality(self.raw))
    }

    pub fn child(self, index: usize) -> LogicalOp<'plan> {
        LogicalOp {
            raw: ffi::lo_child(self.raw, index),
        }
    }

    pub fn children(self) -> impl Iterator<Item = LogicalOp<'plan>> {
        (0..self.child_count()).map(move |i| self.child(i))
    }

    /// Dispatch on the operator's [`LogicalOperatorType`] into a typed [`Operator`]
    /// view that exposes only that operator's accessors. A `LOGICAL_GET` splits
    /// into [`Operator::TableScan`] / [`Operator::TableFunctionScan`] by whether it
    /// reads a base table; any kind the consumer doesn't handle is
    /// [`Operator::Unsupported`].
    pub fn operator(self) -> Operator<'plan> {
        use LogicalOperatorType as L;
        match self.op_type() {
            L::LOGICAL_PROJECTION => Operator::Projection(Projection { raw: self.raw }),
            L::LOGICAL_FILTER => Operator::Filter(Filter { raw: self.raw }),
            L::LOGICAL_AGGREGATE_AND_GROUP_BY => Operator::Aggregate(Aggregate { raw: self.raw }),
            L::LOGICAL_ORDER_BY => Operator::OrderBy(OrderBy { raw: self.raw }),
            L::LOGICAL_TOP_N => Operator::TopN(TopN { raw: self.raw }),
            L::LOGICAL_LIMIT => Operator::Limit(Limit { raw: self.raw }),
            L::LOGICAL_GET if ffi::lo_get_has_table(self.raw) => {
                Operator::TableScan(TableScan { raw: self.raw })
            }
            L::LOGICAL_GET => Operator::TableFunctionScan(TableFunctionScan { raw: self.raw }),
            L::LOGICAL_CREATE_TABLE => Operator::CreateTable(CreateTable { raw: self.raw }),
            L::LOGICAL_SET => Operator::Set(Set { raw: self.raw }),
            L::LOGICAL_RESET => Operator::Reset(Reset { raw: self.raw }),
            L::LOGICAL_COMPARISON_JOIN => {
                Operator::ComparisonJoin(ComparisonJoin { raw: self.raw })
            }
            L::LOGICAL_DUMMY_SCAN => Operator::DummyScan,
            L::LOGICAL_EXPLAIN => Operator::Explain,
            _ => Operator::Unsupported,
        }
    }
}

/// A borrowed, typed view of a [`LogicalOp`], discriminated by operator kind.
/// Each variant exposes only the accessors valid for that kind, so a projection
/// accessor can't be called on a filter. Obtained via [`LogicalOp::operator`].
/// Every variant is a zero-cost borrowed view (one reference), not owned data.
#[derive(Clone, Copy)]
pub enum Operator<'plan> {
    Projection(Projection<'plan>),
    Filter(Filter<'plan>),
    Aggregate(Aggregate<'plan>),
    OrderBy(OrderBy<'plan>),
    TopN(TopN<'plan>),
    Limit(Limit<'plan>),
    /// A base-table scan (`LOGICAL_GET` over a table).
    TableScan(TableScan<'plan>),
    /// A scan over a table-valued function (`LOGICAL_GET` with no base table).
    TableFunctionScan(TableFunctionScan<'plan>),
    CreateTable(CreateTable<'plan>),
    /// `SET name = value`.
    Set(Set<'plan>),
    /// `RESET name`.
    Reset(Reset<'plan>),
    /// A comparison join; the consumer only handles the late-materialization
    /// shape (see [`ComparisonJoin::is_late_materialization`]).
    ComparisonJoin(ComparisonJoin<'plan>),
    /// The single-row source under a `FROM`-less `SELECT`.
    DummyScan,
    /// `EXPLAIN <query>`.
    Explain,
    /// Any operator type the consumer doesn't handle.
    Unsupported,
}

/// Define zero-cost borrowed view newtypes over a raw FFI object: each wraps one
/// `&'plan $raw` reference and is `Copy`. The doc comment on each name carries
/// through. Used for both the [`Operator`] and [`Expression`] views.
macro_rules! define_handles {
    ($raw:ty; $($(#[$attr:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$attr])*
            #[derive(Clone, Copy)]
            pub struct $name<'plan> {
                raw: &'plan $raw,
            }
        )+
    };
}

define_handles! { ffi::LogicalOperator;
    /// A `LogicalProjection`: a list of output expressions.
    Projection,
    /// A `LogicalFilter`: boolean conditions plus an optional projection map.
    Filter,
    /// A `LogicalAggregate`: GROUP BY keys plus aggregate expressions.
    Aggregate,
    /// A `LogicalOrder`: a list of sort keys.
    OrderBy,
    /// A `LogicalTopN`: ORDER BY + LIMIT, optionally a dynamic-filter producer.
    TopN,
    /// A `LogicalLimit`: a bare `LIMIT`/`OFFSET` with no ORDER BY.
    Limit,
    /// A `LogicalGet` over a base table.
    TableScan,
    /// A `LogicalGet` over a table-valued function.
    TableFunctionScan,
    /// A `LogicalCreateTable` with an explicit column list.
    CreateTable,
    /// A `LogicalSet`: `SET name = value`.
    Set,
    /// A `LogicalReset`: `RESET name`.
    Reset,
    /// A `LogicalComparisonJoin`.
    ComparisonJoin,
}

impl<'plan> Projection<'plan> {
    /// The projection's output expressions.
    pub fn exprs(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::lo_projection_expr_count(self.raw)).map(move |i| Expr {
            raw: ffi::lo_projection_expr(self.raw, i),
        })
    }
}

impl<'plan> Filter<'plan> {
    /// The filter's boolean conditions (implicitly ANDed).
    pub fn exprs(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::lo_filter_expr_count(self.raw)).map(move |i| Expr {
            raw: ffi::lo_filter_expr(self.raw, i),
        })
    }

    /// The filter's `projection_map`: each entry is the child-output column index
    /// it keeps, paired with that column's type. Empty when the filter passes all
    /// child columns through.
    pub fn projection_map(self) -> impl Iterator<Item = (usize, LogicalTypeId)> {
        (0..ffi::lo_filter_projection_map_count(self.raw)).map(move |i| {
            (
                ffi::lo_filter_projection_map_index(self.raw, i),
                LogicalTypeId::from_u8(ffi::lo_filter_type_id(self.raw, i)),
            )
        })
    }
}

impl<'plan> Aggregate<'plan> {
    pub fn groups(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::lo_aggregate_group_count(self.raw)).map(move |i| Expr {
            raw: ffi::lo_aggregate_group(self.raw, i),
        })
    }

    pub fn expressions(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::lo_aggregate_expr_count(self.raw)).map(move |i| Expr {
            raw: ffi::lo_aggregate_expr(self.raw, i),
        })
    }
}

impl<'plan> OrderBy<'plan> {
    pub fn keys(self) -> impl Iterator<Item = OrderKey<'plan>> {
        (0..ffi::lo_orderby_count(self.raw)).map(move |i| OrderKey {
            direction: OrderType::from_u8(ffi::lo_orderby_direction(self.raw, i)),
            expression: Expr {
                raw: ffi::lo_orderby_expr(self.raw, i),
            },
        })
    }
}

impl<'plan> TopN<'plan> {
    pub fn keys(self) -> impl Iterator<Item = OrderKey<'plan>> {
        (0..ffi::lo_topn_order_count(self.raw)).map(move |i| OrderKey {
            direction: OrderType::from_u8(ffi::lo_topn_order_direction(self.raw, i)),
            expression: Expr {
                raw: ffi::lo_topn_order_expr(self.raw, i),
            },
        })
    }

    pub fn limit(self) -> usize {
        ffi::lo_topn_limit(self.raw)
    }

    pub fn offset(self) -> usize {
        ffi::lo_topn_offset(self.raw)
    }

    /// The dynamic filter this Top-N publishes as a producer, if its optimizer
    /// installed one. At runtime it pushes its running boundary into the
    /// referenced shared cell so consumer scans elsewhere can prune.
    pub fn dynamic_filter(self) -> Option<DynamicFilterRef> {
        ffi::lo_topn_has_dynamic_filter(self.raw).then(|| DynamicFilterRef {
            data_id: ffi::lo_topn_dynamic_filter_data_id(self.raw),
            column: ffi::lo_topn_dynamic_filter_column(self.raw),
            comparison: ExpressionType::from_u8(ffi::lo_topn_dynamic_filter_comparison(self.raw)),
        })
    }
}

impl<'plan> Limit<'plan> {
    pub fn value_kind(self) -> LimitNodeType {
        LimitNodeType::from_u8(ffi::lo_limit_value_kind(self.raw))
    }

    pub fn value(self) -> usize {
        ffi::lo_limit_value(self.raw)
    }

    pub fn offset_kind(self) -> LimitNodeType {
        LimitNodeType::from_u8(ffi::lo_limit_offset_kind(self.raw))
    }

    pub fn offset(self) -> usize {
        ffi::lo_limit_offset(self.raw)
    }
}

/// The projected output columns shared by base-table and table-function scans:
/// each a storage column index paired with its type.
fn scan_output_columns(
    raw: &ffi::LogicalOperator,
) -> impl Iterator<Item = (usize, LogicalTypeId)> + '_ {
    (0..ffi::lo_get_output_count(raw)).map(move |i| {
        (
            ffi::lo_get_output_column(raw, i),
            LogicalTypeId::from_u8(ffi::lo_get_output_type(raw, i)),
        )
    })
}

impl<'plan> TableScan<'plan> {
    /// Move the pivot table handle out of this scan's catalog entry. Call once
    /// per base-table scan during the walk.
    pub fn take_table(self) -> Box<OptionalTableWrapper> {
        ffi::lo_get_take_table(self.raw)
    }

    /// The scan's projected output columns, each a storage column index paired
    /// with its type.
    pub fn output_columns(self) -> impl Iterator<Item = (usize, LogicalTypeId)> + 'plan {
        scan_output_columns(self.raw)
    }

    /// The static `col op const` predicates DuckDB pushed into `table_filters`,
    /// rebuilt as expressions owned by the returned [`PushedConditions`]. Empty
    /// when there are none.
    pub fn pushed_conditions(self) -> Result<PushedConditions, cxx::Exception> {
        Ok(PushedConditions {
            list: ffi::lo_get_pushed_conditions(self.raw)?,
        })
    }

    /// Dynamic-filter consumers attached to this scan's `table_filters`.
    pub fn dynamic_filters(self) -> impl Iterator<Item = DynamicFilterRef> + 'plan {
        (0..ffi::lo_get_dynamic_filter_count(self.raw)).map(move |i| DynamicFilterRef {
            data_id: ffi::lo_get_dynamic_filter_data_id(self.raw, i),
            column: ffi::lo_get_dynamic_filter_column(self.raw, i),
            comparison: ExpressionType::from_u8(ffi::lo_get_dynamic_filter_comparison(self.raw, i)),
        })
    }
}

impl<'plan> TableFunctionScan<'plan> {
    pub fn function_name(self) -> String {
        ffi::lo_get_function_name(self.raw)
    }

    pub fn has_named_params(self) -> bool {
        ffi::lo_get_has_named_params(self.raw)
    }

    /// The function's bound constant arguments.
    pub fn params(self) -> impl Iterator<Item = ScalarValue> + 'plan {
        (0..ffi::lo_get_param_count(self.raw))
            .map(move |i| scalar_from_value(ffi::lo_get_param(self.raw, i)))
    }

    /// The scan's projected output columns, each a generated column index paired
    /// with its type.
    pub fn output_columns(self) -> impl Iterator<Item = (usize, LogicalTypeId)> + 'plan {
        scan_output_columns(self.raw)
    }
}

impl<'plan> CreateTable<'plan> {
    pub fn name(self) -> String {
        ffi::lo_create_table_name(self.raw)
    }

    pub fn columns(self) -> impl Iterator<Item = (String, LogicalTypeId)> {
        (0..ffi::lo_create_column_count(self.raw)).map(move |i| {
            (
                ffi::lo_create_column_name(self.raw, i),
                LogicalTypeId::from_u8(ffi::lo_create_column_type(self.raw, i)),
            )
        })
    }

    pub fn options(self) -> impl Iterator<Item = (String, String)> {
        (0..ffi::lo_create_option_count(self.raw)).map(move |i| {
            (
                ffi::lo_create_option_key(self.raw, i),
                ffi::lo_create_option_value(self.raw, i),
            )
        })
    }

    pub fn if_not_exists(self) -> bool {
        ffi::lo_create_if_not_exists(self.raw)
    }

    pub fn or_replace(self) -> bool {
        ffi::lo_create_or_replace(self.raw)
    }

    pub fn temporary(self) -> bool {
        ffi::lo_create_temporary(self.raw)
    }

    pub fn has_query(self) -> bool {
        ffi::lo_create_has_query(self.raw)
    }

    pub fn constraint_count(self) -> usize {
        ffi::lo_create_constraint_count(self.raw)
    }

    /// The statement's constraints: `Some(column)` for a NOT NULL on that
    /// column, `None` for any other constraint kind (which the consumer
    /// rejects).
    pub fn constraints(self) -> impl Iterator<Item = Option<usize>> {
        const NOT_NULL: u8 = 1;
        (0..ffi::lo_create_constraint_count(self.raw)).map(move |i| {
            (ffi::lo_create_constraint_kind(self.raw, i) == NOT_NULL)
                .then(|| ffi::lo_create_constraint_column(self.raw, i))
        })
    }
}

impl<'plan> Set<'plan> {
    pub fn name(self) -> String {
        ffi::lo_set_name(self.raw)
    }

    pub fn value(self) -> String {
        ffi::lo_set_value(self.raw)
    }
}

impl<'plan> Reset<'plan> {
    pub fn name(self) -> String {
        ffi::lo_reset_name(self.raw)
    }
}

impl<'plan> ComparisonJoin<'plan> {
    /// Whether this is the SEMI join DuckDB's late_materialization optimizer
    /// produces (vs a user IN/EXISTS), which the consumer collapses into a fetch.
    pub fn is_late_materialization(self) -> bool {
        ffi::lo_is_late_materialization_join(self.raw)
    }

    /// The full-column (LHS) side's output storage columns (row-id excluded) that
    /// the late-materialization fetch re-reads for the surviving rows.
    pub fn columns(self) -> impl Iterator<Item = usize> + 'plan {
        (0..ffi::lo_late_materialization_column_count(self.raw))
            .map(move |i| ffi::lo_late_materialization_column(self.raw, i))
    }

    /// The join's [`JoinType`] (INNER/SEMI/...).
    pub fn join_type(self) -> JoinType {
        JoinType::from_u8(ffi::lo_join_type(self.raw))
    }

    /// The join's conditions. A comparison condition's sides are bound
    /// positionally to their child's output (LHS into child 0, RHS into
    /// child 1); a predicate condition is one boolean expression bound to the
    /// combined (left then right) child outputs.
    pub fn conditions(self) -> impl Iterator<Item = JoinCondition<'plan>> {
        (0..ffi::lo_join_condition_count(self.raw)).map(move |i| {
            if ffi::lo_join_condition_is_comparison(self.raw, i) {
                JoinCondition::Comparison {
                    left: Expr {
                        raw: ffi::lo_join_condition_left(self.raw, i),
                    },
                    right: Expr {
                        raw: ffi::lo_join_condition_right(self.raw, i),
                    },
                    comparison: ExpressionType::from_u8(ffi::lo_join_condition_comparison(
                        self.raw, i,
                    )),
                }
            } else {
                JoinCondition::Predicate(Expr {
                    raw: ffi::lo_join_condition_predicate(self.raw, i),
                })
            }
        })
    }

    /// Which LHS child output columns survive in the join's output, in order.
    /// Empty means all of them. Filled by DuckDB's column-lifetime pass, e.g.
    /// to drop a key column only referenced by the join condition.
    pub fn left_projection_map(self) -> impl Iterator<Item = usize> + 'plan {
        (0..ffi::lo_join_left_projection_map_count(self.raw))
            .map(move |i| ffi::lo_join_left_projection_map_index(self.raw, i))
    }

    /// The RHS twin of [`left_projection_map`](Self::left_projection_map).
    pub fn right_projection_map(self) -> impl Iterator<Item = usize> + 'plan {
        (0..ffi::lo_join_right_projection_map_count(self.raw))
            .map(move |i| ffi::lo_join_right_projection_map_index(self.raw, i))
    }
}

/// One condition of a comparison join.
pub enum JoinCondition<'plan> {
    /// `left <comparison> right`, each side bound positionally to the
    /// corresponding child's output.
    Comparison {
        left: Expr<'plan>,
        right: Expr<'plan>,
        comparison: ExpressionType,
    },
    /// An arbitrary boolean predicate over both sides, bound to the combined
    /// (left then right) child outputs.
    Predicate(Expr<'plan>),
}

/// One sort key of an ORDER BY / TopN: a direction plus the keyed expression.
pub struct OrderKey<'plan> {
    pub direction: OrderType,
    pub expression: Expr<'plan>,
}

/// A dynamic-filter wiring read off a scan or Top-N: the shared cell it
/// references (by address), the column it constrains, and the comparison shape.
/// The constant itself is published into the cell at runtime.
pub struct DynamicFilterRef {
    pub data_id: usize,
    pub column: usize,
    pub comparison: ExpressionType,
}

/// An owning list of synthesized expressions (a scan's pushed-down filter
/// conditions). Borrow the entries as [`Expr`] handles via [`PushedConditions::iter`].
pub struct PushedConditions {
    list: UniquePtr<ffi::ExpressionList>,
}

impl PushedConditions {
    pub fn len(&self) -> usize {
        ffi::expr_list_count(&self.list)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = Expr<'_>> {
        (0..self.len()).map(move |i| Expr {
            raw: ffi::expr_list_get(&self.list, i),
        })
    }
}

/// A borrowed DuckDB bound `Expression`, tethered to the [`Plan`] that owns it.
#[derive(Clone, Copy)]
pub struct Expr<'plan> {
    raw: &'plan ffi::Expression,
}

impl<'plan> Expr<'plan> {
    /// Wrap a raw bound expression. Used by the catalog pushdown callback, which
    /// receives a borrowed `Expression` straight from C++.
    pub(crate) fn from_raw(raw: &'plan ffi::Expression) -> Self {
        Expr { raw }
    }

    /// Dispatch on the expression's [`ExpressionType`] into a typed [`Expression`]
    /// view that exposes only that kind's accessors. Any kind the consumer doesn't
    /// handle is [`Expression::Unsupported`], carrying the raw type.
    pub fn expression(self) -> Expression<'plan> {
        use ExpressionType as T;
        match ExpressionType::from_u8(ffi::expr_type(self.raw)) {
            T::BOUND_REF | T::BOUND_COLUMN_REF => Expression::Ref(Ref { raw: self.raw }),
            T::COMPARE_EQUAL
            | T::COMPARE_NOTEQUAL
            | T::COMPARE_LESSTHAN
            | T::COMPARE_GREATERTHAN
            | T::COMPARE_LESSTHANOREQUALTO
            | T::COMPARE_GREATERTHANOREQUALTO => Expression::Compare(Compare { raw: self.raw }),
            T::COMPARE_BETWEEN => Expression::Between(Between { raw: self.raw }),
            T::VALUE_CONSTANT => Expression::Constant(Constant { raw: self.raw }),
            T::BOUND_AGGREGATE => Expression::AggregateFunc(AggregateFunc { raw: self.raw }),
            T::BOUND_FUNCTION => Expression::Function(Function { raw: self.raw }),
            T::COMPARE_IN => Expression::InList(InList { raw: self.raw }),
            T::CONJUNCTION_AND | T::CONJUNCTION_OR => {
                Expression::Conjunction(Conjunction { raw: self.raw })
            }
            T::CASE_EXPR => Expression::Case(Case { raw: self.raw }),
            T::OPERATOR_NOT => Expression::Not(Not { raw: self.raw }),
            T::OPERATOR_IS_NULL | T::OPERATOR_IS_NOT_NULL => {
                Expression::IsNull(IsNull { raw: self.raw })
            }
            T::OPERATOR_CAST => Expression::Cast(Cast { raw: self.raw }),
            other => Expression::Unsupported(other),
        }
    }
}

/// A borrowed, typed view of an [`Expr`], discriminated by expression kind. Each
/// variant exposes only the accessors valid for that kind, so a comparison
/// accessor can't be called on a constant. Obtained via [`Expr::expression`].
/// Every variant is a zero-cost borrowed view (one reference), not owned data.
pub enum Expression<'plan> {
    /// A bound column reference (`BOUND_REF` / `BOUND_COLUMN_REF`).
    Ref(Ref<'plan>),
    /// A binary comparison (`=`, `<>`, `<`, …).
    Compare(Compare<'plan>),
    Between(Between<'plan>),
    Constant(Constant<'plan>),
    /// An aggregate function call (`SUM`, `COUNT`, …).
    AggregateFunc(AggregateFunc<'plan>),
    /// A scalar function call (`BOUND_FUNCTION`).
    Function(Function<'plan>),
    /// `input IN (v0, v1, …)`.
    InList(InList<'plan>),
    /// A boolean `AND`/`OR`.
    Conjunction(Conjunction<'plan>),
    Case(Case<'plan>),
    /// Logical negation (`NOT expr`).
    Not(Not<'plan>),
    /// `expr IS [NOT] NULL`.
    IsNull(IsNull<'plan>),
    /// A type cast.
    Cast(Cast<'plan>),
    /// Any expression type the consumer doesn't handle, carrying the raw type.
    Unsupported(ExpressionType),
}

define_handles! { ffi::Expression;
    /// A bound column reference.
    Ref,
    /// A binary comparison expression.
    Compare,
    /// A `BETWEEN` range test.
    Between,
    /// A constant value.
    Constant,
    /// An aggregate function call.
    AggregateFunc,
    /// A scalar function call.
    Function,
    /// An `input IN (v0, v1, …)` membership test.
    InList,
    /// A boolean `AND`/`OR` over child predicates.
    Conjunction,
    /// A `CASE WHEN … THEN … ELSE … END` expression.
    Case,
    /// Logical negation (`NOT expr`).
    Not,
    /// `expr IS [NOT] NULL`.
    IsNull,
    /// A type cast (`CAST(child AS return_type)`).
    Cast,
}

impl<'plan> Ref<'plan> {
    /// The referenced column's positional index. A `BOUND_REF` indexes the child
    /// operator's output; a `BOUND_COLUMN_REF` carries its own column index.
    pub fn column_index(self) -> usize {
        if ExpressionType::from_u8(ffi::expr_type(self.raw)) == ExpressionType::BOUND_REF {
            ffi::expr_ref_index(self.raw)
        } else {
            ffi::expr_columnref_index(self.raw)
        }
    }

    /// The column's logical type.
    pub fn return_type(self) -> LogicalTypeId {
        LogicalTypeId::from_u8(ffi::expr_return_type(self.raw))
    }

    /// The column's source name from DuckDB's binding, or `None`. Display-only.
    pub fn alias(self) -> Option<String> {
        ffi::expr_has_alias(self.raw).then(|| ffi::expr_alias(self.raw))
    }
}

impl<'plan> Compare<'plan> {
    /// The specific comparison (one of the `COMPARE_*` [`ExpressionType`]s).
    pub fn comparison_type(self) -> ExpressionType {
        ExpressionType::from_u8(ffi::expr_type(self.raw))
    }

    pub fn left(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_comparison_left(self.raw),
        }
    }

    pub fn right(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_comparison_right(self.raw),
        }
    }

    /// The comparison's result type.
    pub fn return_type(self) -> LogicalTypeId {
        LogicalTypeId::from_u8(ffi::expr_return_type(self.raw))
    }
}

impl<'plan> Between<'plan> {
    pub fn input(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_between_input(self.raw),
        }
    }

    pub fn lower(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_between_lower(self.raw),
        }
    }

    pub fn upper(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_between_upper(self.raw),
        }
    }

    pub fn lower_inclusive(self) -> bool {
        ffi::expr_between_lower_inclusive(self.raw)
    }

    pub fn upper_inclusive(self) -> bool {
        ffi::expr_between_upper_inclusive(self.raw)
    }
}

impl<'plan> Constant<'plan> {
    /// The constant's typed value.
    pub fn value(self) -> ScalarValue {
        scalar_from_value(ffi::expr_constant(self.raw))
    }

    /// The constant's logical type, without decoding its value.
    pub fn return_type(self) -> LogicalTypeId {
        LogicalTypeId::from_u8(ffi::expr_return_type(self.raw))
    }
}

impl<'plan> AggregateFunc<'plan> {
    pub fn name(self) -> String {
        ffi::expr_aggregate_name(self.raw)
    }

    pub fn distinct(self) -> bool {
        ffi::expr_aggregate_distinct(self.raw)
    }

    pub fn children(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::expr_aggregate_child_count(self.raw)).map(move |i| Expr {
            raw: ffi::expr_aggregate_child(self.raw, i),
        })
    }

    /// DuckDB's declared result type for the call.
    pub fn return_type(self) -> LogicalTypeId {
        LogicalTypeId::from_u8(ffi::expr_return_type(self.raw))
    }
}

impl<'plan> Function<'plan> {
    pub fn name(self) -> String {
        ffi::expr_function_name(self.raw)
    }

    pub fn children(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::expr_function_child_count(self.raw)).map(move |i| Expr {
            raw: ffi::expr_function_child(self.raw, i),
        })
    }

    /// The function's result type.
    pub fn return_type(self) -> LogicalTypeId {
        LogicalTypeId::from_u8(ffi::expr_return_type(self.raw))
    }
}

impl<'plan> InList<'plan> {
    /// The `BoundOperatorExpression` children: child 0 is the tested expression,
    /// children 1.. are the list values.
    pub fn children(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::expr_operator_child_count(self.raw)).map(move |i| Expr {
            raw: ffi::expr_operator_child(self.raw, i),
        })
    }
}

impl<'plan> Conjunction<'plan> {
    /// `CONJUNCTION_AND` or `CONJUNCTION_OR`.
    pub fn conjunction_type(self) -> ExpressionType {
        ExpressionType::from_u8(ffi::expr_type(self.raw))
    }

    pub fn children(self) -> impl Iterator<Item = Expr<'plan>> {
        (0..ffi::expr_conjunction_child_count(self.raw)).map(move |i| Expr {
            raw: ffi::expr_conjunction_child(self.raw, i),
        })
    }
}

impl<'plan> Case<'plan> {
    pub fn checks(self) -> impl Iterator<Item = CaseArm<'plan>> {
        (0..ffi::expr_case_check_count(self.raw)).map(move |i| CaseArm {
            when: Expr {
                raw: ffi::expr_case_when(self.raw, i),
            },
            then: Expr {
                raw: ffi::expr_case_then(self.raw, i),
            },
        })
    }

    pub fn else_expr(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_case_else(self.raw),
        }
    }
}

impl<'plan> Not<'plan> {
    /// The negated operand (`OPERATOR_NOT` has a single child).
    pub fn input(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_operator_child(self.raw, 0),
        }
    }
}

impl<'plan> IsNull<'plan> {
    /// The tested operand (a single child).
    pub fn input(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_operator_child(self.raw, 0),
        }
    }

    /// `true` for `IS NOT NULL`, `false` for `IS NULL`.
    pub fn negated(self) -> bool {
        ExpressionType::from_u8(ffi::expr_type(self.raw)) == ExpressionType::OPERATOR_IS_NOT_NULL
    }
}

impl<'plan> Cast<'plan> {
    pub fn child(self) -> Expr<'plan> {
        Expr {
            raw: ffi::expr_cast_child(self.raw),
        }
    }

    /// The cast's target type (a `BoundCastExpression`'s own result type).
    pub fn return_type(self) -> LogicalTypeId {
        LogicalTypeId::from_u8(ffi::expr_return_type(self.raw))
    }
}

/// One `WHEN when THEN then` arm of a `CASE` expression.
pub struct CaseArm<'plan> {
    pub when: Expr<'plan>,
    pub then: Expr<'plan>,
}
