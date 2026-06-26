#include "duckdb-planner/src/duckdb_bridge/cpp/bridge.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_set.hpp"
#include "duckdb/planner/operator/logical_reset.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/common/error_data.hpp"

#include <flatbuffers/flatbuffers.h>
#include <optional>
#include <string>
#include <unordered_map>

using std::string;
using namespace pivot::plan;

struct UnsupportedPlanError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

// Stable id assignment for shared `DynamicFilterData` cells. One cell may be
// referenced by both a producer (TopN; later, hash-join build side) and one or
// more consumers (a LogicalGet's table_filters); deduping by raw pointer
// identity lets the Rust side rebuild the shared-slot graph by index lookup.
using DynamicFilterDedup = std::unordered_map<duckdb::DynamicFilterData *, size_t>;

std::unique_ptr<PlanNodeT> build_plan_node(duckdb::LogicalOperator *op,
                                           rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                                           DynamicFilterDedup &df_dedup);
std::unique_ptr<PlanNodeT> build_late_materialization(duckdb::LogicalComparisonJoin &join,
                                                      rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                                                      DynamicFilterDedup &df_dedup);

// Wrap a concrete expression payload (RefT, CompareT, …) in the ExpressionT
// table whose union tag the Rust reader matches on.
template <typename T>
static std::unique_ptr<ExpressionT> wrap_expr(T value) {
	auto expr = std::make_unique<ExpressionT>();
	expr->kind.Set(std::move(value));
	return expr;
}

// Serialize the finished plan tree (or error) into a FlatBuffers buffer for the
// Rust side. This is the single point where the mutable object-API tree becomes
// an immutable wire buffer.
static rust::Vec<uint8_t> serialize_plan_result(const PlanResultT &result) {
	flatbuffers::FlatBufferBuilder fbb;
	fbb.Finish(PlanResult::Pack(fbb, &result));
	rust::Vec<uint8_t> out;
	out.reserve(fbb.GetSize());
	const uint8_t *data = fbb.GetBufferPointer();
	for (flatbuffers::uoffset_t i = 0; i < fbb.GetSize(); i++) {
		out.push_back(data[i]);
	}
	return out;
}

static PlanResultT make_bridge_error(const string &kind, const string &message,
                                     std::optional<string> position = std::nullopt) {
	BridgeErrorT error;
	error.kind = kind;
	error.exception_message = message;
	if (position.has_value()) {
		// Empty stays null on the Rust side; only set when we actually have one.
		error.position = *position;
	}
	PlanResultT result;
	result.result.Set(std::move(error));
	return result;
}

static PlanResultT build_duckdb_error(const duckdb::Exception &e) {
	duckdb::ErrorData error_data(e);
	string message = error_data.RawMessage();
	std::optional<string> position;
	const auto &extra = error_data.ExtraInfo();
	if (auto it = extra.find("position"); it != extra.end()) {
		position = it->second;
	}
	return make_bridge_error("duckdb_planning", message, position);
}

// Create new context for an in-memory DB
DuckPlannerContext::DuckPlannerContext(rust::Box<CatalogContext> catalog)
    : catalog(std::move(catalog)),
      db(nullptr, &this->config),
      con(db) {
        con.Query("SET disabled_optimizers='compressed_materialization,empty_result_pullup'");

	// Set catalog context on the storage extension and attach the pivot catalog as default
	auto ext = duckdb::StorageExtension::Find(
	    duckdb::DBConfig::GetConfig(*db.instance), "pivotdb");
	ext->storage_info = duckdb::make_shared_ptr<PivotStorageInfo>(&*this->catalog);
	auto attach_result = con.Query("ATTACH '' AS pv (TYPE pivotdb)");
	if (attach_result->HasError()) {
		throw std::runtime_error(attach_result->GetError());
	}
	auto use_result = con.Query("USE pv");
	if (use_result->HasError()) {
		throw std::runtime_error(use_result->GetError());
	}
}

std::unique_ptr<DuckPlannerContext> new_context(rust::Box<CatalogContext> catalog) {
	return std::unique_ptr<DuckPlannerContext>(
	    new DuckPlannerContext(std::move(catalog)));
}

// The column's source name (the binding's alias, which the binder/optimizer
// carry through column resolution), or empty when it has none. An empty string
// is serialized as null, letting the Rust side render a plan with real names
// instead of positional `#idx` references.
static string ref_name(const duckdb::Expression &expr) {
	return expr.GetAlias();
}

std::unique_ptr<ExpressionT> build_ref_expression(duckdb::BoundReferenceExpression *ref) {
	RefT data;
	data.column_idx = ref->index;
	data.return_type = static_cast<uint8_t>(ref->return_type.id());
	data.name = ref_name(*ref);
	return wrap_expr(std::move(data));
}

std::unique_ptr<ExpressionT> build_column_ref_expression(duckdb::BoundColumnRefExpression *col_ref) {
	RefT data;
	data.column_idx = col_ref->binding.column_index.GetIndex();
	data.return_type = static_cast<uint8_t>(col_ref->return_type.id());
	data.name = ref_name(*col_ref);
	return wrap_expr(std::move(data));
}

std::unique_ptr<ExpressionT> build_comparison_expression(duckdb::BoundComparisonExpression *compare) {
	CompareT data;
	data.left = build_expression(compare->left.get());
	data.right = build_expression(compare->right.get());
	data.compare_type = static_cast<uint8_t>(compare->type);
	data.return_type = static_cast<uint8_t>(compare->return_type.id());
	return wrap_expr(std::move(data));
}

std::unique_ptr<ExpressionT> build_between_expression(duckdb::BoundBetweenExpression *between) {
	BetweenT data;
	data.input = build_expression(between->input.get());
	data.lower = build_expression(between->lower.get());
	data.upper = build_expression(between->upper.get());
	data.lower_inclusive = between->lower_inclusive;
	data.upper_inclusive = between->upper_inclusive;
	return wrap_expr(std::move(data));
}

// Encode a DuckDB constant as a ScalarValueT: its logical-type id plus the value
// stringified exactly as DuckDB renders it. Shared by constant expressions and
// table-function arguments.
static ScalarValueT build_scalar_value(const duckdb::Value &value) {
	ScalarValueT data;
	data.logical_type = static_cast<uint8_t>(value.type().id());
	data.raw_value = value.ToString();
	return data;
}

std::unique_ptr<ExpressionT> build_value_constant_expression(duckdb::BoundConstantExpression *constant) {
	return wrap_expr(build_scalar_value(constant->value));
}

std::unique_ptr<ExpressionT> build_aggregate_expression(duckdb::BoundAggregateExpression *aggregate) {
	AggregateFuncT data;
	data.aggregate_function = aggregate->function.name;
	for (auto &param : aggregate->children) {
		data.params.push_back(build_expression(param.get()));
	}
	data.distinct = aggregate->IsDistinct();
	data.return_type = static_cast<uint8_t>(aggregate->return_type.id());
	return wrap_expr(std::move(data));
}

std::unique_ptr<ExpressionT> build_function_expression(duckdb::BoundFunctionExpression *function) {
	FunctionT data;
	data.function = function->function.name;
	for (auto &param : function->children) {
		data.params.push_back(build_expression(param.get()));
	}
	data.return_type = static_cast<uint8_t>(function->return_type.id());
	return wrap_expr(std::move(data));
}

// `a AND b AND ...` / `a OR b OR ...`. The optimizer rewrites a small
// `x IN (a, b)` into `x = a OR x = b`, so this is the usual shape a pushed-down
// IN membership test reaches us in. `conjunction_type` preserves AND vs OR.
std::unique_ptr<ExpressionT> build_conjunction_expression(duckdb::BoundConjunctionExpression *conj) {
	ConjunctionT data;
	data.conjunction_type = static_cast<uint8_t>(conj->type);
	for (auto &child : conj->children) {
		data.children.push_back(build_expression(child.get()));
	}
	return wrap_expr(std::move(data));
}

// `x IN (a, b, ...)` is a BoundOperatorExpression whose first child is the
// tested expression and whose remaining children are the list values.
std::unique_ptr<ExpressionT> build_in_expression(duckdb::BoundOperatorExpression *op) {
	InListT data;
	data.input = build_expression(op->children[0].get());
	for (size_t i = 1; i < op->children.size(); i++) {
		data.values.push_back(build_expression(op->children[i].get()));
	}
	return wrap_expr(std::move(data));
}

// `CASE WHEN c0 THEN r0 WHEN c1 THEN r1 ... ELSE e END`. Each `(when, then)`
// pair becomes a check; a CASE without an explicit ELSE has a NULL else_expr.
std::unique_ptr<ExpressionT> build_case_expression(duckdb::BoundCaseExpression *case_expr) {
	CaseT data;
	for (auto &check : case_expr->case_checks) {
		auto entry = std::make_unique<CaseCheckT>();
		entry->when_expr = build_expression(check.when_expr.get());
		entry->then_expr = build_expression(check.then_expr.get());
		data.checks.push_back(std::move(entry));
	}
	data.else_expr = build_expression(case_expr->else_expr.get());
	return wrap_expr(std::move(data));
}

// `NOT expr` — a BoundOperatorExpression with a single child.
std::unique_ptr<ExpressionT> build_not_expression(duckdb::BoundOperatorExpression *op) {
	NotT data;
	data.input = build_expression(op->children[0].get());
	return wrap_expr(std::move(data));
}

std::unique_ptr<ExpressionT> build_expression(duckdb::Expression *expr) {
	switch (expr->type) {
	case duckdb::ExpressionType::BOUND_REF:
		return build_ref_expression(&expr->Cast<duckdb::BoundReferenceExpression>());
	case duckdb::ExpressionType::BOUND_COLUMN_REF:
		// A column ref carries the same shape as a positional ref; both deserialize
		// into a Rust `Expression::Ref`.
		return build_column_ref_expression(&expr->Cast<duckdb::BoundColumnRefExpression>());
	case duckdb::ExpressionType::COMPARE_EQUAL:
	case duckdb::ExpressionType::COMPARE_NOTEQUAL:
	case duckdb::ExpressionType::COMPARE_LESSTHAN:
	case duckdb::ExpressionType::COMPARE_GREATERTHAN:
	case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return build_comparison_expression(&expr->Cast<duckdb::BoundComparisonExpression>());
	case duckdb::ExpressionType::COMPARE_BETWEEN:
		return build_between_expression(&expr->Cast<duckdb::BoundBetweenExpression>());
	case duckdb::ExpressionType::VALUE_CONSTANT:
		return build_value_constant_expression(&expr->Cast<duckdb::BoundConstantExpression>());
	case duckdb::ExpressionType::BOUND_AGGREGATE:
		return build_aggregate_expression(&expr->Cast<duckdb::BoundAggregateExpression>());
	case duckdb::ExpressionType::BOUND_FUNCTION:
		return build_function_expression(&expr->Cast<duckdb::BoundFunctionExpression>());
	case duckdb::ExpressionType::COMPARE_IN:
		return build_in_expression(&expr->Cast<duckdb::BoundOperatorExpression>());
	case duckdb::ExpressionType::CONJUNCTION_AND:
	case duckdb::ExpressionType::CONJUNCTION_OR:
		return build_conjunction_expression(&expr->Cast<duckdb::BoundConjunctionExpression>());
	case duckdb::ExpressionType::CASE_EXPR:
		return build_case_expression(&expr->Cast<duckdb::BoundCaseExpression>());
	case duckdb::ExpressionType::OPERATOR_NOT:
		return build_not_expression(&expr->Cast<duckdb::BoundOperatorExpression>());
	case duckdb::ExpressionType::OPERATOR_CAST:
		// Unwrap casts: pivot's executor aggregates the underlying column
		// directly (e.g. AVG casts its integer input to DOUBLE in DuckDB,
		// but pivot sums the raw integer values). Serialize the cast's child
		// in place of the cast itself.
		return build_expression(expr->Cast<duckdb::BoundCastExpression>().child.get());
	default:
		throw UnsupportedPlanError("Unsupported expression: " + expr->ToString() + " of type " +
		                           std::to_string(static_cast<int>(expr->type)));
	}
}

ProjectionT build_projection(duckdb::LogicalProjection *projection) {
	ProjectionT data;
	for (auto &expr : projection->expressions) {
		data.projections.push_back(build_expression(expr.get()));
	}
	return data;
}

// Find a DynamicFilter inside a TableFilter, peeling off a single
// OPTIONAL_FILTER wrapper if present. Returns nullptr when there's no
// DynamicFilter at the top (or one level inside an OptionalFilter). More exotic
// shapes — e.g. the `OptionalFilter(ConjunctionOr(IsNull, DynamicFilter))` the
// TopN pass produces when nulls sort first — are intentionally left
// unrecognised; they're dropped by the skip-on-OPTIONAL/DYNAMIC path in
// emit_table_filter rather than mistranslated.
static duckdb::DynamicFilter *extract_dynamic_filter(duckdb::TableFilter &filter) {
	if (filter.filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
		return &filter.Cast<duckdb::DynamicFilter>();
	}
	if (filter.filter_type == duckdb::TableFilterType::OPTIONAL_FILTER) {
		auto &opt = filter.Cast<duckdb::OptionalFilter>();
		if (opt.child_filter && opt.child_filter->filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
			return &opt.child_filter->Cast<duckdb::DynamicFilter>();
		}
	}
	return nullptr;
}

// Result of walking a Get's table_filters: consumer-side dynamic-filter entries
// to attach to the Input, plus synthetic LogicalFilter conditions to sit above
// it.
struct GetTableFilters {
	std::vector<std::unique_ptr<DynamicFilterT>> dynamic_filters;
	std::vector<std::unique_ptr<ExpressionT>> conditions;
};

// Build a reference to a shared dynamic-filter cell: assign (or reuse) a stable
// slot id for `filter_data` via `df_dedup`, recording the comparison and the
// given column index. Used by both the consumer side (a Get's table_filters) and
// the producer side (a Top-N), which must agree on the slot id per cell.
static std::unique_ptr<DynamicFilterT> make_dynamic_filter(duckdb::DynamicFilterData *filter_data,
                                                           uint64_t column_idx,
                                                           DynamicFilterDedup &df_dedup) {
	auto [it, _] = df_dedup.try_emplace(filter_data, df_dedup.size());
	auto entry = std::make_unique<DynamicFilterT>();
	entry->slot_id = it->second;
	entry->column_idx = column_idx;
	entry->compare_type = static_cast<uint8_t>(filter_data->filter->comparison_type);
	return entry;
}

// Recursively split a single column's TableFilter into:
//   * dynamic-filter consumer entries appended to `dynamic_filters`
//     (for `OPTIONAL(DYNAMIC)` / bare `DYNAMIC`), and
//   * synthetic LogicalFilter conditions appended to `conditions`
//     (for static children like `ConstantFilter`, via `ToExpression`).
//
// CONJUNCTION_AND is unwrapped and its children processed independently at the
// same column — exactly the case where a column carries both a static predicate
// (`col <> ''`) and a TopN-installed dynamic filter in one table_filters entry.
// Anything else (CONJUNCTION_OR, IS_NULL, IN, …) flows through `ToExpression`
// and may then be rejected by `build_expression` if Rust doesn't support that
// shape yet — the same conservative behaviour as for any unrecognised filter.
static void emit_table_filter(duckdb::TableFilter &filter, duckdb::idx_t proj_idx,
                              const duckdb::LogicalGet &get, DynamicFilterDedup &df_dedup,
                              GetTableFilters &out) {
	if (filter.filter_type == duckdb::TableFilterType::CONJUNCTION_AND) {
		auto &conj = filter.Cast<duckdb::ConjunctionAndFilter>();
		for (auto &child : conj.child_filters) {
			emit_table_filter(*child, proj_idx, get, df_dedup, out);
		}
		return;
	}

	if (auto *df = extract_dynamic_filter(filter)) {
		auto storage_idx = get.GetColumnIds()[proj_idx].GetPrimaryIndex();
		out.dynamic_filters.push_back(make_dynamic_filter(df->filter_data.get(), storage_idx, df_dedup));
		return;
	}

	// A standalone OPTIONAL/DYNAMIC we couldn't peel above (e.g. the nulls-first
	// `OptionalFilter(ConjunctionOr(IsNull, DynamicFilter))`): drop it rather
	// than ship a placeholder constant to Rust. Dropping a dynamic filter only
	// forgoes an optimization — never changes results.
	if (filter.filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
	    filter.filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
		return;
	}

	duckdb::ColumnBinding binding(get.table_index, duckdb::ProjectionIndex(proj_idx));
	auto col_ref = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(get.types[proj_idx], binding);
	// Tag the synthetic ref with the table's column name so a pushed-down
	// `col op const` filter renders as `col = 5`, not `#idx = 5`, in a plan dump.
	auto storage_idx = get.GetColumnIds()[proj_idx].GetPrimaryIndex();
	if (storage_idx < get.names.size()) {
		col_ref->SetAlias(get.names[storage_idx]);
	}
	auto expr = filter.ToExpression(*col_ref);
	out.conditions.push_back(build_expression(expr.get()));
}

static GetTableFilters split_table_filters(duckdb::LogicalGet &get, DynamicFilterDedup &df_dedup) {
	GetTableFilters out;
	for (auto &entry : get.table_filters) {
		emit_table_filter(entry.Filter(), entry.GetIndex().GetIndex(), get, df_dedup, out);
	}
	return out;
}

// A LogicalGet's projected output columns, each as a positional BOUND_REF.
// Shared by base-table (build_get) and table-function (build_table_function_get)
// gets.
//
// A LogicalGet reads `column_ids` off disk but may *output* only a subset /
// reordering of them, given by `projection_ids` (indices into column_ids).
// This happens when a filter is pushed all the way into the scan: columns
// referenced only by that pushed-down predicate are read but not emitted.
// When `projection_ids` is set, the scan's output order, and therefore the
// positional indices ColumnBindingResolver assigns to every ref above the
// scan, follow projection_ids, NOT column_ids, so we serialize in that
// order. This is the scan-level twin of the LogicalFilter `projection_map`
// handling in build_plan_node.
std::vector<std::unique_ptr<ExpressionT>> build_get_output_columns(duckdb::LogicalGet *get) {
	std::vector<std::unique_ptr<ExpressionT>> columns;
	auto &column_ids = get->GetColumnIds();
	auto emit = [&](size_t column_position, size_t type_position) {
		RefT data;
		data.column_idx = column_ids[column_position].GetPrimaryIndex();
		data.return_type = static_cast<uint8_t>(get->types[type_position].id());
		columns.push_back(wrap_expr(std::move(data)));
	};
	if (!get->projection_ids.empty()) {
		for (auto projection_id : get->projection_ids) {
			emit(projection_id, projection_id);
		}
	} else {
		for (size_t i = 0; i < column_ids.size(); i++) {
			emit(i, i);
		}
	}
	return columns;
}

InputT build_get(duckdb::LogicalGet *get, rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                 std::vector<std::unique_ptr<DynamicFilterT>> dynamic_filters) {
	InputT data;
	data.columns = build_get_output_columns(get);

	if (!get->GetTable()) {
		throw UnsupportedPlanError("LogicalGet without a pivot table entry is not supported");
	}

	auto &pivot_entry = get->GetTable()->Cast<PivotTableCatalogEntry>();
	tables.push_back(std::move(pivot_entry.table));
	data.table_id = tables.size() - 1;
	data.dynamic_filters = std::move(dynamic_filters);
	// Flipped to true by build_late_materialization on a late-mat narrow scan.
	data.emit_row_group_metadata = false;
	return data;
}

// A LogicalGet over a table-valued function (e.g. `generate_series(1, 10)`, or
// one of pivot's own metadata functions) rather than a base table: it has no
// PivotTableCatalogEntry. DuckDB keeps the function name on `get->function` and
// the bound constant arguments on `get->parameters`, so we serialize those plus
// the projected output columns. The Rust side regenerates the rows from the name
// and args, then projects them to `columns`; there is no `table_id` because
// nothing is read off disk.
TableFunctionScanT build_table_function_get(duckdb::LogicalGet *get) {
	// Only positional `parameters` are serialized. None of pivot's table
	// functions take named parameters; reject them rather than silently drop a
	// bound argument (which would compile the function with that argument missing).
	if (!get->named_parameters.empty()) {
		throw UnsupportedPlanError("table function " + get->function.name +
		                           " with named parameters is not supported");
	}

	TableFunctionScanT data;
	data.function_name = get->function.name;
	for (auto &param : get->parameters) {
		data.args.push_back(std::make_unique<ScalarValueT>(build_scalar_value(param)));
	}
	data.columns = build_get_output_columns(get);
	return data;
}

OrderByT build_order_by(duckdb::LogicalOrder *order_by) {
	OrderByT data;
	for (auto &order : order_by->orders) {
		auto node = std::make_unique<OrderByNodeT>();
		node->direction = static_cast<uint8_t>(order.type);
		node->expression = build_expression(order.expression.get());
		data.order_bys.push_back(std::move(node));
	}
	return data;
}

AggregateT build_aggregate(duckdb::LogicalAggregate *aggregate) {
	AggregateT data;
	for (auto &group : aggregate->groups) {
		data.groups.push_back(build_expression(group.get()));
	}
	for (auto &expression : aggregate->expressions) {
		data.expressions.push_back(build_expression(expression.get()));
	}
	return data;
}

FilterT build_filter(duckdb::LogicalFilter *filter) {
	FilterT data;
	for (auto &expr : filter->expressions) {
		data.conditions.push_back(build_expression(expr.get()));
	}
	return data;
}

TopNT build_top_n(duckdb::LogicalTopN *top_n, DynamicFilterDedup &df_dedup) {
	TopNT data;
	for (auto &order : top_n->orders) {
		auto node = std::make_unique<OrderByNodeT>();
		node->direction = static_cast<uint8_t>(order.type);
		node->expression = build_expression(order.expression.get());
		data.order_bys.push_back(std::move(node));
	}
	data.limit = top_n->limit;
	data.offset = top_n->offset;

	// The TopN optimizer only installs a dynamic filter when `orders[0]` is a
	// BoundColumnRefExpression (topn_optimizer.cpp), but by the time the plan is
	// extracted ColumnBindingResolver has rewritten that into a
	// BoundReferenceExpression whose `index` is the column's position in the
	// TopN's child output, the value we publish. The comparison lives on the
	// pre-allocated placeholder ConstantFilter and is stable across runs.
	if (top_n->dynamic_filter) {
		auto &ref = top_n->orders[0].expression->Cast<duckdb::BoundReferenceExpression>();
		data.produces_dynamic_filter =
		    make_dynamic_filter(top_n->dynamic_filter.get(), ref.index, df_dedup);
	}
	return data;
}

// A bare LIMIT/OFFSET (no ORDER BY; an ORDER BY + LIMIT is fused into LogicalTopN
// upstream). pivot only handles constant bounds — a percentage or expression
// limit can't be a fixed row count, so reject it rather than emit a wrong plan.
// An absent LIMIT (offset-only query) serializes as a null limit, which the Rust
// side reads as "unbounded".
LimitT build_limit(duckdb::LogicalLimit *limit) {
	auto bound = [](const duckdb::BoundLimitNode &node,
	                const char *what) -> flatbuffers::Optional<uint64_t> {
		switch (node.Type()) {
		case duckdb::LimitNodeType::CONSTANT_VALUE:
			return flatbuffers::Optional<uint64_t>(node.GetConstantValue());
		case duckdb::LimitNodeType::UNSET:
			return flatbuffers::nullopt;
		default:
			throw UnsupportedPlanError(std::string("Unsupported non-constant LIMIT ") + what);
		}
	};
	LimitT data;
	data.limit = bound(limit->limit_val, "value");
	data.offset = bound(limit->offset_val, "offset");
	return data;
}

string create_table_option_to_string(duckdb::ParsedExpression &expr) {
	if (expr.GetExpressionClass() == duckdb::ExpressionClass::CONSTANT) {
		auto &value = expr.Cast<duckdb::ConstantExpression>().value;
		if (value.IsNull()) {
			return "true";
		}
		return value.ToString();
	}
	return expr.ToString();
}

CreateTableT build_create_table(duckdb::LogicalCreateTable *create_table) {
	CreateTableT data;
	auto &info = create_table->info->Base();

	for (auto &column : info.columns.Logical()) {
		auto col = std::make_unique<CreateTableColumnT>();
		col->name = column.GetName();
		col->col_type = static_cast<uint8_t>(column.Type().id());
		data.columns.push_back(std::move(col));
	}
	for (auto &entry : info.options) {
		auto kv = std::make_unique<KeyValueT>();
		kv->key = entry.first;
		kv->value = create_table_option_to_string(*entry.second);
		data.options.push_back(std::move(kv));
	}

	data.name = info.table;
	data.if_not_exists = info.on_conflict == duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
	data.or_replace = info.on_conflict == duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
	data.temporary = info.temporary;
	data.has_query = create_table->info->query != nullptr;
	data.constraint_count = create_table->info->constraints.size();
	return data;
}

// `SET <name> = <value>`. DuckDB's binder builds a LogicalSet for *any* name —
// it doesn't validate the setting exists until execution, which pivot never runs
// — so the name comes through verbatim and pivot decides what (if anything) it
// means. The value is a bound constant; pivot only ever wants its string form
// (a boolean reads back as "true"/"false"), so serialize just that, not the
// `{logical_type, raw_value}` pair every other scalar carries.
SetVariableT build_set(duckdb::LogicalSet *set) {
	SetVariableT data;
	data.name = set->name;
	// `has_value` distinguishes a real SET (even `SET x = ''`, an empty string)
	// from a RESET, which an empty/null value alone could not.
	data.has_value = true;
	data.value = set->value.ToString();
	return data;
}

// `RESET <name>` — restore the setting's default. pivot has no per-setting
// default machinery, so we model it as a SetVariable with no value (`has_value`
// false); the consumer reads "no value" as "off / default". Emitted under the
// same SetVariable variant so the Rust side needs only one operator.
SetVariableT build_reset(duckdb::LogicalReset *reset) {
	SetVariableT data;
	data.name = reset->name;
	// has_value defaults to false -> decoded as `value: None`.
	return data;
}

// Remove DuckDB's row-id column from an already-built late-mat RHS subtree.
//
// Late materialization threads a row-id column from the narrow scan up to the
// (now-removed) join. pivot can't scan a virtual row-id and doesn't need it —
// its materializer keys off row-group metadata instead. DuckDB appends the
// row-id *last* at every level (the Get's column list and each Projection that
// carries it up), so dropping it never shifts another column's position: we just
// find it at the scan and drop the lone reference to it in each projection on the
// way back up. Returns the output position that held the row-id in `node`, or
// `nullopt` if this subtree has none.
static std::optional<size_t> strip_trailing_rowid(PlanNodeT &node) {
	if (node.op.type == OperatorKind_Input) {
		auto *input = node.op.AsInput();
		auto &cols = input->columns;
		for (size_t i = 0; i < cols.size(); i++) {
			auto *ref = cols[i]->kind.AsRef();
			if (ref && ref->column_idx == duckdb::COLUMN_IDENTIFIER_ROW_ID) {
				cols.erase(cols.begin() + i);
				return i;
			}
		}
		return std::nullopt;
	}

	if (node.inputs.empty()) {
		return std::nullopt;
	}
	auto child_rowid = strip_trailing_rowid(*node.inputs[0]);

	// A projection that carried the row-id up references it positionally in its
	// child's output; drop that one entry. Other operators (Filter, Top-N, and
	// the synthetic PUSHDOWN_FILTER wrapper) pass columns through unchanged.
	if (node.op.type == OperatorKind_Projection && child_rowid) {
		auto *projection = node.op.AsProjection();
		auto &exprs = projection->projections;
		for (size_t i = 0; i < exprs.size(); i++) {
			auto *ref = exprs[i]->kind.AsRef();
			if (ref && ref->column_idx == *child_rowid) {
				exprs.erase(exprs.begin() + i);
				return i;
			}
		}
	}
	return child_rowid;
}

// Walk a late-mat RHS subtree to its scan, flag it to emit row-group metadata
// (so the materializer can fetch survivors), and return its table_id — which the
// Materialize node reuses, so resolution clones the one resolved table for both.
static int64_t prepare_narrow_scan(PlanNodeT &node) {
	if (node.op.type == OperatorKind_Input) {
		auto *input = node.op.AsInput();
		input->emit_row_group_metadata = true;
		return static_cast<int64_t>(input->table_id);
	}
	if (node.inputs.empty()) {
		return -1;
	}
	return prepare_narrow_scan(*node.inputs[0]);
}

// Collapse DuckDB's late-materialization SEMI join into a pivot Materialize node.
//
// DuckDB's plan is `... -> Projection -> SemiJoin(lhs.rowid = rhs.rowid)` where
// the LHS is a full-column Get of the table and the RHS is the original narrow
// `Top-N -> [Projection] -> [Filter] -> Get(+rowid)` pipeline. The SEMI join
// keeps the LHS rows whose row-id survived the narrow Top-N. Pivot can't run a
// join, so we emit `Materialize(<lhs columns>) -> <rhs pipeline>`: the narrow
// pipeline runs as-is (its row-id column is dropped Rust-side, replaced by
// pivot's row-group metadata), and the materializer re-reads the LHS columns for
// the surviving rows. `columns` is the LHS Get's output column set (storage
// indices, row-id excluded), in output order so the Projection kept above the
// (former) join still lines up positionally.
std::unique_ptr<PlanNodeT> build_late_materialization(duckdb::LogicalComparisonJoin &join,
                                                      rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                                                      DynamicFilterDedup &df_dedup) {
	auto &lhs_get = join.children[0]->Cast<duckdb::LogicalGet>();
	auto &col_ids = lhs_get.GetColumnIds();
	std::vector<uint64_t> mat_columns;
	auto emit_col = [&](size_t pos) {
		auto storage = col_ids[pos].GetPrimaryIndex();
		if (storage == duckdb::COLUMN_IDENTIFIER_ROW_ID) {
			return;
		}
		mat_columns.push_back(storage);
	};
	if (!lhs_get.projection_ids.empty()) {
		for (auto pid : lhs_get.projection_ids) {
			emit_col(pid);
		}
	} else {
		for (size_t i = 0; i < col_ids.size(); i++) {
			emit_col(i);
		}
	}

	// The narrow pipeline is the RHS; translate it normally, then drop the row-id
	// column DuckDB threaded through it for the join we're discarding.
	auto child = build_plan_node(join.children[1].get(), tables, df_dedup);
	strip_trailing_rowid(*child);
	// Tag the narrow scan to emit row-group metadata, and reuse its table_id for
	// the Materialize so both resolve to (a clone of) the same table.
	int64_t table_id = prepare_narrow_scan(*child);
	if (table_id < 0) {
		// No base-table scan under the narrow pipeline (should not happen given
		// is_late_materialization_join). Fail cleanly rather than ship a
		// usize::MAX table_id that panics on an out-of-bounds index in Rust.
		throw UnsupportedPlanError("late-materialization narrow pipeline has no base-table scan");
	}

	MaterializeT data;
	data.table_id = static_cast<uint64_t>(table_id);
	data.columns = std::move(mat_columns);

	auto node = std::make_unique<PlanNodeT>();
	node->name = "Materialize";
	node->inputs.push_back(std::move(child));
	node->op.Set(std::move(data));
	return node;
}

// Whether a SEMI join is the one DuckDB's late_materialization optimizer
// produces (vs a user `IN`/`EXISTS`). Late-mat keys the join on the row-id
// virtual column, so its LHS is a bare LogicalGet carrying a column tagged with
// COLUMN_IDENTIFIER_ROW_ID (added by GetOrInsertRowIds). A user semi-join's LHS
// is an arbitrary subtree with no row-id, so it falls through to the normal
// path (and is rejected as an unsupported operator, which is correct — pivot
// can't execute a general join).
static bool is_late_materialization_join(duckdb::LogicalComparisonJoin &join) {
	if (join.children.empty() ||
	    join.children[0]->type != duckdb::LogicalOperatorType::LOGICAL_GET) {
		return false;
	}
	auto &lhs_get = join.children[0]->Cast<duckdb::LogicalGet>();
	for (auto &cid : lhs_get.GetColumnIds()) {
		if (cid.GetPrimaryIndex() == duckdb::COLUMN_IDENTIFIER_ROW_ID) {
			return true;
		}
	}
	return false;
}

// Whether an already-built node is a late-mat Materialize whose narrow scan
// reads no data columns. That's the shape of a plain LIMIT (no ORDER BY/filter
// key): the narrow Get carried only the row-id, which strip_trailing_rowid
// removed, leaving an empty projection. DuckDB also synthesizes a row-id ORDER BY
// above the join in that case — see late_materialization.cpp — which pivot can't
// run (its materializer drops the row id) and doesn't need (materialized rows
// come back in scan/row-id order anyway), so the caller drops it.
static bool is_materialize_over_empty_scan(const PlanNodeT &node) {
	if (node.name != "Materialize") {
		return false;
	}
	const PlanNodeT *cur = &node;
	while (!cur->inputs.empty()) {
		cur = cur->inputs[0].get();
		if (cur->op.type == OperatorKind_Input) {
			return cur->op.AsInput()->columns.empty();
		}
	}
	return false;
}

std::unique_ptr<PlanNodeT> build_plan_node(duckdb::LogicalOperator *op,
                                           rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                                           DynamicFilterDedup &df_dedup) {
	// DuckDB's late_materialization optimizer rewrites a wide Top-N/Limit scan
	// into a row-id SEMI join. Collapse that into pivot's own Materialize node
	// rather than executing a join — but only the late-mat shape, not a user
	// IN/EXISTS semi-join (see is_late_materialization_join).
	if (op->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op->Cast<duckdb::LogicalComparisonJoin>();
		if (join.join_type == duckdb::JoinType::SEMI && is_late_materialization_join(join)) {
			return build_late_materialization(join, tables, df_dedup);
		}
	}

	auto node = std::make_unique<PlanNodeT>();
	node->name = op->GetName();
	for (auto &child : op->children) {
		node->inputs.push_back(build_plan_node(child.get(), tables, df_dedup));
	}

	// Static `col op const` filters DuckDB pushed into a LogicalGet's
	// table_filters, rewritten back into expressions; reattached below as a
	// synthetic LogicalFilter so the Rust side keeps seeing `Filter -> Input`
	// exactly as it would with filter_pushdown disabled.
	std::vector<std::unique_ptr<ExpressionT>> get_pushed_conditions;

	switch (op->type) {
	case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
		node->op.Set(build_projection(&op->Cast<duckdb::LogicalProjection>()));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		// The single-row source under a FROM-less SELECT (e.g.
		// `SELECT drop_cache()`). No payload — pivot emits one empty row.
		node->op.Set(DummyScanT());
		break;
	case duckdb::LogicalOperatorType::LOGICAL_EXPLAIN:
		// `EXPLAIN <query>` wraps the optimized plan as its single child (already
		// built into `inputs` above). No payload: pivot renders that child plan
		// as text rather than running it.
		node->op.Set(ExplainT());
		break;
	case duckdb::LogicalOperatorType::LOGICAL_GET: {
		auto &get = op->Cast<duckdb::LogicalGet>();
		// A table-valued function (generate_series, pivot's metadata functions)
		// has no base-table catalog entry. Emit a TableFunctionScan instead of a
		// base-table Input, with the function name + args rather than a `table_id`.
		if (!get.GetTable()) {
			node->op.Set(build_table_function_get(&get));
			break;
		}
		auto split = split_table_filters(get, df_dedup);
		get_pushed_conditions = std::move(split.conditions);
		node->op.Set(build_get(&get, tables, std::move(split.dynamic_filters)));
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
		// Drop the row-id ORDER BY DuckDB synthesizes above a late-materialized
		// plain LIMIT (see is_materialize_over_empty_scan): pivot can't run it and
		// doesn't need it. Returns the Materialize child directly; the ORDER BY is
		// a pass-through, so the parent's column positions are unchanged.
		if (!node->inputs.empty() && is_materialize_over_empty_scan(*node->inputs[0])) {
			return std::move(node->inputs[0]);
		}
		node->op.Set(build_order_by(&op->Cast<duckdb::LogicalOrder>()));
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		node->op.Set(build_aggregate(&op->Cast<duckdb::LogicalAggregate>()));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_FILTER:
		node->op.Set(build_filter(&op->Cast<duckdb::LogicalFilter>()));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
		node->op.Set(build_top_n(&op->Cast<duckdb::LogicalTopN>(), df_dedup));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
		node->op.Set(build_limit(&op->Cast<duckdb::LogicalLimit>()));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_CREATE_TABLE:
		node->op.Set(build_create_table(&op->Cast<duckdb::LogicalCreateTable>()));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_SET:
		node->op.Set(build_set(&op->Cast<duckdb::LogicalSet>()));
		break;
	case duckdb::LogicalOperatorType::LOGICAL_RESET:
		// Modelled as a SetVariable with no value (see build_reset).
		node->op.Set(build_reset(&op->Cast<duckdb::LogicalReset>()));
		break;
	default:
		throw UnsupportedPlanError("Unsupported operator type: " + op->GetName());
	}

	// Reattach the Get's static pushed-down filters as a LogicalFilter above it.
	// The conditions reference the Get's output columns positionally, which is
	// what a Filter parent expects.
	if (op->type == duckdb::LogicalOperatorType::LOGICAL_GET && !get_pushed_conditions.empty()) {
		FilterT filter;
		filter.conditions = std::move(get_pushed_conditions);
		auto wrapper = std::make_unique<PlanNodeT>();
		wrapper->name = "PUSHDOWN_FILTER";
		wrapper->inputs.push_back(std::move(node));
		wrapper->op.Set(std::move(filter));
		return wrapper;
	}

	// A LogicalFilter may carry a `projection_map`: it does NOT output all of its
	// child's columns, only the subset/reordering listed there (the rest exist
	// solely to evaluate the filter predicates and are dropped). DuckDB's
	// ColumnBindingResolver assigns positional indices to every ref *above* the
	// filter against that projected output — so e.g. with projection_map=[4] the
	// lone surviving column becomes index 0. pivot's filter passes every input
	// column through unchanged, so without replaying the projection those indices
	// point at the wrong columns (e.g. a grouped `date_trunc(timestamp_col)`
	// would read the wrong column and collapse every row into one bucket). Replay it by wrapping
	// the filter in a Projection that selects exactly `projection_map`, positionally,
	// from the filter's (pass-through) output.
	if (op->type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<duckdb::LogicalFilter>();
		if (!filter.projection_map.empty()) {
			ProjectionT projection;
			for (size_t i = 0; i < filter.projection_map.size(); i++) {
				RefT data;
				data.column_idx = static_cast<uint64_t>(filter.projection_map[i]);
				data.return_type = static_cast<uint8_t>(filter.types[i].id());
				projection.projections.push_back(wrap_expr(std::move(data)));
			}
			auto wrapper = std::make_unique<PlanNodeT>();
			wrapper->name = "FILTER_PROJECTION";
			wrapper->inputs.push_back(std::move(node));
			wrapper->op.Set(std::move(projection));
			return wrapper;
		}
	}

	return node;
}

// Replicates `duckdb::ClientContext::ExtractPlan`, additionally returning the
// binder-resolved result column names (in select order) via `result_names`.
// The stock `ExtractPlan` computes those names on its local `Planner` and then
// discards them when it returns only the plan; we reproduce its body here (the
// binder/optimizer/resolver steps are all public API) so the names can be
// captured without patching the bundled DuckDB.
static duckdb::unique_ptr<duckdb::LogicalOperator>
extract_plan_with_names(duckdb::Connection &con, const std::string &query,
                        duckdb::vector<std::string> &result_names) {
	auto statements = con.ExtractStatements(query);
	if (statements.size() != 1) {
		throw duckdb::InvalidInputException("ExtractPlan can only prepare a single statement");
	}
	auto &context = *con.context;
	duckdb::unique_ptr<duckdb::LogicalOperator> plan;
	context.RunFunctionInTransaction([&]() {
		duckdb::Planner planner(context);
		planner.CreatePlan(std::move(statements[0]));
		// The binder resolves the client-facing result column names before
		// optimization rewrites the plan; capture them while still intact.
		result_names = planner.names;
		plan = std::move(planner.plan);
		if (context.config.enable_optimizer) {
			duckdb::Optimizer optimizer(*planner.binder, context);
			plan = optimizer.Optimize(std::move(plan));
		}
		plan->ResolveOperatorTypes();
		duckdb::ColumnBindingResolver resolver;
		resolver.Verify(*plan);
		resolver.VisitOperator(*plan);
	});
	return plan;
}

ExtractPlanResult extract_plan(DuckPlannerContext &ctx, rust::Str query) {
	PlanResultT result;
	rust::Vec<rust::Box<OptionalTableWrapper>> tables;

	try {
		std::string query_str(query.data(), query.size());
		// The result column names DuckDB would hand a client, in select order
		// (e.g. `["hour", "count_star()"]` for `SELECT f(t) AS hour, COUNT(*)`).
		// Captured in the same planning pass as the plan itself: the binder
		// resolves them before optimization pushes expressions around and the
		// optimized plan no longer carries them intact. The Rust side stamps
		// them onto the final output schema.
		duckdb::vector<std::string> name_list;
		auto plan = extract_plan_with_names(ctx.con, query_str, name_list);
		// Rewrite DuckDB's column *bindings* (table_index, column_index) into
		// positional BoundReference indices against each operator's actual child
		// output. Without this we serialized binding.column_index directly, which
		// only lines up for simple linear chains — once the optimizer reorders a
		// scan's column_ids or drops a projection (e.g. a grouped/projected
		// non-filter column above a filter), the raw index points at the wrong
		// column. This is the standard resolution DuckDB runs before execution.
		duckdb::ColumnBindingResolver resolver;
		resolver.VisitOperator(*plan);
		DynamicFilterDedup df_dedup;
		auto root = build_plan_node(plan.get(), tables, df_dedup);

		SuccessPayloadT success;
		success.plan = std::move(root);
		for (const auto &name : name_list) {
			success.output_names.push_back(name);
		}
		result.result.Set(std::move(success));
	} catch (duckdb::Exception &e) {
		result = build_duckdb_error(e);
	} catch (const UnsupportedPlanError &e) {
		result = make_bridge_error("unsupported_plan", e.what());
	} catch (const std::exception &e) {
		result = make_bridge_error("bridge_error", e.what());
	} catch (...) {
		result = make_bridge_error("bridge_error", "unknown C++ exception");
	}

	// Free table entries created during this plan call
	PivotStorageInfo::Get(*ctx.db.instance).ClearTableEntries();

	return ExtractPlanResult{serialize_plan_result(result), std::move(tables)};
}
