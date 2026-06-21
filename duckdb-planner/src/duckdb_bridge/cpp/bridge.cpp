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
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>

using json = nlohmann::json;
using std::string;

struct UnsupportedPlanError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

// Stable id assignment for shared `DynamicFilterData` cells. One cell may be
// referenced by both a producer (TopN; later, hash-join build side) and one or
// more consumers (a LogicalGet's table_filters); deduping by raw pointer
// identity lets the Rust side rebuild the shared-slot graph by index lookup.
using DynamicFilterDedup = std::unordered_map<duckdb::DynamicFilterData *, size_t>;

json build_plan_node_json(duckdb::LogicalOperator *op, rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                          DynamicFilterDedup &df_dedup);
json build_expression(duckdb::Expression *expr);
json build_late_materialization(duckdb::LogicalComparisonJoin &join,
                                rust::Vec<rust::Box<OptionalTableWrapper>> &tables, DynamicFilterDedup &df_dedup);

json build_bridge_error(const string &kind, const string &message, std::optional<string> position = std::nullopt) {
	json error = {
	    {"kind", kind},
	    {"exception_message", message},
	    {"position", nullptr},
	};
	if (position.has_value()) {
		error["position"] = *position;
	}
	return {{"type", "error"}, {"data", std::move(error)}};
}

json build_duckdb_error(const duckdb::Exception &e) {
	auto parsed = json::parse(e.what(), nullptr, false);
	string message = e.what();
	std::optional<string> position;

	if (parsed.is_object()) {
		if (auto message_it = parsed.find("exception_message");
		    message_it != parsed.end() && message_it->is_string()) {
			message = message_it->get<string>();
		}
		if (auto position_it = parsed.find("position"); position_it != parsed.end()) {
			if (position_it->is_string()) {
				position = position_it->get<string>();
			} else if (position_it->is_number_integer()) {
				position = std::to_string(position_it->get<int64_t>());
			} else if (position_it->is_number_unsigned()) {
				position = std::to_string(position_it->get<uint64_t>());
			} else if (!position_it->is_null()) {
				position = position_it->dump();
			}
		}
	}

	return build_bridge_error("duckdb_planning", message, position);
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

json build_ref_expression(duckdb::BoundReferenceExpression *ref) {
	return {
	    {"column_idx", ref->index},
	    {"return_type", ref->return_type.id()},
	};
}

json build_column_ref_expression(duckdb::BoundColumnRefExpression *col_ref) {
	return {
	    {"column_idx", col_ref->binding.column_index.GetIndex()},
	    {"return_type", col_ref->return_type.id()},
	};
}

json build_comparison_expression(duckdb::BoundComparisonExpression *compare) {
	return {
		{"left", build_expression(compare->left.get())},
		{"right", build_expression(compare->right.get())},
		{"compare_type", static_cast<uint8_t>(compare->type)},
		{"return_type", compare->return_type.id()}
	};
}

json build_between_expression(duckdb::BoundBetweenExpression *between) {
	return {
		{"input", build_expression(between->input.get())},
		{"lower", build_expression(between->lower.get())},
		{"upper", build_expression(between->upper.get())},
		{"lower_inclusive", between->lower_inclusive},
		{"upper_inclusive", between->upper_inclusive}
	};
}

json build_value_constant_expression(duckdb::BoundConstantExpression *constant) {
	return {
		{"logical_type", constant->value.type().id()},
		{"raw_value", constant->value.ToString()}
	};
}

json build_aggregate_expression(duckdb::BoundAggregateExpression *aggregate) {
    json params = json::array();

    for (auto &param : aggregate->children) {
        params.push_back(build_expression(param.get()));
    }

    return {
        {"aggregate_function", aggregate->function.name},
        {"params", params},
        {"distinct", aggregate->IsDistinct()},
        {"return_type", aggregate->return_type.id()}
    };
}

json build_function_expression(duckdb::BoundFunctionExpression *function) {
    json params = json::array();
    for (auto &param : function->children) {
        params.push_back(build_expression(param.get()));
    }

    return {
        {"function", function->function.name},
        {"params", params},
        {"return_type", function->return_type.id()}
    };
}

// `a AND b AND ...` / `a OR b OR ...`. The optimizer rewrites a small
// `x IN (a, b)` into `x = a OR x = b`, so this is the usual shape a pushed-down
// IN membership test reaches us in. `conjunction_type` preserves AND vs OR.
json build_conjunction_expression(duckdb::BoundConjunctionExpression *conj) {
	json children = json::array();
	for (auto &child : conj->children) {
		children.push_back(build_expression(child.get()));
	}
	return {
		{"conjunction_type", static_cast<uint8_t>(conj->type)},
		{"children", std::move(children)},
	};
}

// `x IN (a, b, ...)` is a BoundOperatorExpression whose first child is the
// tested expression and whose remaining children are the list values.
json build_in_expression(duckdb::BoundOperatorExpression *op) {
	json values = json::array();
	for (size_t i = 1; i < op->children.size(); i++) {
		values.push_back(build_expression(op->children[i].get()));
	}
	return {
		{"input", build_expression(op->children[0].get())},
		{"values", std::move(values)},
	};
}

// `CASE WHEN c0 THEN r0 WHEN c1 THEN r1 ... ELSE e END`. Each `(when, then)`
// pair becomes a check; a CASE without an explicit ELSE has a NULL else_expr.
json build_case_expression(duckdb::BoundCaseExpression *case_expr) {
	json checks = json::array();
	for (auto &check : case_expr->case_checks) {
		checks.push_back({
			{"when", build_expression(check.when_expr.get())},
			{"then", build_expression(check.then_expr.get())},
		});
	}
	return {
		{"checks", std::move(checks)},
		{"else_expr", build_expression(case_expr->else_expr.get())},
	};
}

// `NOT expr` — a BoundOperatorExpression with a single child.
json build_not_expression(duckdb::BoundOperatorExpression *op) {
	return {
		{"input", build_expression(op->children[0].get())},
	};
}

json build_expression(duckdb::Expression *expr) {
	json new_expression;
	new_expression["type"] = expr->type;

	switch (expr->type) {
	case duckdb::ExpressionType::BOUND_REF: {
		new_expression["data"] = build_ref_expression(&expr->Cast<duckdb::BoundReferenceExpression>());
		break;
	}
	case duckdb::ExpressionType::BOUND_COLUMN_REF: {
		new_expression["type"] = duckdb::ExpressionType::BOUND_REF;
		new_expression["data"] = build_column_ref_expression(&expr->Cast<duckdb::BoundColumnRefExpression>());
		break;
	}
	case duckdb::ExpressionType::COMPARE_EQUAL:
	case duckdb::ExpressionType::COMPARE_NOTEQUAL:
	case duckdb::ExpressionType::COMPARE_LESSTHAN:
	case duckdb::ExpressionType::COMPARE_GREATERTHAN:
	case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
		new_expression["data"] = build_comparison_expression(&expr->Cast<duckdb::BoundComparisonExpression>());
		break;
	}
	case duckdb::ExpressionType::COMPARE_BETWEEN: {
		new_expression["data"] = build_between_expression(&expr->Cast<duckdb::BoundBetweenExpression>());
		break;
	}
	case duckdb::ExpressionType::VALUE_CONSTANT: {
		new_expression["data"] = build_value_constant_expression(&expr->Cast<duckdb::BoundConstantExpression>());
		break;
	}
	case duckdb::ExpressionType::BOUND_AGGREGATE: {
		new_expression["data"] = build_aggregate_expression(&expr->Cast<duckdb::BoundAggregateExpression>());
		break;
	}
	case duckdb::ExpressionType::BOUND_FUNCTION: {
		new_expression["data"] = build_function_expression(&expr->Cast<duckdb::BoundFunctionExpression>());
		break;
	}
	case duckdb::ExpressionType::COMPARE_IN: {
		new_expression["data"] = build_in_expression(&expr->Cast<duckdb::BoundOperatorExpression>());
		break;
	}
	case duckdb::ExpressionType::CONJUNCTION_AND:
	case duckdb::ExpressionType::CONJUNCTION_OR: {
		new_expression["data"] = build_conjunction_expression(&expr->Cast<duckdb::BoundConjunctionExpression>());
		break;
	}
	case duckdb::ExpressionType::CASE_EXPR: {
		new_expression["data"] = build_case_expression(&expr->Cast<duckdb::BoundCaseExpression>());
		break;
	}
	case duckdb::ExpressionType::OPERATOR_NOT: {
		new_expression["data"] = build_not_expression(&expr->Cast<duckdb::BoundOperatorExpression>());
		break;
	}
	case duckdb::ExpressionType::OPERATOR_CAST: {
		// Unwrap casts: pivot's executor aggregates the underlying column
		// directly (e.g. AVG casts its integer input to DOUBLE in DuckDB,
		// but pivot sums the raw integer values). Serialize the cast's child
		// in place of the cast itself.
		return build_expression(expr->Cast<duckdb::BoundCastExpression>().child.get());
	}
	default:
		throw UnsupportedPlanError("Unsupported expression: " + expr->ToString() + " of type " +
		                           std::to_string(static_cast<int>(expr->type)));
	}

	return new_expression;
}

json build_projection(duckdb::LogicalProjection *projection) {
	json projections = json::array();
	for (auto &expr : projection->expressions) {
		projections.push_back(build_expression(expr.get()));
	}

	return {
		{"projections", projections}
	};
}

json build_constant_comparison_filter(duckdb::ConstantFilter &filter, json column_ref) {
    return {
        {"column_ref", std::move(column_ref)},
        {"compare_type", static_cast<uint8_t>(filter.comparison_type)},
        {"constant", {
            {"logical_type", filter.constant.type().id()},
            {"raw_value", filter.constant.ToString()},
        }},
    };
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
// shape yet — the same conservative behaviour as before this pass existed.
static void emit_table_filter(duckdb::TableFilter &filter, duckdb::idx_t proj_idx,
                              const duckdb::LogicalGet &get, DynamicFilterDedup &df_dedup,
                              json &dynamic_filters, json &conditions) {
	if (filter.filter_type == duckdb::TableFilterType::CONJUNCTION_AND) {
		auto &conj = filter.Cast<duckdb::ConjunctionAndFilter>();
		for (auto &child : conj.child_filters) {
			emit_table_filter(*child, proj_idx, get, df_dedup, dynamic_filters, conditions);
		}
		return;
	}

	if (auto *df = extract_dynamic_filter(filter)) {
		auto *filter_data = df->filter_data.get();
		auto [it, _] = df_dedup.try_emplace(filter_data, df_dedup.size());
		auto storage_idx = get.GetColumnIds()[proj_idx].GetPrimaryIndex();
		dynamic_filters.push_back({
		    {"slot_id", it->second},
		    {"column_idx", storage_idx},
		    {"compare_type", static_cast<uint8_t>(filter_data->filter->comparison_type)},
		});
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
	auto expr = filter.ToExpression(*col_ref);
	conditions.push_back(build_expression(expr.get()));
}

// Result of walking a Get's table_filters: consumer-side dynamic-filter entries
// to attach to the Input, plus synthetic LogicalFilter conditions to sit above
// it.
struct GetTableFilters {
	json dynamic_filters;
	json conditions;
};

static GetTableFilters split_table_filters(duckdb::LogicalGet &get, DynamicFilterDedup &df_dedup) {
	GetTableFilters out{json::array(), json::array()};
	for (auto &entry : get.table_filters) {
		emit_table_filter(entry.Filter(), entry.GetIndex().GetIndex(), get, df_dedup,
		                  out.dynamic_filters, out.conditions);
	}
	return out;
}

json build_get(duckdb::LogicalGet *get, rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
               json dynamic_filters) {
	json columns = json::array();
	auto &column_ids = get->GetColumnIds();
    // A LogicalGet reads `column_ids` off disk but may *output* only a subset /
    // reordering of them, given by `projection_ids` (indices into column_ids).
    // This happens when a filter is pushed all the way into the scan: columns
    // referenced only by that pushed-down predicate are read but not emitted.
    // When `projection_ids` is set, the scan's output order — and therefore the
    // positional indices ColumnBindingResolver assigns to every ref above the
    // scan — follow projection_ids, NOT column_ids, so we serialize in that
    // order. This is the scan-level twin of the LogicalFilter `projection_map`
    // handling in build_plan_node_json (the latter is what actually fixed Q42,
    // where the filter stayed a separate node)
	auto emit = [&](size_t col_pos, size_t type_pos) {
		json data;
		data["column_idx"] = column_ids[col_pos].GetPrimaryIndex();
		data["return_type"] = get->types[type_pos].id();
		json column;
		column["type"] = static_cast<uint8_t>(duckdb::ExpressionType::BOUND_REF);
		column["data"] = data;
		columns.push_back(column);
	};
	if (!get->projection_ids.empty()) {
		for (auto pid : get->projection_ids) {
			emit(pid, pid);
		}
	} else {
		for (size_t i = 0; i < column_ids.size(); i++) {
			emit(i, i);
		}
	}

	if (!get->GetTable()) {
		throw UnsupportedPlanError("LogicalGet without a pivot table entry is not supported");
	}

	auto &pivot_entry = get->GetTable()->Cast<PivotTableCatalogEntry>();
	tables.push_back(std::move(pivot_entry.table));
	size_t table_id = tables.size() - 1;

	return {
	    {"table_id", table_id},
	    {"columns", columns},
	    {"dynamic_filters", std::move(dynamic_filters)},
	    // Flipped to true by build_late_materialization on a late-mat narrow scan.
	    {"emit_row_group_metadata", false},
	};
}

json build_order_by(duckdb::LogicalOrder *order_by) {
	json orders = json::array();
	for (auto &order : order_by->orders) {
		orders.push_back({
			{"direction", static_cast<uint8_t>(order.type)},
			{"expression", build_expression(order.expression.get())}
		});
	}

	return {
		{"order_bys", orders}
	};
}

json build_aggregate(duckdb::LogicalAggregate *aggregate) {
    json groups = json::array();
    for (auto &group :aggregate->groups) {
        groups.push_back(build_expression(group.get()));
    }

    json expressions = json::array();
    for (auto &expression :aggregate->expressions ) {
        expressions.push_back(build_expression(expression.get()));
    }

	return {
	    {"groups", groups},
	    {"expressions", expressions}
	};
}

json build_filter(duckdb::LogicalFilter *filter) {
	json filter_expressions = json::array();
	for (auto &expr : filter->expressions) {
		filter_expressions.push_back(build_expression(expr.get()));
	}

	return {
		{"conditions", filter_expressions}
	};
}

json build_top_n(duckdb::LogicalTopN *top_n, DynamicFilterDedup &df_dedup) {
	json orders = json::array();
    for (auto &order : top_n->orders) {
        orders.push_back({
            {"direction", static_cast<uint8_t>(order.type)},
            {"expression", build_expression(order.expression.get())}
        });
    }

    // The TopN optimizer only installs a dynamic filter when `orders[0]` is a
    // BoundColumnRefExpression (topn_optimizer.cpp), but by the time the plan is
    // extracted ColumnBindingResolver has rewritten that into a
    // BoundReferenceExpression whose `index` is the column's position in the
    // TopN's child output — the value we publish. The comparison lives on the
    // pre-allocated placeholder ConstantFilter and is stable across runs.
    json produces_dynamic_filter = nullptr;
    if (top_n->dynamic_filter) {
        auto *filter_data = top_n->dynamic_filter.get();
        auto &ref = top_n->orders[0].expression->Cast<duckdb::BoundReferenceExpression>();
        auto [it, _] = df_dedup.try_emplace(filter_data, df_dedup.size());
        produces_dynamic_filter = {
            {"slot_id", it->second},
            {"column_idx", ref.index},
            {"compare_type", static_cast<uint8_t>(filter_data->filter->comparison_type)},
        };
    }

    return {
        {"limit", top_n->limit},
        {"offset", top_n->offset},
        {"order_bys", orders},
        {"produces_dynamic_filter", produces_dynamic_filter}
	    };
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

json build_create_table(duckdb::LogicalCreateTable *create_table) {
    json columns = json::array();
    json options = json::object();
    auto &info = create_table->info->Base();

    for (auto &column : info.columns.Logical()) {
        columns.push_back({
            {"name", column.GetName()},
            {"col_type", static_cast<uint8_t>(column.Type().id())},
        });
    }
    for (auto &entry : info.options) {
        options[entry.first] = create_table_option_to_string(*entry.second);
    }

    return {
        {"name", info.table},
        {"columns", columns},
        {"options", options},
        {"if_not_exists", info.on_conflict == duckdb::OnCreateConflict::IGNORE_ON_CONFLICT},
        {"or_replace", info.on_conflict == duckdb::OnCreateConflict::REPLACE_ON_CONFLICT},
        {"temporary", info.temporary},
        {"has_query", create_table->info->query != nullptr},
        {"constraint_count", create_table->info->constraints.size()},
    };
}

// `SET <name> = <value>`. DuckDB's binder builds a LogicalSet for *any* name —
// it doesn't validate the setting exists until execution, which pivot never runs
// — so the name comes through verbatim and pivot decides what (if anything) it
// means. The value is a bound constant; pivot only ever wants its string form
// (a boolean reads back as "true"/"false"), so serialize just that, not the
// `{logical_type, raw_value}` pair every other scalar carries.
json build_set(duckdb::LogicalSet *set) {
	return {
		{"name", set->name},
		{"value", set->value.ToString()},
	};
}

// `RESET <name>` — restore the setting's default. pivot has no per-setting
// default machinery, so we model it as a SET with no value (`null`); the consumer
// reads "no value" as "off / default". Emitted under the LOGICAL_SET tag (see the
// switch) so the Rust side needs only one variant.
json build_reset(duckdb::LogicalReset *reset) {
	return {
		{"name", reset->name},
		{"value", nullptr},
	};
}

// Remove DuckDB's row-id column from an already-built late-mat RHS subtree JSON.
//
// Late materialization threads a row-id column from the narrow scan up to the
// (now-removed) join. pivot can't scan a virtual row-id and doesn't need it —
// its materializer keys off row-group metadata instead. DuckDB appends the
// row-id *last* at every level (the Get's column list and each Projection that
// carries it up), so dropping it never shifts another column's position: we just
// find it at the scan and drop the lone reference to it in each projection on the
// way back up. Returns the output position that held the row-id in `node`, or
// `nullopt` if this subtree has none.
static std::optional<size_t> strip_trailing_rowid(json &node) {
	auto type = node["operator"]["type"].get<uint8_t>();
	auto &data = node["operator"]["data"];

	if (type == static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_GET)) {
		auto &cols = data["columns"];
		for (size_t i = 0; i < cols.size(); i++) {
			if (cols[i]["data"]["column_idx"].get<uint64_t>() == duckdb::COLUMN_IDENTIFIER_ROW_ID) {
				cols.erase(cols.begin() + i);
				return i;
			}
		}
		return std::nullopt;
	}

	if (!node.contains("inputs") || node["inputs"].empty()) {
		return std::nullopt;
	}
	auto child_rowid = strip_trailing_rowid(node["inputs"][0]);

	// A projection that carried the row-id up references it positionally in its
	// child's output; drop that one entry. Other operators (Filter, Top-N, and
	// the synthetic PUSHDOWN_FILTER wrapper) pass columns through unchanged.
	if (type == static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_PROJECTION) && child_rowid) {
		auto &exprs = data["projections"];
		for (size_t i = 0; i < exprs.size(); i++) {
			auto &e = exprs[i];
			if (e["type"].get<uint8_t>() == static_cast<uint8_t>(duckdb::ExpressionType::BOUND_REF) &&
			    e["data"]["column_idx"].get<uint64_t>() == *child_rowid) {
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
static int64_t prepare_narrow_scan(json &node) {
	if (node["operator"]["type"].get<uint8_t>() ==
	    static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_GET)) {
		auto &data = node["operator"]["data"];
		data["emit_row_group_metadata"] = true;
		return data["table_id"].get<int64_t>();
	}
	if (!node.contains("inputs") || node["inputs"].empty()) {
		return -1;
	}
	return prepare_narrow_scan(node["inputs"][0]);
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
json build_late_materialization(duckdb::LogicalComparisonJoin &join,
                                rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                                DynamicFilterDedup &df_dedup) {
	auto &lhs_get = join.children[0]->Cast<duckdb::LogicalGet>();
	auto &col_ids = lhs_get.GetColumnIds();
	json mat_columns = json::array();
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
	json child = build_plan_node_json(join.children[1].get(), tables, df_dedup);
	strip_trailing_rowid(child);
	// Tag the narrow scan to emit row-group metadata, and reuse its table_id for
	// the Materialize so both resolve to (a clone of) the same table.
	int64_t table_id = prepare_narrow_scan(child);

	json op = json::object();
	op["type"] = static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
	op["data"] = json{{"table_id", table_id}, {"columns", std::move(mat_columns)}};

	json node = json::object();
	node["name"] = "Materialize";
	node["inputs"] = json::array({std::move(child)});
	node["operator"] = std::move(op);
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

json build_plan_node_json(duckdb::LogicalOperator *op, rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
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

	json new_node = json::object();
	json new_operator = json::object();

	new_node["name"] = op->GetName();

	json inputs = json::array();
	for (auto &child : op->children) {
		inputs.push_back(build_plan_node_json(child.get(), tables, df_dedup));
	}
	new_node["inputs"] = inputs;

	new_operator["type"] = static_cast<uint8_t>(op->type);

	// Static `col op const` filters DuckDB pushed into a LogicalGet's
	// table_filters, rewritten back into expressions; reattached below as a
	// synthetic LogicalFilter so the Rust side keeps seeing `Filter -> Input`
	// exactly as it would with filter_pushdown disabled.
	json get_pushed_conditions = json::array();

	// Add operator-specific properties
	switch (op->type) {
	case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
		new_operator["data"] = build_projection(&op->Cast<duckdb::LogicalProjection>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN: {
		// The single-row source under a FROM-less SELECT (e.g.
		// `SELECT drop_cache()`). No payload — pivot emits one empty row.
		new_operator["data"] = json::object();
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_GET: {
		auto &get = op->Cast<duckdb::LogicalGet>();
		auto split = split_table_filters(get, df_dedup);
		get_pushed_conditions = std::move(split.conditions);
		new_operator["data"] = build_get(&get, tables, std::move(split.dynamic_filters));
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
		new_operator["data"] = build_order_by(&op->Cast<duckdb::LogicalOrder>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		new_operator["data"] = build_aggregate(&op->Cast<duckdb::LogicalAggregate>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_FILTER: {
		new_operator["data"] = build_filter(&op->Cast<duckdb::LogicalFilter>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_TOP_N: {
		new_operator["data"] = build_top_n(&op->Cast<duckdb::LogicalTopN>(), df_dedup);
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_CREATE_TABLE: {
		new_operator["data"] = build_create_table(&op->Cast<duckdb::LogicalCreateTable>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_SET: {
		new_operator["data"] = build_set(&op->Cast<duckdb::LogicalSet>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_RESET: {
		// Relabel RESET to SET (with a null value) so Rust needs one variant.
		new_operator["type"] = static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_SET);
		new_operator["data"] = build_reset(&op->Cast<duckdb::LogicalReset>());
		break;
	}
	default:
		throw UnsupportedPlanError("Unsupported operator type: " + op->GetName());
	}

	new_node["operator"] = new_operator;

	// Reattach the Get's static pushed-down filters as a LogicalFilter above it.
	// The conditions reference the Get's output columns positionally, which is
	// what a Filter parent expects.
	if (op->type == duckdb::LogicalOperatorType::LOGICAL_GET && !get_pushed_conditions.empty()) {
		json filter_op = json::object();
		filter_op["type"] = static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_FILTER);
		filter_op["data"] = json{{"conditions", get_pushed_conditions}};
		json wrapper = json::object();
		wrapper["name"] = "PUSHDOWN_FILTER";
		wrapper["inputs"] = json::array({new_node});
		wrapper["operator"] = filter_op;
		return wrapper;
	}

	// A LogicalFilter may carry a `projection_map`: it does NOT output all of its
	// child's columns, only the subset/reordering listed there (the rest exist
	// solely to evaluate the filter predicates and are dropped). DuckDB's
	// ColumnBindingResolver assigns positional indices to every ref *above* the
	// filter against that projected output — so e.g. with projection_map=[4] the
	// lone surviving column becomes index 0. pivot's filter passes every input
	// column through unchanged, so without replaying the projection those indices
	// point at the wrong columns (this is what made grouped `date_trunc(EventTime)`
	// read CounterID and collapse every row into one bucket). Replay it by wrapping
	// the filter in a Projection that selects exactly `projection_map`, positionally,
	// from the filter's (pass-through) output.
	if (op->type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<duckdb::LogicalFilter>();
		if (!filter.projection_map.empty()) {
			json projections = json::array();
			for (size_t i = 0; i < filter.projection_map.size(); i++) {
				json data;
				data["column_idx"] = static_cast<uint64_t>(filter.projection_map[i]);
				data["return_type"] = filter.types[i].id();
				json col;
				col["type"] = static_cast<uint8_t>(duckdb::ExpressionType::BOUND_REF);
				col["data"] = data;
				projections.push_back(col);
			}
			json proj_op = json::object();
			proj_op["type"] = static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_PROJECTION);
			proj_op["data"] = json{{"projections", projections}};
			json wrapper = json::object();
			wrapper["name"] = "FILTER_PROJECTION";
			wrapper["inputs"] = json::array({new_node});
			wrapper["operator"] = proj_op;
			return wrapper;
		}
	}

	return new_node;
}

ExtractPlanResult extract_plan(DuckPlannerContext &ctx, rust::Str query) {
	json result;
	rust::Vec<rust::Box<OptionalTableWrapper>> tables;

	try {
		auto plan = ctx.con.ExtractPlan(std::string(query.data(), query.size()));
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
		auto root = build_plan_node_json(plan.get(), tables, df_dedup);
		result = {{"type", "success"}, {"data", root}};
	} catch (duckdb::Exception &e) {
		result = build_duckdb_error(e);
	} catch (const UnsupportedPlanError &e) {
		result = build_bridge_error("unsupported_plan", e.what());
	} catch (const std::exception &e) {
		result = build_bridge_error("bridge_error", e.what());
	} catch (...) {
		result = build_bridge_error("bridge_error", "unknown C++ exception");
	}

	// Free table entries created during this plan call
	PivotStorageInfo::Get(*ctx.db.instance).ClearTableEntries();

	return ExtractPlanResult{rust::String(result.dump()), std::move(tables)};
}
