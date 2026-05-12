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
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
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

// Stable id assignment for shared `DynamicFilterData` cells. A single cell may
// be referenced by both the producer (TopN, hash-join build side) and one or
// more consumers (LogicalGet table_filters); deduping by raw pointer identity
// lets the Rust side rebuild the shared-slot graph by index lookup.
using DynamicFilterDedup = std::unordered_map<duckdb::DynamicFilterData *, size_t>;

json build_plan_node_json(duckdb::LogicalOperator *op, rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                          DynamicFilterDedup &df_dedup);
json build_expression(duckdb::Expression *expr);

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
        con.Query("SET disabled_optimizers='compressed_materialization,late_materialization,empty_result_pullup'");

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

// Find a DynamicFilter inside a TableFilter, peeling off OPTIONAL_FILTER if
// present. Returns nullptr if there's no DynamicFilter at the top (or one
// level deep inside an OptionalFilter). More exotic shapes — e.g. the
// `OptionalFilter(ConjunctionOr(IsNull, DynamicFilter))` produced by TopN
// pushdown when nulls land first — are intentionally left unrecognised here;
// they'll fall through to the skip-on-OPTIONAL/DYNAMIC path in the
// per-column walker below.
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

// Recursive walker over the TableFilter tree for a single column. Splits
// `filter` into:
//   * dynamic-filter consumer entries appended to `dynamic_filters` (for
//     `OPTIONAL(DYNAMIC)` / bare `DYNAMIC` children), and
//   * synthetic `LogicalFilter` conditions appended to `conditions` (for
//     static children like `ConstantFilter`).
//
// `CONJUNCTION_AND` is unwrapped — its children are processed independently
// at the same column position. That's exactly the case that triggers when a
// column has BOTH a static predicate (`col <> ''`) and a TopN-installed
// dynamic filter merged into the same `table_filters` entry. Anything else
// (CONJUNCTION_OR, NULL/IS_NOT_NULL, IN, …) flows through `ToExpression`
// and may then fail in `build_expression` if Rust doesn't yet support that
// expression shape — same conservative behaviour as before.
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

	// Standalone OPTIONAL/DYNAMIC we didn't peel above (e.g.
	// `OptionalFilter(ConjunctionOr(IsNull, DynamicFilter))`): drop it
	// rather than ship a `BoundConstant(true)` placeholder to Rust.
	if (filter.filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
	    filter.filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
		return;
	}

	duckdb::ColumnBinding binding(get.table_index, duckdb::ProjectionIndex(proj_idx));
	auto col_ref = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(get.types[proj_idx], binding);
	auto expr = filter.ToExpression(*col_ref);
	conditions.push_back(build_expression(expr.get()));
}

// Result of walking a Get's table_filters: the consumer-side dynamic-filter
// entries to attach to the Input, plus any synthetic LogicalFilter conditions
// that should sit above the Input.
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
	for (size_t i = 0; i < column_ids.size(); i++) {
		auto &column_id = column_ids[i];
		json data;
		data["column_idx"] = column_id.GetPrimaryIndex();
		data["return_type"] = get->types[i].id();
		json column;
		column["type"] = static_cast<uint8_t>(duckdb::ExpressionType::BOUND_REF);
		column["data"] = data;
		columns.push_back(column);
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
    // BoundColumnRefExpression (see topn_optimizer.cpp:75-83), but by the time
    // extract_plan runs, ColumnBindingResolver has already rewritten that into
    // a BoundReferenceExpression. The `index` on that ref is the position in
    // TopN's child output — i.e. the column we're tracking.
    // `comparison_type` lives on the pre-allocated placeholder ConstantFilter
    // and is stable across runs.
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
        {"order_bys", orders},
        {"produces_dynamic_filter", produces_dynamic_filter},
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

json build_plan_node_json(duckdb::LogicalOperator *op, rust::Vec<rust::Box<OptionalTableWrapper>> &tables,
                          DynamicFilterDedup &df_dedup) {
	json new_node = json::object();
	json new_operator = json::object();

	new_node["name"] = op->GetName();

	json inputs = json::array();
	for (auto &child : op->children) {
		inputs.push_back(build_plan_node_json(child.get(), tables, df_dedup));
	}
	new_node["inputs"] = inputs;

	new_operator["type"] = static_cast<uint8_t>(op->type);

	// Add operator-specific properties
	switch (op->type) {
	case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
		new_operator["data"] = build_projection(&op->Cast<duckdb::LogicalProjection>());
		break;
	}
	case duckdb::LogicalOperatorType::LOGICAL_GET: {
		auto &get = op->Cast<duckdb::LogicalGet>();
		auto split = split_table_filters(get, df_dedup);
		new_operator["data"] = build_get(&get, tables, std::move(split.dynamic_filters));
		if (!split.conditions.empty()) {
			// Wrap the get node in a synthetic LogicalFilter carrying the
			// static pushed-down table filters. This mirrors what DuckDB
			// would emit itself if filter_pushdown were false, so the Rust
			// side keeps seeing `Filter(...) -> Input(...)`.
			json get_node = {
			    {"name", op->GetName()},
			    {"inputs", new_node["inputs"]},
			    {"operator", new_operator},
			};
			new_operator = json::object();
			new_operator["type"] = static_cast<uint8_t>(duckdb::LogicalOperatorType::LOGICAL_FILTER);
			new_operator["data"] = {{"conditions", std::move(split.conditions)}};
			new_node["inputs"] = json::array({std::move(get_node)});
		}
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
	default:
		throw UnsupportedPlanError("Unsupported operator type: " + op->GetName());
	}

	new_node["operator"] = new_operator;

	return new_node;
}

ExtractPlanResult extract_plan(DuckPlannerContext &ctx, rust::Str query) {
	json result;
	rust::Vec<rust::Box<OptionalTableWrapper>> tables;

	try {
		auto plan = ctx.con.ExtractPlan(std::string(query.data(), query.size()));
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
