// Eager aggregation below an outer join.
//
// A GROUP BY over an outer join whose group keys come from the preserved side
// and whose aggregates read only the null-extended side computes, per
// preserved row, a summary of that row's join partners. Grouping the
// null-extended side by its join keys first yields the same summaries with the
// join and the GROUP BY processing one row per key instead of one row per
// partner: count becomes a sum of per-key counts (an unmatched preserved row
// contributing the rows it would have produced), sum a sum of sums, min and
// max a min of mins and a max of maxes. Since every preserved row reaches the
// GROUP BY either way, the rewrite cannot lose rows; it is only applied where
// the grouped side is the much larger one, which is where it pays off.
#include "duckdb-planner/src/duckdb_bridge/cpp/eager_aggregation.h"

#include <algorithm>

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"

using namespace duckdb;

namespace {

// The join side every row of which reaches the operators above the join.
bool find_preserved_side(JoinType join_type, idx_t &preserved) {
	switch (join_type) {
	case JoinType::LEFT:
		preserved = 0;
		return true;
	case JoinType::RIGHT:
		preserved = 1;
		return true;
	default:
		return false;
	}
}

void collect_column_bindings(const Expression &expression, vector<ColumnBinding> &bindings) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		bindings.push_back(expression.Cast<BoundColumnRefExpression>().binding);
	}
	ExpressionIterator::EnumerateChildren(
	    expression, [&](const Expression &child) { collect_column_bindings(child, bindings); });
}

bool reads_only(const Expression &expression, const vector<ColumnBinding> &allowed) {
	vector<ColumnBinding> bindings;
	collect_column_bindings(expression, bindings);
	return std::all_of(bindings.begin(), bindings.end(), [&](const ColumnBinding &binding) {
		return std::find(allowed.begin(), allowed.end(), binding) != allowed.end();
	});
}

bool counts_rows(const BoundAggregateExpression &aggregate) {
	return aggregate.function.name == "count" || aggregate.function.name == "count_star";
}

// Whether the aggregate over a key's rows equals the aggregate over per-key
// partial results (a count's partials being combined by summing them).
bool combines_from_partials(const BoundAggregateExpression &aggregate) {
	if (aggregate.IsDistinct() || aggregate.filter || aggregate.order_bys) {
		return false;
	}
	const auto &name = aggregate.function.name;
	return counts_rows(aggregate) || name == "sum" || name == "min" || name == "max";
}

// The rows a join side is expected to produce: a scan answers through its
// table's own estimate, which accounts for the predicates pushed into it,
// rather than the join order optimizer's reading of them; the operators
// above a single scan pass that estimate up.
idx_t estimate_rows(ClientContext &context, LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.cardinality) {
			auto stats = get.function.cardinality(context, get.bind_data.get());
			if (stats && stats->has_estimated_cardinality) {
				return stats->estimated_cardinality;
			}
		}
	} else if (op.children.size() == 1) {
		return estimate_rows(context, *op.children[0]);
	}
	return op.EstimateCardinality(context);
}

unique_ptr<Expression> make_column_ref(const LogicalType &type, ColumnBinding binding) {
	return make_uniq<BoundColumnRefExpression>(type, binding);
}

unique_ptr<BoundAggregateExpression> bind_aggregate(ClientContext &context, const string &name,
                                                    unique_ptr<Expression> argument) {
	auto &entry = Catalog::GetSystemCatalog(context).GetEntry<AggregateFunctionCatalogEntry>(
	    context, DEFAULT_SCHEMA, name);
	auto function = entry.functions.GetFunctionByArguments(context, {argument->return_type});
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(argument));
	FunctionBinder function_binder(context);
	return function_binder.BindAggregateFunction(function, std::move(children));
}

// `CASE WHEN partial IS NULL THEN rows_without_partner ELSE partial END`: a
// preserved row without a join partner has no per-key count, yet the original
// join output held `rows_without_partner` rows for it.
unique_ptr<Expression> count_with_fallback(unique_ptr<Expression> partial, int64_t rows_without_partner) {
	auto is_null = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
	is_null->children.push_back(partial->Copy());
	auto fallback = make_uniq<BoundConstantExpression>(Value::BIGINT(rows_without_partner));
	return make_uniq<BoundCaseExpression>(std::move(is_null), std::move(fallback), std::move(partial));
}

// One aggregate split in two: what the grouped side computes per key and what
// the GROUP BY above the join computes from those partials. `result_type` is
// the original aggregate's type, which the combined value is cast back to.
struct SplitAggregate {
	unique_ptr<Expression> partial;
	unique_ptr<Expression> combined;
	LogicalType result_type;
};

bool split_aggregate(ClientContext &context, const BoundAggregateExpression &aggregate,
                     ColumnBinding partial_binding, SplitAggregate &split) {
	auto partial_ref = make_column_ref(aggregate.return_type, partial_binding);
	if (counts_rows(aggregate)) {
		// `count(*)` counted the null-extended row an unmatched preserved row
		// produced; `count(column)` did not, the column being null there.
		int64_t rows_without_partner = aggregate.function.name == "count_star" ? 1 : 0;
		split.combined =
		    bind_aggregate(context, "sum", count_with_fallback(std::move(partial_ref), rows_without_partner));
	} else {
		split.combined = bind_aggregate(context, aggregate.function.name, std::move(partial_ref));
		if (split.combined->return_type != aggregate.return_type) {
			return false;
		}
	}
	split.partial = aggregate.Copy();
	split.result_type = aggregate.return_type;
	return true;
}

// Rewrites the aggregate in `aggregate_op` when it fits the shape described at
// the top of this file, returning the projection that takes its place and
// listing in `replacements` the bindings its ancestors must now read. Returns
// null, with the plan untouched, otherwise.
unique_ptr<LogicalOperator> rewrite_aggregate(ClientContext &context, Binder &binder,
                                              unique_ptr<LogicalOperator> &aggregate_op,
                                              vector<ReplacementBinding> &replacements) {
	auto &aggregate = aggregate_op->Cast<LogicalAggregate>();
	if (aggregate.children.size() != 1 ||
	    aggregate.children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty() ||
	    aggregate.expressions.empty()) {
		return nullptr;
	}
	auto &join = aggregate.children[0]->Cast<LogicalComparisonJoin>();
	idx_t preserved;
	if (!find_preserved_side(join.join_type, preserved) || join.conditions.empty() ||
	    !join.duplicate_eliminated_columns.empty()) {
		return nullptr;
	}
	idx_t grouped = 1 - preserved;
	auto &preserved_child = *join.children[preserved];
	auto &grouped_child = *join.children[grouped];
	// Grouping a side that is not much larger than the preserved side cannot
	// shrink the join's input enough to pay for the extra aggregate.
	if (estimate_rows(context, grouped_child) < 2 * estimate_rows(context, preserved_child)) {
		return nullptr;
	}
	auto preserved_bindings = preserved_child.GetColumnBindings();
	auto grouped_bindings = grouped_child.GetColumnBindings();
	for (auto &group : aggregate.groups) {
		if (!reads_only(*group, preserved_bindings)) {
			return nullptr;
		}
	}
	for (auto &expression : aggregate.expressions) {
		if (expression->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return nullptr;
		}
		auto &bound_aggregate = expression->Cast<BoundAggregateExpression>();
		if (!combines_from_partials(bound_aggregate) || !reads_only(bound_aggregate, grouped_bindings)) {
			return nullptr;
		}
	}
	// The grouped side's key columns become its groups, one per distinct column.
	vector<ColumnBinding> key_bindings;
	vector<LogicalType> key_types;
	vector<idx_t> condition_group;
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
			return nullptr;
		}
		auto &grouped_key = grouped == 0 ? condition.GetLHS() : condition.GetRHS();
		auto &preserved_key = grouped == 0 ? condition.GetRHS() : condition.GetLHS();
		if (grouped_key.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
		    !reads_only(grouped_key, grouped_bindings) || !reads_only(preserved_key, preserved_bindings)) {
			return nullptr;
		}
		auto binding = grouped_key.Cast<BoundColumnRefExpression>().binding;
		auto position = idx_t(std::find(key_bindings.begin(), key_bindings.end(), binding) - key_bindings.begin());
		if (position == key_bindings.size()) {
			key_bindings.push_back(binding);
			key_types.push_back(grouped_key.return_type);
		}
		condition_group.push_back(position);
	}
	auto partial_group_index = binder.GenerateTableIndex();
	auto partial_aggregate_index = binder.GenerateTableIndex();
	vector<SplitAggregate> splits;
	for (idx_t k = 0; k < aggregate.expressions.size(); k++) {
		SplitAggregate split;
		if (!split_aggregate(context, aggregate.expressions[k]->Cast<BoundAggregateExpression>(),
		                     ColumnBinding(partial_aggregate_index, ProjectionIndex(k)), split)) {
			return nullptr;
		}
		splits.push_back(std::move(split));
	}

	// The shape fits; from here on the plan is rewritten. First the per-key
	// aggregate over the grouped side ...
	vector<unique_ptr<Expression>> partials;
	for (auto &split : splits) {
		partials.push_back(std::move(split.partial));
	}
	auto per_key = make_uniq<LogicalAggregate>(partial_group_index, partial_aggregate_index, std::move(partials));
	GroupingSet all_groups;
	for (idx_t i = 0; i < key_bindings.size(); i++) {
		per_key->groups.push_back(make_column_ref(key_types[i], key_bindings[i]));
		all_groups.insert(ProjectionIndex(i));
	}
	per_key->grouping_sets.push_back(std::move(all_groups));
	per_key->children.push_back(std::move(join.children[grouped]));
	join.children[grouped] = std::move(per_key);
	// ... then the join, keyed on the groups and passing only the partials up ...
	for (idx_t i = 0; i < join.conditions.size(); i++) {
		auto &key = grouped == 0 ? join.conditions[i].LeftReference() : join.conditions[i].RightReference();
		key = make_column_ref(key->return_type,
		                      ColumnBinding(partial_group_index, ProjectionIndex(condition_group[i])));
	}
	auto &grouped_output = grouped == 0 ? join.left_projection_map : join.right_projection_map;
	grouped_output.clear();
	for (idx_t k = 0; k < splits.size(); k++) {
		grouped_output.push_back(ProjectionIndex(key_bindings.size() + k));
	}
	// ... then the GROUP BY above, combining the partials ...
	for (idx_t k = 0; k < splits.size(); k++) {
		aggregate.expressions[k] = std::move(splits[k].combined);
	}
	// ... and a projection reading its results back as the original types, in
	// place of the aggregate's own bindings.
	auto projection_index = binder.GenerateTableIndex();
	vector<unique_ptr<Expression>> outputs;
	for (idx_t i = 0; i < aggregate.groups.size(); i++) {
		ColumnBinding old_binding(aggregate.group_index, ProjectionIndex(i));
		outputs.push_back(make_column_ref(aggregate.groups[i]->return_type, old_binding));
		replacements.emplace_back(old_binding, ColumnBinding(projection_index, ProjectionIndex(i)));
	}
	for (idx_t k = 0; k < splits.size(); k++) {
		ColumnBinding old_binding(aggregate.aggregate_index, ProjectionIndex(k));
		auto combined = make_column_ref(aggregate.expressions[k]->return_type, old_binding);
		outputs.push_back(BoundCastExpression::AddCastToType(context, std::move(combined), splits[k].result_type));
		replacements.emplace_back(old_binding,
		                          ColumnBinding(projection_index, ProjectionIndex(aggregate.groups.size() + k)));
	}
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(outputs));
	projection->children.push_back(std::move(aggregate_op));
	return projection;
}

void rewrite_tree(ClientContext &context, Binder &binder, unique_ptr<LogicalOperator> &root,
                  unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		rewrite_tree(context, binder, root, child);
	}
	if (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return;
	}
	vector<ReplacementBinding> replacements;
	auto projection = rewrite_aggregate(context, binder, op, replacements);
	if (!projection) {
		return;
	}
	ColumnBindingReplacer replacer;
	replacer.replacement_bindings = std::move(replacements);
	// The projection reads the aggregate's bindings itself; only its ancestors
	// move to the projection's.
	replacer.stop_operator = projection.get();
	op = std::move(projection);
	replacer.VisitOperator(*root);
}

} // namespace

void push_aggregates_below_outer_joins(ClientContext &context, Binder &binder, unique_ptr<LogicalOperator> &plan) {
	rewrite_tree(context, binder, plan, plan);
}
