#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/bridge.h"

#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"

#include <nlohmann/json.hpp>
#include <cstdio>

using namespace duckdb;
using json = nlohmann::json;

static BindInfo PivotScanGetBindInfo(const optional_ptr<FunctionData> bind_data) {
	auto &data = bind_data->Cast<PivotScanBindData>();
	return BindInfo(data.catalog_entry);
}

// This function is used to "hook" the filters a logicalget has before stats / anything else runs.
// We use it to try and pushdown filters into the table function so stats will be accurate to the filters.
static void PivotScanPushdownComplexFilter(ClientContext &context, LogicalGet &get, FunctionData *bind_data,
                                           vector<unique_ptr<Expression>> &filters) {
	auto &data = bind_data->Cast<PivotScanBindData>();

	// Complex filters: offer each one to Rust. If the table pushes it down,
	// drop it from the vector so DuckDB doesn't re-apply it on top.
	for (auto it = filters.begin(); it != filters.end();) {
		json serialized_filter = {
		    {"type", static_cast<uint8_t>(duckdb::TableFilterType::EXPRESSION_FILTER)},
		    {"data", build_expression(it->get())},
		};
		if (pushdown_filter(data.table, serialized_filter.dump())) {
			it = filters.erase(it);
		} else {
			++it;
		}
	}
}

PivotTableCatalogEntry::PivotTableCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info,
                                               rust::Box<OptionalTableWrapper> table)
    : TableCatalogEntry(catalog, schema, info), table(std::move(table)) {
}

TableFunction PivotTableCatalogEntry::GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<PivotScanBindData>(*this, *table);
	TableFunction func(name, {}, nullptr, nullptr);
	func.get_bind_info = PivotScanGetBindInfo;
	func.pushdown_complex_filter = PivotScanPushdownComplexFilter;
	// Intentionally disabled even though we implement pushdown_complex_filter.
	//
	// With filter_pushdown=false + pushdown_complex_filter set, DuckDB still
	// routes every filter (simple or complex) through the complex hook, but
	// skips its own post-callback pass that would otherwise extract
	// `col op const` leftovers into get.table_filters. Filters the table
	// rejects stay in the complex-filter vector and DuckDB turns them into a
	// LogicalFilter above the LogicalGet on its own.
	func.filter_pushdown = false;
	func.projection_pushdown = true;
	return func;
}

unique_ptr<BaseStatistics> PivotTableCatalogEntry::GetStatistics(ClientContext &context, column_t column_id) {
	std::fprintf(stderr, "[PivotTableCatalogEntry::GetStatistics] reached, column_id=%llu\n",
	             static_cast<unsigned long long>(column_id));
	return nullptr;
}

TableStorageInfo PivotTableCatalogEntry::GetStorageInfo(ClientContext &context) {
	return TableStorageInfo();
}
