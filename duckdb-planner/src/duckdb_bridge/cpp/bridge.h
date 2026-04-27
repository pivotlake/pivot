#pragma once
#include "rust/cxx.h"
#include "duckdb.hpp"
#include "duckdb/planner/table_filter_set.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include <nlohmann/json.hpp>
#include <memory>

using json = nlohmann::json;

struct CatalogContext;
struct ExtractPlanResult;

struct DuckPlannerContext {
	rust::Box<CatalogContext> catalog;  // owns the CatalogContext; must outlive db
	duckdb::DBConfig config;
	duckdb::DuckDB db;
	duckdb::Connection con;

	explicit DuckPlannerContext(rust::Box<CatalogContext> catalog);
};

std::unique_ptr<DuckPlannerContext> new_context(rust::Box<CatalogContext> catalog);
ExtractPlanResult extract_plan(DuckPlannerContext &ctx, rust::Str query);
json build_expression(duckdb::Expression *expr);