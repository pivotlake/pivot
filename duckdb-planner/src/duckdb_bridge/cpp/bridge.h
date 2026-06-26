#pragma once
#include "rust/cxx.h"
#include "duckdb.hpp"
#include "duckdb/planner/table_filter_set.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "plan_generated.h"
#include <memory>

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

// Build the FlatBuffers expression IR for a single bound expression. Shared with
// the filter-pushdown callback in table_entry.cpp.
std::unique_ptr<pivot::plan::ExpressionT> build_expression(duckdb::Expression *expr);
