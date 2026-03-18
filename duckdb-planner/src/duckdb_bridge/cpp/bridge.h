#pragma once
#include "rust/cxx.h"
#include "duckdb.hpp"
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
