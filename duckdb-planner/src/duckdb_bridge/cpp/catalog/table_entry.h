#pragma once
#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/table_function.hpp"
#include "rust/cxx.h"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

struct PivotScanBindData : public duckdb::TableFunctionData {
	explicit PivotScanBindData(duckdb::TableCatalogEntry &catalog_entry, OptionalTableWrapper &table)
	    : catalog_entry(catalog_entry), table(table) {}
	duckdb::TableCatalogEntry &catalog_entry;
	OptionalTableWrapper &table;
};

class PivotTableCatalogEntry : public duckdb::TableCatalogEntry {
public:
	/// Opaque handle to the Rust `GetDuckDBTypedColumns` object that backs this catalog entry.
	rust::Box<OptionalTableWrapper> table;

	PivotTableCatalogEntry(duckdb::Catalog &catalog, duckdb::SchemaCatalogEntry &schema,
	                duckdb::CreateTableInfo &info, rust::Box<OptionalTableWrapper> table);

	duckdb::TableFunction GetScanFunction(duckdb::ClientContext &context,
	                                      duckdb::unique_ptr<duckdb::FunctionData> &bind_data) override;
	duckdb::unique_ptr<duckdb::BaseStatistics> GetStatistics(duckdb::ClientContext &context,
	                                                         duckdb::column_t column_id) override;
	duckdb::TableStorageInfo GetStorageInfo(duckdb::ClientContext &context) override;
};
