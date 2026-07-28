#pragma once
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"

#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "rust/cxx.h"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

struct PivotStorageInfo : public duckdb::StorageExtensionInfo {
	const CatalogContext *catalog_ctx;
	// The pivot transaction of the plan currently being extracted. Set by
	// `extract_plan` for the duration of one call (planning is single-threaded
	// per context) and stamped onto each `PivotTransaction` the transaction
	// manager starts, so table binding resolves against that plan's snapshot.
	const TransactionContext *current_transaction = nullptr;
	// The table whose generated columns are declared to the binder as GENERATED
	// for the plan currently being extracted, or empty for none. Only a write to
	// a table needs that declaration (it is what excludes generated columns from
	// the inserted column list and rejects an explicit insert into one); every
	// other binding sees ordinary columns and so reads the stored values.
	std::string generated_columns_table;
	std::vector<duckdb::unique_ptr<PivotTableCatalogEntry>> table_entries;
	// Table- and scalar-function catalog entries synthesized on lookup; kept
	// alive for the duration of one plan alongside the table entries.
	std::vector<duckdb::unique_ptr<duckdb::TableFunctionCatalogEntry>> function_entries;
	std::vector<duckdb::unique_ptr<duckdb::ScalarFunctionCatalogEntry>> scalar_function_entries;

	explicit PivotStorageInfo(const CatalogContext *ctx) : catalog_ctx(ctx) {}

	PivotTableCatalogEntry *AddTableEntry(duckdb::unique_ptr<PivotTableCatalogEntry> entry);
	duckdb::TableFunctionCatalogEntry *AddFunctionEntry(
	    duckdb::unique_ptr<duckdb::TableFunctionCatalogEntry> entry);
	duckdb::ScalarFunctionCatalogEntry *AddScalarFunctionEntry(
	    duckdb::unique_ptr<duckdb::ScalarFunctionCatalogEntry> entry);
	void ClearTableEntries();

	static PivotStorageInfo &Get(duckdb::DatabaseInstance &db);
};
