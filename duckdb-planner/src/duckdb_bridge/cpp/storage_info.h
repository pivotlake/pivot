#pragma once
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"

#include "duckdb/storage/storage_extension.hpp"
#include "rust/cxx.h"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

struct PivotStorageInfo : public duckdb::StorageExtensionInfo {
	const CatalogContext *catalog_ctx;
	std::vector<duckdb::unique_ptr<PivotTableCatalogEntry>> table_entries;

	explicit PivotStorageInfo(const CatalogContext *ctx) : catalog_ctx(ctx) {}

	PivotTableCatalogEntry *AddTableEntry(duckdb::unique_ptr<PivotTableCatalogEntry> entry);
	void ClearTableEntries();

	static PivotStorageInfo &Get(duckdb::DatabaseInstance &db);
};
