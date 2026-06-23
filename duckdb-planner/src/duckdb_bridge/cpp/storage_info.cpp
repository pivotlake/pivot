#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"

#include "duckdb/main/config.hpp"

using namespace duckdb;

PivotStorageInfo &PivotStorageInfo::Get(DatabaseInstance &db) {
	auto ext = StorageExtension::Find(DBConfig::GetConfig(db), "pivotdb");
	return dynamic_cast<PivotStorageInfo &>(*ext->storage_info);
}

PivotTableCatalogEntry *PivotStorageInfo::AddTableEntry(unique_ptr<PivotTableCatalogEntry> entry) {
	auto *ptr = entry.get();
	table_entries.push_back(std::move(entry));
	return ptr;
}

TableFunctionCatalogEntry *PivotStorageInfo::AddFunctionEntry(
    unique_ptr<TableFunctionCatalogEntry> entry) {
	auto *ptr = entry.get();
	function_entries.push_back(std::move(entry));
	return ptr;
}

ScalarFunctionCatalogEntry *PivotStorageInfo::AddScalarFunctionEntry(
    unique_ptr<ScalarFunctionCatalogEntry> entry) {
	auto *ptr = entry.get();
	scalar_function_entries.push_back(std::move(entry));
	return ptr;
}

void PivotStorageInfo::ClearTableEntries() {
	table_entries.clear();
	function_entries.clear();
	scalar_function_entries.clear();
}
