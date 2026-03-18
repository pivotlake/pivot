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

void PivotStorageInfo::ClearTableEntries() {
	table_entries.clear();
}
