#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

#include "duckdb/main/config.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

std::string PivotExtension::Name() {
	return "pivotdb";
}

static unique_ptr<Catalog> pivot_catalog_attach(optional_ptr<StorageExtensionInfo> info,
                                                ClientContext &context, AttachedDatabase &db,
                                                const string &name, AttachInfo &attach_info,
                                                AttachOptions &options) {
	auto &pivot_info = dynamic_cast<PivotStorageInfo &>(*info);
	return make_uniq<PivotCatalog>(db, pivot_info.catalog_ctx);
}

void PivotExtension::Load(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto ext = make_shared_ptr<StorageExtension>();
	ext->attach = pivot_catalog_attach;
	ext->create_transaction_manager = create_pivot_transaction_manager;
	StorageExtension::Register(DBConfig::GetConfig(db), "pivotdb", ext);

	// DuckDB provides VARIANT casts and field access. Pivot functions are loaded
	// on demand from the Rust registry through LookupEntry.
}
