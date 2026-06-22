#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

#include "duckdb/main/config.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

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

	// Register `drop_cache()` so the binder accepts `SELECT drop_cache()`. The
	// body never runs — pivot only *optimizes* through DuckDB, then re-plans:
	// the call comes through as an ordinary scalar-function expression over a
	// DummyScan, which pivot's planner maps to its own `drop_cache` expression.
	// Marked VOLATILE so the optimizer can't constant-fold the no-arg call away.
	ScalarFunction drop_cache("drop_cache", {}, LogicalType::BIGINT,
	                          [](DataChunk &, ExpressionState &, Vector &result) {
		                          result.Reference(Value::BIGINT(0));
	                          });
	drop_cache.SetStability(FunctionStability::VOLATILE);
	loader.RegisterFunction(drop_cache);

	// Table functions (e.g. `metadata`) are NOT registered here. They resolve on
	// demand through the catalog's LookupEntry, which builds their entry from the
	// Rust provider's registry, so adding one needs no change in this bridge.
}
