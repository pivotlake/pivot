#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

#include "duckdb/main/config.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

std::string PivotExtension::Name() {
	return "pivotdb";
}

static unique_ptr<Catalog> pivot_catalog_attach(optional_ptr<StorageExtensionInfo> info,
                                                ClientContext &context, AttachedDatabase &db,
                                                const string &name, AttachInfo &attach_info,
                                                AttachOptions &options) {
	return make_uniq<PivotCatalog>(db);
}

// `has_token(text, token)` is evaluated by Pivot, on the text or from a text
// index. DuckDB only has to know it exists and what it returns, to bind it.
static void has_token_is_evaluated_by_pivot(DataChunk &, ExpressionState &, Vector &) {
	throw NotImplementedException("has_token is evaluated by Pivot, not by DuckDB");
}

void PivotExtension::Load(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("has_token", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                       LogicalType::BOOLEAN, has_token_is_evaluated_by_pivot));
	auto &db = loader.GetDatabaseInstance();
	auto ext = make_shared_ptr<StorageExtension>();
	ext->attach = pivot_catalog_attach;
	ext->create_transaction_manager = create_pivot_transaction_manager;
	StorageExtension::Register(DBConfig::GetConfig(db), "pivotdb", ext);
}
