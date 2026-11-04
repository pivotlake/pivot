#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

std::string PivotExtension::Name() {
	return "pivotdb";
}

// The ATTACH option naming the schema the database's unqualified names resolve
// in. The planner context attaches every datastore with it.
static constexpr const char *DEFAULT_SCHEMA_OPTION = "default_schema";

static unique_ptr<Catalog> pivot_catalog_attach(optional_ptr<StorageExtensionInfo> info,
                                                ClientContext &context, AttachedDatabase &db,
                                                const string &name, AttachInfo &attach_info,
                                                AttachOptions &options) {
	auto default_schema = options.options.find(DEFAULT_SCHEMA_OPTION);
	if (default_schema == options.options.end()) {
		throw InvalidInputException("ATTACH of pivotdb database \"%s\" names no DEFAULT_SCHEMA", name);
	}
	return make_uniq<PivotCatalog>(
	    db, StringValue::Get(default_schema->second.DefaultCastAs(LogicalType::VARCHAR)));
}

void PivotExtension::Load(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto ext = make_shared_ptr<StorageExtension>();
	ext->attach = pivot_catalog_attach;
	ext->create_transaction_manager = create_pivot_transaction_manager;
	StorageExtension::Register(DBConfig::GetConfig(db), "pivotdb", ext);
}
