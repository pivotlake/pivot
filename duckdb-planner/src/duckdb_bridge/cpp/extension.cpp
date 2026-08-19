#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

#include "duckdb/main/config.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

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

// `SELECT * FROM 's3://bucket/events/*.parquet'`: a name no table resolves is
// read as the location it looks like, by rewriting the reference into the
// `read_parquet` call the user would otherwise have written. Reached only after
// the catalog has failed to find a table of that name, so a real table always
// wins over a file that happens to share its name.
static unique_ptr<TableRef> pivot_parquet_replacement_scan(ClientContext &, ReplacementScanInput &input,
                                                           optional_ptr<ReplacementScanData>) {
	auto location = ReplacementScan::GetFullPath(input);
	if (!ReplacementScan::CanReplace(location, {"parquet"})) {
		return nullptr;
	}
	auto table_function = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_uniq<ConstantExpression>(Value(location)));
	table_function->function = make_uniq<FunctionExpression>("read_parquet", std::move(children));
	table_function->alias = location;
	return std::move(table_function);
}

void PivotExtension::Load(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto ext = make_shared_ptr<StorageExtension>();
	ext->attach = pivot_catalog_attach;
	ext->create_transaction_manager = create_pivot_transaction_manager;
	auto &config = DBConfig::GetConfig(db);
	StorageExtension::Register(config, "pivotdb", ext);
	config.replacement_scans.emplace_back(pivot_parquet_replacement_scan);
}
