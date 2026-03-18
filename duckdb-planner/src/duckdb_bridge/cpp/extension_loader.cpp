#include "duckdb/main/extension_helper.hpp"
#include "core_functions_extension.hpp"
#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"

namespace duckdb {

void ExtensionHelper::LoadAllExtensions(DuckDB &db) {
	db.LoadStaticExtension<CoreFunctionsExtension>();
	db.LoadStaticExtension<PivotExtension>();
}

vector<string> ExtensionHelper::LoadedExtensionTestPaths() {
	return {};
}

} // namespace duckdb
