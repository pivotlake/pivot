#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"

#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/common/exception.hpp"

using namespace duckdb;

struct PivotScanBindData : public TableFunctionData {
	explicit PivotScanBindData(TableCatalogEntry &table) : table(table) {}
	TableCatalogEntry &table;
};

static BindInfo PivotScanGetBindInfo(const optional_ptr<FunctionData> bind_data) {
	auto &data = bind_data->Cast<PivotScanBindData>();
	return BindInfo(data.table);
}

PivotTableCatalogEntry::PivotTableCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info,
                                               rust::Box<OptionalTableWrapper> table)
    : TableCatalogEntry(catalog, schema, info), table(std::move(table)) {
}

TableFunction PivotTableCatalogEntry::GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<PivotScanBindData>(*this);
	TableFunction func(name, {}, nullptr, nullptr);
	func.get_bind_info = PivotScanGetBindInfo;
	return func;
}

unique_ptr<BaseStatistics> PivotTableCatalogEntry::GetStatistics(ClientContext &context, column_t column_id) {
	return nullptr;
}

TableStorageInfo PivotTableCatalogEntry::GetStorageInfo(ClientContext &context) {
	return TableStorageInfo();
}
