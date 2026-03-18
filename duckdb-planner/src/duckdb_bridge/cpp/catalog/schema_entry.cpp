#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/common.h"

#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/common/exception.hpp"

using namespace duckdb;

PivotSchemaCatalogEntry::PivotSchemaCatalogEntry(Catalog &catalog, CreateSchemaInfo &info)
    : SchemaCatalogEntry(catalog, info) {
}

optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::LookupEntry(CatalogTransaction transaction,
                                                         const EntryLookupInfo &lookup_info) {
	auto &table_name = lookup_info.GetEntryName();
	auto &pivot_catalog = ParentCatalog().Cast<PivotCatalog>();
	auto result = catalog_get_table(*pivot_catalog.catalog_ctx, table_name);

	if (!result.found) {
		return nullptr;
	}

	CreateTableInfo table_info(*this, table_name);
	for (const auto &col : result.columns) {
		auto col_name = std::string(col.name);
		auto type_id = static_cast<duckdb::LogicalTypeId>(col.duckdb_logical_type_id);
		table_info.columns.AddColumn(ColumnDefinition(col_name, LogicalType(type_id)));
	}

	auto &db_instance = ParentCatalog().GetAttached().GetDatabase();
	auto &storage_info = PivotStorageInfo::Get(db_instance);
	auto entry = make_uniq<PivotTableCatalogEntry>(ParentCatalog(), *this, table_info, std::move(result.table));

	return storage_info.AddTableEntry(std::move(entry));
}

void PivotSchemaCatalogEntry::Scan(ClientContext &context, CatalogType type,
                            const std::function<void(CatalogEntry &)> &callback) {
}

void PivotSchemaCatalogEntry::Scan(CatalogType type,
                            const std::function<void(CatalogEntry &)> &callback) {
}

optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateTable(CatalogTransaction, BoundCreateTableInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateFunction(CatalogTransaction, CreateFunctionInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateIndex(CatalogTransaction, CreateIndexInfo &, TableCatalogEntry &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateView(CatalogTransaction, CreateViewInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateSequence(CatalogTransaction, CreateSequenceInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateTableFunction(CatalogTransaction, CreateTableFunctionInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateCopyFunction(CatalogTransaction, CreateCopyFunctionInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreatePragmaFunction(CatalogTransaction, CreatePragmaFunctionInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateCollation(CatalogTransaction, CreateCollationInfo &) { RUST_NOT_IMPLEMENTED; }
optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::CreateType(CatalogTransaction, CreateTypeInfo &) { RUST_NOT_IMPLEMENTED; }
void PivotSchemaCatalogEntry::DropEntry(ClientContext &, DropInfo &) { RUST_NOT_IMPLEMENTED; }
void PivotSchemaCatalogEntry::Alter(CatalogTransaction, AlterInfo &) { RUST_NOT_IMPLEMENTED; }
