#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/common.h"

#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/exception.hpp"

using namespace duckdb;

namespace {

// Carries a table function's output schema from the lookup (where Rust supplied
// it) through to its bind. pivot only plans, never executes, so the bind just
// republishes that schema and the bind data is a trivial placeholder.
struct PivotTableFunctionInfo : public TableFunctionInfo {
	vector<string> names;
	vector<LogicalType> return_types;
};

struct PivotTableFunctionBindData : public TableFunctionData {};

unique_ptr<FunctionData> pivot_table_function_bind(ClientContext &, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types,
                                                   vector<string> &names) {
	auto &info = input.info->Cast<PivotTableFunctionInfo>();
	names = info.names;
	return_types = info.return_types;
	return make_uniq<PivotTableFunctionBindData>();
}

// Body of a pivot scalar function stub. It never runs: pivot re-plans the call
// into its own expression, and the only pivot scalar (drop_cache) is VOLATILE so
// the optimizer can't fold it. Emits a constant NULL so DuckDB has a valid,
// type-agnostic result if it ever does evaluate the call.
void pivot_scalar_function_stub(DataChunk &, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

} // namespace

PivotSchemaCatalogEntry::PivotSchemaCatalogEntry(Catalog &catalog, CreateSchemaInfo &info)
    : SchemaCatalogEntry(catalog, info) {
}

optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::LookupEntry(CatalogTransaction transaction,
                                                         const EntryLookupInfo &lookup_info) {
	auto &table_name = lookup_info.GetEntryName();
	auto &pivot_catalog = ParentCatalog().Cast<PivotCatalog>();

	// A table-function reference (e.g. `metadata('t')`): resolve it against the
	// Rust provider's registry. Unknown names (DuckDB built-ins like
	// generate_series) return nullptr, so the binder falls through to the system
	// catalog. The function's schema comes entirely from Rust; nothing about it
	// is declared in this bridge.
	if (lookup_info.GetCatalogType() == CatalogType::TABLE_FUNCTION_ENTRY) {
		auto function = catalog_get_table_function(*pivot_catalog.catalog_ctx, table_name);
		if (!function.found) {
			return nullptr;
		}

		auto info = make_shared_ptr<PivotTableFunctionInfo>();
		vector<LogicalType> arguments;
		for (auto type_id : function.arg_type_ids) {
			arguments.emplace_back(static_cast<LogicalTypeId>(type_id));
		}
		for (const auto &col : function.columns) {
			info->names.emplace_back(std::string(col.name));
			info->return_types.emplace_back(
			    static_cast<LogicalTypeId>(col.duckdb_logical_type_id));
		}

		TableFunction func(std::string(table_name), std::move(arguments), nullptr,
		                   pivot_table_function_bind);
		func.function_info = info;

		CreateTableFunctionInfo create_info(func);
		auto entry =
		    make_uniq<TableFunctionCatalogEntry>(ParentCatalog(), *this, create_info);
		auto &db_instance = ParentCatalog().GetAttached().GetDatabase();
		return PivotStorageInfo::Get(db_instance).AddFunctionEntry(std::move(entry));
	}

	// A scalar-function reference (e.g. `drop_cache()`): same idea as table
	// functions. Unknown names (DuckDB's own built-ins like `+`/`length`) return
	// nullptr and resolve against the system catalog.
	if (lookup_info.GetCatalogType() == CatalogType::SCALAR_FUNCTION_ENTRY) {
		auto function = catalog_get_scalar_function(*pivot_catalog.catalog_ctx, table_name);
		if (!function.found) {
			return nullptr;
		}

		vector<LogicalType> arguments;
		for (auto type_id : function.arg_type_ids) {
			arguments.emplace_back(static_cast<LogicalTypeId>(type_id));
		}
		LogicalType return_type(static_cast<LogicalTypeId>(function.return_type_id));

		ScalarFunction func(std::string(table_name), std::move(arguments), return_type,
		                    pivot_scalar_function_stub);
		if (function.is_volatile) {
			func.SetStability(FunctionStability::VOLATILE);
		}

		CreateScalarFunctionInfo create_info(func);
		auto entry =
		    make_uniq<ScalarFunctionCatalogEntry>(ParentCatalog(), *this, create_info);
		auto &db_instance = ParentCatalog().GetAttached().GetDatabase();
		return PivotStorageInfo::Get(db_instance).AddScalarFunctionEntry(std::move(entry));
	}

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
