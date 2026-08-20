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

// Materialize the type behind a LogicalTypeId the Rust provider described.
// Most ids are complete types on their own; VARIANT carries internal type
// structure, so it must be built through its factory.
LogicalType logical_type_from(uint8_t type_id) {
	auto id = static_cast<LogicalTypeId>(type_id);
	if (id == LogicalTypeId::DECIMAL) {
		throw InternalException("a DECIMAL column needs a width and scale, not just a type id");
	}
	if (id == LogicalTypeId::VARIANT) {
		return LogicalType::VARIANT();
	}
	return LogicalType(id);
}

// Materialize a column's type, completing a DECIMAL with the width and scale
// the Rust provider carried alongside the id.
LogicalType logical_type_from_column(const DuckDBColumn &col) {
	if (static_cast<LogicalTypeId>(col.duckdb_logical_type_id) == LogicalTypeId::DECIMAL) {
		return LogicalType::DECIMAL(col.decimal_width, col.decimal_scale);
	}
	return logical_type_from(col.duckdb_logical_type_id);
}

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
// into its own expression, and every pivot scalar whose value DuckDB could ask
// for is registered VOLATILE so the optimizer can't fold it. Emits a constant
// NULL so DuckDB has a valid, type-agnostic result if it ever does evaluate the
// call — a fold that reached here would rewrite the call to NULL, so a new
// pivot scalar that leaves out the volatile flag is a silently wrong answer.
void pivot_scalar_function_stub(DataChunk &, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

} // namespace

PivotSchemaCatalogEntry::PivotSchemaCatalogEntry(Catalog &catalog, CreateSchemaInfo &info)
    : SchemaCatalogEntry(catalog, info) {
}

// The pivot transaction of the plan currently being extracted, published on
// the storage info by `extract_plan` for the duration of that one call
// (planning is single-threaded per context). Table and table-function lookups
// resolve through it, so everything one plan binds comes from the same catalog
// snapshot (never from the live catalog, which a background refresh may be
// updating concurrently). A lookup without one is a bridge bug, never a
// legitimate binder probe: the only statements run outside `extract_plan` are
// the ATTACHes at context construction, which bind no pivot entries.
static const ::TransactionContext &pivot_transaction_ctx(duckdb::Catalog &catalog) {
	auto &storage_info = PivotStorageInfo::Get(catalog.GetAttached().GetDatabase());
	if (!storage_info.current_transaction) {
		throw InternalException("pivot catalog lookup with no pivot transaction published");
	}
	return *storage_info.current_transaction;
}

optional_ptr<CatalogEntry> PivotSchemaCatalogEntry::LookupEntry(CatalogTransaction transaction,
                                                         const EntryLookupInfo &lookup_info) {
	auto &table_name = lookup_info.GetEntryName();
	auto &pivot_catalog = ParentCatalog().Cast<PivotCatalog>();

	// A table-function reference (e.g. `metadata('t')`): resolve it against the
	// transaction's snapshot, since such a function reads catalog data. Unknown
	// names (DuckDB built-ins like generate_series) return nullptr, so the
	// binder falls through to the system catalog. The function's schema comes
	// entirely from Rust; nothing about it is declared in this bridge.
	if (lookup_info.GetCatalogType() == CatalogType::TABLE_FUNCTION_ENTRY) {
		auto function = catalog_get_table_function(
		    pivot_transaction_ctx(ParentCatalog()), ParentCatalog().GetName(), table_name);
		if (!function.found) {
			return nullptr;
		}

		auto info = make_shared_ptr<PivotTableFunctionInfo>();
		vector<LogicalType> arguments;
		for (auto type_id : function.arg_type_ids) {
			arguments.emplace_back(logical_type_from(type_id));
		}
		for (const auto &col : function.columns) {
			info->names.emplace_back(std::string(col.name));
			info->return_types.emplace_back(logical_type_from_column(col));
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
		for (auto type_id : function.function.arg_type_ids) {
			arguments.emplace_back(logical_type_from(type_id));
		}
		LogicalType return_type = logical_type_from(function.function.return_type_id);

		ScalarFunction func(std::string(table_name), std::move(arguments), return_type,
		                    pivot_scalar_function_stub);
		if (function.function.is_volatile) {
			func.SetStability(FunctionStability::VOLATILE);
		}

		CreateScalarFunctionInfo create_info(func);
		auto entry =
		    make_uniq<ScalarFunctionCatalogEntry>(ParentCatalog(), *this, create_info);
		auto &db_instance = ParentCatalog().GetAttached().GetDatabase();
		return PivotStorageInfo::Get(db_instance).AddScalarFunctionEntry(std::move(entry));
	}

	// Pivot holds only tables and the functions handled above; probes for any
	// other entry type (types, sequences, macros, ...) can't resolve here, so
	// fall through to the system catalog exactly as a not-found table does.
	if (lookup_info.GetCatalogType() != CatalogType::TABLE_ENTRY) {
		return nullptr;
	}

	// A base-table reference: resolve it through the transaction's snapshot,
	// within this schema. The catalog only hands out an entry for a schema the
	// datastore defines, so `name` here is always one of its own schemas.
	auto result = catalog_get_table(
	    pivot_transaction_ctx(ParentCatalog()), ParentCatalog().GetName(), this->name, table_name);

	if (!result.found) {
		return nullptr;
	}

	CreateTableInfo table_info(*this, table_name);
	for (const auto &col : result.columns) {
		auto col_name = std::string(col.name);
		table_info.columns.AddColumn(
		    ColumnDefinition(col_name, logical_type_from_column(col)));
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
