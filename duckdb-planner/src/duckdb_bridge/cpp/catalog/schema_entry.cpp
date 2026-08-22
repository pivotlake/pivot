#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/common.h"

#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
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

// The pivot transaction of the plan currently being extracted, published on
// the storage info by `extract_plan` for the duration of that one call
// (planning is single-threaded per context). Table lookups resolve through it,
// so everything one plan binds comes from the same catalog snapshot (never from
// the live catalog, which a background refresh may be updating concurrently). A
// lookup without one is a bridge bug, never a legitimate binder probe: the only
// statements run outside `extract_plan` are the ATTACHes at context
// construction, which bind no pivot entries.
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

	// A scalar-function reference (e.g. `drop_cache()`): resolve it against the
	// provider's own functions. Unknown names (DuckDB's own built-ins like
	// `+`/`length`) return nullptr and resolve against the system catalog.
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
