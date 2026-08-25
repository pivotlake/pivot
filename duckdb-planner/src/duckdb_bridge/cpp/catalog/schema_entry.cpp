#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/common.h"

#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/exception.hpp"

#include <map>

using namespace duckdb;

namespace {

// Materialize the type behind a LogicalTypeId described by Rust.
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
// Rust carried alongside the id.
LogicalType logical_type_from_column(const DuckDBColumn &col) {
	if (static_cast<LogicalTypeId>(col.duckdb_logical_type_id) == LogicalTypeId::DECIMAL) {
		return LogicalType::DECIMAL(col.decimal_width, col.decimal_scale);
	}
	return logical_type_from(col.duckdb_logical_type_id);
}

// Carries the invocation-independent lookup state into the bind callback. The
// output schema is not known until Rust opens the argument's files, so binding
// returns both the columns and the regular bound table that will scan them.
struct PivotTableFunctionInfo : public TableFunctionInfo {
	string name;
};

unique_ptr<FunctionData> pivot_table_function_bind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types,
                                                   vector<string> &names) {
	auto &info = input.info->Cast<PivotTableFunctionInfo>();
	if (!input.named_parameters.empty()) {
		throw BinderException("pivot table functions do not support named arguments");
	}
	auto &storage_info = PivotStorageInfo::Get(*context.db);
	if (!storage_info.current_transaction) {
		throw InternalException("table-function bind with no pivot transaction published");
	}
	auto binding = catalog_bind_table_function(
	    *storage_info.current_transaction, info.name, input.inputs);
	for (const auto &col : binding.columns) {
		names.emplace_back(std::string(col.name));
		return_types.emplace_back(logical_type_from_column(col));
	}
	return make_uniq<PivotTableFunctionBindData>(std::move(binding.table));
}

unique_ptr<NodeStatistics> pivot_table_function_cardinality(ClientContext &,
                                                            const FunctionData *bind_data) {
	auto &data = bind_data->Cast<PivotTableFunctionBindData>();
	auto estimate = table_estimate_row_count(*data.table);
	if (!estimate.known) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(estimate.rows, estimate.rows);
}

// A Pivot table function binds to the same Rust BoundTable abstraction as a
// catalog scan, so offer its filters through the same storage-index remapping
// path. A table that returns false leaves the filter above the scan.
void pivot_table_function_pushdown_complex_filter(
    ClientContext &, LogicalGet &get, FunctionData *bind_data,
    vector<unique_ptr<Expression>> &filters) {
	auto &data = bind_data->Cast<PivotTableFunctionBindData>();
	PivotPushdownComplexFilters(get, *data.table, filters);
}

// Pivot's executor identifies late-materialization survivors through its own
// scan metadata. DuckDB still requires a row-id-shaped virtual column to prove
// that a table function is eligible for the logical rewrite; it remains a
// planning marker and is never read by the Pivot scan.
virtual_column_map_t pivot_table_function_get_virtual_columns(
    ClientContext &, optional_ptr<FunctionData>) {
	virtual_column_map_t result;
	result.insert(make_pair(COLUMN_IDENTIFIER_ROW_ID,
	                        TableColumn("rowid", LogicalType::ROW_TYPE)));
	return result;
}

vector<column_t> pivot_table_function_get_row_id_columns(
    ClientContext &, optional_ptr<FunctionData>) {
	return {COLUMN_IDENTIFIER_ROW_ID};
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

void RegisterPivotTableFunctions(ClientContext &context, const CatalogContext &catalog_context) {
	std::map<string, TableFunctionSet> function_sets;
	for (const auto &definition : catalog_table_functions(catalog_context)) {
		string name(definition.name);
		auto entry = function_sets.find(name);
		if (entry == function_sets.end()) {
			entry = function_sets.emplace(name, TableFunctionSet(name)).first;
		}

		vector<LogicalType> arguments;
		for (auto type_id : definition.arg_type_ids) {
			arguments.emplace_back(logical_type_from(type_id));
		}
		auto info = make_shared_ptr<PivotTableFunctionInfo>();
		info->name = name;
		TableFunction function(name, std::move(arguments), nullptr, pivot_table_function_bind);
		function.function_info = std::move(info);
		function.pushdown_complex_filter = pivot_table_function_pushdown_complex_filter;
		// Let DuckDB lower simple comparisons into table_filters after the
		// callback has offered them to the bound table. The plan bridge restores
		// those filters above the Pivot scan because a bound table may use them
		// only as a metadata optimization.
		function.filter_pushdown = true;
		function.projection_pushdown = true;
		function.cardinality = pivot_table_function_cardinality;
		if (definition.supports_late_materialization) {
			function.late_materialization = true;
			function.get_virtual_columns = pivot_table_function_get_virtual_columns;
			function.get_row_id_columns = pivot_table_function_get_row_id_columns;
		}
		entry->second.AddFunction(std::move(function));
	}

	context.RunFunctionInTransaction([&]() {
		auto &catalog = Catalog::GetSystemCatalog(context);
		for (auto &entry : function_sets) {
			CreateTableFunctionInfo create_info(std::move(entry.second));
			create_info.internal = true;
			create_info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
			catalog.CreateTableFunction(context, create_info);
		}
	});
}

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

	// A scalar-function reference (e.g. `drop_cache()`): resolve it against the
	// current transaction's functions. Unknown names (DuckDB's own built-ins like
	// `+`/`length`) return nullptr and resolve against the system catalog.
	if (lookup_info.GetCatalogType() == CatalogType::SCALAR_FUNCTION_ENTRY) {
		auto function = catalog_get_scalar_function(
		    pivot_transaction_ctx(ParentCatalog()), table_name);
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
