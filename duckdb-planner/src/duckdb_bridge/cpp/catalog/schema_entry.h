#pragma once
#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

/// A Pivot table function binds to the same Rust table object as a catalog
/// scan. DuckDB may copy bind data while optimizing, so every copy gets a fresh
/// wrapper around the bound table's own clone.
struct PivotTableFunctionBindData : public duckdb::TableFunctionData {
	explicit PivotTableFunctionBindData(rust::Box<OptionalTableWrapper> table)
	    : table(std::move(table)) {}

	rust::Box<OptionalTableWrapper> table;

	duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
		auto copy = duckdb::make_uniq<PivotTableFunctionBindData>(
		    clone_table_function_table(*table));
		copy->column_ids = column_ids;
		return duckdb::unique_ptr_cast<PivotTableFunctionBindData, duckdb::FunctionData>(
		    std::move(copy));
	}
};

/// Register every Pivot-supported table-function overload in DuckDB's system
/// catalog. Their bind callbacks return regular Rust table handles.
void RegisterPivotTableFunctions(duckdb::ClientContext &context,
                                 const CatalogContext &catalog);

class PivotSchemaCatalogEntry : public duckdb::SchemaCatalogEntry {
public:
	PivotSchemaCatalogEntry(duckdb::Catalog &catalog, duckdb::CreateSchemaInfo &info);

	// Table lookup — calls back to Rust
	duckdb::optional_ptr<duckdb::CatalogEntry> LookupEntry(duckdb::CatalogTransaction transaction,
	                                                        const duckdb::EntryLookupInfo &lookup_info) override;
	void Scan(duckdb::ClientContext &context, duckdb::CatalogType type,
	          const std::function<void(duckdb::CatalogEntry &)> &callback) override;
	void Scan(duckdb::CatalogType type,
	          const std::function<void(duckdb::CatalogEntry &)> &callback) override;

	// Not supported — all throw NotImplementedException
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateTable(duckdb::CatalogTransaction, duckdb::BoundCreateTableInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateFunction(duckdb::CatalogTransaction, duckdb::CreateFunctionInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateIndex(duckdb::CatalogTransaction, duckdb::CreateIndexInfo &, duckdb::TableCatalogEntry &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateView(duckdb::CatalogTransaction, duckdb::CreateViewInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateSequence(duckdb::CatalogTransaction, duckdb::CreateSequenceInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateTableFunction(duckdb::CatalogTransaction, duckdb::CreateTableFunctionInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateCopyFunction(duckdb::CatalogTransaction, duckdb::CreateCopyFunctionInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreatePragmaFunction(duckdb::CatalogTransaction, duckdb::CreatePragmaFunctionInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateCollation(duckdb::CatalogTransaction, duckdb::CreateCollationInfo &) override;
	duckdb::optional_ptr<duckdb::CatalogEntry> CreateType(duckdb::CatalogTransaction, duckdb::CreateTypeInfo &) override;
	void DropEntry(duckdb::ClientContext &, duckdb::DropInfo &) override;
	void Alter(duckdb::CatalogTransaction, duckdb::AlterInfo &) override;
};
