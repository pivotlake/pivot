#pragma once
#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/table_function.hpp"
#include "rust/cxx.h"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

struct PivotScanBindData : public duckdb::TableFunctionData {
	explicit PivotScanBindData(duckdb::TableCatalogEntry &catalog_entry, OptionalTableWrapper &table)
	    : catalog_entry(catalog_entry), table(table) {}
	duckdb::TableCatalogEntry &catalog_entry;
	OptionalTableWrapper &table;

	// Late materialization clones the scan's bind data to build a second
	// (full-column) Get of the same table. Both copies reference the same
	// catalog entry and table handle — fine, since we only plan (never execute)
	// with DuckDB, and the bridge collapses the resulting join back into a
	// single pivot scan + materializer.
	duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
		return duckdb::make_uniq<PivotScanBindData>(catalog_entry, table);
	}
};

class PivotTableCatalogEntry : public duckdb::TableCatalogEntry {
public:
	/// Opaque handle to the Rust `GetDuckDBTypedColumns` object that backs this catalog entry.
	rust::Box<OptionalTableWrapper> table;

	PivotTableCatalogEntry(duckdb::Catalog &catalog, duckdb::SchemaCatalogEntry &schema,
	                duckdb::CreateTableInfo &info, rust::Box<OptionalTableWrapper> table);

	duckdb::TableFunction GetScanFunction(duckdb::ClientContext &context,
	                                      duckdb::unique_ptr<duckdb::FunctionData> &bind_data) override;
	duckdb::unique_ptr<duckdb::BaseStatistics> GetStatistics(duckdb::ClientContext &context,
	                                                         duckdb::column_t column_id) override;
	duckdb::TableStorageInfo GetStorageInfo(duckdb::ClientContext &context) override;
};

// Give `func` the hooks a pivot scan is planned with: filter and projection
// pushdown into the Rust table, pushed field extracts, and the table's real
// cardinality. Shared by the two ways a pivot scan is reached — a table
// reference, and a function naming a location (`read_parquet`) whose bind
// resolves to a table — so both are planned by exactly the same rules.
void ConfigurePivotScanFunction(duckdb::TableFunction &func);

// Add the hooks that let DuckDB rewrite a wide Top-N over `table` into a narrow
// row-id scan whose survivors pivot materializes. Separate from
// `ConfigurePivotScanFunction` because it is the one part that depends on the
// bound table: a table with no materialization implementation must not
// advertise them. `entry` supplies the row-id virtual column.
void ConfigurePivotLateMaterialization(duckdb::TableFunction &func, const OptionalTableWrapper &table);
