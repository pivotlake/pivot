#pragma once
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "rust/cxx.h"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

class PivotCatalog : public duckdb::Catalog {
	duckdb::unique_ptr<PivotSchemaCatalogEntry> main_schema;
public:
	const CatalogContext *catalog_ctx;

	PivotCatalog(duckdb::AttachedDatabase &db, const CatalogContext *catalog_ctx);

	void Initialize(bool load_builtin) override;
	std::string GetCatalogType() override;

	duckdb::optional_ptr<duckdb::CatalogEntry> CreateSchema(duckdb::CatalogTransaction transaction,
	                                                         duckdb::CreateSchemaInfo &info) override;
	duckdb::optional_ptr<duckdb::SchemaCatalogEntry> LookupSchema(
	    duckdb::CatalogTransaction transaction, const duckdb::EntryLookupInfo &schema_lookup,
	    duckdb::OnEntryNotFound if_not_found) override;
	void ScanSchemas(duckdb::ClientContext &context,
	                 std::function<void(duckdb::SchemaCatalogEntry &)> callback) override;
	void DropSchema(duckdb::ClientContext &context, duckdb::DropInfo &info) override;

	duckdb::PhysicalOperator &PlanCreateTableAs(duckdb::ClientContext &, duckdb::PhysicalPlanGenerator &,
	                                            duckdb::LogicalCreateTable &, duckdb::PhysicalOperator &) override;
	duckdb::PhysicalOperator &PlanInsert(duckdb::ClientContext &, duckdb::PhysicalPlanGenerator &,
	                                     duckdb::LogicalInsert &, duckdb::optional_ptr<duckdb::PhysicalOperator>) override;
	duckdb::PhysicalOperator &PlanDelete(duckdb::ClientContext &, duckdb::PhysicalPlanGenerator &,
	                                     duckdb::LogicalDelete &, duckdb::PhysicalOperator &) override;
	duckdb::PhysicalOperator &PlanUpdate(duckdb::ClientContext &, duckdb::PhysicalPlanGenerator &,
	                                     duckdb::LogicalUpdate &, duckdb::PhysicalOperator &) override;

	duckdb::DatabaseSize GetDatabaseSize(duckdb::ClientContext &context) override;
	bool InMemory() override;
	std::string GetDBPath() override;
};
