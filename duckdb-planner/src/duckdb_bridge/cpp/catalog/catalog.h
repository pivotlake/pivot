#pragma once
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "rust/cxx.h"
#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"

// A `CREATE TEXT INDEX` statement, carried through DuckDB's planning as it
// was written, for the planner to resolve. It is an extension operator
// because DuckDB takes an operator typed CREATE INDEX for its own, with its
// own fields, wherever it walks a plan.
struct LogicalCreateTextIndex : public duckdb::LogicalExtensionOperator {
	explicit LogicalCreateTextIndex(duckdb::unique_ptr<duckdb::CreateIndexInfo> info) : info(std::move(info)) {
	}

	duckdb::unique_ptr<duckdb::CreateIndexInfo> info;

	duckdb::PhysicalOperator &CreatePlan(duckdb::ClientContext &, duckdb::PhysicalPlanGenerator &) override {
		throw duckdb::NotImplementedException("CREATE TEXT INDEX is run by Pivot, not by DuckDB");
	}

	std::string GetExtensionName() const override {
		return "pivot_create_text_index";
	}

protected:
	void ResolveTypes() override {
		types.emplace_back(duckdb::LogicalType::BIGINT);
	}
};

class PivotCatalog : public duckdb::Catalog {
	// The schema entries materialized by lookups so far, keyed by schema name.
	// Pivot has no static schema list: which schemas exist is answered per
	// lookup by the datastore, through the transaction being planned. An entry
	// is created the first time a schema resolves and then kept, because
	// DuckDB's binder holds plain references to it for the rest of the plan.
	// Planning is single-threaded per context, so the map needs no lock.
	duckdb::unordered_map<std::string, duckdb::unique_ptr<PivotSchemaCatalogEntry>> schemas;

	// The entry for `name`, creating it on first resolution.
	PivotSchemaCatalogEntry &FindOrCreateSchema(const std::string &name);

	// Whether this datastore defines `name`, asked of the datastore through the
	// transaction currently being planned.
	bool DoesSchemaExist(const std::string &name);

public:
	explicit PivotCatalog(duckdb::AttachedDatabase &db);

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

	// `CREATE TEXT INDEX` is the one kind of index there is. DuckDB would bind
	// the statement into a plan that builds one of its own indexes; this hands
	// the statement on as it was written, in a `LogicalCreateTextIndex`.
	duckdb::unique_ptr<duckdb::LogicalOperator> BindCreateIndex(duckdb::Binder &binder, duckdb::CreateStatement &stmt,
	                                                           duckdb::TableCatalogEntry &table,
	                                                           duckdb::unique_ptr<duckdb::LogicalOperator> plan) override;

	duckdb::DatabaseSize GetDatabaseSize(duckdb::ClientContext &context) override;
	bool InMemory() override;
	std::string GetDBPath() override;
};
