#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/common.h"

#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/common/exception.hpp"

using namespace duckdb;

PivotCatalog::PivotCatalog(AttachedDatabase &db, const CatalogContext *catalog_ctx)
    : Catalog(db), catalog_ctx(catalog_ctx) {
}

void PivotCatalog::Initialize(bool load_builtin) {
	CreateSchemaInfo info;
	info.schema = DEFAULT_SCHEMA;
	main_schema = make_uniq<PivotSchemaCatalogEntry>(*this, info);
}

string PivotCatalog::GetCatalogType() {
	return "pivot";
}

optional_ptr<CatalogEntry> PivotCatalog::CreateSchema(CatalogTransaction, CreateSchemaInfo &) { RUST_NOT_IMPLEMENTED; }

optional_ptr<SchemaCatalogEntry> PivotCatalog::LookupSchema(CatalogTransaction transaction,
                                                            const EntryLookupInfo &schema_lookup,
                                                            OnEntryNotFound if_not_found) {
	auto &name = schema_lookup.GetEntryName();
	if (name == DEFAULT_SCHEMA || name == INVALID_SCHEMA) {
		return main_schema.get();
	}
	if (if_not_found == OnEntryNotFound::THROW_EXCEPTION) {
		throw CatalogException("Schema \"%s\" not found", name);
	}
	return nullptr;
}

void PivotCatalog::ScanSchemas(ClientContext &context,
                               std::function<void(SchemaCatalogEntry &)> callback) {
	callback(*main_schema);
}

void PivotCatalog::DropSchema(ClientContext &, DropInfo &) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanCreateTableAs(ClientContext &, PhysicalPlanGenerator &, LogicalCreateTable &, PhysicalOperator &) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanInsert(ClientContext &, PhysicalPlanGenerator &, LogicalInsert &, optional_ptr<PhysicalOperator>) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanDelete(ClientContext &, PhysicalPlanGenerator &, LogicalDelete &, PhysicalOperator &) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanUpdate(ClientContext &, PhysicalPlanGenerator &, LogicalUpdate &, PhysicalOperator &) { RUST_NOT_IMPLEMENTED; }

DatabaseSize PivotCatalog::GetDatabaseSize(ClientContext &context) { return DatabaseSize(); }
bool PivotCatalog::InMemory() { return true; }
string PivotCatalog::GetDBPath() { return ""; }
