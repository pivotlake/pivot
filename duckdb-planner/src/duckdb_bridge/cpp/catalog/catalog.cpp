#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/common.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/common/exception.hpp"

using namespace duckdb;

PivotCatalog::PivotCatalog(AttachedDatabase &db) : Catalog(db) {
}

void PivotCatalog::Initialize(bool load_builtin) {
}

string PivotCatalog::GetCatalogType() {
	return "pivot";
}

optional_ptr<CatalogEntry> PivotCatalog::CreateSchema(CatalogTransaction, CreateSchemaInfo &) { RUST_NOT_IMPLEMENTED; }

unique_ptr<LogicalOperator> PivotCatalog::BindCreateIndex(Binder &, CreateStatement &stmt, TableCatalogEntry &table,
                                                          unique_ptr<LogicalOperator>) {
	auto &info = stmt.info->Cast<CreateIndexInfo>();
	if (info.index_type != "TEXT") {
		throw NotImplementedException("Only CREATE TEXT INDEX is supported, not an index of type %s",
		                              info.index_type);
	}
	// The statement may name its table without a schema or a database. The
	// binder resolved it to `table`, so the plan carries the names that did.
	info.catalog = table.ParentCatalog().GetName();
	info.schema = table.ParentSchema().name;
	return make_uniq<LogicalCreateTextIndex>(unique_ptr_cast<CreateInfo, CreateIndexInfo>(std::move(stmt.info)));
}

PivotSchemaCatalogEntry &PivotCatalog::FindOrCreateSchema(const std::string &name) {
	auto existing = schemas.find(name);
	if (existing != schemas.end()) {
		return *existing->second;
	}
	CreateSchemaInfo info;
	info.schema = name;
	auto entry = make_uniq<PivotSchemaCatalogEntry>(*this, info);
	auto &reference = *entry;
	schemas[name] = std::move(entry);
	return reference;
}

bool PivotCatalog::DoesSchemaExist(const std::string &name) {
	// Only a published transaction can answer: schema existence always comes
	// from the datastore's snapshot, never from this bridge. A lookup without
	// one is a bridge bug, so it fails rather than reporting the schema absent.
	auto &storage_info = PivotStorageInfo::Get(GetAttached().GetDatabase());
	if (!storage_info.current_transaction) {
		throw InternalException("pivot schema lookup with no pivot transaction published");
	}
	return catalog_does_schema_exist(*storage_info.current_transaction, GetName(), name);
}

optional_ptr<SchemaCatalogEntry> PivotCatalog::LookupSchema(CatalogTransaction transaction,
                                                            const EntryLookupInfo &schema_lookup,
                                                            OnEntryNotFound if_not_found) {
	// A lookup that names no schema is a reference to the default one.
	auto &requested = schema_lookup.GetEntryName();
	auto name = requested == INVALID_SCHEMA ? std::string(DEFAULT_SCHEMA) : requested;

	if (!DoesSchemaExist(name)) {
		if (if_not_found == OnEntryNotFound::THROW_EXCEPTION) {
			throw CatalogException("Schema \"%s\" not found", name);
		}
		return nullptr;
	}
	return &FindOrCreateSchema(name);
}

// Reports the schemas resolved so far rather than every schema the datastore
// holds: pivot answers schema existence by name through the transaction and
// never enumerates a datastore's schemas.
void PivotCatalog::ScanSchemas(ClientContext &context,
                               std::function<void(SchemaCatalogEntry &)> callback) {
	for (auto &entry : schemas) {
		callback(*entry.second);
	}
}

void PivotCatalog::DropSchema(ClientContext &, DropInfo &) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanCreateTableAs(ClientContext &, PhysicalPlanGenerator &, LogicalCreateTable &, PhysicalOperator &) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanInsert(ClientContext &, PhysicalPlanGenerator &, LogicalInsert &, optional_ptr<PhysicalOperator>) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanDelete(ClientContext &, PhysicalPlanGenerator &, LogicalDelete &, PhysicalOperator &) { RUST_NOT_IMPLEMENTED; }
PhysicalOperator &PivotCatalog::PlanUpdate(ClientContext &, PhysicalPlanGenerator &, LogicalUpdate &, PhysicalOperator &) { RUST_NOT_IMPLEMENTED; }

DatabaseSize PivotCatalog::GetDatabaseSize(ClientContext &context) { return DatabaseSize(); }
bool PivotCatalog::InMemory() { return true; }
string PivotCatalog::GetDBPath() { return ""; }
