#include "duckdb-planner/src/duckdb_bridge/cpp/extension.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/catalog.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

#include "duckdb/main/config.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/secret/secret.hpp"
#include "duckdb/main/secret/secret_manager.hpp"

using namespace duckdb;

std::string PivotExtension::Name() {
	return "pivotdb";
}

static unique_ptr<Catalog> pivot_catalog_attach(optional_ptr<StorageExtensionInfo> info,
                                                ClientContext &context, AttachedDatabase &db,
                                                const string &name, AttachInfo &attach_info,
                                                AttachOptions &options) {
	auto &pivot_info = dynamic_cast<PivotStorageInfo &>(*info);
	return make_uniq<PivotCatalog>(db, pivot_info.catalog_ctx);
}

// Build the KeyValueSecret for a bound `CREATE SECRET (TYPE s3, ...)`. Pivot
// intercepts the CREATE SECRET plan before execution, so this only runs if the
// embedded planner ever executes one itself; it still builds a faithful secret.
static unique_ptr<BaseSecret> create_s3_secret_from_config(ClientContext &context, CreateSecretInput &input) {
	auto scope = input.scope;
	if (scope.empty()) {
		scope = {"s3://", "s3n://", "s3a://"};
	}
	auto secret = make_uniq<KeyValueSecret>(scope, input.type, input.provider, input.name);
	for (const auto &option : input.options) {
		secret->secret_map[option.first] = option.second;
	}
	secret->redact_keys = {"secret", "session_token"};
	return std::move(secret);
}

// Register the `s3` secret type with its `config` provider so the binder
// accepts `CREATE SECRET (TYPE s3, KEY_ID ..., SECRET ..., ...)` and emits a
// LOGICAL_CREATE_SECRET for pivot to walk.
//
// Named parameters: the store consumes key_id/secret/session_token when it
// signs requests. region and endpoint are stored and listed but describe the
// secret's target rather than re-pointing the store, whose endpoint and
// signing region are fixed when the database opens (see catalog's store/s3.rs).
// The type's other defaults (scope, redacted keys) live on the Rust side, in
// the planner, which resolves every CREATE SECRET before pivot applies it.
static void register_s3_secret_type(DatabaseInstance &db) {
	auto &secret_manager = db.GetSecretManager();

	SecretType s3_type;
	s3_type.name = "s3";
	s3_type.deserializer = KeyValueSecret::Deserialize<KeyValueSecret>;
	s3_type.default_provider = "config";
	s3_type.extension = "pivotdb";
	secret_manager.RegisterSecretType(s3_type);

	CreateSecretFunction config_function;
	config_function.secret_type = "s3";
	config_function.provider = "config";
	config_function.function = create_s3_secret_from_config;
	config_function.named_parameters["key_id"] = LogicalType::VARCHAR;
	config_function.named_parameters["secret"] = LogicalType::VARCHAR;
	config_function.named_parameters["region"] = LogicalType::VARCHAR;
	config_function.named_parameters["session_token"] = LogicalType::VARCHAR;
	config_function.named_parameters["endpoint"] = LogicalType::VARCHAR;
	secret_manager.RegisterSecretFunction(config_function, OnCreateConflict::ERROR_ON_CONFLICT);
}

void PivotExtension::Load(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto ext = make_shared_ptr<StorageExtension>();
	ext->attach = pivot_catalog_attach;
	ext->create_transaction_manager = create_pivot_transaction_manager;
	StorageExtension::Register(DBConfig::GetConfig(db), "pivotdb", ext);

	register_s3_secret_type(db);

	// No other functions are registered here. Pivot's own scalar and table
	// functions (drop_cache, metadata, ...) resolve on demand through the
	// catalog's LookupEntry, which builds their entry from the Rust provider's
	// registry, so adding one needs no change in this bridge.
}
