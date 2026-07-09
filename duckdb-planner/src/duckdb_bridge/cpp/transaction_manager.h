#pragma once
#include "duckdb.hpp"
#include "duckdb/transaction/transaction_manager.hpp"
#include "duckdb/transaction/transaction.hpp"

#include <mutex>

namespace duckdb {
class StorageExtensionInfo;
}

class PivotTransaction : public duckdb::Transaction {
public:
	PivotTransaction(duckdb::TransactionManager &manager, duckdb::ClientContext &context);

	// Opaque per-statement context pointer the Rust planner threaded into
	// `extract_plan`. Set on the active transaction at the top of planning and
	// read back in `PivotSchemaCatalogEntry::LookupEntry` to resolve tables
	// against that statement's catalog snapshot. `0` when none was set.
	size_t statement_context = 0;
};

class PivotTransactionManager : public duckdb::TransactionManager {
	std::mutex lock;
	std::vector<duckdb::unique_ptr<PivotTransaction>> transactions;

public:
	explicit PivotTransactionManager(duckdb::AttachedDatabase &db);
	duckdb::Transaction &StartTransaction(duckdb::ClientContext &context) override;
	duckdb::ErrorData CommitTransaction(duckdb::ClientContext &context, duckdb::Transaction &transaction) override;
	void RollbackTransaction(duckdb::Transaction &transaction) override;
	void Checkpoint(duckdb::ClientContext &context, bool force) override;
};

duckdb::unique_ptr<duckdb::TransactionManager> create_pivot_transaction_manager(
    duckdb::optional_ptr<duckdb::StorageExtensionInfo> info, duckdb::AttachedDatabase &db, duckdb::Catalog &catalog);
