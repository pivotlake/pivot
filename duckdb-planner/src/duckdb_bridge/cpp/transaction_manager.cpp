#include "duckdb-planner/src/duckdb_bridge/cpp/transaction_manager.h"

using namespace duckdb;

PivotTransaction::PivotTransaction(TransactionManager &manager, ClientContext &context)
    : Transaction(manager, context) {
}

PivotTransactionManager::PivotTransactionManager(AttachedDatabase &db) : TransactionManager(db) {
}

Transaction &PivotTransactionManager::StartTransaction(ClientContext &context) {
	lock_guard<mutex> guard(lock);
	auto transaction = make_uniq<PivotTransaction>(*this, context);
	auto &ref = *transaction;
	transactions.push_back(std::move(transaction));
	return ref;
}

ErrorData PivotTransactionManager::CommitTransaction(ClientContext &context, Transaction &transaction) {
	lock_guard<mutex> guard(lock);
	transactions.erase(
	    std::remove_if(transactions.begin(), transactions.end(),
	                   [&](auto &t) { return t.get() == &transaction; }),
	    transactions.end());
	return ErrorData();
}

void PivotTransactionManager::RollbackTransaction(Transaction &transaction) {
	lock_guard<mutex> guard(lock);
	transactions.erase(
	    std::remove_if(transactions.begin(), transactions.end(),
	                   [&](auto &t) { return t.get() == &transaction; }),
	    transactions.end());
}

void PivotTransactionManager::Checkpoint(ClientContext &context, bool force) {
}

unique_ptr<TransactionManager> create_pivot_transaction_manager(
    optional_ptr<StorageExtensionInfo> info, AttachedDatabase &db, Catalog &catalog) {
	return make_uniq<PivotTransactionManager>(db);
}
