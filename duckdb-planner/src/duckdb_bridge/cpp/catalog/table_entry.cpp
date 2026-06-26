#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/bridge.h"

#include "duckdb/common/column_index.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/table_storage_info.hpp"

#include <flatbuffers/flatbuffers.h>
#include <cstdio>

using namespace duckdb;

static BindInfo PivotScanGetBindInfo(const optional_ptr<FunctionData> bind_data) {
	auto &data = bind_data->Cast<PivotScanBindData>();
	return BindInfo(data.catalog_entry);
}

// Walk an expression tree and rewrite every BoundReferenceExpression's index
// from "projected output" space (i.e. an index into LogicalGet's bound output)
// into "storage" space (an index into the table's full column list), using
// the get's `column_ids` mapping (`projected_index → storage_index`).
//
// Why this is necessary:
//
// By the time `pushdown_complex_filter` runs, DuckDB's binder has already
// pruned `LogicalGet` to only the columns the query references. The filter
// expressions handed to us reference columns by their position in *that
// pruned output*, not by their position in the underlying table. The Rust
// catalog, on the other hand, indexes parquet row-group statistics by
// storage position — so without this remap a filter on a non-leading column (storage
// index 9, projected index 0) would look up stats for whatever storage
// column happens to live at parquet index 0, which is
// nonsense and over-prunes (or under-prunes) row groups.
//
// We mutate a *clone* of the expression rather than the filter in-place: if
// the table refuses pushdown, DuckDB keeps the original filter and reapplies
// it as a `LogicalFilter` above the scan, where projected-space indices are
// the right ones again.
static void RewriteRefsToStorage(Expression &expr, const vector<ColumnIndex> &column_ids) {
	// Filters arriving in `pushdown_complex_filter` reference columns via
	// `BoundColumnRefExpression`, whose `binding.column_index` is a
	// `ProjectionIndex` into the LogicalGet's projected output. Remap to
	// storage with `column_ids`.
	if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		auto &col_ref = expr.Cast<BoundColumnRefExpression>();
		auto projected = col_ref.binding.column_index.GetIndex();
		D_ASSERT(projected < column_ids.size());
		col_ref.binding.column_index =
		    ProjectionIndex(column_ids[projected].GetPrimaryIndex());
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { RewriteRefsToStorage(child, column_ids); });
}

// This function is used to "hook" the filters a LogicalGet has before stats /
// anything else runs. We use it to try and push filters into the table function
// so stats will be accurate to the filters.
//
// References inside `filters` are in projected space (see RewriteRefsToStorage
// above for the gory details); we clone + remap each filter before handing it
// to Rust so the catalog sees storage-space column indices and can index into
// parquet row-group stats correctly.
static void PivotScanPushdownComplexFilter(ClientContext &context, LogicalGet &get, FunctionData *bind_data,
                                           vector<unique_ptr<Expression>> &filters) {
	auto &data = bind_data->Cast<PivotScanBindData>();
	const auto &column_ids = get.GetColumnIds();

	// Complex filters: offer each one to Rust. If the table pushes it down,
	// drop it from the vector so DuckDB doesn't re-apply it on top.
	for (auto it = filters.begin(); it != filters.end();) {
		auto remapped = (*it)->Copy();
		RewriteRefsToStorage(*remapped, column_ids);
		pivot::plan::TableFilterT filter;
		filter.kind.Set(std::move(*build_expression(remapped.get())));
		flatbuffers::FlatBufferBuilder fbb;
		fbb.Finish(pivot::plan::TableFilter::Pack(fbb, &filter));
		rust::Slice<const uint8_t> filter_fb(fbb.GetBufferPointer(), fbb.GetSize());
		if (pushdown_filter(data.table, filter_fb)) {
			it = filters.erase(it);
		} else {
			++it;
		}
	}
}

// Expose the table's virtual columns (notably the default `rowid`) so DuckDB's
// late-materialization optimizer can reference a row-id when it rewrites a wide
// Top-N/Limit scan into a row-id SEMI join.
static virtual_column_map_t PivotScanGetVirtualColumns(ClientContext &context,
                                                       optional_ptr<FunctionData> bind_data_p) {
	auto &data = bind_data_p->Cast<PivotScanBindData>();
	return data.catalog_entry.GetVirtualColumns();
}

// The row-id column(s) late materialization joins on — the standard single
// `rowid` from TableCatalogEntry. Plan-time only; pivot never executes a
// DuckDB scan, so no real row-id values are produced.
static vector<column_t> PivotScanGetRowIdColumns(ClientContext &context,
                                                 optional_ptr<FunctionData> bind_data_p) {
	auto &data = bind_data_p->Cast<PivotScanBindData>();
	return data.catalog_entry.GetRowIdColumns();
}

PivotTableCatalogEntry::PivotTableCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info,
                                               rust::Box<OptionalTableWrapper> table)
    : TableCatalogEntry(catalog, schema, info), table(std::move(table)) {
}

TableFunction PivotTableCatalogEntry::GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<PivotScanBindData>(*this, *table);
	TableFunction func(name, {}, nullptr, nullptr);
	func.get_bind_info = PivotScanGetBindInfo;
	func.pushdown_complex_filter = PivotScanPushdownComplexFilter;
	// Enabling filter_pushdown lets DuckDB's optimizer passes install filters
	// into get.table_filters — both `col op const` leftovers and the
	// DynamicFilters produced by the Top-N pushdown pass. The Rust catalog
	// doesn't consume TableFilters directly; `build_plan_node` in
	// bridge.cpp splits each Get's table_filters back into a synthetic
	// LogicalFilter (for the static predicates, keeping the Rust-facing shape
	// identical to filter_pushdown=false) plus dynamic-filter slot references
	// attached to the Input and its producing Top-N.
	func.filter_pushdown = true;
	func.projection_pushdown = true;
	// Advertise row-id / late-materialization support so DuckDB's
	// `late_materialization` optimizer fires for `SELECT <wide> ... ORDER BY ...
	// LIMIT n` queries over this table: it rewrites them into a SEMI join on the
	// row-id whose narrow side scans only the predicate/sort columns. The bridge
	// recognizes that join and collapses it onto pivot's own materializer. We
	// only PLAN with DuckDB (never execute its scan), so the row-id is purely a
	// plan-time marker; the default `rowid` virtual column from TableCatalogEntry
	// is enough.
	func.late_materialization = true;
	func.get_virtual_columns = PivotScanGetVirtualColumns;
	func.get_row_id_columns = PivotScanGetRowIdColumns;
	return func;
}

unique_ptr<BaseStatistics> PivotTableCatalogEntry::GetStatistics(ClientContext &context, column_t column_id) {
	std::fprintf(stderr, "[PivotTableCatalogEntry::GetStatistics] reached, column_id=%llu\n",
	             static_cast<unsigned long long>(column_id));
	return nullptr;
}

TableStorageInfo PivotTableCatalogEntry::GetStorageInfo(ClientContext &context) {
	return TableStorageInfo();
}
