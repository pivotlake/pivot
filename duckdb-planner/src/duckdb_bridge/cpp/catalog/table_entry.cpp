#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/bridge.h"

#include "duckdb/common/column_index.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table_storage_info.hpp"

using namespace duckdb;

static optional_ptr<TableCatalogEntry> PivotScanGetTableEntry(optional_ptr<const FunctionData> bind_data) {
	return &bind_data->Cast<PivotScanBindData>().catalog_entry;
}

// Feed DuckDB's cost model the table's real size. The optimizer reaches this
// through LogicalGet::EstimateCardinality, and the join order optimizer's
// relation stats build on it — so join ordering (and with it the hash join's
// build/probe side choice) sees actual row counts instead of the default
// guess. Returning null means "unknown" and keeps DuckDB's defaults.
static unique_ptr<NodeStatistics> PivotScanCardinality(ClientContext &context, const FunctionData *bind_data) {
	auto &data = bind_data->Cast<PivotScanBindData>();
	auto estimate = table_estimate_row_count(data.table);
	if (!estimate.known) {
		return nullptr;
	}
	// The count is reported as an estimate and not as an upper bound. DuckDB drops a LIMIT above an
	// input it knows to be no larger, and pivot's LIMIT is what stops a scan early.
	return make_uniq<NodeStatistics>(estimate.rows);
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
static unique_ptr<Expression> RewriteRefsToStorage(unique_ptr<Expression> expr,
                                                   const vector<ColumnIndex> &column_ids) {
	// Filters arriving in `pushdown_complex_filter` reference columns via
	// `BoundColumnRefExpression`, whose binding's column index is a
	// `ProjectionIndex` into the LogicalGet's projected output. Remap to
	// storage with `column_ids`.
	return ExpressionIterator::PreorderReplace<BoundColumnRefExpression>(
	    std::move(expr), [&](const BoundColumnRefExpression &column_ref) {
		    auto &binding = column_ref.Binding();
		    auto projected = binding.column_index.GetIndex();
		    D_ASSERT(projected < column_ids.size());
		    return column_ref.WithBinding(
		        ColumnBinding(binding.table_index, ProjectionIndex(column_ids[projected].GetPrimaryIndex())));
	    });
}

// This function is used to "hook" the filters a LogicalGet has before stats /
// anything else runs. We use it to try and push filters into the table function
// so stats will be accurate to the filters.
//
// References inside `filters` are in projected space (see RewriteRefsToStorage
// above for the gory details); we clone + remap each filter before handing it
// to Rust so the catalog sees storage-space column indices and can index into
// parquet row-group stats correctly.
void PivotPushdownComplexFilters(LogicalGet &get, OptionalTableWrapper &table,
                                 vector<unique_ptr<Expression>> &filters) {
	const auto &column_ids = get.GetColumnIds();

	// Complex filters: offer each one to Rust. If the table pushes it down,
	// drop it from the vector so DuckDB doesn't re-apply it on top.
	for (auto it = filters.begin(); it != filters.end();) {
		auto remapped = RewriteRefsToStorage((*it)->Copy(), column_ids);
		// Hand the (storage-remapped) DuckDB expression straight to Rust, which
		// reads it through the same `expr_*` accessors the plan walk uses.
		if (pushdown_filter(table, *remapped)) {
			it = filters.erase(it);
		} else {
			++it;
		}
	}
}

static void PivotScanPushdownComplexFilter(ClientContext &context, LogicalGet &get,
                                           optional_ptr<const FunctionData> bind_data,
                                           vector<unique_ptr<Expression>> &filters) {
	auto &data = bind_data->Cast<PivotScanBindData>();
	PivotPushdownComplexFilters(get, data.table, filters);
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
                                                 optional_ptr<const FunctionData> bind_data_p) {
	auto &data = bind_data_p->Cast<PivotScanBindData>();
	return data.catalog_entry.GetRowIdColumns();
}

PivotTableCatalogEntry::PivotTableCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info,
                                               rust::Box<OptionalTableWrapper> table)
    : TableCatalogEntry(catalog, schema, info), table(std::move(table)), columns(info.columns.Copy()) {
}

const ColumnList &PivotTableCatalogEntry::GetColumns() const {
	return columns;
}

// DuckDB's projection-pushdown pass asks, per column, whether this scanner can
// satisfy a field extract (`variant_extract`/`struct_extract`) by reading only
// the referenced sub-column instead of the whole column. We can: a variant
// column reads only the referenced shredded leaves, and a column with no
// extracts simply has nothing to push. Returning true lets the pass rewrite the
// scan's column_ids into a ColumnIndex path tree the bridge then reads.
static bool PivotScanSupportsPushdownExtract(const FunctionData &, const LogicalIndex &) {
	return true;
}

TableFunction PivotTableCatalogEntry::GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<PivotScanBindData>(*this, *table);
	TableFunction func(name, {}, nullptr, nullptr);
	func.get_table_entry = PivotScanGetTableEntry;
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
	// Push variant/struct field extracts into this scan's column_ids so we read
	// only the referenced leaves. Gated by the optimizer on `func.statistics`
	// being unset, which it is.
	func.supports_pushdown_extract = PivotScanSupportsPushdownExtract;
	// A capable table lets DuckDB rewrite a wide Top-N into a narrow row-id
	// scan whose survivors Pivot materializes. The row-id is a plan-time marker;
	// the default virtual column from TableCatalogEntry is sufficient. Tables
	// without a materialization implementation must not advertise these hooks.
	if (table_supports_late_materialization(*table)) {
		func.late_materialization = true;
		func.get_virtual_columns = PivotScanGetVirtualColumns;
		func.get_row_id_columns = PivotScanGetRowIdColumns;
	}
	func.cardinality = PivotScanCardinality;
	return func;
}

// Per-column statistics (min/max, distinct counts) are not plumbed from the
// Rust catalog; table-level cardinality (PivotScanCardinality) is. Null means
// "unknown" and is always safe.
unique_ptr<BaseStatistics> PivotTableCatalogEntry::GetStatistics(ClientContext &context, column_t column_id) {
	return nullptr;
}

TableStorageInfo PivotTableCatalogEntry::GetStorageInfo(ClientContext &context) {
	return TableStorageInfo();
}
