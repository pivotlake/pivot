#include "duckdb-planner/src/duckdb_bridge/cpp/pivot_config.h"

PivotConfig::PivotConfig(const CatalogContext *catalog_ctx)
    : catalog_ctx(catalog_ctx) {
}
