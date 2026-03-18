#pragma once
#include "duckdb.hpp"
#include "duckdb/main/config.hpp"

struct CatalogContext;

class PivotConfig : public duckdb::DBConfig {
public:
	explicit PivotConfig(const CatalogContext *catalog_ctx);

	const CatalogContext *catalog_ctx;
};
