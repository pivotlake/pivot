#pragma once
#include "duckdb/main/extension.hpp"

class PivotExtension : public duckdb::Extension {
public:
	void Load(duckdb::ExtensionLoader &loader) override;
	std::string Name() override;
};
