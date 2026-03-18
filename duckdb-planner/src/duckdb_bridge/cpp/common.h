#pragma once
#include "duckdb/common/exception.hpp"

#define RUST_NOT_IMPLEMENTED throw duckdb::NotImplementedException("Not supported in Rust catalog")
