#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/logical_operator.hpp"

// Rewrites each GROUP BY whose aggregates read only the null-extended side of
// an outer equality join: that side is grouped by its join keys before the
// join, so the join and the GROUP BY above it see one row per key instead of
// one row per input row. Runs on the optimized plan, before its column
// bindings are resolved to positions.
void push_aggregates_below_outer_joins(duckdb::ClientContext &context, duckdb::Binder &binder,
                                       duckdb::unique_ptr<duckdb::LogicalOperator> &plan);
