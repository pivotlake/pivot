#!/usr/bin/env bash
# Runs the window queries on DuckDB, over the same lineitem rows the suite's
# setup.sql gives pivot, and prints DuckDB's own timer for each run.
#
# The rows are loaded once into a DuckDB database file, so the queries read
# DuckDB's own storage. Only the columns the queries use are loaded.
#
# Usage: run-duckdb.sh <directory holding lineitem/*.parquet> <database file> [query ...]
set -euo pipefail
source_directory="$1"
database="$2"
shift 2
queries=("${@:-q01 q02 q03 q04 q05 q06}")
suite_directory="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$database" ]; then
    {
        echo "CREATE TABLE lineitem AS SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity,"
        echo "    l_extendedprice, l_discount, l_shipdate, l_shipmode"
        echo "FROM read_parquet('$source_directory/lineitem/*.parquet');"
    } | duckdb "$database"
fi

{
    echo ".timer on"
    for query in ${queries[@]}; do
        for iteration in 1 2 3 4 5; do
            echo ".print == $query iteration $iteration"
            cat "$suite_directory/$query.sql"
        done
    done
} | duckdb -readonly "$database"
