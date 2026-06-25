#!/usr/bin/env bash
#
# prep-tpch-flat-data.sh - generate the data for the tpch_flat suite: one
# fully denormalised, lineitem-grain parquet file. DuckDB's tpch extension
# generates the eight normalised tables, then a single query joins them all
# into one wide `lineitem_flat` table (one row per lineitem, carrying its
# order / customer / supplier / part / partsupp / nation / region attributes,
# both the supplier and customer nation+region). The suite's queries then run
# as pure single-table scans - no joins - so the same parquet is read the same
# way by pivot, DuckDB and ClickHouse.
#
# Monetary / quantity columns are written as DOUBLE rather than TPC-H's
# DECIMAL: pivot evaluates decimal arithmetic in floating point, so storing
# DOUBLE makes all three engines compute the same way (and sidesteps any
# decimal-reader differences). Keys stay integers, dates stay DATE.
#
# Usage:
#   ./prep-tpch-flat-data.sh                       # sf 1 → ~/tpch-flat
#   ./prep-tpch-flat-data.sh --sf 10 --root ~/tpch-flat
#   ./prep-tpch-flat-data.sh --force              # rebuild even if present
#
# The result is <root>/lineitem_flat.parquet; point the suite at the directory:
#   cargo run --release -- --suite tpch_flat --source <root>
#
# Needs `duckdb` on PATH.

set -euo pipefail

root="$HOME/tpch-flat"
sf="1"
force=0

# Print the leading comment block (everything after the shebang up to the first
# non-comment line), so the help text tracks the header without magic line ranges.
usage() {
    awk 'NR==1 && /^#!/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

# A value-taking flag given as the last argument leaves $2 unset, which aborts
# with a cryptic "$2: unbound variable" under `set -u`; check before reading it.
need_val() { [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)    need_val "$@"; root="$2"; shift 2 ;;
        --sf)      need_val "$@"; sf="$2"; shift 2 ;;
        --force)   force=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH" >&2; exit 1; }

mkdir -p "$root"
out="$root/lineitem_flat.parquet"

if [[ -s "$out" && $force -eq 0 ]]; then
    # Non-emptiness alone is not "finished": a run killed mid-COPY leaves a
    # truncated parquet. Read it back to confirm it is a valid, complete file;
    # if the read fails, fall through and rebuild rather than reusing garbage.
    if rows=$(duckdb -noheader -list -c "SELECT count(*) FROM read_parquet('$out');" 2>/dev/null) \
        && [[ -n "$rows" ]]; then
        echo "have  $out ($(du -h "$out" | cut -f1), $rows rows); pass --force to rebuild"
        exit 0
    fi
    echo "existing $out is unreadable (partial/corrupt); rebuilding"
fi

echo "generating tpch sf=$sf and denormalising → $out"
rm -f "$out"

# dbgen builds the normalised tables; the COPY joins them into one lineitem-grain
# row set. The two nation/region pairs distinguish the supplier's nation/region
# (cn/cr below are the customer's, sn/sr the supplier's). Measures are cast to
# DOUBLE so every engine evaluates the same arithmetic.
duckdb <<SQL
INSTALL tpch; LOAD tpch;
CALL dbgen(sf=$sf);
COPY (
    SELECT
        l.l_orderkey::BIGINT             AS l_orderkey,
        l.l_partkey::INTEGER             AS l_partkey,
        l.l_suppkey::INTEGER             AS l_suppkey,
        l.l_linenumber::INTEGER          AS l_linenumber,
        l.l_quantity::DOUBLE             AS l_quantity,
        l.l_extendedprice::DOUBLE        AS l_extendedprice,
        l.l_discount::DOUBLE             AS l_discount,
        l.l_tax::DOUBLE                  AS l_tax,
        l.l_returnflag                   AS l_returnflag,
        l.l_linestatus                   AS l_linestatus,
        l.l_shipdate                     AS l_shipdate,
        l.l_commitdate                   AS l_commitdate,
        l.l_receiptdate                  AS l_receiptdate,
        l.l_shipinstruct                 AS l_shipinstruct,
        l.l_shipmode                     AS l_shipmode,
        o.o_orderstatus                  AS o_orderstatus,
        o.o_totalprice::DOUBLE           AS o_totalprice,
        o.o_orderdate                    AS o_orderdate,
        o.o_orderpriority                AS o_orderpriority,
        o.o_clerk                        AS o_clerk,
        o.o_shippriority::INTEGER        AS o_shippriority,
        c.c_custkey::INTEGER             AS c_custkey,
        c.c_name                         AS c_name,
        c.c_address                      AS c_address,
        c.c_phone                        AS c_phone,
        c.c_acctbal::DOUBLE              AS c_acctbal,
        c.c_mktsegment                   AS c_mktsegment,
        cn.n_name                        AS c_nation,
        cr.r_name                        AS c_region,
        s.s_suppkey::INTEGER             AS s_suppkey,
        s.s_name                         AS s_name,
        s.s_address                      AS s_address,
        s.s_phone                        AS s_phone,
        s.s_acctbal::DOUBLE              AS s_acctbal,
        sn.n_name                        AS s_nation,
        sr.r_name                        AS s_region,
        p.p_name                         AS p_name,
        p.p_mfgr                         AS p_mfgr,
        p.p_brand                        AS p_brand,
        p.p_type                         AS p_type,
        p.p_size::INTEGER                AS p_size,
        p.p_container                    AS p_container,
        p.p_retailprice::DOUBLE          AS p_retailprice,
        ps.ps_supplycost::DOUBLE         AS ps_supplycost
    FROM lineitem l
    JOIN orders   o  ON l.l_orderkey  = o.o_orderkey
    JOIN customer c  ON o.o_custkey   = c.c_custkey
    JOIN nation   cn ON c.c_nationkey = cn.n_nationkey
    JOIN region   cr ON cn.n_regionkey = cr.r_regionkey
    JOIN supplier s  ON l.l_suppkey   = s.s_suppkey
    JOIN nation   sn ON s.s_nationkey = sn.n_nationkey
    JOIN region   sr ON sn.n_regionkey = sr.r_regionkey
    JOIN part     p  ON l.l_partkey   = p.p_partkey
    JOIN partsupp ps ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
) TO '$out' (FORMAT PARQUET);
SQL

rows=$(duckdb -noheader -list -c "SELECT count(*) FROM read_parquet('$out');")
echo
echo "ready: $out ($(du -h "$out" | cut -f1), $rows rows)"
echo "run:   cargo run --release -- --suite tpch_flat --source $root"
