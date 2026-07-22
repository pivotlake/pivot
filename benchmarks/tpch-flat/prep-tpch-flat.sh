#!/usr/bin/env bash
#
# prep-tpch-flat.sh - build the denormalized "flat" TPC-H dataset the `tpch`
# benchmark suite reads. Two steps:
#
#   1. Generate the 8 normalized TPC-H tables as parquet with `tpchgen-cli`
#      (the fast Rust generator, github.com/clflushopt/tpchgen-rs) into
#      <root>/base/.
#   2. Join them lineitem-centric into a single wide table with DuckDB and write
#      it to <root>/flat/ as a directory of parquet files. Money and quantity
#      columns are cast to DOUBLE; both nation+region names are folded in on the
#      customer and supplier sides. Column order matches benchmarks/tpch/setup.sql.
#
# The flat table is one row per lineitem (dims join 1:1, so no fan-out): at
# SF100 that is ~600M rows. Point the suite at it:
#
#   cargo run --release -- --suite tpch-flat --source <root>/flat --iterations 3
#
# The denormalizing join is memory-hungry; DuckDB spills to --temp-dir. Give it a
# fast disk with a few hundred GB free.
#
# Usage:
#   ./prep-tpch-flat.sh                              # SF100 → ~/tpch-data
#   ./prep-tpch-flat.sh --scale-factor 10 --root ~/tpch-sf10
#   ./prep-tpch-flat.sh --threads 32 --temp-dir /mnt/fast/duck-tmp
#   ./prep-tpch-flat.sh --tpchgen ~/.cargo/bin/tpchgen-cli
#
# Idempotent: existing base tables and the built flat dir are reused; --force
# rebuilds the flat dir from the (regenerated) base tables.
#
# Install the generator first if needed: `cargo install tpchgen-cli`.

set -euo pipefail

root="$HOME/tpch-data"
scale_factor=100
threads=""
temp_dir=""
tpchgen="tpchgen-cli"
force=0

usage() { sed -n '3,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)         root="$2"; shift 2 ;;
        --scale-factor) scale_factor="$2"; shift 2 ;;
        --threads)      threads="$2"; shift 2 ;;
        --temp-dir)     temp_dir="$2"; shift 2 ;;
        --tpchgen)      tpchgen="$2"; shift 2 ;;
        --force)        force=1; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

command -v "$tpchgen" >/dev/null 2>&1 || {
    echo "error: '$tpchgen' not on PATH - install with 'cargo install tpchgen-cli' or pass --tpchgen <path>" >&2
    exit 1
}
command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH" >&2; exit 1; }

base_dir="$root/base"
flat_dir="$root/flat"
mkdir -p "$base_dir"

echo "TPC-H flat prep (SF$scale_factor) → $root"

# 1. Generate the normalized base tables as parquet. tpchgen-cli writes one
#    <table>.parquet per table into --output-dir.
if [[ -s "$base_dir/lineitem.parquet" && $force -eq 0 ]]; then
    echo "  have  base tables ($base_dir)"
else
    echo "  generate base tables (tpchgen-cli)"
    "$tpchgen" --scale-factor "$scale_factor" --output-dir "$base_dir" --format=parquet
fi

# 2. Denormalize lineitem-centric into the flat table.
if [[ -d "$flat_dir" && -n "$(ls -A "$flat_dir" 2>/dev/null)" && $force -eq 0 ]]; then
    echo "  have  flat table ($flat_dir)"
    echo
    echo "ready: --suite tpch-flat --source $flat_dir"
    exit 0
fi

echo "  build flat table (DuckDB denormalize → $flat_dir)"
rm -rf "$flat_dir"
mkdir -p "$flat_dir"

# PER_THREAD_OUTPUT writes one parquet per thread into the directory, which pivot
# globs like any partitioned source.
pragma_sql=""
[[ -n "$threads" ]]  && pragma_sql+="PRAGMA threads=$threads;"$'\n'
[[ -n "$temp_dir" ]] && { mkdir -p "$temp_dir"; pragma_sql+="PRAGMA temp_directory='$temp_dir';"$'\n'; }

duckdb <<SQL
${pragma_sql}
COPY (
    SELECT
        -- lineitem
        l.l_orderkey,
        l.l_partkey,
        l.l_suppkey,
        l.l_linenumber,
        l.l_quantity::DOUBLE      AS l_quantity,
        l.l_extendedprice::DOUBLE AS l_extendedprice,
        l.l_discount::DOUBLE      AS l_discount,
        l.l_tax::DOUBLE           AS l_tax,
        l.l_returnflag,
        l.l_linestatus,
        l.l_shipdate,
        l.l_commitdate,
        l.l_receiptdate,
        l.l_shipinstruct,
        l.l_shipmode,
        l.l_comment,
        -- orders
        o.o_custkey,
        o.o_orderstatus,
        o.o_totalprice::DOUBLE    AS o_totalprice,
        o.o_orderdate,
        o.o_orderpriority,
        o.o_clerk,
        o.o_shippriority,
        o.o_comment,
        -- customer
        c.c_name,
        c.c_address,
        c.c_nationkey,
        c.c_phone,
        c.c_acctbal::DOUBLE       AS c_acctbal,
        c.c_mktsegment,
        c.c_comment,
        -- part
        p.p_name,
        p.p_mfgr,
        p.p_brand,
        p.p_type,
        p.p_size,
        p.p_container,
        p.p_retailprice::DOUBLE   AS p_retailprice,
        p.p_comment,
        -- supplier
        s.s_name,
        s.s_address,
        s.s_nationkey,
        s.s_phone,
        s.s_acctbal::DOUBLE       AS s_acctbal,
        s.s_comment,
        -- partsupp
        ps.ps_availqty,
        ps.ps_supplycost::DOUBLE  AS ps_supplycost,
        ps.ps_comment,
        -- geography (names on both sides)
        cn.n_name AS c_nation,
        cr.r_name AS c_region,
        sn.n_name AS s_nation,
        sr.r_name AS s_region
    FROM read_parquet('$base_dir/lineitem.parquet') l
    JOIN read_parquet('$base_dir/orders.parquet')   o  ON l.l_orderkey  = o.o_orderkey
    JOIN read_parquet('$base_dir/customer.parquet') c  ON o.o_custkey   = c.c_custkey
    JOIN read_parquet('$base_dir/part.parquet')     p  ON l.l_partkey   = p.p_partkey
    JOIN read_parquet('$base_dir/supplier.parquet') s  ON l.l_suppkey   = s.s_suppkey
    JOIN read_parquet('$base_dir/partsupp.parquet') ps ON ps.ps_partkey = l.l_partkey
                                                      AND ps.ps_suppkey = l.l_suppkey
    JOIN read_parquet('$base_dir/nation.parquet')   cn ON c.c_nationkey = cn.n_nationkey
    JOIN read_parquet('$base_dir/region.parquet')   cr ON cn.n_regionkey = cr.r_regionkey
    JOIN read_parquet('$base_dir/nation.parquet')   sn ON s.s_nationkey = sn.n_nationkey
    JOIN read_parquet('$base_dir/region.parquet')   sr ON sn.n_regionkey = sr.r_regionkey
) TO '$flat_dir' (FORMAT PARQUET, PER_THREAD_OUTPUT true);
SQL

echo
echo "ready: --suite tpch-flat --source $flat_dir"
