#!/usr/bin/env bash
#
# prep-ssb-data.sh - build the Star Schema Benchmark dataset the `ssb` suite
# reads. Three steps:
#
#   1. Clone and build ssb-dbgen (the maintained cmake fork,
#      github.com/eyalroz/ssb-dbgen) under <root>/ssb-dbgen/.
#   2. Generate the 5 tables as .tbl files into <root>/tbl/. The lineorder
#      fact table is generated as --chunks parallel dbgen processes; the four
#      dimension tables are small and generate in one process each.
#   3. Convert each .tbl to parquet with DuckDB into <root>/<table>/, one
#      directory per table, the layout `setup.sql` expects. Types match
#      setup.sql: BIGINT keys, INTEGER measures and yyyymmdd date keys,
#      VARCHAR text.
#
# At SF100 lineorder is ~600M rows: ~60 GB of .tbl text plus ~30 GB of
# parquet, so give <root> a disk with ~120 GB free. The .tbl files are kept
# for reuse; delete <root>/tbl to reclaim the space.
#
# Usage:
#   ./prep-ssb-data.sh                                   # SF100 → ~/ssb-sf100
#   ./prep-ssb-data.sh --scale-factor 10 --root ~/ssb-sf10
#   ./prep-ssb-data.sh --chunks 8
#
# Then run the suite (from the crate root):
#   cargo run --release -- --suite ssb --source <root> --iterations 3
#
# Idempotent: the dbgen build, existing .tbl files and existing parquet table
# directories are reused; --force regenerates everything.

set -euo pipefail

root="$HOME/ssb-sf100"
scale_factor=100
chunks="$(nproc)"
force=0

usage() { sed -n '3,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)         root="$2"; shift 2 ;;
        --scale-factor) scale_factor="$2"; shift 2 ;;
        --chunks)       chunks="$2"; shift 2 ;;
        --force)        force=1; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH" >&2; exit 1; }
command -v cmake  >/dev/null 2>&1 || { echo "error: cmake not on PATH" >&2; exit 1; }

dbgen_dir="$root/ssb-dbgen"
tbl_dir="$root/tbl"
mkdir -p "$tbl_dir"

echo "SSB prep (SF$scale_factor) → $root"

# 1. Build the generator.
if [[ -x "$dbgen_dir/build/dbgen" && $force -eq 0 ]]; then
    echo "  have  dbgen ($dbgen_dir/build/dbgen)"
else
    echo "  build dbgen (eyalroz/ssb-dbgen)"
    [[ -d "$dbgen_dir/.git" ]] || git clone --depth 1 https://github.com/eyalroz/ssb-dbgen.git "$dbgen_dir"
    cmake -S "$dbgen_dir" -B "$dbgen_dir/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$dbgen_dir/build" --target dbgen >/dev/null
fi
dbgen="$dbgen_dir/build/dbgen"

# dbgen reads dists.dss via DSS_CONFIG and writes .tbl files into DSS_PATH.
export DSS_CONFIG="$dbgen_dir/src"
[[ -f "$DSS_CONFIG/dists.dss" ]] || export DSS_CONFIG="$dbgen_dir"
[[ -f "$DSS_CONFIG/dists.dss" ]] || { echo "error: dists.dss not found under $dbgen_dir" >&2; exit 1; }
export DSS_PATH="$tbl_dir"

# 2. Generate the .tbl files.
if compgen -G "$tbl_dir/lineorder.tbl*" >/dev/null && [[ $force -eq 0 ]]; then
    echo "  have  .tbl files ($tbl_dir)"
else
    rm -f "$tbl_dir"/*.tbl*
    echo "  generate dimensions (customer, supplier, part, date)"
    for table_flag in c s p d; do
        "$dbgen" -f -s "$scale_factor" -T "$table_flag"
    done
    echo "  generate lineorder ($chunks parallel chunks)"
    if [[ "$chunks" -gt 1 ]]; then
        seq 1 "$chunks" \
            | xargs -P "$chunks" -I{} "$dbgen" -f -s "$scale_factor" -T l -C "$chunks" -S {}
    else
        "$dbgen" -f -s "$scale_factor" -T l
    fi
fi

# 3. Convert to parquet, one directory per table. The .tbl format is
# pipe-separated with a trailing pipe; the extra `extra_sep` column swallows it
# (and null_padding keeps a build without the trailing pipe working too).
convert() {
    local table="$1" glob="$2" columns="$3" per_thread="$4"
    local out_dir="$root/$table"
    if [[ -d "$out_dir" && -n "$(ls -A "$out_dir" 2>/dev/null)" && $force -eq 0 ]]; then
        echo "  have  $table parquet ($out_dir)"
        return
    fi
    echo "  convert $table → parquet"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    local target="'$out_dir/$table.parquet' (FORMAT PARQUET)"
    [[ "$per_thread" == "1" ]] && target="'$out_dir' (FORMAT PARQUET, PER_THREAD_OUTPUT true)"
    duckdb -c "
        COPY (
            SELECT * EXCLUDE (extra_sep)
            FROM read_csv('$tbl_dir/$glob', delim='|', header=false, null_padding=true,
                          columns={$columns, 'extra_sep':'VARCHAR'})
        ) TO $target;
    "
}

convert lineorder 'lineorder.tbl*' "
    'lo_orderkey':'BIGINT', 'lo_linenumber':'INTEGER', 'lo_custkey':'BIGINT',
    'lo_partkey':'BIGINT', 'lo_suppkey':'BIGINT', 'lo_orderdate':'INTEGER',
    'lo_orderpriority':'VARCHAR', 'lo_shippriority':'VARCHAR',
    'lo_quantity':'INTEGER', 'lo_extendedprice':'INTEGER',
    'lo_ordtotalprice':'INTEGER', 'lo_discount':'INTEGER',
    'lo_revenue':'INTEGER', 'lo_supplycost':'INTEGER', 'lo_tax':'INTEGER',
    'lo_commitdate':'INTEGER', 'lo_shipmode':'VARCHAR'" 1

convert customer 'customer.tbl' "
    'c_custkey':'BIGINT', 'c_name':'VARCHAR', 'c_address':'VARCHAR',
    'c_city':'VARCHAR', 'c_nation':'VARCHAR', 'c_region':'VARCHAR',
    'c_phone':'VARCHAR', 'c_mktsegment':'VARCHAR'" 0

convert supplier 'supplier.tbl' "
    's_suppkey':'BIGINT', 's_name':'VARCHAR', 's_address':'VARCHAR',
    's_city':'VARCHAR', 's_nation':'VARCHAR', 's_region':'VARCHAR',
    's_phone':'VARCHAR'" 0

convert part 'part.tbl' "
    'p_partkey':'BIGINT', 'p_name':'VARCHAR', 'p_mfgr':'VARCHAR',
    'p_category':'VARCHAR', 'p_brand1':'VARCHAR', 'p_color':'VARCHAR',
    'p_type':'VARCHAR', 'p_size':'INTEGER', 'p_container':'VARCHAR'" 0

convert date 'date.tbl' "
    'd_datekey':'INTEGER', 'd_date':'VARCHAR', 'd_dayofweek':'VARCHAR',
    'd_month':'VARCHAR', 'd_year':'INTEGER', 'd_yearmonthnum':'INTEGER',
    'd_yearmonth':'VARCHAR', 'd_daynuminweek':'INTEGER',
    'd_daynuminmonth':'INTEGER', 'd_daynuminyear':'INTEGER',
    'd_monthnuminyear':'INTEGER', 'd_weeknuminyear':'INTEGER',
    'd_sellingseason':'VARCHAR', 'd_lastdayinweekfl':'INTEGER',
    'd_lastdayinmonthfl':'INTEGER', 'd_holidayfl':'INTEGER',
    'd_weekdayfl':'INTEGER'" 0

echo
echo "ready: --suite ssb --source $root"
