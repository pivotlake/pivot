#!/usr/bin/env bash
#
# prep-ssb-data.sh — build the Star Schema Benchmark parquet dataset the ssb
# suite reads (see setup.sql): one directory per table, each holding that
# table's parquet files.
#
# Usage:
#   ./prep-ssb-data.sh --scale 1 --output ~/ssb-sf1
#   ./prep-ssb-data.sh --scale 100 --output ~/ssb-sf100 --keep-tbl
#
# Clones and builds ssb-dbgen (the maintained eyalroz fork of the official
# generator), generates the .tbl files at the requested scale factor, then
# converts them to parquet with DuckDB. The intermediate .tbl files are
# deleted afterwards unless --keep-tbl is given (at SF100 they are ~60GB, so
# make sure the disk fits both them and the parquet while the script runs).
#
# Requires: git, cmake, a C compiler, duckdb on PATH.

set -euo pipefail

scale=1
output=""
keep_tbl=0
work_dir="${TMPDIR:-/tmp}/ssb-dbgen-work"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale)   scale="$2"; shift 2 ;;
        --output)  output="$2"; shift 2 ;;
        --work-dir) work_dir="$2"; shift 2 ;;
        --keep-tbl) keep_tbl=1; shift ;;
        -h|--help) sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$output" ]] || { echo "error: --output is required" >&2; exit 1; }
command -v duckdb >/dev/null || { echo "error: duckdb not found on PATH" >&2; exit 1; }

mkdir -p "$work_dir" "$output"

if [[ ! -x "$work_dir/ssb-dbgen/build/dbgen" ]]; then
    [[ -d "$work_dir/ssb-dbgen" ]] || git clone --depth 1 \
        https://github.com/eyalroz/ssb-dbgen.git "$work_dir/ssb-dbgen"
    cmake -S "$work_dir/ssb-dbgen" -B "$work_dir/ssb-dbgen/build" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$work_dir/ssb-dbgen/build"
fi

# dbgen writes its .tbl files into the cwd and only generates one table per
# invocation (-T a is not accepted despite the usage string).
cd "$work_dir/ssb-dbgen/build"
for t in c p s d l; do
    ./dbgen -f -q -s "$scale" -T "$t"
done

# dbgen ends every line with a trailing '|', which reads as one extra empty
# column; each schema below carries a trailing_sep column that is dropped on
# conversion. Money and quantity columns are integral in the SSB spec, and
# every key fits INTEGER at any published scale factor.
duckdb <<SQL
COPY (SELECT c_custkey, c_name, c_address, c_city, c_nation, c_region, c_phone,
             c_mktsegment
      FROM read_csv('customer.tbl', delim='|', header=false, columns={
          'c_custkey': 'INTEGER', 'c_name': 'VARCHAR', 'c_address': 'VARCHAR',
          'c_city': 'VARCHAR', 'c_nation': 'VARCHAR', 'c_region': 'VARCHAR',
          'c_phone': 'VARCHAR', 'c_mktsegment': 'VARCHAR',
          'trailing_sep': 'VARCHAR'}))
TO '$output/customer' (FORMAT parquet, PER_THREAD_OUTPUT true,
                       FILENAME_PATTERN 'customer.{i}');

COPY (SELECT s_suppkey, s_name, s_address, s_city, s_nation, s_region, s_phone
      FROM read_csv('supplier.tbl', delim='|', header=false, columns={
          's_suppkey': 'INTEGER', 's_name': 'VARCHAR', 's_address': 'VARCHAR',
          's_city': 'VARCHAR', 's_nation': 'VARCHAR', 's_region': 'VARCHAR',
          's_phone': 'VARCHAR', 'trailing_sep': 'VARCHAR'}))
TO '$output/supplier' (FORMAT parquet, PER_THREAD_OUTPUT true,
                       FILENAME_PATTERN 'supplier.{i}');

COPY (SELECT p_partkey, p_name, p_mfgr, p_category, p_brand1, p_color, p_type,
             p_size, p_container
      FROM read_csv('part.tbl', delim='|', header=false, columns={
          'p_partkey': 'INTEGER', 'p_name': 'VARCHAR', 'p_mfgr': 'VARCHAR',
          'p_category': 'VARCHAR', 'p_brand1': 'VARCHAR', 'p_color': 'VARCHAR',
          'p_type': 'VARCHAR', 'p_size': 'INTEGER', 'p_container': 'VARCHAR',
          'trailing_sep': 'VARCHAR'}))
TO '$output/part' (FORMAT parquet, PER_THREAD_OUTPUT true,
                   FILENAME_PATTERN 'part.{i}');

COPY (SELECT d_datekey, d_date, d_dayofweek, d_month, d_year, d_yearmonthnum,
             d_yearmonth, d_daynuminweek, d_daynuminmonth, d_daynuminyear,
             d_monthnuminyear, d_weeknuminyear, d_sellingseason,
             d_lastdayinweekfl, d_lastdayinmonthfl, d_holidayfl, d_weekdayfl
      FROM read_csv('date.tbl', delim='|', header=false, columns={
          'd_datekey': 'INTEGER', 'd_date': 'VARCHAR', 'd_dayofweek': 'VARCHAR',
          'd_month': 'VARCHAR', 'd_year': 'INTEGER', 'd_yearmonthnum': 'INTEGER',
          'd_yearmonth': 'VARCHAR', 'd_daynuminweek': 'INTEGER',
          'd_daynuminmonth': 'INTEGER', 'd_daynuminyear': 'INTEGER',
          'd_monthnuminyear': 'INTEGER', 'd_weeknuminyear': 'INTEGER',
          'd_sellingseason': 'VARCHAR', 'd_lastdayinweekfl': 'INTEGER',
          'd_lastdayinmonthfl': 'INTEGER', 'd_holidayfl': 'INTEGER',
          'd_weekdayfl': 'INTEGER', 'trailing_sep': 'VARCHAR'}))
TO '$output/date' (FORMAT parquet, PER_THREAD_OUTPUT true,
                   FILENAME_PATTERN 'date.{i}');

COPY (SELECT lo_orderkey, lo_linenumber, lo_custkey, lo_partkey, lo_suppkey,
             lo_orderdate, lo_orderpriority, lo_shippriority, lo_quantity,
             lo_extendedprice, lo_ordtotalprice, lo_discount, lo_revenue,
             lo_supplycost, lo_tax, lo_commitdate, lo_shipmode
      FROM read_csv('lineorder.tbl', delim='|', header=false, columns={
          'lo_orderkey': 'INTEGER', 'lo_linenumber': 'INTEGER',
          'lo_custkey': 'INTEGER', 'lo_partkey': 'INTEGER',
          'lo_suppkey': 'INTEGER', 'lo_orderdate': 'INTEGER',
          'lo_orderpriority': 'VARCHAR', 'lo_shippriority': 'INTEGER',
          'lo_quantity': 'INTEGER', 'lo_extendedprice': 'INTEGER',
          'lo_ordtotalprice': 'INTEGER', 'lo_discount': 'INTEGER',
          'lo_revenue': 'INTEGER', 'lo_supplycost': 'INTEGER',
          'lo_tax': 'INTEGER', 'lo_commitdate': 'INTEGER',
          'lo_shipmode': 'VARCHAR', 'trailing_sep': 'VARCHAR'}))
TO '$output/lineorder' (FORMAT parquet, PER_THREAD_OUTPUT true,
                        FILENAME_PATTERN 'lineorder.{i}');
SQL

if [[ "$keep_tbl" != "1" ]]; then
    rm -f customer.tbl supplier.tbl part.tbl date.tbl lineorder.tbl
fi

echo "dataset written to $output:"
for t in lineorder customer supplier part date; do
    n=$(duckdb -noheader -list -c "SELECT count(*) FROM '$output/$t/*.parquet'")
    echo "  $t: $n rows"
done
