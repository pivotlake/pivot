#!/usr/bin/env bash
#
# bench-tpch-flat.sh - run the tpch_flat suite through pivotdb and, side by side,
# DuckDB and ClickHouse over the same flat parquet, printing cold/hot timings and
# pivot's speedup per query.
#
# All three engines read one denormalised lineitem_flat.parquet (built by
# prep-tpch-flat-data.sh), so every query is a single-table scan - no joins. The
# suite's qNN.sql are run unchanged: DuckDB and ClickHouse each expose the parquet
# as a view named after the suite's CREATE TABLE (parsed from setup.sql).
#
# Usage:
#   ./bench-tpch-flat.sh --source ~/tpch-flat                 # all queries
#   ./bench-tpch-flat.sh --source ~/tpch-flat --query 4       # just q04
#   ./bench-tpch-flat.sh --source ~/tpch-flat --iterations 5
#   ./bench-tpch-flat.sh --source ~/tpch-flat --clickhouse clickhouse
#
# Each engine runs every iteration in one warm session-equivalent: pivot's server
# stays up across iterations; DuckDB and ClickHouse re-open the parquet per
# iteration (cold engine, warm OS cache after iteration 1), matching how the
# ClickBench harness here measures them. cold = iteration 1, hot = min of the rest.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite_dir="$here/tpch_flat"

source_path="$HOME/tpch-flat"
queries=""
iterations=5
clickhouse_bin=""

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
        --source)     need_val "$@"; source_path="$2"; shift 2 ;;
        --query)      need_val "$@"; queries="$2"; shift 2 ;;
        --iterations) need_val "$@"; iterations="$2"; shift 2 ;;
        --clickhouse) need_val "$@"; clickhouse_bin="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || { echo "error: --iterations must be a positive integer (got '$iterations')" >&2; exit 1; }
command -v duckdb >/dev/null 2>&1 || { echo "error: duckdb not on PATH" >&2; exit 1; }
[[ -d "$source_path" ]] || { echo "error: --source must be a directory of *.parquet" >&2; exit 1; }
# ClickHouse is optional: use --clickhouse, else clickhouse on PATH if present.
[[ -z "$clickhouse_bin" ]] && command -v clickhouse >/dev/null 2>&1 && clickhouse_bin="clickhouse"

# The table the suite scans, parsed from setup.sql's CREATE TABLE - DuckDB and
# ClickHouse expose the parquet under this name so the qNN.sql run unchanged.
table="$(sed -n 's/^[[:space:]]*CREATE TABLE[[:space:]]\{1,\}\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$suite_dir/setup.sql" | head -1)"
[[ -n "$table" ]] || { echo "error: could not parse table name from setup.sql" >&2; exit 1; }
parquet_glob="${source_path%/}/*.parquet"

# Build the list of query stems (q04) to run.
declare -a ids=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra raw <<< "$queries"
    for id in "${raw[@]}"; do
        id="${id#q}"
        # Reject non-numeric ids up front: `$((10#$id))` on e.g. "all" aborts
        # with a raw bash arithmetic error before the friendly check below.
        [[ "$id" =~ ^[0-9]+$ ]] || { echo "error: invalid query id '$id' (expected a number like 4 or q04)" >&2; exit 1; }
        printf -v stem 'q%02d' "$((10#$id))"
        [[ -f "$suite_dir/$stem.sql" ]] || { echo "error: no query $stem" >&2; exit 1; }
        ids+=("$stem")
    done
else
    # `-e` guards against a queryless dir, where the unmatched glob would
    # otherwise stay literal and add a bogus "q*" id.
    for f in "$suite_dir"/q*.sql; do [[ -e "$f" ]] || continue; ids+=("$(basename "$f" .sql)"); done
fi
[[ ${#ids[@]} -gt 0 ]] || { echo "error: no queries found in $suite_dir" >&2; exit 1; }

echo "suite: tpch_flat   table: $table   source: $parquet_glob"
echo "queries: ${#ids[@]}   iterations: $iterations"
engines="pivot, duckdb"; [[ -n "$clickhouse_bin" ]] && engines="$engines, clickhouse"
echo "engines: $engines"
echo

# Run pivot for one query, scraping "[i/N] Query qNN - Xms" lines into "cold hot"
# (cold = iter 1, hot = min of the rest, "-" if only one iteration). Uses the
# prebuilt binary in $PIVOT_BIN when set (e.g. on a box where `cargo run` can't
# build), else `cargo run --release`.
pivot_run() {  # pivot_run <stem>
    local errf out; errf="$(mktemp)"
    out="$(
        if [[ -n "${PIVOT_BIN:-}" ]]; then
            "$PIVOT_BIN" --suite tpch_flat --source "$source_path" \
                --query "$1" --iterations "$iterations" --skip-check 2>"$errf"
        else
            ( cd "$here" && cargo run --release -- --suite tpch_flat --source "$source_path" \
                --query "$1" --iterations "$iterations" --skip-check 2>"$errf" )
        fi \
        | awk '
            $1 ~ /^\[/ && $2 == "Query" { t = $NF; sub(/ms$/, "", t); t += 0; n++
                if (n == 1) cold = t; else if (t < hot || hot == "") hot = t }
            END { printf "%s %s", (n ? cold : "ERR"), (n > 1 ? hot : "-") }'
    )"
    # No timing parsed: the runner failed (build error, panic, bad --source).
    # Surface its stderr so a real outage is not mistaken for an unsupported query.
    if [[ "${out%% *}" == "ERR" && -s "$errf" ]]; then
        echo "  pivot failed for $1:" >&2
        tail -5 "$errf" | sed 's/^/    /' >&2
    fi
    rm -f "$errf"
    printf '%s' "$out"
}

# Run one engine query in a fresh process per iteration; $1 emits a command that
# prints a "real <secs>" line we scrape. cold = iter 1, hot = min of the rest.
timed_run() {  # timed_run <runner-fn> <stem>
    local runner="$1" stem="$2" sql cold="" hot="" i v
    sql="$(cat "$suite_dir/$stem.sql")"
    for ((i = 1; i <= iterations; i++)); do
        v="$("$runner" "$sql")"
        if [[ $i -eq 1 ]]; then
            # Only a failed cold read is fatal for the engine; a later empty
            # result is skipped so an already-valid cold time is not discarded.
            [[ -z "$v" ]] && { echo "ERR -"; return; }
            cold="$v"
        else
            [[ -z "$v" ]] && continue
            [[ -z "$hot" || $(awk -v a="$v" -v b="$hot" 'BEGIN{print (a<b)}') -eq 1 ]] && hot="$v"
        fi
    done
    echo "$cold ${hot:--}"
}

# DuckDB: a view over the parquet + the query, one fresh process per call, with
# .timer on; scrape its "Run Time (s): real X" into milliseconds.
duck_once() {  # duck_once <sql>
    printf 'CREATE VIEW %s AS SELECT * FROM read_parquet(%s);\n.timer on\n%s\n' \
        "$table" "'$parquet_glob'" "$1" \
    | duckdb 2>&1 | awk '/Run Time/ { t = $5 } END { if (t != "") printf "%.1f", t * 1000 }'
}

# ClickHouse: clickhouse-local exposing the parquet as a view, --time prints the
# query seconds to stderr; scrape the last bare number into milliseconds.
ch_once() {  # ch_once <sql>
    ( cd "$source_path" && "$clickhouse_bin" local --time --format=Null --query="
        CREATE VIEW $table AS SELECT * FROM file('*.parquet', Parquet);
        $1" ) 2>&1 | tr '\r' '\n' \
    | awk '/^[0-9]+(\.[0-9]+)?$/ { v = $1 } END { if (v != "") printf "%.1f", v * 1000 }'
}

# Header. Columns align with the row format below: engine timings are %13s
# (an 11-wide value plus "ms"), the speedup column %12s.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then grn=$'\033[32m'; red=$'\033[31m'; rst=$'\033[0m'; else grn=""; red=""; rst=""; fi
hdr=$(printf "%-6s %13s %13s" "query" "pivot(c)" "duckdb(c)")
[[ -n "$clickhouse_bin" ]] && hdr="$hdr$(printf " %13s" "clickh(c)")"
hdr="$hdr $(printf "%12s" "cold")  $(printf "%13s %13s" "pivot(h)" "duckdb(h)")"
[[ -n "$clickhouse_bin" ]] && hdr="$hdr$(printf " %13s" "clickh(h)")"
hdr="$hdr $(printf "%12s" "hot")"
echo "$hdr"
printf '%*s\n' "${#hdr}" '' | tr ' ' '-'

# speedup colour: green when pivot is at/above parity with the fastest competitor.
speed() {  # speed <pivot> <comp>
    local p="$1" c="$2"
    [[ "$p" == "ERR" || "$p" == "-" || "$c" == "-" || -z "$c" ]] && { printf "%12s" "-"; return; }
    awk -v p="$p" -v c="$c" -v g="$grn" -v r="$red" -v x="$rst" \
        'BEGIN {
            # A 0ms pivot timing means the query ran faster than the ms-granularity
            # clock; the true ratio is unbounded, so show a ">Nx" marker rather than
            # fabricating a number (matching benchmark.sh).
            if (p <= 0) { printf "%s%12s%s", g, ">99x", x; exit }
            s = c / p; col = (s >= 1.0) ? g : r; printf "%s%11.1fx%s", col, s, x
        }'
}

# The smaller of two timings (the faster competitor), or "-" when neither is a
# number. Non-numeric inputs (ERR / "-") drop out so only real timings compete.
fastest() {  # fastest <a> <b>
    awk -v a="$1" -v b="$2" 'BEGIN {
        m = ""
        if (a ~ /^[0-9.]+$/) m = a
        if (b ~ /^[0-9.]+$/ && (m == "" || b < m)) m = b
        print (m == "" ? "-" : m)
    }'
}

for stem in "${ids[@]}"; do
    read -r pc ph <<< "$(pivot_run "$stem")"
    read -r dc dh <<< "$(timed_run duck_once "$stem")"
    cc="-"; chh="-"
    [[ -n "$clickhouse_bin" ]] && read -r cc chh <<< "$(timed_run ch_once "$stem")"

    bc="$(fastest "$dc" "$cc")"; bh="$(fastest "$dh" "$chh")"

    row=$(printf "%-6s %11sms %11sms" "$stem" "$pc" "$dc")
    [[ -n "$clickhouse_bin" ]] && row="$row$(printf " %11sms" "$cc")"
    row="$row $(speed "$pc" "$bc")  "
    row="$row$(printf "%11sms %11sms" "$ph" "$dh")"
    [[ -n "$clickhouse_bin" ]] && row="$row$(printf " %11sms" "$chh")"
    row="$row $(speed "$ph" "$bh")"
    echo "$row"
done
