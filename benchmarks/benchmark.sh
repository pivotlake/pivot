#!/usr/bin/env bash
#
# benchmark.sh — run the ClickBench queries through pivotdb and, optionally, one
# or both comparison engines (DuckDB, ClickHouse), printing a side-by-side table
# of cold and hot timings plus the speedup (how many times faster pivot is than
# the fastest competitor) per query.
#
# pivot always runs (via `just pgo-use run --release -- ...`, so generate the PGO
# profile first with `just pgo-gen ...`). DuckDB is opt-in with --duckdb (or
# --native) and driven by run-duckdb.sh; ClickHouse is opt-in with --clickhouse
# and driven by run-clickhouse.sh. A bare run is pivot-only.
#
# pivot runs its whole query set first, then DuckDB, then ClickHouse.
#
# By default each engine runs the whole query set in one session: pivot boots
# its server once and runs every query, DuckDB likewise. The page cache is
# dropped once up front, so only the first query is a true cold read; the rest
# measure a warm-cache session.
#
# With --restart-server each query is isolated — the pivot server is restarted
# (and DuckDB re-launched) per query, with the page cache dropped before each —
# so every query's iteration 1 is a true cold read. Slower, but the fairest
# cold comparison. (Cache drop is Linux only, via sudo.)
#
# The speedup column is green when pivot is at or above parity (>= 1.0x) and
# red when it is slower.
#
# Usage:
#   ./benchmark.sh --source ~/hits                     # pivot only
#   ./benchmark.sh --source ~/hits --duckdb            # pivot vs DuckDB
#   ./benchmark.sh --source ~/hits --duckdb --clickhouse ~/clickhouse  # all three
#   ./benchmark.sh --hits ~/hits --duckdb --query 7,20 # subset; --hits == --source
#   ./benchmark.sh --source ~/hits --iterations 5      # 1 cold + 4 hot runs
#   ./benchmark.sh --source ~/hits --sleep 500         # 500ms between iterations
#   ./benchmark.sh --source ~/hits --restart-server    # isolate each query
#   ./benchmark.sh --source ~/hits --skip-check        # don't verify pivot output
#   ./benchmark.sh --source ~/hits --no-drop-caches    # skip the cache drop
#   ./benchmark.sh --source ~/single --native ~/hits.db  # pivot parquet vs DuckDB native
#   ./benchmark.sh --source ~/hits --duckdb-process single  # DuckDB engine-warm (symmetric)
#   ./benchmark.sh --source ~/hits --clickhouse ~/clickhouse  # add ClickHouse-over-parquet
#
# --source/--hits accepts a directory (globbed for *.parquet), a single
# .parquet file, or an explicit glob.
#
# --clickhouse <binary> adds ClickHouse as a third engine. By default it reads
# the parquet --source via the official clickhouse-parquet method (clickhouse-local
# over File(Parquet, ...)). With --clickhouse-native it instead queries the running
# ClickHouse server's native MergeTree `hits` table (the official clickhouse method;
# set it up first with prep-clickhouse-native.sh) — a persistent warm server, like
# pivot. Either way the table gains clickh(c)/clickh(h) columns and the score a
# clickhouse row. See run-clickhouse.sh.
#
# --duckdb adds DuckDB as a comparison engine (off by default), reading the
# parquet --source. --native <db> instead points DuckDB at a persistent .db's
# native `hits` table (ClickBench-native style) and implies --duckdb; pivot still
# reads --source, so that compares pivot-on-parquet to DuckDB-on-native.
#
# --duckdb-process per-iteration|single  (default per-iteration) — only affects
# the DuckDB side. per-iteration spawns a fresh duckdb process per iteration,
# exactly how ClickBench measures embedded DuckDB (engine-cold each hot run).
# ⚠️ That default is UNFAIR to DuckDB here: pivot's iterations run against ONE
# warm, persistent server while DuckDB is forced engine-cold every iteration.
# `single` runs a query's iterations in one duckdb process (engine-warm, like
# pivot's server) for a symmetric comparison — but it is NOT how ClickBench
# measures, and DuckDB looks far faster on light queries. See run-duckdb.sh.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite_dir="$here/clickbench"

source_path=""
native_db=""
queries=""
iterations=3
drop_caches=1
restart_server=0
sleep_ms=0
skip_check=0
duckdb_process="per-iteration"
duck_enabled=0
clickhouse_bin=""
clickhouse_native=0

usage() {
    sed -n '3,65p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source|--hits)  source_path="$2"; shift 2 ;;
        --duckdb)         duck_enabled=1; shift ;;
        --native)         native_db="$2"; duck_enabled=1; shift 2 ;;  # native implies DuckDB
        --duckdb-process) duckdb_process="$2"; shift 2 ;;
        --clickhouse)     clickhouse_bin="$2"; shift 2 ;;
        --clickhouse-native) clickhouse_native=1; shift ;;
        --query)          queries="$2"; shift 2 ;;
        --iterations)     iterations="$2"; shift 2 ;;
        --sleep)          sleep_ms="$2"; shift 2 ;;
        --skip-check)     skip_check=1; shift ;;
        --restart-server) restart_server=1; shift ;;
        --no-drop-caches) drop_caches=0; shift ;;
        -h|--help)        usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$source_path" ]]; then
    echo "error: --source/--hits is required" >&2
    usage 1
fi

# pivot-bench flag: skip its output-vs-expected check (DuckDB doesn't verify).
skip_flag=""
[[ $skip_check -eq 1 ]] && skip_flag="--skip-check"

# Build the list of query IDs to run (stems like "q07"), accepting "7", "q07",
# or nothing (= every qNN.sql in the suite).
declare -a ids=()
if [[ -n "$queries" ]]; then
    IFS=',' read -ra raw <<< "$queries"
    for id in "${raw[@]}"; do
        id="${id#q}"
        printf -v stem 'q%02d' "$((10#$id))"
        [[ -f "$suite_dir/$stem.sql" ]] || { echo "error: no query $stem" >&2; exit 1; }
        ids+=("$stem")
    done
else
    for f in "$suite_dir"/q*.sql; do
        [[ "$f" == *-duckdb.sql ]] && continue   # DuckDB-only overrides, not queries
        ids+=("$(basename "$f" .sql)")
    done
fi

# Flush the Linux page cache so the next run reads cold from disk, the way
# ClickBench measures. Needs root, so it goes through sudo.
flush_page_cache() {
    [[ "$drop_caches" == "1" ]] || return 0
    # Don't let a sudo hiccup abort the whole run under set -e — warn and go on.
    if ! { sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }; then
        echo "  warning: page-cache drop failed (need sudo/Linux); continuing" >&2
    fi
}

# Run pivot for one or more queries (comma-joined), raw output on stdout.
# pivot-bench prints "[i/N] Query q07 — 12ms" lines (integer ms).
pivot_invoke() {
    local IFS=,
    # Unless cache-dropping is disabled, have pivot-bench evict its file cache
    # *and* the OS page cache before each query, so every query's iteration 1 is
    # a true cold read within the one warm server session (no restart needed).
    local cold_flag=""
    [[ $drop_caches -eq 1 ]] && cold_flag="--drop-caches"
    ( cd "$here" && just pgo-use run --release -- \
        --source "$source_path" --query "$*" --iterations "$iterations" \
        --sleep "$sleep_ms" $skip_flag $cold_flag ) 2>&1
}

# Run DuckDB for one or more queries (comma-joined), raw output on stdout.
# run-duckdb.sh prints "Run Time (s): real 0.008 ..." lines. Unless dropping is
# disabled, let run-duckdb.sh flush the page cache before each query (it drops
# once, then runs each iteration in a fresh duckdb process — exactly the official
# ClickBench harness), so iteration 1 is the cold read and the rest are hot.
duck_invoke() {
    local IFS=,
    local duck_cache_flag="--no-drop-caches"
    [[ $drop_caches -eq 1 ]] && duck_cache_flag=""
    # In native mode DuckDB reads its own .db storage (--native); otherwise the
    # same parquet --source pivot reads. Pivot's source is unchanged either way,
    # so native mode is pivot-on-parquet vs DuckDB-on-native by design.
    local duck_source=(--source "$source_path")
    [[ -n "$native_db" ]] && duck_source=(--native "$native_db")
    "$here/run-duckdb.sh" "${duck_source[@]}" --query "$*" \
        --iterations "$iterations" --sleep "$sleep_ms" \
        --duckdb-process "$duckdb_process" $duck_cache_flag 2>&1
}

# Run ClickHouse-over-parquet for one or more queries. ClickHouse always reads
# the parquet --source (the upstream clickhouse-parquet method), even in --native
# mode (where DuckDB reads its native .db) — so it's pivot/DuckDB vs ClickHouse
# all on the same parquet. run-clickhouse.sh prints the same "Run Time" lines.
ch_invoke() {
    local IFS=,
    local ch_cache_flag="--no-drop-caches"
    [[ $drop_caches -eq 1 ]] && ch_cache_flag=""
    # --clickhouse-native: query the running ClickHouse server's MergeTree (the
    # official native engine); otherwise clickhouse-local over the parquet --source.
    local ch_src=(--source "$source_path")
    [[ $clickhouse_native -eq 1 ]] && ch_src=(--native)
    "$here/run-clickhouse.sh" "${ch_src[@]}" --query "$*" \
        --iterations "$iterations" --sleep "$sleep_ms" \
        --clickhouse "$clickhouse_bin" $ch_cache_flag 2>&1
}

# Parse pivot output (any number of queries) into "id cold hot it1 it2 …"
# lines: cold is iteration 1, hot is the min of the rest ("-" if there is no
# rest), then every per-iteration ms value. ClickBench's hot metric is the
# minimum of the warm runs, not their mean. The "=== Query qNN ===" headers are
# skipped — only "[i/N] ..." lines count. (The table reads just cold/hot; the
# trailing iterations are for the per-query recap.)
parse_pivot() {
    awk '
    $1 ~ /^\[/ && $2 == "Query" {
        id = $3
        t = $NF; sub(/ms$/, "", t); t += 0
        split($1, a, "/"); itn = a[1]; gsub(/[^0-9]/, "", itn)
        if (!(id in seen)) { seen[id] = 1; order[++n] = id }
        if (itn + 0 == 1) cold[id] = t
        else { if (!(id in cnt) || t < hmin[id]) hmin[id] = t; cnt[id]++ }
        iters[id] = (id in iters ? iters[id] " " : "") sprintf("%.1f", t)
    }
    END {
        for (i = 1; i <= n; i++) {
            k = order[i]
            h = (cnt[k] > 0) ? sprintf("%.1f", hmin[k]) : "-"
            printf "%s %.1f %s %s\n", k, cold[k], h, iters[k]
        }
    }'
}

# Parse run-duckdb.sh output into "id cold hot it1 it2 …" lines (seconds → ms).
# Hot is the min of the warm runs (ClickBench's hot metric), not their mean.
parse_duck() {
    awk '
    /^=== / { id = $2; if (!(id in seen)) { seen[id] = 1; order[++n] = id } idx = 0; next }
    /Run Time/ && id != "" {
        v = $5 * 1000; idx++
        if (idx == 1) cold[id] = v
        else { if (!(id in cnt) || v < hmin[id]) hmin[id] = v; cnt[id]++ }
        iters[id] = (id in iters ? iters[id] " " : "") sprintf("%.1f", v)
    }
    END {
        for (i = 1; i <= n; i++) {
            k = order[i]
            if (!(k in cold)) { printf "%s ERR ERR\n", k; continue }
            h = (cnt[k] > 0) ? sprintf("%.1f", hmin[k]) : "-"
            printf "%s %.1f %s %s\n", k, cold[k], h, iters[k]
        }
    }'
}

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then color=1; else color=0; fi
if [[ -t 1 ]]; then tty=1; else tty=0; fi

pivot_tsv="$(mktemp)"; duck_tsv="$(mktemp)"; ch_tsv="$(mktemp)"
data="$(mktemp)"; spin_out="$(mktemp)"
trap 'rm -f "$pivot_tsv" "$duck_tsv" "$ch_tsv" "$data" "$spin_out"' EXIT
# ClickHouse is an optional 3rd engine (always over parquet, like upstream
# clickhouse-parquet). Enabled by --clickhouse; ch_tsv stays empty otherwise.
ch_enabled=0; [[ -n "$clickhouse_bin" ]] && ch_enabled=1

# Run a command in the background while animating a one-line spinner, then wait
# for it. The command's stdout lands in $spin_out for the caller to read; its
# stderr (build/server noise) is discarded. The spinner scrapes the live output
# for the most recent "qNN" marker so it shows which query is running right now
# (both engines print "=== Query qNN ===" as they reach each one). On a non-tty
# we just print a static line. Failures are swallowed — a missing query becomes
# "ERR" downstream.
frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
spin() {
    local label="$1"; shift
    "$@" >"$spin_out" 2>/dev/null &
    local pid=$! i=0 cur
    if [[ $tty -eq 1 ]]; then
        while kill -0 "$pid" 2>/dev/null; do
            # Track only the live "=== Query qNN ===" / "=== qNN ===" headers
            # each engine prints when it *starts* a query. Matching bare qNN
            # tokens anywhere would also catch the cargo "Running" line and the
            # end-of-run comparison table, making the label jump around.
            # `|| true`: no header yet (build/startup) → grep exits 1 → set -e.
            cur=$(grep -oE '^=== (Query )?q[0-9]+' "$spin_out" 2>/dev/null \
                | grep -oE 'q[0-9]+' | tail -1) || true
            printf '\r\033[K  %s Running %s  %s' \
                "${frames[i++ % ${#frames[@]}]}" "$label" "${cur:-…}"
            sleep 0.1
        done
        printf '\r\033[K'
    else
        printf '  Running %s …\n' "$label"
    fi
    wait "$pid" 2>/dev/null || true
}

tick=$([[ $color -eq 1 ]]  && printf '\033[32m✓\033[0m' || printf 'done')
cross=$([[ $color -eq 1 ]] && printf '\033[31m✗\033[0m' || printf 'FAIL')

# Echo the parsed "id cold hot it1 it2 …" lines on stdin as tidy result rows,
# listing every iteration (the first is the cold run, the rest are hot).
print_results() {
    local id c h iters
    while read -r id c h iters; do
        if [[ "$c" == "ERR" || -z "$iters" ]]; then
            printf '  %s %-4s  %s\n' "$tick" "$id" "(no result)"
        else
            printf '  %s %-4s  %s ms\n' "$tick" "$id" "${iters// /, }"
        fi
    done
}

# True if the parsed chunk in $1 has a real (non-ERR) row for every id in $2….
chunk_ok() {
    local cf="$1"; shift
    local id
    for id in "$@"; do
        awk -v q="$id" '$1 == q && $2 != "ERR" { ok = 1 } END { exit ok ? 0 : 1 }' \
            "$cf" || return 1
    done
    return 0
}

# A run produced no usable timings — surface the captured output (build errors,
# panics, planner errors, …) indented, so the failure is actually visible.
show_error() {
    local label="$1" what="$2"
    printf '  %s %s %s failed — captured output:\n' "$cross" "$label" "$what"
    if [[ -s "$spin_out" ]]; then
        sed 's/^/      /' "$spin_out"
    else
        printf '      (no output captured)\n'
    fi
}

# Collect one engine's timings. $1 = label, $2 = invoke fn, $3 = parse fn,
# $4 = out tsv. Honours --restart-server: per-query (cache drop each) vs one
# session (single upfront drop). When a run yields no usable timings, the
# captured output is printed so the underlying error is visible.
collect() {
    local label="$1" invoke="$2" parse="$3" out="$4"
    echo "▸ $label"
    : > "$out"
    local chunk; chunk="$(mktemp)"
    if [[ $restart_server -eq 1 ]]; then
        for id in "${ids[@]}"; do
            flush_page_cache
            spin "$label" "$invoke" "$id"
            "$parse" < "$spin_out" > "$chunk"
            cat "$chunk" >> "$out"
            chunk_ok "$chunk" "$id" || show_error "$label" "$id"
        done
    else
        flush_page_cache
        spin "$label" "$invoke" "${ids[@]}"
        "$parse" < "$spin_out" > "$out"
        chunk_ok "$out" "${ids[@]}" || show_error "$label" "one or more queries"
    fi
    rm -f "$chunk"
    print_results < "$out"
}

case "$duckdb_process" in
    per-iteration|single) ;;
    *) echo "error: --duckdb-process must be per-iteration|single (got '$duckdb_process')" >&2; usage 1 ;;
esac

mode=$([[ $restart_server -eq 1 ]] && echo "restart per query" || echo "one session")
echo "queries: ${#ids[@]}, iterations: $iterations, mode: $mode, drop_caches: $drop_caches"
engines="pivot"; [[ $duck_enabled -eq 1 ]] && engines="$engines, duckdb"; [[ $ch_enabled -eq 1 ]] && engines="$engines, clickhouse"
echo "engines: $engines"
if [[ $duck_enabled -eq 1 ]]; then
    dp_note=$([[ "$duckdb_process" == "per-iteration" ]] && echo "ClickBench-faithful; unfair to DuckDB" || echo "engine-warm; symmetric, non-ClickBench")
    echo "duckdb-process: $duckdb_process ($dp_note)"
fi
echo "pivot source:  $source_path"
if [[ $duck_enabled -eq 1 ]]; then
    if [[ -n "$native_db" ]]; then
        echo "duckdb source: native db $native_db (hits table)"
    else
        echo "duckdb source: $source_path"
    fi
fi
if [[ $ch_enabled -eq 1 ]]; then
    if [[ $clickhouse_native -eq 1 ]]; then
        echo "clickhouse:    $clickhouse_bin NATIVE MergeTree server (official clickhouse)"
    else
        echo "clickhouse:    $clickhouse_bin over $source_path/*.parquet (official clickhouse-parquet)"
    fi
fi
[[ $clickhouse_native -eq 1 ]] && echo "isolation:     ClickHouse server stopped while pivot/DuckDB run (started only for its own phase)"
echo

# ISOLATION: a persistent ClickHouse server (native mode) must NOT be resident
# while pivot/DuckDB are measured — its memory footprint and post-load background
# merges contaminate their cold and hot numbers. So with --clickhouse-native we
# stop the CH server before pivot/DuckDB and start it only for ClickHouse's own
# phase (ClickBench measures each engine in isolation). ClickHouse-parquet
# (clickhouse-local) has no resident server, so there is nothing to manage there.
ch_server() {   # ch_server start|stop  — no-op unless --clickhouse-native
    [[ $clickhouse_native -eq 1 ]] || return 0
    if [[ "$1" == stop ]]; then
        sudo clickhouse stop >/dev/null 2>&1 || true
    else
        sudo clickhouse start >/dev/null 2>&1 || true
        local i; for i in $(seq 1 60); do
            "$clickhouse_bin" client --query "SELECT 1" >/dev/null 2>&1 && return 0; sleep 1
        done
        echo "  warning: clickhouse server did not come up for its phase" >&2
    fi
}

ch_server stop                  # ensure the CH server is down while pivot/DuckDB run
collect pivot  pivot_invoke parse_pivot "$pivot_tsv"
[[ $duck_enabled -eq 1 ]] && collect DuckDB duck_invoke parse_duck "$duck_tsv"
if [[ $ch_enabled -eq 1 ]]; then
    ch_server start             # bring CH up ONLY for its own measurement
    # parse_duck reads the shared "=== qNN ===" / "Run Time (s): real X" format.
    collect ClickHouse ch_invoke parse_duck "$ch_tsv"
    ch_server stop              # leave it down so it never contends with a later run
fi
echo

# Merge the result sets by query id, in canonical order. Columns: id dc dh pc ph
# cc chh (duckdb/pivot/clickhouse cold+hot). An engine that is enabled but missing
# a query is "ERR" (it ran and failed); an engine that wasn't run at all (its flag
# off → empty tsv) is "-" (not present), so the table/score skip it cleanly.
awk -v order="$(IFS=,; echo "${ids[*]}")" -v duck="$duck_enabled" -v ch="$ch_enabled" \
    -v df="$duck_tsv" -v pf="$pivot_tsv" -v cf="$ch_tsv" '
FILENAME == df { dc[$1] = $2; dh[$1] = $3; next }
FILENAME == pf { pc[$1] = $2; ph[$1] = $3; next }
FILENAME == cf { cc[$1] = $2; chh[$1] = $3; next }
END {
    n = split(order, o, ",")
    for (i = 1; i <= n; i++) {
        k = o[i]
        printf "%s %s %s %s %s %s %s\n", k, \
            (k in dc ? dc[k] : (duck ? "ERR" : "-")), (k in dh ? dh[k] : (duck ? "ERR" : "-")), \
            (k in pc ? pc[k] : "ERR"), (k in ph ? ph[k] : "ERR"), \
            (k in cc ? cc[k] : (ch ? "ERR" : "-")), (k in chh ? chh[k] : (ch ? "ERR" : "-"))
    }
}' "$duck_tsv" "$pivot_tsv" "$ch_tsv" > "$data"

# Render the comparison table. A column is shown per engine actually run (pivot
# always; duckdb with --duckdb/--native; clickhouse with --clickhouse). The
# cold/hot "speedup" column is how many times faster pivot is than the FASTEST
# competitor present (so it reduces to duckdb/pivot in the classic 2-engine run);
# it is omitted when pivot runs alone. Green = pivot at/above parity, red = slower.
awk -v color="$color" -v duck="$duck_enabled" -v ch="$ch_enabled" '
function isnum(x) { return x ~ /^[0-9]+(\.[0-9]+)?$/ }
function tm(v) { return (v == "-" || v == "ERR") ? sprintf("%11s", v) : sprintf("%9sms", v) }
function bestcomp(d, c,   b) {   # fastest present competitor value, "" if none
    b = ""
    if (duck && isnum(d)) b = d
    if (ch && isnum(c) && (b == "" || c < b)) b = c
    return b
}
function speed(comp, piv) {
    if (comp == "" || !isnum(piv)) return "-"
    if (piv + 0 <= 0) return ">99x"          # pivot too fast to measure at ms granularity
    return sprintf("%.1fx", comp / piv)
}
function colored(txt, comp, piv,   col, vis) {
    vis = sprintf("%8s", txt)
    if (!color || txt == "-" || txt == ">99x") {
        if (txt == ">99x" && color) return green vis rst
        return vis
    }
    col = (comp / piv >= 1.0) ? green : red
    return col vis rst
}
BEGIN {
    green = color ? "\033[32m" : ""
    red   = color ? "\033[31m" : ""
    rst   = color ? "\033[0m"  : ""
    comp = (duck || ch)          # is there any competitor to show a speedup against
    h = sprintf("%-6s", "query")
    if (duck) h = h sprintf(" %11s", "duckdb(c)")
    h = h sprintf(" %11s", "pivot(c)")
    if (ch)   h = h sprintf(" %11s", "clickh(c)")
    if (comp) h = h sprintf(" %9s", "cold")
    h = h "  "
    if (duck) h = h sprintf(" %11s", "duckdb(h)")
    h = h sprintf(" %11s", "pivot(h)")
    if (ch)   h = h sprintf(" %11s", "clickh(h)")
    if (comp) h = h sprintf(" %9s", "hot")
    print h
    dash = ""; n = length(h); for (i = 0; i < n; i++) dash = dash "-"; print dash
}
{
    id=$1; dc=$2; dh=$3; pc=$4; ph=$5; cc=$6; chh=$7
    bcc = bestcomp(dc, cc); bch = bestcomp(dh, chh)
    row = sprintf("%-6s", id)
    if (duck) row = row " " tm(dc)
    row = row " " tm(pc)
    if (ch)   row = row " " tm(cc)
    if (comp) row = row " " colored(speed(bcc, pc), bcc, pc)
    row = row "  "
    if (duck) row = row " " tm(dh)
    row = row " " tm(ph)
    if (ch)   row = row " " tm(chh)
    if (comp) row = row " " colored(speed(bch, ph), bch, ph)
    print row
}
' "$data"

# ClickBench-style weighted score: the geometric mean over queries of
# (t + 10ms) / (best_for_that_query + 10ms) — ClickBench's summary metric. The
# +10ms regularises near-zero queries; lower is better and 1.00 means fastest
# on every scored query. The per-query baseline is the fastest engine present.
# So all engines stay comparable, a query is scored only when EVERY active engine
# has a timing there (intersection); ERR / single-iteration "-" drop it for all.
awk -v color="$color" -v duck="$duck_enabled" -v ch="$ch_enabled" '
function isnum(x) { return x ~ /^[0-9]+(\.[0-9]+)?$/ }
function gm(logsum, n) { return n ? exp(logsum / n) : 0 }
function cell(mine, best, n,   s) {
    if (!n) return sprintf("%8s", "-")
    s = sprintf("%.2f", mine)
    if (color && mine <= best + 1e-9) return sprintf("%s%8s%s", grn, s, rst)
    return sprintf("%8s", s)
}
BEGIN { grn = color ? "\033[32m" : ""; rst = color ? "\033[0m" : "" }
{
    dc = $2; dh = $3; pc = $4; ph = $5; cc = $6; chh = $7
    if (isnum(pc) && (!duck || isnum(dc)) && (!ch || isnum(cc))) {
        b = pc; if (duck && dc < b) b = dc; if (ch && cc < b) b = cc
        lpc += log((pc + 10) / (b + 10)); nc++
        if (duck) ldc += log((dc + 10) / (b + 10))
        if (ch)   lcc += log((cc + 10) / (b + 10))
    }
    if (isnum(ph) && (!duck || isnum(dh)) && (!ch || isnum(chh))) {
        b = ph; if (duck && dh < b) b = dh; if (ch && chh < b) b = chh
        lph += log((ph + 10) / (b + 10)); nh++
        if (duck) ldh += log((dh + 10) / (b + 10))
        if (ch)   lch += log((chh + 10) / (b + 10))
    }
}
END {
    gpc = gm(lpc,nc); gdc = gm(ldc,nc); gcc = gm(lcc,nc)
    gph = gm(lph,nh); gdh = gm(ldh,nh); gch = gm(lch,nh)
    bc = gpc; if (duck && gdc < bc) bc = gdc; if (ch && gcc < bc) bc = gcc
    bh = gph; if (duck && gdh < bh) bh = gdh; if (ch && gch < bh) bh = gch
    print ""
    print "ClickBench score — geomean of (t+10ms)/(best+10ms) per query, lower is better"
    print "(1.00 = fastest on every scored query):"
    printf "  %-11s %8s %8s\n", "", "cold", "hot"
    printf "  %-11s %s %s\n", "pivot",  cell(gpc, bc, nc), cell(gph, bh, nh)
    if (duck) printf "  %-11s %s %s\n", "duckdb",     cell(gdc, bc, nc), cell(gdh, bh, nh)
    if (ch)   printf "  %-11s %s %s\n", "clickhouse", cell(gcc, bc, nc), cell(gch, bh, nh)
    printf "  scored %d/%d queries (cold/hot)\n", nc, nh
}
' "$data"
