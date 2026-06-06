#!/usr/bin/env bash
#
# benchmark.sh — run the same ClickBench queries through DuckDB and through
# pivotdb, and print a side-by-side table of cold and hot timings plus the
# speedup (how many times faster pivot is) per query.
#
# DuckDB is driven by run-duckdb.sh; pivot is driven by
# `just pgo-use run --release -- ...` (so it runs the PGO-optimised build —
# generate the profile first with `just pgo-gen ...`).
#
# All pivot queries run first, then all DuckDB queries.
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
#   ./benchmark.sh --source ~/hits                     # all queries, one session
#   ./benchmark.sh --hits ~/hits --query 7,20          # subset; --hits == --source
#   ./benchmark.sh --source ~/hits --iterations 5      # 1 cold + 4 hot runs
#   ./benchmark.sh --source ~/hits --sleep 500         # 500ms between iterations
#   ./benchmark.sh --source ~/hits --restart-server    # isolate each query
#   ./benchmark.sh --source ~/hits --skip-check        # don't verify pivot output
#   ./benchmark.sh --source ~/hits --no-drop-caches    # skip the cache drop
#
# --source/--hits accepts a directory (globbed for *.parquet), a single
# .parquet file, or an explicit glob.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite_dir="$here/clickbench"

source_path=""
queries=""
iterations=3
drop_caches=1
restart_server=0
sleep_ms=0
skip_check=0

usage() {
    sed -n '3,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source|--hits)  source_path="$2"; shift 2 ;;
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
# disabled, let run-duckdb.sh flush the page cache before each query (it runs
# each query's iterations in one process), matching pivot's per-query cold.
duck_invoke() {
    local IFS=,
    local duck_cache_flag="--no-drop-caches"
    [[ $drop_caches -eq 1 ]] && duck_cache_flag=""
    "$here/run-duckdb.sh" --source "$source_path" --query "$*" \
        --iterations "$iterations" --sleep "$sleep_ms" $duck_cache_flag 2>&1
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

pivot_tsv="$(mktemp)"; duck_tsv="$(mktemp)"
data="$(mktemp)"; spin_out="$(mktemp)"
trap 'rm -f "$pivot_tsv" "$duck_tsv" "$data" "$spin_out"' EXIT

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

mode=$([[ $restart_server -eq 1 ]] && echo "restart per query" || echo "one session")
echo "queries: ${#ids[@]}, iterations: $iterations, mode: $mode, drop_caches: $drop_caches"
echo "source: $source_path"
echo
collect pivot  pivot_invoke parse_pivot "$pivot_tsv"
collect DuckDB duck_invoke  parse_duck  "$duck_tsv"
echo

# Merge the two result sets by query id, in the canonical query order, filling
# "ERR" for any query an engine failed to produce. Columns: id dc dh pc ph.
awk -v order="$(IFS=,; echo "${ids[*]}")" '
FNR == NR { dc[$1] = $2; dh[$1] = $3; next }   # first file: duckdb
          { pc[$1] = $2; ph[$1] = $3 }         # second file: pivot
END {
    n = split(order, o, ",")
    for (i = 1; i <= n; i++) {
        k = o[i]
        printf "%s %s %s %s %s\n", k, \
            (k in dc ? dc[k] : "ERR"), (k in dh ? dh[k] : "ERR"), \
            (k in pc ? pc[k] : "ERR"), (k in ph ? ph[k] : "ERR")
    }
}' "$duck_tsv" "$pivot_tsv" > "$data"

# Render the comparison table. Speedup = duckdb / pivot (>1 means pivot faster).
awk -v color="$color" '
function speed(duck, piv,   r) {
    if (duck == "ERR" || piv == "ERR" || duck == "-" || piv == "-") return "-"
    if (piv + 0 <= 0) return ">99x"          # pivot too fast to measure at ms granularity
    r = duck / piv
    return sprintf("%.1fx", r)
}
function colored(txt, duck, piv,   col, vis) {
    vis = sprintf("%8s", txt)
    if (!color || txt == "-" || txt == ">99x") {
        if (txt == ">99x" && color) return green vis rst
        return vis
    }
    col = ((duck / piv) >= 1.0) ? green : red
    return col vis rst
}
function tm(v) { return (v == "-" || v == "ERR") ? sprintf("%11s", v) : sprintf("%9sms", v) }
BEGIN {
    green = color ? "\033[32m" : ""
    red   = color ? "\033[31m" : ""
    rst   = color ? "\033[0m"  : ""
    printf "%-6s %11s %11s %9s   %11s %11s %9s\n", \
        "query", "duckdb(c)", "pivot(c)", "cold", "duckdb(h)", "pivot(h)", "hot"
    printf "%s\n", "-----------------------------------------------------------------------------"
}
{
    id=$1; dc=$2; dh=$3; pc=$4; ph=$5
    cs = speed(dc, pc); hs = speed(dh, ph)
    printf "%-6s %s %s %s   %s %s %s\n", \
        id, tm(dc), tm(pc), colored(cs, dc, pc), tm(dh), tm(ph), colored(hs, dh, ph)
}
' "$data"

# ClickBench-style weighted score: the geometric mean over queries of
# (t + 10ms) / (best_for_that_query + 10ms) — ClickBench's summary metric. The
# +10ms regularises near-zero queries; lower is better and 1.00 means fastest
# on every scored query. With two engines the per-query baseline is the faster
# of the two. Computed over our cold and hot numbers; a query is skipped for a
# metric when either engine has no timing there (ERR or single-iteration "-").
awk -v color="$color" '
function isnum(x) { return x ~ /^[0-9]+(\.[0-9]+)?$/ }
function gm(logsum, n) { return n ? exp(logsum / n) : 0 }
function cell(mine, other, n,   s) {
    if (!n) return sprintf("%8s", "-")
    s = sprintf("%.2f", mine)
    if (color && mine <= other) return sprintf("%s%8s%s", grn, s, rst)
    return sprintf("%8s", s)
}
BEGIN { grn = color ? "\033[32m" : ""; rst = color ? "\033[0m" : "" }
{
    dc = $2; dh = $3; pc = $4; ph = $5
    if (isnum(dc) && isnum(pc)) {
        b = dc < pc ? dc : pc
        ldc += log((dc + 10) / (b + 10)); lpc += log((pc + 10) / (b + 10)); nc++
    }
    if (isnum(dh) && isnum(ph)) {
        b = dh < ph ? dh : ph
        ldh += log((dh + 10) / (b + 10)); lph += log((ph + 10) / (b + 10)); nh++
    }
}
END {
    print ""
    print "ClickBench score — geomean of (t+10ms)/(best+10ms) per query, lower is better"
    print "(1.00 = fastest on every scored query):"
    printf "  %-7s %8s %8s\n", "", "cold", "hot"
    printf "  %-7s %s %s\n", "pivot",  cell(gm(lpc,nc), gm(ldc,nc), nc), cell(gm(lph,nh), gm(ldh,nh), nh)
    printf "  %-7s %s %s\n", "duckdb", cell(gm(ldc,nc), gm(lpc,nc), nc), cell(gm(ldh,nh), gm(lph,nh), nh)
    printf "  scored %d/%d queries (cold/hot)\n", nc, nh
}
' "$data"
