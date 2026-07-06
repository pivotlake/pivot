#!/usr/bin/env bash
#
# bench-ab.sh — A/B performance comparison of two pivotdb source trees, measured
# through the ClickBench pivot-parquet harness.
#
# Runs entirely on the benchmark box. For each of the two trees ("before" and
# "after") it builds a PGO pivotdb-server, then hands that binary to the
# ClickBench harness (via PIVOT_SERVER_BIN), which times the query set the
# faithful ClickBench way: for every query it restarts the server and drops the
# OS page cache, then runs it BENCH_TRIES times. Per query the first try is the
# cold number and the min of the rest is hot. The two runs are then diffed; if
# any query's hot time regressed past the threshold, the script exits non-zero.
#
# It is meant to be launched detached and polled from a short-lived ssh session,
# so it writes its PID and emits a sentinel on every exit path.
#
# Usage:
#   bench-ab.sh \
#     --before-dir ~/perf-ab/before --after-dir ~/perf-ab/after \
#     --clickbench-dir ~/perf-ab/ClickBench \
#     --source ~/hits --pgo-subset ~/hits-pgo-subset \
#     --iterations 3 --regression-pct 5 --report /tmp/ab-report.txt \
#     [--before-label <sha>] [--after-label <sha>] \
#     [--query 7,20]     # ClickBench query numbers (0-based), empty = all
#     [--server-env 'PIVOT_X=1 PIVOT_Y=true']   # env applied to both sides

set -uo pipefail

pid_file="${PID_FILE:-/tmp/bench-ab.pid}"
echo $$ >"$pid_file"
# Fire on every exit path carrying the real exit code, so a poller watching the
# log never hangs on a silent failure.
trap 'echo "=== BENCH-AB COMPLETE exit=$? ==="' EXIT
set -e

before_dir=""
after_dir=""
clickbench_dir=""
source_path=""
pgo_subset=""
iterations=3
regression_pct=5
report="/tmp/ab-report.txt"
before_label="before"
after_label="after"
queries=""
duckdb_data=""
server_env=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --before-dir)     before_dir="$2"; shift 2 ;;
        --after-dir)      after_dir="$2"; shift 2 ;;
        --clickbench-dir) clickbench_dir="$2"; shift 2 ;;
        --source)         source_path="$2"; shift 2 ;;
        --pgo-subset)     pgo_subset="$2"; shift 2 ;;
        --iterations)     iterations="$2"; shift 2 ;;
        --regression-pct) regression_pct="$2"; shift 2 ;;
        --report)         report="$2"; shift 2 ;;
        --before-label)   before_label="$2"; shift 2 ;;
        --after-label)    after_label="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        # Optional DuckDB reference: directory holding the partitioned
        # hits_*.parquet. When set, DuckDB is also timed (once) as a baseline.
        --duckdb-data)    duckdb_data="$2"; shift 2 ;;
        # Extra environment for both server builds and harness runs, given as
        # space-separated KEY=VALUE pairs. Applied to both sides identically so
        # the comparison stays a like-for-like A/B (e.g. PIVOT_* runtime knobs).
        --server-env)     server_env="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for req in before_dir after_dir clickbench_dir source_path pgo_subset; do
    if [[ -z "${!req}" ]]; then
        echo "error: --${req//_/-} is required" >&2
        exit 2
    fi
done

# Paths may arrive with a leading ~ (a launcher can't expand it against this
# box's home), so expand it here against $HOME.
expand_tilde() {
    # The "~" patterns match a literal leading tilde in the argument.
    # shellcheck disable=SC2088
    case "$1" in
        "~")   printf '%s' "$HOME" ;;
        "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}
before_dir="$(expand_tilde "$before_dir")"
after_dir="$(expand_tilde "$after_dir")"
clickbench_dir="$(expand_tilde "$clickbench_dir")"
source_path="$(expand_tilde "$source_path")"
pgo_subset="$(expand_tilde "$pgo_subset")"
[[ -n "$duckdb_data" ]] && duckdb_data="$(expand_tilde "$duckdb_data")"

# cargo / just / llvm-profdata live under the login home but not on the
# non-interactive ssh PATH, so add them explicitly.
export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin"
export NO_COLOR=1

# Extra environment requested for the run. Exported into this process so it is
# inherited by everything downstream: the PGO profiling run (so the profile is
# generated under the same configuration it is later measured under) and the
# harness server runs. Both sides see the identical set, keeping the A/B fair.
if [[ -n "$server_env" ]]; then
    for kv in $server_env; do
        export "${kv?}"
    done
fi

adapter="$clickbench_dir/pivot-parquet"
[[ -x "$adapter/benchmark.sh" ]] || { echo "error: no ClickBench pivot-parquet adapter at $adapter" >&2; exit 2; }

# If a query subset was requested, build a filtered queries file (ClickBench
# query numbers are 0-based line offsets into queries.sql) and remember which
# original number each kept line maps back to, for labelling the report.
queries_file=""
declare -a query_labels=()
if [[ -n "$queries" ]]; then
    queries_file="$(mktemp)"
    : >"$queries_file"
    IFS=',' read -ra want <<<"$queries"
    for n in "${want[@]}"; do
        line=$(sed -n "$((n + 1))p" "$adapter/queries.sql")
        [[ -n "$line" ]] || { echo "error: no ClickBench query $n in queries.sql" >&2; exit 2; }
        printf '%s\n' "$line" >>"$queries_file"
        query_labels+=("$n")
    done
else
    # All queries: label by their 0-based line offset.
    n=0
    while IFS= read -r _; do query_labels+=("$n"); n=$((n + 1)); done <"$adapter/queries.sql"
fi

# Build a PGO pivotdb-server for the tree in $1; echo the binary path on stdout
# (build chatter goes to stderr). Mirrors the adapter's own install: profile is
# generated by running pivot-bench over the small subset, then the server is
# built profile-use. The PGO dir is shared, so it is cleared per tree.
build_server() {
    local tree="$1"
    (
        cd "$tree/benchmarks"
        just pgo-clean
        just pgo-gen run --release -- --source "$pgo_subset" --iterations 2 --skip-check
        just pgo-use build --release -p server --bin pivotdb-server
    ) >&2
    printf '%s' "$tree/benchmarks/target-pgouse/release/pivotdb-server"
}

# Run the ClickBench harness for a server binary ($1), writing its raw output to
# $2. $3/$4 are a dedicated port and a fresh catalog dir so the two sides never
# collide. The harness restarts the server and drops caches per query itself.
run_harness() {
    local bin="$1" out="$2" port="$3" catalog="$4"
    rm -rf "$catalog"
    (
        cd "$adapter"
        export PIVOT_SERVER_BIN="$bin" PIVOT_SOURCE="$source_path" \
               PIVOT_PORT="$port" PIVOT_CATALOG="$catalog" BENCH_TRIES="$iterations"
        [[ -n "$queries_file" ]] && export BENCH_QUERIES_FILE="$queries_file"
        ./benchmark.sh
    ) >"$out" 2>&1 || true   # a nonzero exit (e.g. the QPS watchdog) is fine; we read the timings
    # Make sure the server is down before the other side reuses the box.
    ( cd "$adapter"; PIVOT_PORT="$port" ./stop >/dev/null 2>&1 || true )
}

# Run the DuckDB (parquet, partitioned) ClickBench adapter once, writing its raw
# output to $1. The adapter reads hits_*.parquet from its own directory, so the
# partitioned dataset ($duckdb_data) is symlinked in for the run and removed
# after. For a query subset, the DuckDB dialect queries are filtered at the same
# 0-based indices as the pivot side (queries correspond positionally).
run_duckdb_harness() {
    local out="$1"
    local dd="$clickbench_dir/duckdb-parquet-partitioned"
    [[ -x "$dd/benchmark.sh" ]] || { echo "error: no duckdb-parquet-partitioned adapter at $dd" >&2; return 1; }

    local n=0 f
    find "$dd" -maxdepth 1 -type l -name 'hits_*.parquet' -delete 2>/dev/null || true
    for f in "$duckdb_data"/hits_*.parquet; do
        [[ -e "$f" ]] || { echo "error: no hits_*.parquet in $duckdb_data" >&2; return 1; }
        ln -sf "$f" "$dd/"; n=$((n + 1))
    done
    echo "  linked $n partitioned parquet files into the DuckDB adapter" >&2

    local duck_qfile=""
    if [[ -n "$queries" ]]; then
        duck_qfile="$(mktemp)"
        local idx line
        for idx in "${query_labels[@]}"; do
            line=$(sed -n "$((idx + 1))p" "$dd/queries.sql")
            printf '%s\n' "$line" >>"$duck_qfile"
        done
    fi
    (
        cd "$dd"
        # duckdb is not on the non-interactive PATH; expose the CLI so the
        # adapter's ./install sees it and skips a network install.
        export PATH="$PATH:$HOME/.duckdb/cli/latest:$HOME/.duckdb/cli/1.5.3"
        export BENCH_TRIES="$iterations"
        [[ -n "$duck_qfile" ]] && export BENCH_QUERIES_FILE="$duck_qfile"
        ./benchmark.sh
    ) >"$out" 2>&1 || true
    find "$dd" -maxdepth 1 -type l -name 'hits_*.parquet' -delete 2>/dev/null || true
}

# Turn a harness log into "idx cold hot" rows: idx is the 0-based position of the
# query, cold is the first try, hot is the min of the remaining tries ("null" if
# any needed value is missing). The harness prints one "[t1,t2,t3]," line per
# query, in order.
parse_timings() {
    awk '
    /^\[/ {
        line = $0
        gsub(/[][,]+$/, "", line)      # strip trailing "],"
        gsub(/^\[/, "", line)          # strip leading "["
        n = split(line, t, ",")
        cold = t[1]
        hot = ""
        for (i = 2; i <= n; i++) {
            if (t[i] == "null") continue
            if (hot == "" || t[i] + 0 < hot + 0) hot = t[i]
        }
        if (hot == "") hot = "null"
        printf "%d %s %s\n", idx++, cold, hot
    }' "$1"
}

before_out="/tmp/ab-before.out"; after_out="/tmp/ab-after.out"
before_tsv="/tmp/ab-before.tsv"; after_tsv="/tmp/ab-after.tsv"
duck_out="/tmp/ab-duck.out";     duck_tsv="/tmp/ab-duck.tsv"

echo "=== A/B via ClickBench pivot-parquet: '$before_label' (before) vs '$after_label' (after) ===" >"$report"
echo "source=$source_path tries=$iterations regression_pct=$regression_pct queries=${queries:-all}" >>"$report"
[[ -n "$server_env" ]] && echo "server_env=$server_env" >>"$report"
echo >>"$report"

echo ">>> building BEFORE server ($before_label)"
before_bin="$(build_server "$before_dir")"
echo ">>> timing BEFORE through ClickBench harness"
run_harness "$before_bin" "$before_out" 7799 /tmp/ab-cat-before
parse_timings "$before_out" >"$before_tsv"

echo ">>> building AFTER server ($after_label)"
after_bin="$(build_server "$after_dir")"
echo ">>> timing AFTER through ClickBench harness"
run_harness "$after_bin" "$after_out" 7798 /tmp/ab-cat-after
parse_timings "$after_out" >"$after_tsv"

# Join by query index and render the cold/hot diff. Gate on hot: a query whose
# hot time is >= regression_pct slower after than before fails the run. Labels
# come from query_labels[idx].
labels_csv="$(IFS=,; echo "${query_labels[*]}")"
failures="$(awk \
    -v bf="$before_tsv" -v af="$after_tsv" -v t="$regression_pct" -v labels="$labels_csv" \
    -v blabel="$before_label" -v alabel="$after_label" '
# Timings are in seconds; floor the divisor at 1ms so sub-millisecond
# baselines do not blow the percentage up (they are not meaningful anyway).
function pct(b, a) { b = (b < 0.001 ? 0.001 : b); return (a - b) / b * 100 }
function valid(x) { return (x != "" && x != "null") }
BEGIN {
    nl = split(labels, lab, ",")
    printf "%-6s %10s %10s %8s   %10s %10s %8s  %s\n", \
        "query", "cold_b", "cold_a", "cold_Δ%", "hot_b", "hot_a", "hot_Δ%", "status" > "/dev/stderr"
}
FILENAME == bf { bc[$1] = $2; bh[$1] = $3; seen[$1] = 1; next }
FILENAME == af { ac[$1] = $2; ah[$1] = $3; seen[$1] = 1; next }
END {
    for (i = 0; i in seen; i++) {
        q = "Q" (i < nl ? lab[i + 1] : i)
        cb = bc[i]; ca = ac[i]; hb = bh[i]; ha = ah[i]
        # Cold diff (informational).
        if (valid(cb) && valid(ca)) cd = sprintf("%+.1f%%", pct(cb, ca)); else cd = "-"
        # Hot diff (the gate). A missing timing on either side (query errored
        # or produced no result) is a failure too, not a silent pass.
        status = "ok"; hd = "-"
        if (!valid(hb) || !valid(ha)) {
            status = "ERR"; print q " ERR (missing timing)"
        } else {
            hp = pct(hb, ha)
            hd = sprintf("%+.1f%%", hp)
            if (hp >= t) { status = "REGRESSION"; print q " " hd }
            else if (hp <= -t) status = "improved"
        }
        printf "%-6s %10s %10s %8s   %10s %10s %8s  %s\n", \
            q, cb, ca, cd, hb, ha, hd, status > "/dev/stderr"

        # ClickBench score: each build relative to the per-query best (min),
        # regularised with +10ms (10ms = 0.01s since timings are seconds), then
        # geomean over queries. Fastest build on every query = 1.00.
        if (valid(cb) && valid(ca)) {
            b = (cb < ca ? cb : ca)
            bcl += log((cb + 0.01) / (b + 0.01))
            acl += log((ca + 0.01) / (b + 0.01)); cn++
        }
        if (valid(hb) && valid(ha)) {
            b = (hb < ha ? hb : ha)
            bhl += log((hb + 0.01) / (b + 0.01))
            ahl += log((ha + 0.01) / (b + 0.01)); hn++
        }
    }
    printf "\nClickBench score - geomean of (t+10ms)/(best+10ms) per query (1.00 = fastest on every query):\n" > "/dev/stderr"
    printf "  %-14s %8s %8s\n", "", "cold", "hot" > "/dev/stderr"
    printf "  %-14s %7.2fx %7.2fx\n", blabel, (cn ? exp(bcl / cn) : 0), (hn ? exp(bhl / hn) : 0) > "/dev/stderr"
    printf "  %-14s %7.2fx %7.2fx\n", alabel, (cn ? exp(acl / cn) : 0), (hn ? exp(ahl / hn) : 0) > "/dev/stderr"
    printf "  scored %d/%d queries (cold/hot)\n", cn, hn > "/dev/stderr"
}' "$before_tsv" "$after_tsv" 2>>"$report")"

{
    echo
    echo "=== failures: hot regressions (>= ${regression_pct}%) or missing timings ==="
    if [[ -n "$failures" ]]; then echo "$failures"; else echo "none"; fi
} >>"$report"

# Optional DuckDB reference: time DuckDB (parquet, partitioned) once and compare
# the after build against it. DuckDB is not part of the pass/fail gate — it is a
# baseline, reported with a ClickBench-style score (geomean of pivot/duckdb).
if [[ -n "$duckdb_data" ]]; then
    echo ">>> timing DuckDB (parquet, partitioned) through ClickBench harness"
    run_duckdb_harness "$duck_out"
    parse_timings "$duck_out" >"$duck_tsv"

    { echo; echo "=== pivot ($after_label) vs DuckDB (parquet, partitioned) ==="; } >>"$report"
    awk -v af="$after_tsv" -v dk="$duck_tsv" -v labels="$labels_csv" '
    function valid(x) { return (x != "" && x != "null") }
    function ratio(p, d) { return (p + 0.01) / (d + 0.01) }   # <1 => pivot faster
    BEGIN {
        nl = split(labels, lab, ",")
        printf "%-6s %10s %10s %9s   %10s %10s %9s\n", \
            "query", "piv_cold", "duck_cold", "c p/d", "piv_hot", "duck_hot", "h p/d" > "/dev/stderr"
    }
    FILENAME == af { pc[$1] = $2; ph[$1] = $3; seen[$1] = 1; next }
    FILENAME == dk { dc[$1] = $2; dh[$1] = $3; seen[$1] = 1; next }
    END {
        for (i = 0; i in seen; i++) {
            q = "Q" (i < nl ? lab[i + 1] : i)
            cr = (valid(pc[i]) && valid(dc[i])) ? sprintf("%.2fx", ratio(pc[i], dc[i])) : "-"
            hr = (valid(ph[i]) && valid(dh[i])) ? sprintf("%.2fx", ratio(ph[i], dh[i])) : "-"
            printf "%-6s %10s %10s %9s   %10s %10s %9s\n", \
                q, pc[i], dc[i], cr, ph[i], dh[i], hr > "/dev/stderr"
            # ClickBench score: each system relative to the per-query best
            # (min), regularised with +10ms, then geomean over queries.
            if (valid(pc[i]) && valid(dc[i])) {
                b = (pc[i] < dc[i] ? pc[i] : dc[i])
                pcl += log((pc[i] + 0.01) / (b + 0.01))
                dcl += log((dc[i] + 0.01) / (b + 0.01)); cn++
            }
            if (valid(ph[i]) && valid(dh[i])) {
                b = (ph[i] < dh[i] ? ph[i] : dh[i])
                phl += log((ph[i] + 0.01) / (b + 0.01))
                dhl += log((dh[i] + 0.01) / (b + 0.01)); hn++
            }
        }
        printf "\nClickBench score - geomean of (t+10ms)/(best+10ms) per query (1.00 = fastest on every query):\n" > "/dev/stderr"
        printf "  %-8s %8s %8s\n", "", "cold", "hot" > "/dev/stderr"
        printf "  %-8s %7.2fx %7.2fx\n", "pivot",  (cn ? exp(pcl / cn) : 0), (hn ? exp(phl / hn) : 0) > "/dev/stderr"
        printf "  %-8s %7.2fx %7.2fx\n", "duckdb", (cn ? exp(dcl / cn) : 0), (hn ? exp(dhl / hn) : 0) > "/dev/stderr"
        printf "  scored %d/%d queries (cold/hot)\n", cn, hn > "/dev/stderr"
    }' "$after_tsv" "$duck_tsv" 2>>"$report"
fi

cat "$report"

if [[ -n "$failures" ]]; then
    exit 3
fi
