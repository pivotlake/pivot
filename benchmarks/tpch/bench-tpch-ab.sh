#!/usr/bin/env bash
#
# bench-tpch-ab.sh - A/B performance comparison of two pivotdb commits on the
# normalized TPC-H suite, run on a freshly launched instance-store machine.
#
# Owns the whole box side: checks both commits out of the AMI-baked repo clone,
# formats and mounts the instance-store NVMe, syncs the per-table datasets from
# S3 (small PGO scale first, then the measurement scale) while both sides
# build, builds a per-side pivot-bench (PGO by default, plain release with
# --mode release), and restores/saves warm cargo target dirs from S3 so only
# the commit diff and the profile-dependent Rust units recompile.
#
# The measurement is a cold power run per side and pass: one pivot-bench
# process streams the whole suite, each query once, a quiet gap between
# queries, caches dropped only before the stream starts. Later queries reuse
# whatever earlier ones left in pivot's file cache and the OS page cache.
# Sides alternate within each pass and the per-query cold time is the min
# across passes.
#
# The gate: any query whose cold time regresses after vs before by at least
# --regression-pct fails the run, as does a missing timing or an output
# mismatch between the two sides.
#
# With --duckdb, DuckDB gets the same stream shape as a reference column via
# run-duckdb.sh: caches dropped once before the stream, the same gap between
# queries, no drops in between. It never gates.
#
# The box-side machinery it shares with the other A/B harnesses (NVMe mount,
# checkouts, S3 dataset sync, warm-cache restore/save, PGO builds,
# correctness) lives in benchmarks/lib/bench-ab-common.sh; what is specific to
# the normalized TPC-H suite stays here: the power-run measurement.
#
# Meant to be launched detached and polled over ssh, so it writes its PID and
# emits a sentinel on every exit path.
#
# Usage:
#   bench-tpch-ab.sh \
#     --mirror /opt/pivotdb --repo-url <tokenised fetch url> \
#     --before <sha> --after <sha> \
#     --data-bucket s3://bucket --cache-prefix s3://bucket/cache \
#     [--mode pgo|release] [--passes 2] [--regression-pct 5] \
#     [--query q06,q12] [--duckdb] [--bench-env 'PIVOT_X=1 PIVOT_Y=true'] \
#     [--report /tmp/ab-report.txt] [--before-label <sha>] [--after-label <sha>]

set -uo pipefail

# The shared library and the DuckDB runner sit next to this script in the
# repo, but the workflow ships all three flat into one /tmp directory on the
# box; find them either way.
_here="${BASH_SOURCE[0]%/*}"
_common="$_here/bench-ab-common.sh"
[[ -f "$_common" ]] || _common="$_here/../lib/bench-ab-common.sh"
# shellcheck source=../lib/bench-ab-common.sh
source "$_common"
duckdb_runner="$_here/run-duckdb.sh"

pid_file="${PID_FILE:-/tmp/bench-tpch-ab.pid}"
echo $$ >"$pid_file"
# Fire on every exit path carrying the real exit code, so a poller watching the
# log never hangs on a silent failure.
trap 'echo "=== BENCH-TPCH-AB COMPLETE exit=$? ==="' EXIT
set -e

mirror="/opt/pivotdb"
repo_url=""
before_sha=""
after_sha=""
work_dir="$HOME/ab"
data_bucket=""
data_root="/mnt/nvme/tpch"
cache_prefix=""
mode="pgo"
passes=2
regression_pct=5
queries=""
run_duckdb=0
bench_env=""
report="/tmp/ab-report.txt"
before_label="before"
after_label="after"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)         mirror="$2"; shift 2 ;;
        --repo-url)       repo_url="$2"; shift 2 ;;
        --before)         before_sha="$2"; shift 2 ;;
        --after)          after_sha="$2"; shift 2 ;;
        --work-dir)       work_dir="$2"; shift 2 ;;
        --data-bucket)    data_bucket="$2"; shift 2 ;;
        --data-root)      data_root="$2"; shift 2 ;;
        --cache-prefix)   cache_prefix="$2"; shift 2 ;;
        --mode)           mode="$2"; shift 2 ;;
        --passes)         passes="$2"; shift 2 ;;
        --regression-pct) regression_pct="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        --duckdb)         run_duckdb=1; shift ;;
        # Extra environment for both sides' profiling and measurement runs,
        # space-separated KEY=VALUE pairs, applied identically to keep the
        # comparison like for like.
        --bench-env)      bench_env="$2"; shift 2 ;;
        --report)         report="$2"; shift 2 ;;
        --before-label)   before_label="$2"; shift 2 ;;
        --after-label)    after_label="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for req in before_sha after_sha data_bucket; do
    if [[ -z "${!req}" ]]; then
        echo "error: --${req%_sha} is required" >&2
        exit 2
    fi
done
if [[ "$mode" != "pgo" && "$mode" != "release" ]]; then
    echo "error: --mode must be pgo or release" >&2
    exit 2
fi

ab_common_init

power_sleep=500
sf_pgo="sf1"
# The pivot-written variant of SF100: the same rows as pivot's own writer
# lays them out after an INSERT, which is what a real table looks like.
sf_measure="sf100-pivot"
pgo_data="$data_root/$sf_pgo"
measure_data="$data_root/$sf_measure"

# ---------------------------------------------------------------------------
# Measurement helpers. The dataset sync (sync_scale/wait_for_scale) and the
# io_ticks disk-state signal come from bench-ab-common.sh.
# ---------------------------------------------------------------------------
data_dev="" # set after mount, for the io sanity signal

io_delta() {
    awk -v a="${1:-0}" -v b="$(io_ticks)" 'BEGIN{printf "%.1f", (b - a) / 1000}'
}

# Streams one side's whole suite in a single pivot-bench process: each query
# once, ${power_sleep}ms of quiet before the next (with --iterations 1 the
# runner's --sleep fires in its per-query loop, skipping the first query).
# Caches are NOT dropped between queries; the caller drops them once before
# the stream, so later queries keep whatever earlier ones cached. Echoes
# "<query> <ms>" per line, in suite order.
run_power() {
    local bin="$1" server="$2" dir="$3" out
    out="$("$bin" --suite tpch --suite-dir "$dir/benchmarks/tpch" \
        --server-bin "$server" \
        --source "$measure_data" --query "$queries" \
        --iterations 1 --sleep "$power_sleep" --skip-check 2>&1)" || true
    # Timing lines end in "<ms>ms"; take those alone (the runner also prints a
    # per-query header) without depending on the separator glyph it prints
    # between the query id and the time.
    grep -E "Query q[0-9]+.*[0-9]ms$" <<<"$out" \
        | sed -E 's/^.*Query (q[0-9]+).*[[:space:]]([0-9]+)ms$/\1 \2/'
}

# The same stream shape for DuckDB via run-duckdb.sh: the caller drops caches
# once before the stream, the runner never does (--no-drop-caches), and the
# same gap separates queries. Query files come from the after checkout.
# Echoes "<query> <ms>" per line.
run_duckdb_power() {
    local out
    out="$(bash "$duckdb_runner" --suite-dir "$after_dir/benchmarks/tpch" \
        --source "$measure_data" --query "$queries" \
        --iterations 1 --sleep "$power_sleep" --no-drop-caches 2>&1)" || true
    awk '/^=== q[0-9]+ ===$/ { query = $2; next }
        /Run Time/ {
            for (i = 1; i <= NF; i++)
                if ($i == "real") { printf "%s %d\n", query, $(i + 1) * 1000; break }
        }' <<<"$out"
}

# ---------------------------------------------------------------------------
# Phase 0: disk, data, checkouts.
# ---------------------------------------------------------------------------
mkdir -p "$work_dir"
mount_nvme
data_dev="$(df --output=source "$(dirname "$data_root")" | tail -1 | sed 's|/dev/||')"

( sync_scale "$sf_pgo"; sync_scale "$sf_measure" ) &
sync_pid=$!

echo ">>> fetching commits into the baked clone"
if [[ -n "$repo_url" ]]; then
    git -C "$mirror" fetch --quiet "$repo_url" "$before_sha" "$after_sha" \
        || git -C "$mirror" fetch --quiet "$repo_url" '+refs/heads/*:refs/ab/heads/*'
fi
before_dir="$work_dir/before"
after_dir="$work_dir/after"
checkout_side "$before_sha" "$before_dir"
checkout_side "$after_sha" "$after_dir"

# ---------------------------------------------------------------------------
# Phase 1: cache restore + builds, overlapping the dataset sync.
# ---------------------------------------------------------------------------
restore_pids=()
restore_cargo_home & restore_pids+=($!)
if [[ "$mode" == "pgo" ]]; then
    restore_cache "pgogen-before" "$before_dir/benchmarks/target-pgogen" & restore_pids+=($!)
    restore_cache "pgogen-after" "$after_dir/benchmarks/target-pgogen" & restore_pids+=($!)
    restore_cache "pgouse-before" "$before_dir/benchmarks/target-pgouse" & restore_pids+=($!)
    restore_cache "pgouse-after" "$after_dir/benchmarks/target-pgouse" & restore_pids+=($!)
    restore_cache "client-before" "$before_dir/benchmarks/target-client" & restore_pids+=($!)
    restore_cache "client-after" "$after_dir/benchmarks/target-client" & restore_pids+=($!)
else
    restore_cache "release-before" "$before_dir/target" & restore_pids+=($!)
    restore_cache "release-after" "$after_dir/target" & restore_pids+=($!)
fi
wait "${restore_pids[@]}"

if [[ "$mode" == "pgo" ]]; then
    echo ">>> instrumented builds (A and B in parallel)"
    build_gen "$before_dir" & b1=$!
    build_gen "$after_dir" & b2=$!
    wait "$b1"; wait "$b2"

    echo ">>> waiting for the $sf_pgo dataset"
    wait_for_scale "$sf_pgo"
    echo ">>> profiling runs on $sf_pgo"
    profile_side "$before_dir" before tpch "$pgo_data"
    profile_side "$after_dir" after tpch "$pgo_data"

    echo ">>> profile-use builds (A and B in parallel)"
    build_use "$before_dir" before & b1=$!
    build_use "$after_dir" after & b2=$!
    wait "$b1"; wait "$b2"

    before_bin="$before_dir/benchmarks/target-client/release/pivot-bench"
    after_bin="$after_dir/benchmarks/target-client/release/pivot-bench"
    before_server="$before_dir/benchmarks/target-pgouse/$host_target/release/pivotdb-server"
    after_server="$after_dir/benchmarks/target-pgouse/$host_target/release/pivotdb-server"
else
    echo ">>> release builds (A and B in parallel)"
    build_release "$before_dir" & b1=$!
    build_release "$after_dir" & b2=$!
    wait "$b1"; wait "$b2"
    before_bin="$before_dir/target/release/pivot-bench"
    after_bin="$after_dir/target/release/pivot-bench"
    before_server="$before_dir/target/release/pivotdb-server"
    after_server="$after_dir/target/release/pivotdb-server"
fi
for built in "$before_bin" "$after_bin" "$before_server" "$after_server"; do
    [[ -x "$built" ]] || { echo "error: build produced no $built" >&2; exit 1; }
done

echo ">>> saving warm caches"
save_cache "cargo-home" "$HOME/.cargo" registry git &
if [[ "$mode" == "pgo" ]]; then
    save_cache "pgogen-before" "$before_dir/benchmarks/target-pgogen" &
    save_cache "pgogen-after" "$after_dir/benchmarks/target-pgogen" &
    save_cache "pgouse-before" "$before_dir/benchmarks/target-pgouse" &
    save_cache "pgouse-after" "$after_dir/benchmarks/target-pgouse" &
    save_cache "client-before" "$before_dir/benchmarks/target-client" &
    save_cache "client-after" "$after_dir/benchmarks/target-client" &
else
    save_cache "release-before" "$before_dir/target" &
    save_cache "release-after" "$after_dir/target" &
fi
# The uploads overlap the wait for the measurement dataset; collect them
# before measuring so they don't steal bandwidth or CPU from the timed runs.
wait_for_scale "$sf_measure"
wait

# ---------------------------------------------------------------------------
# Phase 2: burn-in + correctness. One untimed full pass per side writes each
# side's result tsvs and, as a side effect, walks the instance store out of
# its fresh-boot throughput state before anything is measured. The two sides'
# outputs are then compared with a float tolerance (summation order makes the
# last digit unstable); a real mismatch fails the run.
# ---------------------------------------------------------------------------
if [[ -z "$queries" ]]; then
    queries="$(cd "$after_dir/benchmarks/tpch" \
        && ls q*.sql | grep -v -- '-duckdb' | sed 's/\.sql$//' | paste -sd,)"
fi
echo ">>> burn-in + result capture (queries: $queries)"
drop_caches
"$before_bin" --suite tpch --suite-dir "$before_dir/benchmarks/tpch" \
    --server-bin "$before_server" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null
drop_caches
"$after_bin" --suite tpch --suite-dir "$after_dir/benchmarks/tpch" \
    --server-bin "$after_server" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null

correctness="ok"
if ! compare_outputs "$before_dir/benchmarks/tpch" "$after_dir/benchmarks/tpch"; then
    correctness="MISMATCH"
fi

# ---------------------------------------------------------------------------
# Phase 3: measurement, one power stream per side and pass, sides alternating
# within a pass so slow drift in instance-store throughput hits both equally.
# A stream is a single cold sample per query; several passes with a per-query
# min give the gate a stable number.
# ---------------------------------------------------------------------------
rows="/tmp/ab-rows.tsv"
: >"$rows"
stream="/tmp/ab-stream.txt"
echo ">>> measuring (power passes=$passes, ${power_sleep}ms between queries, mode=$mode)"
for pass in $(seq "$passes"); do
    drop_caches; sleep 3
    io_start="$(io_ticks)"
    run_power "$before_bin" "$before_server" "$before_dir" >"$stream"
    awk -v side=before 'NF == 2 {print $1 "\t" side "\t" $2}' "$stream" >>"$rows"
    echo "    pass $pass before: $(wc -l <"$stream") queries timed, io $(io_delta "$io_start")s"
    drop_caches; sleep 3
    io_start="$(io_ticks)"
    run_power "$after_bin" "$after_server" "$after_dir" >"$stream"
    awk -v side=after 'NF == 2 {print $1 "\t" side "\t" $2}' "$stream" >>"$rows"
    echo "    pass $pass after: $(wc -l <"$stream") queries timed, io $(io_delta "$io_start")s"
    if [[ "$run_duckdb" == "1" ]]; then
        drop_caches; sleep 3
        run_duckdb_power >"$stream"
        awk -v side=duckdb 'NF == 2 {print $1 "\t" side "\t" $2}' "$stream" >>"$rows"
        echo "    pass $pass duckdb: $(wc -l <"$stream") queries timed"
    fi
done

# ---------------------------------------------------------------------------
# Report + gate. Cold, the only number a power stream yields, is the gate.
# DuckDB, when present, is a reference only.
# ---------------------------------------------------------------------------
{
    echo "=== TPC-H A/B: '$before_label' (before) vs '$after_label' (after) ==="
    echo "mode=$mode source=$measure_data passes=$passes regression_pct=$regression_pct"
    echo "shape: cold power runs; per side and pass, one process streams every query once, ${power_sleep}ms apart, caches dropped only before the stream; cold = per-query min across passes"
    [[ -n "$bench_env" ]] && echo "bench_env=$bench_env"
    [[ "$mode" == "release" ]] && echo "NOTE: release mode is non-PGO; numbers carry code-alignment noise, use for iteration only"
    echo "correctness (before vs after, 1e-6 relative): $correctness"
    echo
} >"$report"

failures="$(awk -F'\t' -v t="$regression_pct" -v duck="$run_duckdb" '
function pct(b, a) { b = (b < 1 ? 1 : b); return (a - b) / b * 100 }
function valid(x) { return (x != "") }
function keepmin(arr, k, v) { if (!(k in arr) || v + 0 < arr[k] + 0) arr[k] = v }
{ keepmin(c, $1 "," $2, $3); if (!seen[$1]++) order[n++] = $1 }
END {
    hdr = sprintf("%-5s %8s %8s %8s", "query", "cold_b", "cold_a", "coldD%")
    if (duck == 1) hdr = hdr sprintf("   %8s %7s", "duckdb", "a/dk")
    print hdr "  status" > "/dev/stderr"
    for (i = 0; i < n; i++) {
        q = order[i]
        cb = c[q ",before"]; ca = c[q ",after"]
        status = "ok"; cd = "-"
        if (!valid(cb) || !valid(ca)) {
            status = "ERR"; print q " ERR (missing timing)"
        } else {
            cp = pct(cb, ca); cd = sprintf("%+.1f%%", cp)
            if (cp >= t) { status = "REGRESSION"; print q " cold " cd }
            else if (cp <= -t) status = "improved"
        }
        line = sprintf("%-5s %8s %8s %8s", q, cb, ca, cd)
        if (duck == 1) {
            dc = c[q ",duckdb"]; ratio = "-"
            if (valid(ca) && valid(dc)) ratio = sprintf("%.2fx", (ca + 10) / (dc + 10))
            line = line sprintf("   %8s %7s", dc, ratio)
        }
        print line "  " status > "/dev/stderr"
        if (valid(cb) && valid(ca)) {
            best = (cb < ca ? cb : ca)
            bcl += log((cb + 10) / (best + 10)); acl += log((ca + 10) / (best + 10)); cn++
            sb += cb; sa += ca
        }
    }
    printf "\ntotals: cold before %dms, after %dms (%+.1f%%)\n", sb, sa, pct(sb, sa) > "/dev/stderr"
    printf "cold score, geomean of (t+10ms)/(best+10ms), 1.00 = fastest every query:\n" > "/dev/stderr"
    printf "  before %.3fx  after %.3fx  (%d queries)\n", \
        (cn ? exp(bcl / cn) : 0), (cn ? exp(acl / cn) : 0), cn > "/dev/stderr"
}' "$rows" 2>>"$report")"

{
    echo
    echo "=== failures: cold regressions (>= ${regression_pct}%) or missing timings ==="
    if [[ "$correctness" != "ok" ]]; then echo "output mismatch between sides"; fi
    if [[ -n "$failures" ]]; then echo "$failures"; else echo "none"; fi
} >>"$report"

cat "$report"
[[ -z "$failures" && "$correctness" == "ok" ]]
