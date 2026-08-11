#!/usr/bin/env bash
#
# bench-tpch-flat-ab.sh - A/B performance comparison of two pivotdb commits on the
# TPC-H flat suite, run on a freshly launched instance-store machine.
#
# Owns the whole box side: checks both commits out of the AMI-baked repo clone,
# formats and mounts the instance-store NVMe, syncs the datasets from S3 (small
# PGO scale first, then the measurement scale) while both sides build, builds a
# per-side pivot-bench (PGO by default, plain release with --mode release),
# restores/saves warm cargo target dirs from S3 so only the commit diff and the
# profile-dependent Rust units recompile, then times the suite cold and hot
# with the two sides interleaved per query so slow drift in instance-store
# throughput (fresh boots read measurably faster than steady state) hits both
# sides equally. Optionally times DuckDB over the same parquet as a reference
# column; DuckDB never gates the run.
#
# The gate: any query whose COLD time regresses after vs before by at least
# --regression-pct fails the run, as does a missing timing or an output
# mismatch between the two sides.
#
# The box-side machinery it shares with the other A/B harnesses (NVMe mount,
# checkouts, S3 dataset sync, warm-cache restore/save, PGO builds,
# correctness) lives in benchmarks/lib/bench-ab-common.sh; what is specific to
# the TPC-H flat suite stays here: the per-query interleaved measurement.
#
# Meant to be launched detached and polled over ssh, so it writes its PID and
# emits a sentinel on every exit path.
#
# Usage:
#   bench-tpch-flat-ab.sh \
#     --mirror /opt/pivotdb --repo-url <tokenised fetch url> \
#     --before <sha> --after <sha> \
#     --data-bucket s3://bucket/prefix --cache-prefix s3://bucket/cache \
#     [--mode pgo|release] [--iterations 3] [--passes 2] [--regression-pct 5] \
#     [--query q06,q12] [--duckdb] [--bench-env 'PIVOT_X=1 PIVOT_Y=true'] \
#     [--report /tmp/ab-report.txt] [--before-label <sha>] [--after-label <sha>]

set -uo pipefail

# The shared library sits at ../lib in the repo, but the workflow ships both
# files flat into one /tmp directory on the box; find it either way.
_here="${BASH_SOURCE[0]%/*}"
_common="$_here/bench-ab-common.sh"
[[ -f "$_common" ]] || _common="$_here/../lib/bench-ab-common.sh"
# shellcheck source=../lib/bench-ab-common.sh
source "$_common"

pid_file="${PID_FILE:-/tmp/bench-tpch-flat-ab.pid}"
echo $$ >"$pid_file"
# Fire on every exit path carrying the real exit code, so a poller watching the
# log never hangs on a silent failure.
trap 'echo "=== BENCH-TPCH-FLAT-AB COMPLETE exit=$? ==="' EXIT
set -e

mirror="/opt/pivotdb"
repo_url=""
before_sha=""
after_sha=""
# Defaults to <nvme mount>/ab once data_root is known: the checkouts and their
# target dirs run to well over 100GB across the sides while the AMI's root
# volume is 40GB, so the builds must live on the instance store too.
work_dir=""
data_bucket=""
data_root="/mnt/nvme/tpch"
cache_prefix=""
mode="pgo"
iterations=3
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
        --iterations)     iterations="$2"; shift 2 ;;
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

work_dir="${work_dir:-$(dirname "$data_root")/ab}"
sf_pgo="sf1"
sf_measure="sf100"
pgo_data="$data_root/$sf_pgo/flat"
measure_data="$data_root/$sf_measure/flat"

# ---------------------------------------------------------------------------
# Measurement helpers. The dataset sync (sync_scale/wait_for_scale) and the
# io_ticks disk-state signal come from bench-ab-common.sh.
# ---------------------------------------------------------------------------
data_dev="" # set after mount, for the io_ticks sanity column

# Runs one side's pivot-bench for one query with $iterations tries in one
# process, echoing "cold hot io_seconds": cold is try 1 in ms, hot the min of
# the rest ("null" when missing), io_seconds the device's io_ticks delta as a
# disk-state sanity signal.
run_pivot() {
    local bin="$1" server="$2" dir="$3" query="$4"
    local t0 t1 out times
    t0="$(io_ticks)"
    out="$("$bin" --suite tpch-flat --suite-dir "$dir/benchmarks/tpch-flat" \
        --server-bin "$server" \
        --source "$measure_data" --query "$query" \
        --iterations "$iterations" --skip-check 2>&1)" || true
    t1="$(io_ticks)"
    # Timing lines end in "<ms>ms"; take the numbers without depending on the
    # separator glyph the runner prints.
    times="$(grep -E "Query q[0-9]+" <<<"$out" | grep -oE '[0-9]+ms$' | sed 's/ms//')"
    local cold="null" hot="null" t
    while IFS= read -r t; do
        [[ -n "$t" ]] || continue
        if [[ "$cold" == "null" ]]; then cold="$t"
        elif [[ "$hot" == "null" || "$t" -lt "$hot" ]]; then hot="$t"; fi
    done <<<"$times"
    echo "$cold $hot $(awk -v a="${t0:-0}" -v b="${t1:-0}" 'BEGIN{printf "%.1f", (b-a)/1000}')"
}

# Same contract for DuckDB: one process, the view created once, the query run
# $iterations times with .timer on; times parsed from "Run Time (s): real X".
run_duckdb_query() {
    local sql_file="$1"
    local script times
    script="$(mktemp)"
    {
        echo ".bail on"
        echo "CREATE VIEW tpch_flat AS SELECT * FROM read_parquet('$measure_data/*.parquet');"
        echo ".timer on"
        for _ in $(seq "$iterations"); do
            # Exactly one trailing semicolon regardless of how the file ends.
            sed -e '$ s/[[:space:];]*$/;/' "$sql_file"
        done
    } >"$script"
    times="$(duckdb <"$script" 2>&1 | grep -oE 'real[[:space:]]+[0-9.]+' | awk '{print int($2 * 1000)}')" || true
    rm -f "$script"
    local cold="null" hot="null" t
    while IFS= read -r t; do
        [[ -n "$t" ]] || continue
        if [[ "$cold" == "null" ]]; then cold="$t"
        elif [[ "$hot" == "null" || "$t" -lt "$hot" ]]; then hot="$t"; fi
    done <<<"$times"
    echo "$cold $hot"
}

# ---------------------------------------------------------------------------
# Phase 0: disk, data, checkouts.
# ---------------------------------------------------------------------------
mount_nvme
mkdir -p "$work_dir"
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
    profile_side "$before_dir" before tpch-flat "$pgo_data"
    profile_side "$after_dir" after tpch-flat "$pgo_data"

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
    queries="$(cd "$after_dir/benchmarks/tpch-flat" && ls q*.sql | sed 's/\.sql$//' | paste -sd,)"
fi
echo ">>> burn-in + result capture (queries: $queries)"
drop_caches
"$before_bin" --suite tpch-flat --suite-dir "$before_dir/benchmarks/tpch-flat" \
    --server-bin "$before_server" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null
drop_caches
"$after_bin" --suite tpch-flat --suite-dir "$after_dir/benchmarks/tpch-flat" \
    --server-bin "$after_server" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null

correctness="ok"
if ! compare_outputs "$before_dir/benchmarks/tpch-flat" "$after_dir/benchmarks/tpch-flat"; then
    correctness="MISMATCH"
fi

# ---------------------------------------------------------------------------
# Phase 3: measurement, sides interleaved per query. Every timed run starts
# from a dropped page cache; hot iterations reuse the warm process exactly the
# way the suite is normally run.
# ---------------------------------------------------------------------------
# Cold times off a dropped cache are single samples and instance-store reads
# drift; several interleaved passes with a per-query min give a stable cold
# number the gate can trust.
rows="/tmp/ab-rows.tsv"
: >"$rows"
IFS=',' read -ra qlist <<<"$queries"
echo ">>> measuring (iterations=$iterations, passes=$passes, mode=$mode)"
for pass in $(seq "$passes"); do
    for q in "${qlist[@]}"; do
        drop_caches; sleep 3
        read -r cold hot io <<<"$(run_pivot "$before_bin" "$before_server" "$before_dir" "$q")"
        echo -e "$q\tbefore\t$cold\t$hot\t$io" >>"$rows"
        drop_caches; sleep 3
        read -r cold hot io <<<"$(run_pivot "$after_bin" "$after_server" "$after_dir" "$q")"
        echo -e "$q\tafter\t$cold\t$hot\t$io" >>"$rows"
        if [[ "$run_duckdb" == "1" ]]; then
            drop_caches; sleep 3
            read -r cold hot <<<"$(run_duckdb_query "$after_dir/benchmarks/tpch-flat/$q.sql")"
            echo -e "$q\tduckdb\t$cold\t$hot\t-" >>"$rows"
        fi
        echo "    $q pass $pass done"
    done
done

# ---------------------------------------------------------------------------
# Report + gate. Cold is the gate (this suite optimizes cold reads); hot and
# the io column are informational. DuckDB, when present, is a reference only.
# ---------------------------------------------------------------------------
{
    echo "=== TPC-H flat A/B: '$before_label' (before) vs '$after_label' (after) ==="
    echo "mode=$mode source=$measure_data iterations=$iterations passes=$passes regression_pct=$regression_pct (cold/hot are per-query mins across passes)"
    [[ -n "$bench_env" ]] && echo "bench_env=$bench_env"
    [[ "$mode" == "release" ]] && echo "NOTE: release mode is non-PGO; numbers carry code-alignment noise, use for iteration only"
    echo "correctness (before vs after, 1e-6 relative): $correctness"
    echo
} >"$report"

failures="$(awk -F'\t' -v t="$regression_pct" -v duck="$run_duckdb" '
function pct(b, a) { b = (b < 1 ? 1 : b); return (a - b) / b * 100 }
function valid(x) { return (x != "" && x != "null") }
function keepmin(arr, k, v) {
    if (v == "null") { if (!(k in arr)) arr[k] = v; return }
    if (!(k in arr) || arr[k] == "null" || v + 0 < arr[k] + 0) arr[k] = v
}
{ keepmin(c, $1 "," $2, $3); keepmin(h, $1 "," $2, $4); io[$1 "," $2] = $5
  if (!seen[$1]++) order[n++] = $1 }
END {
    hdr = sprintf("%-5s %8s %8s %8s   %8s %8s %8s   %7s", \
        "query", "cold_b", "cold_a", "coldD%", "hot_b", "hot_a", "hotD%", "io_b/a")
    if (duck == 1) hdr = hdr sprintf("   %8s %8s %7s", "dk_cold", "dk_hot", "a/dk")
    print hdr "  status" > "/dev/stderr"
    for (i = 0; i < n; i++) {
        q = order[i]
        cb = c[q ",before"]; ca = c[q ",after"]; hb = h[q ",before"]; ha = h[q ",after"]
        status = "ok"; cd = "-"; hd = "-"
        if (!valid(cb) || !valid(ca)) {
            status = "ERR"; print q " ERR (missing timing)"
        } else {
            cp = pct(cb, ca); cd = sprintf("%+.1f%%", cp)
            if (cp >= t) { status = "REGRESSION"; print q " cold " cd }
            else if (cp <= -t) status = "improved"
        }
        if (valid(hb) && valid(ha)) hd = sprintf("%+.1f%%", pct(hb, ha))
        line = sprintf("%-5s %8s %8s %8s   %8s %8s %8s   %7s", \
            q, cb, ca, cd, hb, ha, hd, io[q ",before"] "/" io[q ",after"])
        if (duck == 1) {
            dc = c[q ",duckdb"]; dh = h[q ",duckdb"]; r = "-"
            if (valid(ca) && valid(dc)) r = sprintf("%.2fx", (ca + 10) / (dc + 10))
            line = line sprintf("   %8s %8s %7s", dc, dh, r)
        }
        print line "  " status > "/dev/stderr"
        if (valid(cb) && valid(ca)) {
            b = (cb < ca ? cb : ca)
            bcl += log((cb + 10) / (b + 10)); acl += log((ca + 10) / (b + 10)); cn++
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
