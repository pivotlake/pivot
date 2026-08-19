#!/usr/bin/env bash
#
# bench-concurrent-ab.sh - A/B comparison of two pivotdb commits under
# concurrent load, run on a freshly launched instance-store machine.
#
# Uses the same box-side machinery as the other A/B harnesses
# (benchmarks/lib/bench-ab-common.sh): checkouts from the AMI-baked repo
# clone, NVMe mount, S3 dataset sync overlapping the builds, warm cargo
# caches, per-side PGO or plain release builds.
#
# The measurement is pivot-bench's --clients mode: per pass, per side and per
# client count, one fresh server does an untimed warmup sweep and then N
# clients (each with its own deterministic shuffled order) run every query
# concurrently. The per-count metric is the concurrent wall, min across
# passes; sides alternate within each pass.
#
# Both sides' servers are driven by the AFTER side's pivot-bench binary: the
# client links no engine code (only the server binary is being measured), and
# a before commit that predates the --clients mode could not drive itself.
#
# The gate: any client count whose after wall regresses past
# --regression-pct fails the run. Query outputs are not checked (the
# concurrency mode is timings-only), and aborted executions are reported per
# side but do not gate.
#
# Meant to be launched detached and polled over ssh, so it writes its PID and
# emits a sentinel on every exit path.
#
# Usage:
#   bench-concurrent-ab.sh \
#     --mirror /opt/pivotdb --repo-url <tokenised fetch url> \
#     --before <sha> --after <sha> \
#     --data-bucket s3://bucket --cache-prefix s3://bucket/cache \
#     [--mode release|pgo] [--passes 3] [--regression-pct 10] \
#     [--clients "1 3 6"] [--order shuffled|same] [--scale clickbench-quarter] \
#     [--bench-env 'PIVOT_X=1'] [--report /tmp/ab-report.txt] \
#     [--before-label <sha>] [--after-label <sha>]

set -uo pipefail

_here="${BASH_SOURCE[0]%/*}"
_common="$_here/bench-ab-common.sh"
[[ -f "$_common" ]] || _common="$_here/../lib/bench-ab-common.sh"
# shellcheck source=../lib/bench-ab-common.sh
source "$_common"

pid_file="${PID_FILE:-/tmp/bench-concurrent-ab.pid}"
echo $$ >"$pid_file"
trap 'echo "=== BENCH-CONCURRENT-AB COMPLETE exit=$? ==="' EXIT
set -e

mirror="/opt/pivotdb"
repo_url=""
before_sha=""
after_sha=""
work_dir=""
data_bucket=""
data_root="/mnt/nvme/clickbench"
cache_prefix=""
mode="release"
passes=3
regression_pct=10
client_counts="1 3 6"
order="shuffled"
scale="clickbench-quarter"
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
        --clients)        client_counts="$2"; shift 2 ;;
        --order)          order="$2"; shift 2 ;;
        --scale)          scale="$2"; shift 2 ;;
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
measure_data="$data_root/$scale"

# ---------------------------------------------------------------------------
# Phase 0: disk, data, checkouts.
# ---------------------------------------------------------------------------
mount_nvme
mkdir -p "$work_dir"
data_dev="$(df --output=source "$(dirname "$data_root")" | tail -1 | sed 's|/dev/||')"

( sync_scale "$scale" ) &
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

    echo ">>> waiting for the $scale dataset (profiling runs on it too)"
    wait_for_scale "$scale"
    profile_side "$before_dir" before clickbench "$measure_data"
    profile_side "$after_dir" after clickbench "$measure_data"

    echo ">>> profile-use builds (A and B in parallel)"
    build_use "$before_dir" before & b1=$!
    build_use "$after_dir" after & b2=$!
    wait "$b1"; wait "$b2"

    client_bin="$after_dir/benchmarks/target-client/release/pivot-bench"
    before_server="$before_dir/benchmarks/target-pgouse/$host_target/release/pivotdb-server"
    after_server="$after_dir/benchmarks/target-pgouse/$host_target/release/pivotdb-server"
else
    echo ">>> release builds (A and B in parallel)"
    build_release "$before_dir" & b1=$!
    build_release "$after_dir" & b2=$!
    wait "$b1"; wait "$b2"
    client_bin="$after_dir/target/release/pivot-bench"
    before_server="$before_dir/target/release/pivotdb-server"
    after_server="$after_dir/target/release/pivotdb-server"
fi
for built in "$client_bin" "$before_server" "$after_server"; do
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
wait_for_scale "$scale"
wait

# ---------------------------------------------------------------------------
# Phase 2: measurement. Sides alternate within each pass; every (side, count)
# run boots a fresh server, warms it with one untimed sweep, and measures the
# concurrent wall of N shuffled clients. Walls land in per-run JSONs the
# report step reads.
# ---------------------------------------------------------------------------
suite_dir="$after_dir/benchmarks/clickbench"
results_dir="$work_dir/results"
rm -rf "$results_dir"; mkdir -p "$results_dir"

run_side_count() {
    local server="$1" side="$2" count="$3" pass="$4"
    "$client_bin" --suite clickbench --suite-dir "$suite_dir" \
        --server-bin "$server" --source "$measure_data" \
        --clients "$count" --order "$order" --warmup-sweep --skip-check \
        --json-out "$results_dir/$side-c$count-p$pass.json" >/dev/null
}

for pass in $(seq 1 "$passes"); do
    for count in $client_counts; do
        echo ">>> pass $pass: $count clients, before then after"
        run_side_count "$before_server" before "$count" "$pass"
        run_side_count "$after_server" after "$count" "$pass"
    done
done

# ---------------------------------------------------------------------------
# Phase 3: report + gate, from the JSONs. Per count: min wall across passes
# per side, the after/before delta, and each side's aborted executions.
# ---------------------------------------------------------------------------
gate=0
python3 - "$results_dir" "$regression_pct" "$before_label" "$after_label" \
    "$passes" "$order" "$scale" $client_counts >"$report" <<'EOF' || gate=$?
import glob, json, sys
results_dir, pct, before_label, after_label, passes, order, scale = sys.argv[1:8]
counts = [int(c) for c in sys.argv[8:]]

def stats(side, count):
    walls, failed = [], 0
    for path in sorted(glob.glob(f"{results_dir}/{side}-c{count}-p*.json")):
        data = json.load(open(path))
        walls.append(data["concurrent_wall_ms"])
        failed += sum(
            1
            for client in data["concurrent"]
            for e in client["executions"]
            if not e["completed"]
        )
    return min(walls), failed

print(f"concurrency A/B on {scale} ({order} order, wall = min over {passes} passes)")
print(f"{'clients':<8} {before_label:>14} {after_label:>14} {'delta':>8} {'fail b/a':>9}")
failed_gate = []
for count in counts:
    before_wall, before_failed = stats("before", count)
    after_wall, after_failed = stats("after", count)
    delta = (after_wall - before_wall) / before_wall * 100
    marker = ""
    if delta >= float(pct):
        marker = "  REGRESSION"
        failed_gate.append(count)
    print(
        f"{count:<8} {before_wall / 1000:>13.2f}s {after_wall / 1000:>13.2f}s "
        f"{delta:>+7.1f}% {before_failed:>4}/{after_failed}{marker}"
    )
sys.exit(1 if failed_gate else 0)
EOF
cat "$report"
exit "$gate"
