#!/usr/bin/env bash
#
# Run the PGO training workload against an instrumented `pivot` and collect
# the raw profiles. Mirrors what the perf-box A/B harness trains with, plus
# the write path:
#
#   ClickBench   43 queries x 2 iterations over a small hits subset
#   TPC-H        22 queries x 2 iterations over SF1
#   bulk-insert  one INSERT ... SELECT of 1M hits rows into a native table
#
# pivot-bench boots the server itself over a scratch datastore, so this needs
# nothing installed beyond the two binaries. Training data is a directory
# holding `clickbench-hits-subset/` (flat parquet) and `tpch-sf1/` (one
# subdirectory per table); the canonical copy lives in the deploy GCS bucket,
# see the release workflow.
#
# The datasets are deliberately small: instrumented code runs its hot loops
# ~80x slower, and a profile needs branch bias from representative data, not
# scale. Never point this at a measurement dataset.
#
# Usage:
#   packaging/pgo/train.sh --server-bin BIN --bench-bin BIN \
#       --data-dir DIR --out-dir DIR

set -euo pipefail

fail() {
    echo "pgo-train: $*" >&2
    exit 1
}

server_bin=""
bench_bin=""
data_dir=""
out_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-bin) server_bin="$2"; shift 2 ;;
        --bench-bin)  bench_bin="$2"; shift 2 ;;
        --data-dir)   data_dir="$2"; shift 2 ;;
        --out-dir)    out_dir="$2"; shift 2 ;;
        *) fail "unknown argument: $1" ;;
    esac
done
[[ -x "$server_bin" ]] || fail "--server-bin '$server_bin' is not an executable"
[[ -x "$bench_bin" ]] || fail "--bench-bin '$bench_bin' is not an executable"
[[ -d "$data_dir" ]] || fail "--data-dir '$data_dir' does not exist"
[[ -n "$out_dir" ]] || fail "--out-dir is required"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hits_subset="$data_dir/clickbench-hits-subset"
tpch_sf1="$data_dir/tpch-sf1"
[[ -d "$hits_subset" ]] || fail "$hits_subset does not exist"
[[ -d "$tpch_sf1" ]] || fail "$tpch_sf1 does not exist"

mkdir -p "$out_dir"
rm -f "$out_dir"/*.profraw

# PIVOT_SPIN_LIMIT=0 parks idle workers immediately instead of spinning, so
# the profile records the wait-heavy control-flow mix that cold runs on full
# datasets execute, deterministically rather than as a timing-dependent draw
# per build.
train_suite() {
    local suite="$1" source="$2" iterations="$3"
    echo ">>> training: $suite ($iterations iteration(s)) over $source"
    LLVM_PROFILE_FILE="$out_dir/%m-%p.profraw" PIVOT_SPIN_LIMIT=0 \
        "$bench_bin" \
        --suite "$suite" --suite-dir "$repository_root/benchmarks/$suite" \
        --server-bin "$server_bin" \
        --source "$source" --iterations "$iterations" --skip-check >/dev/null
}

train_suite clickbench "$hits_subset" 2
train_suite tpch "$tpch_sf1" 2
train_suite bulk-insert "$hits_subset" 1

raw_count="$(find "$out_dir" -name '*.profraw' | wc -l)"
[[ "$raw_count" -gt 0 ]] || fail "the training runs produced no .profraw files under $out_dir"
echo ">>> training done: $raw_count profraw files under $out_dir"
