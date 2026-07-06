#!/usr/bin/env bash
#
# bench-ab.sh — A/B performance comparison of two pivotdb source trees.
#
# Runs entirely on the benchmark box. For each of the two trees ("before" and
# "after") it builds a PGO binary (profile generated from a small subset, then a
# profile-use build), then times the ClickBench query set against the full
# dataset. The "before" timings are written to a baseline JSON; the "after" run
# loads that baseline and prints a per-query cold/hot comparison table. If any
# query's hot time regressed past the threshold, the script exits non-zero.
#
# It is meant to be launched detached and polled from a short-lived ssh session,
# so it writes its PID and emits a sentinel on every exit path (success, failure,
# early error) for the poller to match.
#
# Usage:
#   bench-ab.sh \
#     --before-dir ~/perf-ab/before --after-dir ~/perf-ab/after \
#     --source ~/hits --pgo-subset ~/hits-pgo-subset \
#     --iterations 6 --regression-pct 5 --report /tmp/ab-report.txt \
#     [--before-label <sha>] [--after-label <sha>] \
#     [--query 7,20] [--no-skip-check] [--no-drop-caches]

set -uo pipefail

pid_file="${PID_FILE:-/tmp/bench-ab.pid}"
echo $$ >"$pid_file"
# Fire on every exit path (success, `set -e` abort, early error) carrying the
# real exit code, so a poller watching the log never hangs on a silent failure.
trap 'echo "=== BENCH-AB COMPLETE exit=$? ==="' EXIT
set -e

before_dir=""
after_dir=""
source_path=""
pgo_subset=""
iterations=6
regression_pct=5
report="/tmp/ab-report.txt"
before_label="before"
after_label="after"
queries=""
skip_check=1
drop_caches=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --before-dir)     before_dir="$2"; shift 2 ;;
        --after-dir)      after_dir="$2"; shift 2 ;;
        --source)         source_path="$2"; shift 2 ;;
        --pgo-subset)     pgo_subset="$2"; shift 2 ;;
        --iterations)     iterations="$2"; shift 2 ;;
        --regression-pct) regression_pct="$2"; shift 2 ;;
        --report)         report="$2"; shift 2 ;;
        --before-label)   before_label="$2"; shift 2 ;;
        --after-label)    after_label="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        --no-skip-check)  skip_check=0; shift ;;
        --no-drop-caches) drop_caches=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for req in before_dir after_dir source_path pgo_subset; do
    if [[ -z "${!req}" ]]; then
        echo "error: --${req//_/-} is required" >&2
        exit 2
    fi
done

# Paths may arrive with a leading ~ (a launcher can't expand it against this
# box's home), so expand it here against $HOME.
expand_tilde() {
    # The "~" patterns match a literal leading tilde in the argument; the
    # expansion below uses $HOME, so SC2088 does not apply here.
    # shellcheck disable=SC2088
    case "$1" in
        "~")   printf '%s' "$HOME" ;;
        "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}
before_dir="$(expand_tilde "$before_dir")"
after_dir="$(expand_tilde "$after_dir")"
source_path="$(expand_tilde "$source_path")"
pgo_subset="$(expand_tilde "$pgo_subset")"

# The baseline the "before" run writes and the "after" run compares against.
baseline_json="/tmp/ab-baseline.json"
rm -f "$baseline_json"

# cargo / just / llvm-profdata live under the login home but not on the
# non-interactive ssh PATH, so add them explicitly.
export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin"
# Deterministic, parseable table output (no ANSI colour, no terminal probing).
export NO_COLOR=1

query_args=()
[[ -n "$queries" ]] && query_args=(--query "$queries")
common_args=(--source "$source_path" --iterations "$iterations")
[[ $skip_check -eq 1 ]] && common_args+=(--skip-check)
[[ $drop_caches -eq 1 ]] && common_args+=(--drop-caches)

# Build a PGO binary for the tree in $1 and run the query set. $2 selects the
# baseline mode: "record" writes the timings as the baseline; "compare" loads
# the baseline and renders the cold/hot diff (its stdout is the report). Each
# tree keeps its own target dirs, but the PGO profile dir is shared, so it is
# cleared before generating a fresh profile for this tree.
run_tree() {
    local tree="$1" mode="$2"
    cd "$tree/benchmarks"

    # Fresh profile for this exact build: a stale profraw from the other tree
    # would be merged in and skew the profile-use build. Profile the same query
    # set that will be measured, from the small subset (instrumented is slow).
    just pgo-clean
    just pgo-gen run --release -- \
        --source "$pgo_subset" --iterations 1 --update-results "${query_args[@]}"

    if [[ "$mode" == "record" ]]; then
        just pgo-use run --release -- \
            "${common_args[@]}" "${query_args[@]}" \
            --regression-pct "$regression_pct" \
            --baseline "$baseline_json" --force-save
    else
        just pgo-use run --release -- \
            "${common_args[@]}" "${query_args[@]}" \
            --regression-pct "$regression_pct" \
            --baseline "$baseline_json"
    fi
}

echo "=== A/B: '$before_label' (before) vs '$after_label' (after) ===" >"$report"
echo "source=$source_path iterations=$iterations regression_pct=$regression_pct" >>"$report"
echo >>"$report"

echo ">>> building + timing BEFORE ($before_label)"
run_tree "$before_dir" record >/dev/null

echo ">>> building + timing AFTER ($after_label)"
# The compare run's stdout carries the comparison table; keep it in the report.
run_tree "$after_dir" compare | tee -a "$report"

# Gate on hot regressions only: the comparison table's per-query row is
# "id cold_base cold_new cold_Δ% hot_base hot_new hot_Δ% status"; field 7 is the
# hot delta as a signed percent (e.g. "+6.2%"), and a positive value past the
# threshold is a slowdown. Cold-only wobble (page-cache noise) does not fail.
regressions="$(awk -v t="$regression_pct" '
    $1 ~ /^q[0-9]+$/ && NF >= 8 {
        d = $7; gsub(/[%+]/, "", d)
        if (d ~ /^-?[0-9.]+$/ && d + 0 >= t) print $1 " " d "%"
    }' "$report")"

{
    echo
    echo "=== hot regressions (>= ${regression_pct}%) ==="
    if [[ -n "$regressions" ]]; then
        echo "$regressions"
    else
        echo "none"
    fi
} | tee -a "$report"

if [[ -n "$regressions" ]]; then
    exit 3
fi
