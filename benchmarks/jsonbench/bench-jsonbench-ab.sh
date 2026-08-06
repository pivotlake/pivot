#!/usr/bin/env bash
#
# bench-jsonbench-ab.sh - A/B performance comparison of two pivotdb commits on
# the JSONBench suite, run on a freshly launched instance-store machine.
#
# Shares its box-side machinery (NVMe mount, checkouts, warm-cache restore/save,
# PGO builds, correctness) with the TPC-H harness via benchmarks/lib/bench-ab-common.sh;
# what is specific to JSONBench lives here: the public Bluesky ndjson download,
# and the load-once-then-query measurement.
#
# JSONBench loads differently from the TPC-H flat suite. There the source is
# Parquet read in place, so each query is a fresh cold process. Here the source
# is ndjson and pivot-bench loads the whole table through `INSERT`+compaction
# inside one process before it queries. So a side is measured with ONE process
# that loads once and runs every query with `--drop-caches`, giving a true cold
# (first iteration off a dropped cache) and hot (min of the rest) per query,
# plus the ingest time the load itself took. The written Parquet goes to TMPDIR,
# pointed at the NVMe here; the source dir holds only the ndjson.
#
# The gate: any query whose COLD time, or the LOAD (ingest) time, regresses
# after vs before by at least --regression-pct fails the run, as does a missing
# timing or an output mismatch between the two sides.
#
# Meant to be launched detached and polled over ssh, so it writes its PID and
# emits a sentinel on every exit path.
#
# Usage:
#   bench-jsonbench-ab.sh \
#     --mirror /opt/pivotdb --repo-url <tokenised fetch url> \
#     --before <sha> --after <sha> \
#     --cache-prefix s3://bucket/cache \
#     [--scale 1m|10m|100m] [--mode pgo|release] [--iterations 3] [--passes 1] \
#     [--regression-pct 5] [--query q01,q02] [--duckdb] \
#     [--bench-env 'PIVOT_X=1'] [--report /tmp/ab-report.txt] \
#     [--before-label <sha>] [--after-label <sha>]

set -uo pipefail

# The shared library sits at ../lib in the repo, but the workflow ships both
# files flat into one /tmp directory on the box; find it either way.
_here="${BASH_SOURCE[0]%/*}"
_common="$_here/bench-ab-common.sh"
[[ -f "$_common" ]] || _common="$_here/../lib/bench-ab-common.sh"
# shellcheck source=../lib/bench-ab-common.sh
source "$_common"
# Absolute, because the suite scripts this one drives sit beside it and later
# phases do not necessarily run from here.
own_dir="$(cd "$_here" && pwd)"

pid_file="${PID_FILE:-/tmp/bench-jsonbench-ab.pid}"
echo $$ >"$pid_file"
# Fire on every exit path carrying the real exit code, so a poller watching the
# log never hangs on a silent failure.
trap 'echo "=== BENCH-JSONBENCH-AB COMPLETE exit=$? ==="' EXIT
set -e

mirror="/opt/pivotdb"
repo_url=""
before_sha=""
after_sha=""
work_dir="$HOME/ab"
data_root="/mnt/nvme/jsonbench"
cache_prefix=""
scale="10m"
mode="pgo"
iterations=3
passes=1
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
        --data-root)      data_root="$2"; shift 2 ;;
        --cache-prefix)   cache_prefix="$2"; shift 2 ;;
        --scale)          scale="$2"; shift 2 ;;
        --mode)           mode="$2"; shift 2 ;;
        --iterations)     iterations="$2"; shift 2 ;;
        --passes)         passes="$2"; shift 2 ;;
        --regression-pct) regression_pct="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        --duckdb)         run_duckdb=1; shift ;;
        --bench-env)      bench_env="$2"; shift 2 ;;
        --report)         report="$2"; shift 2 ;;
        --before-label)   before_label="$2"; shift 2 ;;
        --after-label)    after_label="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for req in before_sha after_sha; do
    if [[ -z "${!req}" ]]; then
        echo "error: --${req%_sha} is required" >&2
        exit 2
    fi
done
if [[ "$mode" != "pgo" && "$mode" != "release" ]]; then
    echo "error: --mode must be pgo or release" >&2
    exit 2
fi
case "$scale" in
    1m)   measure_files=1 ;;
    10m)  measure_files=10 ;;
    100m) measure_files=100 ;;
    *) echo "error: --scale must be 1m|10m|100m (got '$scale')" >&2; exit 2 ;;
esac

ab_common_init

# One file for the PGO profiling run, the requested scale for measurement, in
# separate directories because pivot-bench loads every ndjson under --source.
# `prep-jsonbench-data.sh` owns the download and lays the files out under
# `<dir>/ndjson`.
pgo_root="$data_root/pgo"
measure_root="$data_root/measure"
pgo_data="$pgo_root/ndjson"
measure_data="$measure_root/ndjson"
# The table's Parquet is written under TMPDIR (catalog store root); keep it on
# the NVMe, not the small root volume.
export TMPDIR="$data_root/tmp"

# ---------------------------------------------------------------------------
# Dataset download, backgrounded: the one PGO file lands first (it gates the
# profiling runs), the measurement scale after (it gates the bench phase). Each
# finished scale is marked with a .done sentinel.
#
# The suite's own `prep-jsonbench-data.sh` fetches it, rather than a second
# downloader here. It is the only thing that rejoins the records upstream cut in
# half at a 64KB boundary, and both engines refuse those: DuckDB rejects the
# whole file and pivot's cast to VARIANT fails on the fragment. A separate
# download would hand them the raw files and every scale past four would fail.
# ---------------------------------------------------------------------------
download_scale() {
    local count="$1" root="$2"
    if [[ -f "$root.done" ]]; then return; fi
    "$own_dir/prep-jsonbench-data.sh" --files "$count" --dir "$root" --engines none
    touch "$root.done"
    echo ">>> dataset ($count file(s)) ready in $root/ndjson"
}

wait_for_download() {
    local root="$1"
    while [[ ! -f "$root.done" ]]; do
        if ! kill -0 "$download_pid" 2>/dev/null; then
            echo "error: dataset download died before $root finished" >&2
            exit 1
        fi
        sleep 5
    done
}

# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------
# Runs one side's whole suite in one process: loads the table once, then runs
# every selected query $iterations times with the file/page caches dropped
# before each query. Appends a row per query ("qNN <side> <cold> <hot>") and
# one "__load__ <side> <load_ms> <load_ms>" row to $rows. cold is iteration 1
# (a true cold read off the dropped cache), hot the min of the rest.
run_suite_side() {
    local bin="$1" dir="$2" side="$3"
    local out
    out="$("$bin" --suite jsonbench --suite-dir "$dir/benchmarks/jsonbench" \
        --source "$measure_data" --query "$queries" \
        --iterations "$iterations" --drop-caches --skip-check 2>&1)" || true

    local load
    load="$(grep -E '=== Load ' <<<"$out" | grep -oE '[0-9]+ms' | grep -oE '[0-9]+' | tail -1)"
    echo -e "__load__\t$side\t${load:-null}\t${load:-null}" >>"$rows"

    local q times cold hot t
    for q in "${qlist[@]}"; do
        # Timing lines read "[i/N] Query qNN — <ms>ms"; take this query's times
        # in order without depending on the separator glyph.
        times="$(grep -E "Query $q " <<<"$out" | grep -oE '[0-9]+ms' | grep -oE '[0-9]+')"
        cold="null"; hot="null"
        while IFS= read -r t; do
            [[ -n "$t" ]] || continue
            if [[ "$cold" == "null" ]]; then cold="$t"
            elif [[ "$hot" == "null" || "$t" -lt "$hot" ]]; then hot="$t"; fi
        done <<<"$times"
        echo -e "$q\t$side\t$cold\t$hot" >>"$rows"
    done
}

# DuckDB reference (never gates): its official ndjson load once into a temp db,
# then each query's -duckdb.sql run $iterations times with .timer on.
run_duckdb_side() {
    local suite_dir="$1"
    local db="$TMPDIR/duckdb-ref.db"
    rm -f "$db" "$db.wal"
    drop_caches
    duckdb "$db" -c "$(cat "$suite_dir/duckdb-official/ddl.sql")" >/dev/null 2>&1 || true
    local load_start load_ms
    load_start="$(date +%s%3N)"
    duckdb "$db" -c "INSERT INTO bluesky SELECT * FROM read_ndjson_objects('$measure_data/*.json.gz', ignore_errors=false, maximum_object_size=1048576000);" >/dev/null 2>&1 || true
    load_ms=$(( $(date +%s%3N) - load_start ))
    echo -e "__load__\tduckdb\t$load_ms\t$load_ms" >>"$rows"

    local q script times cold hot t
    for q in "${qlist[@]}"; do
        local sql_file="$suite_dir/$q-duckdb.sql"
        [[ -f "$sql_file" ]] || { echo -e "$q\tduckdb\tnull\tnull" >>"$rows"; continue; }
        drop_caches
        script="$(mktemp)"
        {
            echo ".bail on"
            echo ".timer on"
            for _ in $(seq "$iterations"); do
                sed -e '$ s/[[:space:];]*$/;/' "$sql_file"
            done
        } >"$script"
        times="$(duckdb "$db" <"$script" 2>&1 | grep -oE 'real[[:space:]]+[0-9.]+' | awk '{print int($2 * 1000)}')" || true
        rm -f "$script"
        cold="null"; hot="null"
        while IFS= read -r t; do
            [[ -n "$t" ]] || continue
            if [[ "$cold" == "null" ]]; then cold="$t"
            elif [[ "$hot" == "null" || "$t" -lt "$hot" ]]; then hot="$t"; fi
        done <<<"$times"
        echo -e "$q\tduckdb\t$cold\t$hot" >>"$rows"
    done
}

# ---------------------------------------------------------------------------
# Phase 0: disk, data, checkouts.
# ---------------------------------------------------------------------------
mkdir -p "$work_dir"
mount_nvme
mkdir -p "$TMPDIR"

( download_scale 1 "$pgo_root"; download_scale "$measure_files" "$measure_root" ) &
download_pid=$!

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
# Phase 1: cache restore + builds, overlapping the download.
# ---------------------------------------------------------------------------
restore_pids=()
restore_cargo_home & restore_pids+=($!)
if [[ "$mode" == "pgo" ]]; then
    restore_cache "pgogen-before" "$before_dir/benchmarks/target-pgogen" & restore_pids+=($!)
    restore_cache "pgogen-after" "$after_dir/benchmarks/target-pgogen" & restore_pids+=($!)
    restore_cache "pgouse-before" "$before_dir/benchmarks/target-pgouse" & restore_pids+=($!)
    restore_cache "pgouse-after" "$after_dir/benchmarks/target-pgouse" & restore_pids+=($!)
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

    echo ">>> waiting for the PGO dataset"
    wait_for_download "$pgo_root"
    echo ">>> profiling runs (1 file)"
    profile_side "$before_dir" before jsonbench "$pgo_data"
    profile_side "$after_dir" after jsonbench "$pgo_data"

    echo ">>> profile-use builds (A and B in parallel)"
    build_use "$before_dir" before & b1=$!
    build_use "$after_dir" after & b2=$!
    wait "$b1"; wait "$b2"

    before_bin="$before_dir/benchmarks/target-pgouse/$host_target/release/pivot-bench"
    after_bin="$after_dir/benchmarks/target-pgouse/$host_target/release/pivot-bench"
else
    echo ">>> release builds (A and B in parallel)"
    build_release "$before_dir" & b1=$!
    build_release "$after_dir" & b2=$!
    wait "$b1"; wait "$b2"
    before_bin="$before_dir/target/release/pivot-bench"
    after_bin="$after_dir/target/release/pivot-bench"
fi
[[ -x "$before_bin" && -x "$after_bin" ]] || { echo "error: build produced no binary" >&2; exit 1; }

echo ">>> saving warm caches"
save_cache "cargo-home" "$HOME/.cargo" registry git &
if [[ "$mode" == "pgo" ]]; then
    save_cache "pgogen-before" "$before_dir/benchmarks/target-pgogen" &
    save_cache "pgogen-after" "$after_dir/benchmarks/target-pgogen" &
    save_cache "pgouse-before" "$before_dir/benchmarks/target-pgouse" &
    save_cache "pgouse-after" "$after_dir/benchmarks/target-pgouse" &
else
    save_cache "release-before" "$before_dir/target" &
    save_cache "release-after" "$after_dir/target" &
fi
wait_for_download "$measure_root"
wait

# ---------------------------------------------------------------------------
# Phase 2: burn-in + correctness. One untimed load+query per side writes each
# side's result tsvs and, as a side effect, walks the instance store out of its
# fresh-boot throughput state before anything is measured. The two sides'
# outputs are then compared with a float tolerance; a real mismatch fails.
# ---------------------------------------------------------------------------
if [[ -z "$queries" ]]; then
    queries="$(cd "$after_dir/benchmarks/jsonbench" && ls q*.sql | grep -v -- '-duckdb' | sed 's/\.sql$//' | paste -sd,)"
fi
IFS=',' read -ra qlist <<<"$queries"
echo ">>> burn-in + result capture (queries: $queries)"
"$before_bin" --suite jsonbench --suite-dir "$before_dir/benchmarks/jsonbench" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null
"$after_bin" --suite jsonbench --suite-dir "$after_dir/benchmarks/jsonbench" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null

correctness="ok"
if ! compare_outputs "$before_dir/benchmarks/jsonbench" "$after_dir/benchmarks/jsonbench"; then
    correctness="MISMATCH"
fi

# ---------------------------------------------------------------------------
# Phase 3: measurement. Each pass loads and queries the whole suite once per
# side; per-query cold/hot and the load time are per-key mins across passes, so
# a slow single load or a drifting instance-store read is not the number that
# gates.
# ---------------------------------------------------------------------------
rows="/tmp/ab-rows.tsv"
: >"$rows"
echo ">>> measuring (scale=$scale iterations=$iterations passes=$passes mode=$mode)"
for pass in $(seq "$passes"); do
    run_suite_side "$before_bin" "$before_dir" before
    run_suite_side "$after_bin" "$after_dir" after
    if [[ "$run_duckdb" == "1" ]]; then
        run_duckdb_side "$after_dir/benchmarks/jsonbench"
    fi
    echo "    pass $pass done"
done

# ---------------------------------------------------------------------------
# Report + gate. Cold query time and LOAD (ingest) time both gate; hot is
# informational. DuckDB, when present, is a reference only.
# ---------------------------------------------------------------------------
{
    echo "=== JSONBench A/B: '$before_label' (before) vs '$after_label' (after) ==="
    echo "scale=$scale mode=$mode iterations=$iterations passes=$passes regression_pct=$regression_pct (cold/hot/load are per-key mins across passes)"
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
{ keepmin(c, $1 "," $2, $3); keepmin(h, $1 "," $2, $4)
  if (!seen[$1]++ && $1 != "__load__") order[n++] = $1 }
END {
    hdr = sprintf("%-6s %9s %9s %8s   %9s %9s %8s", \
        "query", "cold_b", "cold_a", "coldD%", "hot_b", "hot_a", "hotD%")
    if (duck == 1) hdr = hdr sprintf("   %9s %9s", "dk_cold", "dk_hot")
    print hdr "  status" > "/dev/stderr"

    # Query rows first, then the LOAD row, both gated on the cold/load column.
    order[n++] = "__load__"
    for (i = 0; i < n; i++) {
        q = order[i]
        label = (q == "__load__" ? "LOAD" : q)
        cb = c[q ",before"]; ca = c[q ",after"]; hb = h[q ",before"]; ha = h[q ",after"]
        status = "ok"; cd = "-"; hd = "-"
        if (!valid(cb) || !valid(ca)) {
            status = "ERR"; print label " ERR (missing timing)"
        } else {
            cp = pct(cb, ca); cd = sprintf("%+.1f%%", cp)
            if (cp >= t) { status = "REGRESSION"; print label (q == "__load__" ? " load " : " cold ") cd }
            else if (cp <= -t) status = "improved"
        }
        if (valid(hb) && valid(ha)) hd = sprintf("%+.1f%%", pct(hb, ha))
        line = sprintf("%-6s %9s %9s %8s   %9s %9s %8s", label, cb, ca, cd, hb, ha, hd)
        if (duck == 1) {
            dc = c[q ",duckdb"]; dh = h[q ",duckdb"]
            line = line sprintf("   %9s %9s", dc, dh)
        }
        print line "  " status > "/dev/stderr"
        if (q != "__load__" && valid(cb) && valid(ca)) { sb += cb; sa += ca }
    }
    printf "\ntotals: cold before %dms, after %dms (%+.1f%%)\n", sb, sa, pct(sb, sa) > "/dev/stderr"
}' "$rows" 2>>"$report")"

{
    echo
    echo "=== failures: cold/load regressions (>= ${regression_pct}%) or missing timings ==="
    if [[ "$correctness" != "ok" ]]; then echo "output mismatch between sides"; fi
    if [[ -n "$failures" ]]; then echo "$failures"; else echo "none"; fi
} >>"$report"

cat "$report"
[[ -z "$failures" && "$correctness" == "ok" ]]
