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
# --scale picks the measured dataset: sf100 (default) syncs the pivot-written
# SF100 from S3; sf1000 generates SF1000 on the box with tpchgen-cli (~396GB,
# faster than any sync), which needs a large instance-store array such as
# c8gd.metal-48xl's six striped disks. The PGO profile is taken on SF1 either
# way. q11's HAVING fraction is rewritten to the scale's 0.0001/SF.
#
# --engines adds reference columns, which never gate, from this list (or
# "all"):
#   duckdb-parquet      DuckDB over the same parquet directories pivot reads
#   clickhouse-parquet  ClickHouse, File(Parquet) tables over those directories
#   datafusion-parquet  DataFusion (datafusion-cli), external parquet tables
#   duckdb-native       DuckDB over its own database file
#   clickhouse-native   ClickHouse over MergeTree tables
# The native formats are loaded on the box from the measured parquet by
# load-native-dbs.sh, so every engine holds exactly the rows pivot reads.
# Each engine streams the suite once, after the first pass, in pivot's shape
# via run-duckdb.sh / run-clickhouse.sh / run-datafusion.sh: caches dropped
# once before the stream, the same gap between queries, one process (server)
# for the whole stream. DataFusion is the exception: a fresh datafusion-cli per
# query, because a query that exhausts its memory pool leaves the process
# unusable for the ones after it. ClickHouse and DataFusion queries are capped
# at --reference-timeout seconds; a capped or failed query shows no time.
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
#     [--mode pgo|release] [--scale sf100|sf1000] [--passes 2] [--regression-pct 5] \
#     [--query q06,q12] [--engines duckdb-parquet,clickhouse-native|all] \
#     [--reference-timeout 600] \
#     [--bench-env 'PIVOT_X=1 PIVOT_Y=true'] \
#     [--report /tmp/ab-report.txt] [--before-label <sha>] [--after-label <sha>]

set -uo pipefail

# The shared library and the suite's helper scripts sit next to this script in
# the repo, but the workflow ships them all flat into one /tmp directory on
# the box; find them either way.
_here="${BASH_SOURCE[0]%/*}"
_common="$_here/bench-ab-common.sh"
[[ -f "$_common" ]] || _common="$_here/../lib/bench-ab-common.sh"
# shellcheck source=../lib/bench-ab-common.sh
source "$_common"
duckdb_runner="$_here/run-duckdb.sh"
clickhouse_runner="$_here/run-clickhouse.sh"
datafusion_runner="$_here/run-datafusion.sh"
native_loader="$_here/load-native-dbs.sh"
data_generator="$_here/gen-tpch-data.sh"

# Pinned: the generated SF1000 must be the same data run over run.
tpchgen_version="2.0.1"
all_engines="duckdb-parquet,clickhouse-parquet,datafusion-parquet,duckdb-native,clickhouse-native"

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
# Defaults to <nvme mount>/ab once data_root is known: the checkouts and their
# target dirs run to well over 100GB across the sides while the AMI's root
# volume is 40GB, so the builds must live on the instance store too.
work_dir=""
data_bucket=""
data_root="/mnt/nvme/tpch"
cache_prefix=""
mode="pgo"
scale="sf100"
passes=2
regression_pct=5
queries=""
engines=""
reference_timeout=600
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
        --scale)          scale="$2"; shift 2 ;;
        --passes)         passes="$2"; shift 2 ;;
        --regression-pct) regression_pct="$2"; shift 2 ;;
        --query)          queries="$2"; shift 2 ;;
        --engines)        engines="$2"; shift 2 ;;
        --reference-timeout) reference_timeout="$2"; shift 2 ;;
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

# Per scale: the measured dataset, q11's HAVING fraction (0.0001 / SF; the
# suite's q11.sql carries SF100's), and the instance-store space the run needs
# for the dataset, both native databases and the two sides' build trees.
case "$scale" in
    sf100)
        # The pivot-written variant of SF100: the same rows as pivot's own
        # writer lays them out after an INSERT, which is what a real table
        # looks like.
        sf_measure="sf100-pivot"
        q11_fraction="0.000001"
        required_disk_gb=400
        ;;
    sf1000)
        sf_measure="sf1000"
        q11_fraction="0.0000001"
        required_disk_gb=2000
        ;;
    *) echo "error: --scale must be sf100 or sf1000 (got '$scale')" >&2; exit 2 ;;
esac

[[ "$engines" == "all" ]] && engines="$all_engines"
reference_engines=()
IFS=',' read -ra requested_engines <<<"$engines"
for engine in "${requested_engines[@]}"; do
    engine="${engine// /}"
    [[ -n "$engine" ]] || continue
    [[ ",$all_engines," == *",$engine,"* ]] \
        || { echo "error: unknown engine '$engine' (known: $all_engines, or all)" >&2; exit 2; }
    reference_engines+=("$engine")
done
# Which provisioning and loads the requested engines need.
wants_engine() { [[ " ${reference_engines[*]} " == *" $1 "* ]]; }
wants_family() { [[ " ${reference_engines[*]} " == *" $1-"* ]]; }

ab_common_init
# A many-core cargo build opens more files than the default soft limit allows.
ulimit -n "$(ulimit -Hn)"

work_dir="${work_dir:-$(dirname "$data_root")/ab}"
power_sleep=500
sf_pgo="sf1"
pgo_data="$data_root/$sf_pgo"
measure_data="$data_root/$sf_measure"
native_root="$(dirname "$data_root")"
native_db="$native_root/tpch-native.duckdb"
native_clickhouse="$native_root/clickhouse"
reference_scratch="$native_root/reference-scratch"
datafusion_bin="$work_dir/datafusion/bin/datafusion-cli"

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

# The reference runners print "=== qNN ===" before a query and a
# "Run Time (s): real <seconds>" line for each timed run; turn that into
# "<query> <ms>" lines. A query that errored prints no Run Time and is left out.
parse_reference_stream() {
    awk '/^=== q[0-9]+ ===$/ { query = $2; next }
        /Run Time/ {
            for (i = 1; i <= NF; i++)
                if ($i == "real") { printf "%s %d\n", query, $(i + 1) * 1000; break }
        }'
}

# One reference engine's stream: the caller drops caches once before it, the
# runners never do (--no-drop-caches), and the same gap separates queries.
# Query files come from the after checkout. The runners work from a scratch
# directory on the instance store, where DuckDB (in its working directory) and
# DataFusion and ClickHouse's parquet mode (in $TMPDIR) put their spill files
# rather than on the small root volume. Echoes "<query> <ms>" per line; a
# runner that fails outright leaves its queries without timings, which the
# report shows, and never stops the run.
run_reference_power() {
    local engine="$1"
    local common=(--suite-dir "$after_dir/benchmarks/tpch" --query "$queries"
                  --iterations 1 --sleep "$power_sleep" --no-drop-caches)
    local runner
    case "$engine" in
        duckdb-parquet)
            runner=("$duckdb_runner" --data parquet --source "$measure_data") ;;
        duckdb-native)
            runner=("$duckdb_runner" --data native --source "$native_db") ;;
        clickhouse-parquet)
            runner=("$clickhouse_runner" --data parquet --source "$measure_data"
                    --clickhouse-process per-run --timeout "$reference_timeout") ;;
        clickhouse-native)
            runner=("$clickhouse_runner" --data native --source "$native_clickhouse"
                    --clickhouse-process per-run --timeout "$reference_timeout") ;;
        datafusion-parquet)
            runner=("$datafusion_runner" --source "$measure_data"
                    --datafusion "$datafusion_bin" --timeout "$reference_timeout"
                    --memory-limit "$datafusion_memory_limit"
                    --disk-limit "$datafusion_disk_limit") ;;
    esac
    { (cd "$reference_scratch" && TMPDIR="$reference_scratch" \
        bash "${runner[@]}" "${common[@]}" 2>&1) || true; } \
        | parse_reference_stream
}

# ---------------------------------------------------------------------------
# Provisioning. The scale's generator is pinned, so the data never changes;
# the reference engines are each project's latest stable release, resolved
# once at startup and recorded in the report, so the columns compare against
# what those engines ship today.
# ---------------------------------------------------------------------------
case "$(uname -m)" in
    aarch64) release_arch="arm64" ;;
    x86_64)  release_arch="amd64" ;;
    *) echo "error: no engine release builds for $(uname -m)" >&2; exit 1 ;;
esac

# The tag GitHub marks as a project's latest release (never a pre-release).
latest_github_release() {
    curl -sSfL "https://api.github.com/repos/$1/releases/latest" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])'
}

latest_crate_version() {
    curl -sSfL -A pivot-bench "https://crates.io/api/v1/crates/$1" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["crate"]["max_stable_version"])'
}

# The cargo-installed tools build under their own CARGO_HOME: the warm-cache
# restore extracts into ~/.cargo concurrently and must not race a second
# writer there.
install_tpchgen() {
    CARGO_HOME="$work_dir/tpchgen-cargo" cargo install tpchgen-cli \
        --version "$tpchgen_version" --locked --quiet --root "$work_dir/tpchgen"
}

generate_scale() {
    local scale_name="$1"
    if [[ -f "$data_root/$scale_name.done" ]]; then return; fi
    install_tpchgen
    PATH="$work_dir/tpchgen/bin:$PATH" bash "$data_generator" \
        --scale "${scale_name#sf}" --root "$data_root/$scale_name"
    touch "$data_root/$scale_name.done"
    echo ">>> dataset $scale_name generated"
}

# ClickHouse tags its releases v<version>-<channel> (v26.9.4.3-stable).
install_clickhouse() {
    local release="clickhouse-common-static-$clickhouse_version"
    curl -sSfL "https://github.com/ClickHouse/ClickHouse/releases/download/$clickhouse_tag/$release-$release_arch.tgz" \
        | tar -xz -C "$work_dir"
    sudo install "$work_dir/$release/usr/bin/clickhouse" /usr/local/bin/clickhouse
    echo ">>> clickhouse $clickhouse_version installed"
}

# Installed over the AMI's DuckDB, so every duckdb on PATH is the latest.
install_duckdb() {
    curl -sSfL -o "$work_dir/duckdb.zip" \
        "https://github.com/duckdb/duckdb/releases/download/$duckdb_tag/duckdb_cli-linux-$release_arch.zip"
    unzip -o -q "$work_dir/duckdb.zip" -d "$work_dir/duckdb"
    sudo install "$work_dir/duckdb/duckdb" /usr/local/bin/duckdb
    echo ">>> duckdb $duckdb_tag installed"
}

# datafusion-cli ships no prebuilt binaries and takes minutes to build, so the
# built binary rides the warm-cache prefix like the target dirs do.
install_datafusion() {
    local cache_name="datafusion-cli-$datafusion_version"
    local root="$work_dir/datafusion"
    restore_cache "$cache_name" "$root"
    if [[ ! -x "$root/bin/datafusion-cli" ]]; then
        CARGO_HOME="$work_dir/datafusion-cargo" cargo install datafusion-cli \
            --version "$datafusion_version" --locked --quiet --root "$root"
        save_cache "$cache_name" "$root" bin
    fi
    echo ">>> datafusion-cli $datafusion_version ready"
}

# q11's HAVING fraction is 0.0001 / SF; the suite writes SF100's. Rewrite it in
# a checkout for the measured scale, failing loudly if the text moved.
set_q11_fraction() {
    local suite="$1/benchmarks/tpch"
    [[ "$q11_fraction" == "0.000001" ]] && return 0
    grep -q ' \* 0\.000001$' "$suite/q11.sql" \
        || { echo "error: q11.sql no longer carries the SF100 fraction to rewrite" >&2; exit 1; }
    sed -i "s/ \* 0\.000001\$/ * $q11_fraction/" "$suite/q11.sql"
}

# ---------------------------------------------------------------------------
# Phase 0: disk, data, checkouts.
# ---------------------------------------------------------------------------
mount_nvme
mkdir -p "$work_dir"
data_dev="$(df --output=source "$native_root" | tail -1 | sed 's|/dev/||')"
available_disk_gb="$(df --output=avail -BG "$native_root" | tail -1 | tr -dc '0-9')"
if [[ "$available_disk_gb" -lt "$required_disk_gb" ]]; then
    echo "error: $scale needs ${required_disk_gb}GB of instance store, $native_root has ${available_disk_gb}GB" >&2
    exit 1
fi

if [[ "$scale" == "sf1000" ]]; then
    ( sync_scale "$sf_pgo"; generate_scale "$sf_measure" ) &
else
    ( sync_scale "$sf_pgo"; sync_scale "$sf_measure" ) &
fi
sync_pid=$!

declare -A install_pids=()
if wants_family duckdb; then
    duckdb_tag="$(latest_github_release duckdb/duckdb)"
    install_duckdb &
    install_pids[duckdb]=$!
fi
if wants_family clickhouse; then
    clickhouse_tag="$(latest_github_release ClickHouse/ClickHouse)"
    clickhouse_version="${clickhouse_tag#v}"
    clickhouse_version="${clickhouse_version%-*}"
    install_clickhouse &
    install_pids[clickhouse]=$!
fi
if wants_family datafusion; then
    datafusion_version="$(latest_crate_version datafusion-cli)"
    install_datafusion &
    install_pids[datafusion]=$!
fi

echo ">>> fetching commits into the baked clone"
if [[ -n "$repo_url" ]]; then
    git -C "$mirror" fetch --quiet "$repo_url" "$before_sha" "$after_sha" \
        || git -C "$mirror" fetch --quiet "$repo_url" '+refs/heads/*:refs/ab/heads/*'
fi
before_dir="$work_dir/before"
after_dir="$work_dir/after"
checkout_side "$before_sha" "$before_dir"
checkout_side "$after_sha" "$after_dir"
set_q11_fraction "$before_dir"
set_q11_fraction "$after_dir"

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
    before_server="$before_dir/benchmarks/target-pgouse/$host_target/release/pivot"
    after_server="$after_dir/benchmarks/target-pgouse/$host_target/release/pivot"
else
    echo ">>> release builds (A and B in parallel)"
    build_release "$before_dir" & b1=$!
    build_release "$after_dir" & b2=$!
    wait "$b1"; wait "$b2"
    before_bin="$before_dir/target/release/pivot-bench"
    after_bin="$after_dir/target/release/pivot-bench"
    before_server="$before_dir/target/release/pivot"
    after_server="$after_dir/target/release/pivot"
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
# Reap the engine installs explicitly first, while their exit statuses are
# still retrievable. The bare wait below collects every remaining background
# job at once and returns success even if one of them failed, so a later wait
# on these pids would find them already reaped and report a spurious failure.
for tool in "${!install_pids[@]}"; do
    if ! wait "${install_pids[$tool]}"; then
        echo "error: $tool install failed" >&2
        exit 1
    fi
done
wait

# The reference engines' native storage, loaded from the measured parquet so
# they hold exactly the rows pivot reads. Loading is CPU-heavy, so it runs
# after the builds and cache uploads and before anything is timed.
for family in duckdb clickhouse; do
    wants_engine "$family-native" || continue
    echo ">>> loading the native $family tables from $sf_measure"
    bash "$native_loader" --source "$measure_data" --root "$native_root" --only "$family"
done
mkdir -p "$reference_scratch"
# DataFusion's pool: the share of memory DuckDB and ClickHouse default to, and
# half the free instance store for its spills.
datafusion_memory_limit="$(awk '/^MemTotal:/ {printf "%dg", $2 * 0.8 / 1048576}' /proc/meminfo)"
datafusion_disk_limit="$(( $(df --output=avail -BG "$native_root" | tail -1 | tr -dc '0-9') / 2 ))g"

# ---------------------------------------------------------------------------
# Phase 2: burn-in + correctness. One untimed full pass per side writes each
# side's result tsvs and, as a side effect, walks the instance store out of
# its fresh-boot throughput state before anything is measured. The two sides'
# outputs are then compared with a float tolerance (summation order makes the
# last digit unstable); a real mismatch fails the run.
# ---------------------------------------------------------------------------
if [[ -z "$queries" ]]; then
    queries="$(cd "$after_dir/benchmarks/tpch" \
        && ls q*.sql | grep -v -- '-duckdb\|-clickhouse' | sed 's/\.sql$//' | paste -sd,)"
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
    # The references never gate, so one stream each is enough; with several
    # engines at the larger scale, repeating them would dominate the run.
    [[ "$pass" == "1" ]] || continue
    for engine in "${reference_engines[@]}"; do
        drop_caches; sleep 3
        start=$SECONDS
        run_reference_power "$engine" >"$stream"
        awk -v side="$engine" 'NF == 2 {print $1 "\t" side "\t" $2}' "$stream" >>"$rows"
        echo "    $engine: $(wc -l <"$stream") queries timed in $((SECONDS - start))s"
    done
done

# ---------------------------------------------------------------------------
# Report + gate. Cold, the only number a power stream yields, is the gate.
# The reference engines, when present, never gate.
# ---------------------------------------------------------------------------
{
    echo "=== TPC-H A/B: '$before_label' (before) vs '$after_label' (after) ==="
    echo "mode=$mode scale=$scale source=$measure_data passes=$passes regression_pct=$regression_pct"
    echo "box: $(nproc) cores, $(free -g | awk '/^Mem:/ {print $2}')GB RAM, instance store ${available_disk_gb}GB on $data_dev"
    echo "shape: cold power runs; per side and pass, one process streams every query once, ${power_sleep}ms apart, caches dropped only before the stream; cold = per-query min across passes"
    if [[ ${#reference_engines[@]} -gt 0 ]]; then
        echo "reference columns (one stream each, never gate; a/<col> = after / that engine; - = capped at ${reference_timeout}s or failed):"
        for engine in "${reference_engines[@]}"; do
            case "$engine" in
                duckdb-parquet)     echo "  dk-pq  DuckDB $duckdb_tag over the same parquet (uncapped)" ;;
                duckdb-native)      echo "  dk-nat DuckDB $duckdb_tag over its native database of the same rows (uncapped)" ;;
                clickhouse-parquet) echo "  ch-pq  ClickHouse $clickhouse_version, File(Parquet) tables over the same parquet" ;;
                clickhouse-native)  echo "  ch-nat ClickHouse $clickhouse_version over MergeTree tables of the same rows" ;;
                datafusion-parquet) echo "  df-pq  DataFusion $datafusion_version over the same parquet, a process per query, pool $datafusion_memory_limit, spill $datafusion_disk_limit" ;;
            esac
        done
    fi
    [[ -n "$bench_env" ]] && echo "bench_env=$bench_env"
    [[ "$mode" == "release" ]] && echo "NOTE: release mode is non-PGO; numbers carry code-alignment noise, use for iteration only"
    echo "correctness (before vs after, 1e-6 relative): $correctness"
    echo
} >"$report"

failures="$(awk -F'\t' -v t="$regression_pct" -v engines="${reference_engines[*]}" -v sf="${scale#sf}" '
function pct(b, a) { b = (b < 1 ? 1 : b); return (a - b) / b * 100 }
function with_commas(x,    s) {
    s = sprintf("%d", x)
    while (s ~ /[0-9][0-9][0-9][0-9]/) sub(/[0-9][0-9][0-9]($|,)/, ",&", s)
    return s
}
# One scoreboard row: the total of the side'"'"'s query times, and the TPC-H
# power metric, 3600 * SF / geometric mean of the query times in seconds. As
# the spec prescribes, times below 1/1000 of the longest are raised to it, so
# near-zero queries cannot dominate the mean. Power needs every query timed.
function score(name, side,    i, q, v, longest, timed, total, logsum, missing, power) {
    longest = 0; timed = 0; total = 0; logsum = 0; missing = ""
    for (i = 0; i < n; i++) {
        v = c[order[i] "," side]
        if (valid(v) && v + 0 > longest) longest = v + 0
    }
    for (i = 0; i < n; i++) {
        q = order[i]; v = c[q "," side]
        if (!valid(v)) { missing = missing " " q; continue }
        timed++; total += v
        v = (v + 0 < longest / 1000 ? longest / 1000 : v + 0)
        logsum += log((v < 1 ? 1 : v) / 1000)
    }
    power = (timed == n && n > 0 ? with_commas(3600 * sf / exp(logsum / n)) : "-")
    printf "  %-20s %10.1fs %13s   %d/%d%s\n", name, total / 1000, power, timed, n, \
        (missing != "" ? "  no time:" missing : "") > "/dev/stderr"
}
function valid(x) { return (x != "") }
function keepmin(arr, k, v) { if (!(k in arr) || v + 0 < arr[k] + 0) arr[k] = v }
BEGIN {
    n_refs = split(engines, refs, " ")
    label["duckdb-parquet"] = "dk-pq"; label["duckdb-native"] = "dk-nat"
    label["clickhouse-parquet"] = "ch-pq"; label["clickhouse-native"] = "ch-nat"
    label["datafusion-parquet"] = "df-pq"
}
{ keepmin(c, $1 "," $2, $3); if (!seen[$1]++) order[n++] = $1 }
END {
    hdr = sprintf("%-5s %8s %8s %8s", "query", "cold_b", "cold_a", "coldD%")
    for (r = 1; r <= n_refs; r++) hdr = hdr sprintf("  %7s %8s", label[refs[r]], "a/" label[refs[r]])
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
        for (r = 1; r <= n_refs; r++) {
            rc = c[q "," refs[r]]; ratio = "-"
            if (valid(ca) && valid(rc)) {
                ratio = sprintf("%.2fx", (ca + 10) / (rc + 10))
                ref_log[r] += log((ca + 10) / (rc + 10)); ref_n[r]++
            }
            line = line sprintf("  %7s %8s", (valid(rc) ? rc : "-"), ratio)
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
    for (r = 1; r <= n_refs; r++)
        printf "after vs %s, geomean of (after+10ms)/(%s+10ms): %.3fx  (%d queries both timed)\n", \
            refs[r], refs[r], (ref_n[r] ? exp(ref_log[r] / ref_n[r]) : 0), ref_n[r] > "/dev/stderr"

    printf "\n=== scoreboard: SF%s, one cold stream per engine (pivot: per-query min across passes) ===\n", sf > "/dev/stderr"
    printf "Power@%s = 3600 x %s / geomean(query seconds), the spec power metric without its refresh functions; higher is better\n", sf, sf > "/dev/stderr"
    printf "  %-20s %11s %13s   %s\n", "engine", "total", "Power@" sf, "queries" > "/dev/stderr"
    score("pivot (after)", "after")
    score("pivot (before)", "before")
    for (r = 1; r <= n_refs; r++) score(refs[r], refs[r])
}' "$rows" 2>>"$report")"

{
    echo
    echo "=== failures: cold regressions (>= ${regression_pct}%) or missing timings ==="
    if [[ "$correctness" != "ok" ]]; then echo "output mismatch between sides"; fi
    if [[ -n "$failures" ]]; then echo "$failures"; else echo "none"; fi
} >>"$report"

cat "$report"
[[ -z "$failures" && "$correctness" == "ok" ]]
