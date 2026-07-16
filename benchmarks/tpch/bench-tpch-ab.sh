#!/usr/bin/env bash
#
# bench-tpch-ab.sh - A/B performance comparison of two pivotdb commits on the
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
# Meant to be launched detached and polled over ssh, so it writes its PID and
# emits a sentinel on every exit path.
#
# Usage:
#   bench-tpch-ab.sh \
#     --mirror /opt/pivotdb --repo-url <tokenised fetch url> \
#     --before <sha> --after <sha> \
#     --data-bucket s3://bucket/prefix --cache-prefix s3://bucket/cache \
#     [--mode pgo|release] [--iterations 3] [--passes 2] [--regression-pct 5] \
#     [--query q06,q12] [--duckdb] [--bench-env 'PIVOT_X=1 PIVOT_Y=true'] \
#     [--report /tmp/ab-report.txt] [--before-label <sha>] [--after-label <sha>]

set -uo pipefail

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

export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin"
export NO_COLOR=1

if [[ -n "$bench_env" ]]; then
    for kv in $bench_env; do
        export "${kv?}"
    done
fi

run_id="$(date +%s)-$$"
host_target="$(rustc -vV | sed -n 's/^host: //p')"
# Cache entries are only valid for the exact compiler (fingerprints) and this
# instance family's target-cpu=native output; the toolchain string covers both
# since the AMI pins the toolchain and the workflow pins the instance type.
cache_key="$(rustc -V | tr ' ()' '__.')"

sf_pgo="sf10"
sf_measure="sf100"
pgo_data="$data_root/$sf_pgo/flat"
measure_data="$data_root/$sf_measure/flat"

# ---------------------------------------------------------------------------
# Instance-store NVMe: find the unformatted ephemeral disk and mount it.
# Device names are not stable across instance types, so pick by model string.
# ---------------------------------------------------------------------------
mount_nvme() {
    local mnt
    mnt="$(dirname "$data_root")"
    if mountpoint -q "$mnt"; then
        echo ">>> $mnt already mounted"
        return
    fi
    local dev
    dev="$(lsblk -dno NAME,MODEL | awk '/Instance Storage/ {print $1; exit}')"
    [[ -n "$dev" ]] || { echo "error: no instance-store NVMe device found" >&2; exit 1; }
    echo ">>> formatting /dev/$dev and mounting at $mnt"
    sudo mkfs.ext4 -q -E lazy_itable_init=1 "/dev/$dev"
    sudo mkdir -p "$mnt"
    sudo mount -o noatime "/dev/$dev" "$mnt"
    sudo chown "$(id -u):$(id -g)" "$mnt"
}

# ---------------------------------------------------------------------------
# Dataset sync, backgrounded: the small PGO scale lands first (it gates the
# profiling runs), the measurement scale after (it gates the bench phase).
# Each finished scale is marked with a .done sentinel. s5cmd parallelises far
# past what aws-cli does; the second pass catches anything a first pass
# dropped.
# ---------------------------------------------------------------------------
sync_scale() {
    local scale="$1"
    if [[ -f "$data_root/$scale.done" ]]; then return; fi
    mkdir -p "$data_root/$scale"
    if command -v s5cmd >/dev/null; then
        for _ in 1 2; do
            s5cmd --log error sync "$data_bucket/$scale/*" "$data_root/$scale/" >/dev/null
        done
    else
        for _ in 1 2 3; do
            aws s3 sync --only-show-errors "$data_bucket/$scale" "$data_root/$scale"
        done
    fi
    touch "$data_root/$scale.done"
    echo ">>> dataset $scale synced"
}

wait_for_scale() {
    local scale="$1"
    while [[ ! -f "$data_root/$scale.done" ]]; do
        if ! kill -0 "$sync_pid" 2>/dev/null; then
            echo "error: dataset sync died before $scale finished" >&2
            exit 1
        fi
        sleep 5
    done
}

# ---------------------------------------------------------------------------
# Source checkouts: local clones of the AMI-baked repo (object hardlinks, no
# network), with submodules resolved against the baked clone's modules. A
# commit that bumps a submodule past the baked state falls back to the
# network, which all submodules allow (public forks).
# ---------------------------------------------------------------------------
checkout_side() {
    local sha="$1" dir="$2"
    rm -rf "$dir"
    git -c protocol.file.allow=always clone --quiet --no-checkout "$mirror" "$dir"
    git -C "$dir" checkout --quiet "$sha"
    git -C "$dir" \
        -c submodule.alternateLocation=superproject \
        -c submodule.alternateErrorStrategy=info \
        -c protocol.file.allow=always \
        submodule update --quiet --init --recursive
}

# ---------------------------------------------------------------------------
# Warm-cache tarballs in S3. Restores are best-effort: a miss just means a
# cold build. Saves happen after the builds so even a failed measurement
# leaves the next run warm.
# ---------------------------------------------------------------------------
s3_get() { aws s3 cp --only-show-errors "$1" - 2>/dev/null; }
s3_put() { aws s3 cp --only-show-errors - "$1"; }

restore_cache() {
    local name="$1" dest="$2"
    [[ -n "$cache_prefix" ]] || return 0
    mkdir -p "$dest"
    if s3_get "$cache_prefix/$cache_key/$name.tar.zst" | tar -I 'zstd -d' -x -C "$dest" 2>/dev/null; then
        echo ">>> cache restored: $name"
    else
        echo ">>> cache miss: $name (cold build)"
        # A partial extract from a truncated stream must not poison the build.
        rm -rf "${dest:?}"/*
    fi
}

# Restoring into ~/.cargo must never touch bin/ (the AMI's toolchain), so the
# cargo-home tarball holds only the registry and git checkouts.
restore_cargo_home() {
    [[ -n "$cache_prefix" ]] || return 0
    if s3_get "$cache_prefix/$cache_key/cargo-home.tar.zst" | tar -I 'zstd -d' -x -C "$HOME/.cargo" 2>/dev/null; then
        echo ">>> cache restored: cargo-home"
    else
        echo ">>> cache miss: cargo-home"
    fi
}

save_cache() {
    local name="$1" src="$2"; shift 2
    local paths=("${@:-.}")
    [[ -n "$cache_prefix" && -d "$src" ]] || return 0
    (cd "$src" && mkdir -p "${paths[@]}")
    tar -I 'zstd -3 -T0' -c -C "$src" "${paths[@]}" | s3_put "$cache_prefix/$cache_key/$name.tar.zst"
    echo ">>> cache saved: $name"
}

# ---------------------------------------------------------------------------
# Builds. PGO: instrumented build (shared warm cache; the profile-generate
# path is the fixed pgo_dir from benchmarks/justfile, so its RUSTFLAGS never
# change and restored artifacts stay valid), profiling run on the small scale
# with LLVM_PROFILE_FILE separating the sides, then a profile-use build whose
# profdata path embeds this run's id: that flag change is what forces every
# Rust unit to rebuild under the fresh profile while the restored target dir
# donates its profile-independent build-script outputs (the DuckDB C++ build).
# ---------------------------------------------------------------------------
# Fixed profile-generate path, matching benchmarks/justfile's pgo_dir: the
# path sits inside RUSTFLAGS, and identical flags are what keep the restored
# target-pgogen artifacts valid run over run.
pgo_dir="/tmp/benchmarks-pgo"

# Mirrors `just pgo-gen-build` / `just pgo-use-with` (pgo.just), inlined so
# both sides build identically even when the before commit predates those
# recipes. Keep the flags in sync with pgo.just.
build_gen() {
    local dir="$1"
    mkdir -p "$pgo_dir"
    # lld: instrumentation grows the text section past the 128MB aarch64
    # branch range and GNU ld fails the link with relocation overflows; lld
    # inserts range-extension thunks. Only the throwaway instrumented binary
    # needs it, the measured profile-use build links like any release build.
    (cd "$dir/benchmarks" && \
        RUSTC_WRAPPER= \
        RUSTFLAGS="-Cprofile-generate=$pgo_dir -Ctarget-cpu=native -Clink-arg=-fuse-ld=lld" \
        CARGO_TARGET_DIR=target-pgogen cargo build --target "$host_target" --release)
}

profile_side() {
    local dir="$1" side="$2"
    local prof_dir="$work_dir/prof-$side"
    rm -rf "$prof_dir"; mkdir -p "$prof_dir"
    LLVM_PROFILE_FILE="$prof_dir/%m-%p.profraw" \
        "$dir/benchmarks/target-pgogen/$host_target/release/pivot-bench" \
        --suite tpch --suite-dir "$dir/benchmarks/tpch" \
        --source "$pgo_data" --iterations 2 --skip-check >/dev/null
    "$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata" \
        merge -o "$work_dir/$side-$run_id.profdata" "$prof_dir"
}

build_use() {
    local dir="$1" side="$2"
    (cd "$dir/benchmarks" && \
        RUSTC_WRAPPER= \
        RUSTFLAGS="-Cprofile-use=$work_dir/$side-$run_id.profdata -Ctarget-cpu=native" \
        CARGO_TARGET_DIR=target-pgouse cargo build --target "$host_target" --release)
}

build_release() {
    local dir="$1"
    (cd "$dir/benchmarks" && RUSTC_WRAPPER= cargo build --release)
}

# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }

data_dev="" # set after mount, for the io_ticks sanity column
io_ticks() {
    [[ -n "$data_dev" ]] || return 0
    awk '{print $10}' "/sys/block/$data_dev/stat"
}

# Runs one side's pivot-bench for one query with $iterations tries in one
# process, echoing "cold hot io_seconds": cold is try 1 in ms, hot the min of
# the rest ("null" when missing), io_seconds the device's io_ticks delta as a
# disk-state sanity signal.
run_pivot() {
    local bin="$1" dir="$2" query="$3"
    local t0 t1 out times
    t0="$(io_ticks)"
    out="$("$bin" --suite tpch --suite-dir "$dir/benchmarks/tpch" \
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
    profile_side "$before_dir" before
    profile_side "$after_dir" after

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
    queries="$(cd "$after_dir/benchmarks/tpch" && ls q*.sql | sed 's/\.sql$//' | paste -sd,)"
fi
echo ">>> burn-in + result capture (queries: $queries)"
drop_caches
"$before_bin" --suite tpch --suite-dir "$before_dir/benchmarks/tpch" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null
drop_caches
"$after_bin" --suite tpch --suite-dir "$after_dir/benchmarks/tpch" \
    --source "$measure_data" --query "$queries" --iterations 1 --update-results >/dev/null

correctness="ok"
if ! python3 - "$before_dir/benchmarks/tpch" "$after_dir/benchmarks/tpch" <<'EOF'
import glob, sys
before, after = sys.argv[1], sys.argv[2]
ok = True
for f in sorted(glob.glob(before + "/q*.tsv")):
    name = f.rsplit("/", 1)[1]
    a = open(f).read().strip().split("\n")
    b = open(after + "/" + name).read().strip().split("\n")
    if len(a) != len(b):
        print(f"MISMATCH {name}: {len(a)} vs {len(b)} rows"); ok = False; continue
    for ra, rb in zip(a, b):
        for va, vb in zip(ra.split("\t"), rb.split("\t")):
            try:
                x, y = float(va), float(vb)
                if abs(x - y) > 1e-6 * max(1, abs(x)):
                    print(f"MISMATCH {name}: {va} vs {vb}"); ok = False
            except ValueError:
                if va != vb:
                    print(f"MISMATCH {name}: {va!r} vs {vb!r}"); ok = False
sys.exit(0 if ok else 1)
EOF
then
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
        read -r cold hot io <<<"$(run_pivot "$before_bin" "$before_dir" "$q")"
        echo -e "$q\tbefore\t$cold\t$hot\t$io" >>"$rows"
        drop_caches; sleep 3
        read -r cold hot io <<<"$(run_pivot "$after_bin" "$after_dir" "$q")"
        echo -e "$q\tafter\t$cold\t$hot\t$io" >>"$rows"
        if [[ "$run_duckdb" == "1" ]]; then
            drop_caches; sleep 3
            read -r cold hot <<<"$(run_duckdb_query "$after_dir/benchmarks/tpch/$q.sql")"
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
