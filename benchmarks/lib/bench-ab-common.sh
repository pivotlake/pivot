#!/usr/bin/env bash
#
# bench-ab-common.sh - shared box-side machinery for the A/B benchmark harnesses
# (benchmarks/tpch/bench-tpch-ab.sh, benchmarks/tpch-flat/bench-tpch-flat-ab.sh
# and benchmarks/jsonbench/bench-jsonbench-ab.sh).
#
# Sourced, not executed. It defines functions only; it runs nothing at source
# time and does not set shell options, so each harness keeps its own `set`.
#
# The functions read these globals, which the sourcing harness must set first
# (ab_common_init sets the ones it can derive):
#   mirror        - path to the AMI-baked repo clone (source of local checkouts)
#   work_dir      - scratch dir for checkouts, profiles, profdata
#   data_root     - root under the instance-store mount (its parent is the mount)
#   data_bucket   - s3://.../prefix holding one directory per dataset scale
#   sync_pid      - pid of the backgrounded sync_scale chain (for wait_for_scale)
#   data_dev      - block device under data_root, without /dev/ (for io_ticks)
#   cache_prefix  - s3://.../key prefix for warm caches ("" disables them)
#   cache_key     - compiler+target fingerprint (set by ab_common_init)
#   host_target   - rustc host target triple (set by ab_common_init)
#   run_id        - unique per run, used in the profdata filename (set by ab_common_init)
#   pgo_dir       - fixed profile-generate path (set by ab_common_init)

# Common environment + derived identifiers. `bench_env` (space-separated
# KEY=VALUE) is applied to this process so every build and run below inherits
# it. Call once, early.
ab_common_init() {
    export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin"
    export NO_COLOR=1
    if [[ -n "${bench_env:-}" ]]; then
        for kv in $bench_env; do
            export "${kv?}"
        done
    fi
    run_id="$(date +%s)-$$"
    host_target="$(rustc -vV | sed -n 's/^host: //p')"
    # Cache entries are only valid for the exact compiler (fingerprints) and
    # this instance family's target-cpu=native output; the toolchain string
    # covers both since the AMI pins the toolchain and the workflow pins the
    # instance type.
    cache_key="$(rustc -V | tr ' ()' '__.')"
    # Fixed profile-generate path, matching benchmarks/justfile's pgo_dir: the
    # path sits inside RUSTFLAGS, and identical flags are what keep the restored
    # target-pgogen artifacts valid run over run.
    pgo_dir="/tmp/benchmarks-pgo"
}

# ---------------------------------------------------------------------------
# Instance-store NVMe: find the unformatted ephemeral disk and mount it at the
# parent of $data_root. Device names are not stable across instance types, so
# pick by model string.
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
# Source checkouts: local clones of the AMI-baked repo (object hardlinks, no
# network), with submodules resolved against the baked clone's modules. A
# commit that bumps a submodule past the baked state falls back to the network,
# which all submodules allow (public forks). mtimes are restored so cargo's
# fingerprints match a warm target dir restored from another machine.
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
    git -C "$dir" restore-mtime --quiet
    git -C "$dir" submodule foreach --quiet --recursive 'git restore-mtime --quiet'
}

# ---------------------------------------------------------------------------
# Dataset sync from S3, one directory per scale under $data_bucket. Meant to
# run backgrounded so the builds overlap it; each finished scale is marked
# with a .done sentinel that wait_for_scale blocks on. s5cmd parallelises far
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

# Milliseconds the data device has spent with IO in flight, cumulative since
# boot; deltas around a measured run give a disk-state sanity signal. Returns
# nothing until the harness sets data_dev after mounting.
io_ticks() {
    [[ -n "$data_dev" ]] || return 0
    awk '{print $10}' "/sys/block/$data_dev/stat"
}

# ---------------------------------------------------------------------------
# Warm-cache tarballs in S3. Restores are best-effort: a miss just means a cold
# build. Saves happen after the builds so even a failed measurement leaves the
# next run warm.
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
# Builds. PGO: an instrumented build (shared warm cache; the profile-generate
# path is fixed so its RUSTFLAGS never change and restored artifacts stay
# valid), a profiling run whose LLVM_PROFILE_FILE separates the sides, then a
# profile-use build. The profdata path embeds this run's id so the rebuild of
# every Rust unit under the fresh profile follows from the flags hash alone;
# the restored target dir still donates its profile-independent build-script
# outputs (the DuckDB C++ build), which the explicit --target keeps unflagged.
# The suite's own harness supplies the profiling workload via profile_side.
# ---------------------------------------------------------------------------
# Both builds go through the pgo.just recipes rather than repeating their
# RUSTFLAGS here. The flags are subtle (lld for the instrumented link, an
# explicit --target so build scripts stay unflagged and cached) and a second
# copy of them drifts from the recipe silently, changing what is measured
# without changing anything that looks like a measurement.
# Every build produces both binaries the harnesses need: the pivotdb-server
# being measured, and the pivot-bench that launches it (--server-bin) and
# drives it over pgwire. Only the server takes profile flags, mirroring
# build-ab-servers.sh: the client links no engine code, so instrumenting it
# would grow the build and rebuild it under every fresh profile for nothing.
# It gets a plain build in its own target-client dir instead.
build_gen() {
    local dir="$1"
    mkdir -p "$pgo_dir"
    (cd "$dir/benchmarks" && \
        PGO_DIR="$pgo_dir" PGO_GEN_TARGET_DIR=target-pgogen \
        just pgo-gen-build build --release -p server --bin pivotdb-server && \
        CARGO_TARGET_DIR=target-client RUSTC_WRAPPER= \
        cargo build --release -p benchmarks --bin pivot-bench)
}

# Produce a profile for one side: the plain client drives the instrumented
# server, which inherits LLVM_PROFILE_FILE through the environment pivot-bench
# spawns it with and is stopped with SIGINT so it flushes its counters. The
# client is not instrumented and writes nothing. The suite name, suite dir and
# source come from the caller so the profile is taken on that suite's own
# small-scale workload.
profile_side() {
    local dir="$1" side="$2" suite="$3" source="$4"
    local prof_dir="$work_dir/prof-$side"
    rm -rf "$prof_dir"; mkdir -p "$prof_dir"
    LLVM_PROFILE_FILE="$prof_dir/%m-%p.profraw" \
        "$dir/benchmarks/target-client/release/pivot-bench" \
        --suite "$suite" --suite-dir "$dir/benchmarks/$suite" \
        --server-bin "$dir/benchmarks/target-pgogen/$host_target/release/pivotdb-server" \
        --source "$source" --iterations 2 --skip-check >/dev/null
    "$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata" \
        merge -o "$work_dir/$side-$run_id.profdata" "$prof_dir"
}

build_use() {
    local dir="$1" side="$2"
    (cd "$dir/benchmarks" && \
        PGO_USE_TARGET_DIR=target-pgouse \
        just pgo-use-with "$work_dir/$side-$run_id.profdata" build --release -p server --bin pivotdb-server)
    verify_pgo_applied \
        "$dir/benchmarks/target-pgouse/$host_target/release/pivotdb-server" \
        "$work_dir/$side-$run_id.profdata" "$side"
}

# Tripwire for the profile applying at all, same as build-ab-servers.sh: the
# decode family's monomorphization hashes in the built server must appear in
# the profile it was compiled against. Zero overlap means the server was built
# outside the profiled symbol universe and is effectively un-PGOed, which is
# silent at compile time and shows up only as a mystery regression.
verify_pgo_applied() {
    local server="$1" profdata="$2" side="$3"
    local family="RleDecoder4read"
    local binary_hashes profile_hashes covered
    # The greps legitimately match nothing when the family is fully inlined,
    # and an empty match must not kill the harness through set -e; the
    # zero-overlap case below is the loud failure.
    binary_hashes=$(nm "$server" | grep "$family" | grep -oE '17h[0-9a-f]+E' | sort -u || true)
    if [[ -z "$binary_hashes" ]]; then
        echo ">>> $side server: no $family symbols visible to nm, skipping the profile-overlap check"
        return 0
    fi
    profile_hashes=$("$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata" \
        show -all-functions "$profdata" 2>/dev/null \
        | grep "$family" | grep -oE '17h[0-9a-f]+E' | sort -u || true)
    covered=$(comm -12 <(printf '%s\n' "$binary_hashes") \
                       <(printf '%s\n' "$profile_hashes") | wc -l)
    if [[ "$covered" -eq 0 ]]; then
        echo "error: the $side server shares no $family symbols with its profile;" >&2
        echo "       the profile did not apply and the binary is effectively un-PGOed" >&2
        return 1
    fi
    echo ">>> $side server: $covered $family monomorphizations carry profile records"
}

build_release() {
    local dir="$1"
    (cd "$dir/benchmarks" && RUSTC_WRAPPER= \
        cargo build --release -p server --bin pivotdb-server -p benchmarks --bin pivot-bench)
}

# Drop the OS page cache (and sync first) so the next read is cold. Needs
# passwordless sudo.
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }

# ---------------------------------------------------------------------------
# Correctness: compare every qNN.tsv the two sides wrote, with a float tolerance
# (summation order makes the last digit unstable). Prints each mismatch and
# returns non-zero if any differ.
# ---------------------------------------------------------------------------
compare_outputs() {
    local before_suite_dir="$1" after_suite_dir="$2"
    python3 - "$before_suite_dir" "$after_suite_dir" <<'EOF'
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
}
