#!/usr/bin/env bash
#
# bench-build.sh - rebuild the working side of an A/B after editing the source.
#
# The inner loop of a perf run. setup-bench.sh builds baseline/ once and clones
# it to working/; this rebuilds working/ against whatever the source tree holds
# now, and never falls back to a clean rebuild. The measured binary is pivot,
# launched as `pivot server`; pivot-bench is the plain pgwire client that
# drives it (see setup-bench.sh for the split). Costs:
#
#   nothing changed                       ~0s
#   a workspace crate edited            ~229s   (that crate and the ones above it)
#   --regen-profile                     ~229s   (our 8 crates, not all 333)
#
# Both sides compile against one path, pgo/active.profdata, holding whichever
# profile is currently in force. The path is part of RUSTFLAGS, so giving
# working/ its own would change the flags hash and rebuild all 333 crates,
# throwing away everything the clone bought. Each side still keeps its own
# profile under its pgo/ directory; only the copy in the slot is overwritten,
# so baseline's remains exactly as setup-bench produced it.
#
# Which leaves the question of how a regenerated profile rebuilds our crates
# without dragging the other 326 along, since rustc records the profile in
# every crate's dep-info. Cargo compares dep-info by mtime, not by content, so
# the new profile goes into the slot with the old mtime put back: nothing looks
# stale, and the dependencies stay cached. Our crates are then rebuilt by
# touching their sources, and pick the new profile up when they do.
#
# The cost of that is drift. A dependency rebuilt later for some unrelated
# reason compiles against whatever profile is at that path then, so the
# dependency set slowly becomes a mix of profile vintages. Re-run setup-bench
# to put everything back on one profile.
#
# --profile picks the cargo profile for the measured binary, and defaults to
# profiling: it inherits release and adds debug info, which perf annotate needs
# for line attribution and for resolving inlined frames. Codegen is unchanged,
# so these are release timings. Plain --profile release drops the debug info and
# keeps only the symbol table, which still resolves function names in perf
# report and perf diff. A profile has its own artifact directory, so the first
# build under a new one rebuilds everything and the clone's warmth does not
# carry over; pass setup-bench the same profile so both sides match.
#
# Usage:
#   just bench-build
#   just bench-build --regen-profile --pgo-source ~/hits-pgo-subset
#   just bench-build --profile release

set -uo pipefail

pid_file="${PID_FILE:-/tmp/bench-build.pid}"
echo $$ >"$pid_file"
trap 'echo "=== BENCH-BUILD COMPLETE exit=$? ==="' EXIT
set -e

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

root="$HOME/bench"
pgo_source=""
suite="clickbench"
iterations=2
queries=""
regen_profile=0
cargo_profile="profiling"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)          root="$2"; shift 2 ;;
        # Run the profiling workload again so the build uses a profile taken
        # from the current code, rather than the one setup-bench left behind.
        --regen-profile) regen_profile=1; shift ;;
        --pgo-source)    pgo_source="$2"; shift 2 ;;
        --suite)         suite="$2"; shift 2 ;;
        --iterations)    iterations="$2"; shift 2 ;;
        --query)         queries="$2"; shift 2 ;;
        # Cargo profile for the measured binary. The default, profiling,
        # inherits release and adds debug info, which perf needs for line-level
        # attribution and for resolving inlined frames. It does not change
        # codegen, so these are release timings. Each profile has its own
        # artifact directory, so the first build under a new one is a full one.
        --profile)       cargo_profile="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

expand_tilde() {
    # The "~" patterns match a literal leading tilde in the argument.
    # shellcheck disable=SC2088
    case "$1" in
        "~")   printf '%s' "$HOME" ;;
        "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}
root="$(expand_tilde "$root")"
[[ -n "$pgo_source" ]] && pgo_source="$(expand_tilde "$pgo_source")"

export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin"
export NO_COLOR=1

baseline="$root/baseline"
working="$root/working"
client="$root/target-client/release/pivot-bench"
# The profile in force. Its contents change; its name and its mtime do not.
active="$root/pgo/active.profdata"
# This side's own profile, kept for provenance and copied into the slot above.
profdata="$working/pgo/merged.profdata"
profile_marker="$working/pgo/cargo-profile"
host_target="$(rustc -vV | sed -n 's/^host: //p')"
# Cargo names most profiles' directories after the profile, but not the two
# built-in ones that predate named profiles.
case "$cargo_profile" in
    dev|test) profile_dir="debug" ;;
    bench)    profile_dir="release" ;;
    *)        profile_dir="$cargo_profile" ;;
esac

for required in "$working" "$active"; do
    if [[ ! -e "$required" ]]; then
        echo "error: $required is missing; run 'just setup-bench' first" >&2
        exit 2
    fi
done

if [[ $regen_profile -eq 0 ]] && \
    { [[ ! -f "$profile_marker" ]] || [[ "$(cat "$profile_marker")" != "$cargo_profile" ]]; }; then
    echo "error: working PGO data was not generated for --profile $cargo_profile; rerun with --regen-profile --pgo-source <directory>" >&2
    exit 2
fi

phase_start=0
begin() { phase_start=$SECONDS; echo; echo ">>> $*"; }
elapsed() { echo "<<< $1 took $((SECONDS - phase_start))s"; }

# Every workspace crate's root source file. Read from cargo rather than listed
# here so a new crate needs no upkeep. arrow-rs is its own workspace and so is
# absent, which is what keeps it on the profile it was built with.
workspace_crate_roots() {
    cargo metadata --no-deps --format-version 1 --manifest-path "$crate_dir/Cargo.toml" \
        | python3 -c '
import json, sys
for package in json.load(sys.stdin)["packages"]:
    for target in package["targets"]:
        if {"lib", "bin"} & set(target["kind"]):
            print(target["src_path"])
            break
'
}

echo "=== bench-build ==="
echo "source tree : $crate_dir"
echo "working     : $working"
echo "profile     : $profdata"
echo "in force    : $active"

if [[ $regen_profile -eq 1 ]]; then
    if [[ -z "$pgo_source" ]]; then
        echo "error: --regen-profile needs --pgo-source" >&2
        exit 2
    fi
    pgo_source="$(expand_tilde "$pgo_source")"

    # -Cprofile-generate keeps naming baseline's directory so this build has the
    # flags the cloned artifacts were made with and stays incremental. Where the
    # profiles land is settled at run time instead, by LLVM_PROFILE_FILE.
    begin "building instrumented pivot server"
    ( cd "$crate_dir" \
        && PGO_DIR="$baseline/pgo" PGO_GEN_TARGET_DIR="$working/target-pgogen" \
           just pgo-gen-build build --profile "$cargo_profile" -p bin --bin pivot --features unbounded-park )
    elapsed "instrumented build"

    begin "building the client (plain release, no profile flags)"
    ( cd "$crate_dir" && CARGO_TARGET_DIR="$root/target-client" \
        cargo build --release -p benchmarks --bin pivot-bench )
    elapsed "client build"

    begin "profiling run over $pgo_source"
    rm -f "$working"/pgo/*.profraw "$profile_marker"
    query_args=()
    [[ -n "$queries" ]] && query_args=(--query "$queries")
    # PIVOT_SPIN_LIMIT=0 parks idle workers immediately instead of spinning,
    # so the profile records the wait-heavy control-flow mix that cold runs
    # on the full dataset execute, deterministically rather than as a
    # timing-dependent draw per build. Both variables reach the instrumented
    # server through the environment pivot-bench spawns it with; the client
    # itself is not instrumented and writes nothing.
    LLVM_PROFILE_FILE="$working/pgo/default_%m_%p.profraw" PIVOT_SPIN_LIMIT=0 \
        "$client" --suite "$suite" \
        --server-bin "$working/target-pgogen/$host_target/$profile_dir/pivot" \
        --source "$pgo_source" \
        --iterations "$iterations" --skip-check "${query_args[@]}"
    elapsed "profiling run"

    begin "installing the new profile"
    # The shared recipe fails loudly when the run produced no .profraw files,
    # so a build can never quietly proceed un-PGOed.
    ( cd "$crate_dir" && just pgo-merge "$profdata" "$working/pgo" )
    # Hold on to the slot's mtime across the overwrite. Cargo decides a crate is
    # stale by comparing dep-info entries' mtimes against the artifact, so
    # restoring it is what stops the 326 dependencies from rebuilding. Only the
    # slot is written; baseline's own profile is left alone.
    mtime_ref="$(mktemp)"
    touch -r "$active" "$mtime_ref"
    cp "$profdata" "$active"
    touch -r "$mtime_ref" "$active"
    rm -f "$mtime_ref"
    printf '%s\n' "$cargo_profile" >"$profile_marker"

    # Cargo now considers everything fresh, our crates included, so nothing
    # would pick the new profile up. Touching their roots is what forces them.
    # All of them, not just the lowest: a crate that depends on none of the
    # others would otherwise keep compiling against the old profile.
    touched=0
    while IFS= read -r src; do
        touch "$src"
        touched=$((touched + 1))
    done < <(workspace_crate_roots)
    echo "    touched $touched workspace crate roots"
    elapsed "profile install"
fi

begin "building profile-use pivot server (--profile $cargo_profile)"
( cd "$crate_dir" \
    && PGO_USE_TARGET_DIR="$working/target-pgouse" \
       just pgo-use-with "$active" build --profile "$cargo_profile" -p bin --bin pivot --features unbounded-park )
elapsed "profile-use build"

# Tripwire: fail loudly if the profile did not actually apply to the build.
( cd "$crate_dir" && just pgo-verify "$working/target-pgouse/$host_target/$profile_dir/pivot" "$active" )

# The client is rebuilt even without --regen-profile so a client-side change
# (a new flag, a protocol tweak) is picked up; it is a no-op when nothing
# changed. The regen path above already built it.
if [[ $regen_profile -eq 0 ]]; then
    begin "building the client (plain release, no profile flags)"
    ( cd "$crate_dir" && CARGO_TARGET_DIR="$root/target-client" \
        cargo build --release -p benchmarks --bin pivot-bench )
    elapsed "client build"
fi

echo
echo "=== ready ==="
echo "client binary   : $client"
echo "baseline server : $baseline/target-pgouse/$host_target/$profile_dir/pivot"
echo "working  server : $working/target-pgouse/$host_target/$profile_dir/pivot"
# An A/B compares the two servers, so baseline has to have been built under
# the same profile. Say so plainly rather than leaving a missing path to be
# discovered when the comparison is run.
if [[ ! -f "$baseline/target-pgouse/$host_target/$profile_dir/pivot" ]]; then
    echo
    echo "warning: baseline has no $cargo_profile server, so there is nothing to"
    echo "         compare against. Re-run setup-bench with --profile $cargo_profile."
    echo "         baseline currently holds:"
    for built in "$baseline/target-pgouse/$host_target"/*/pivot; do
        [[ -f "$built" ]] && echo "           $built"
    done
fi
