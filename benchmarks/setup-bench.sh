#!/usr/bin/env bash
#
# setup-bench.sh - build the baseline artifact tree an A/B run measures from,
# then clone it to the working tree. Run it once per box; after that an A/B
# iteration is an incremental rebuild of the working side instead of two PGO
# builds from scratch.
#
# The measured binary is pivot, launched as `pivot server`. pivot-bench is a
# thin pgwire client that spawns the server, drives the workload and times it,
# and carries no engine code, so the server is what gets instrumented and
# profile-used here, mirroring lib/bench-ab-common.sh and build-ab-servers.sh. The
# client gets one plain release build under target-client/, shared by both
# sides: it is not what is measured, and giving it profile flags would rebuild
# it under every fresh profile for nothing.
#
# Layout under --root (default ~/bench):
#
#   baseline/  pgo/  target-pgogen/  target-pgouse/
#   working/   pgo/  target-pgogen/  target-pgouse/
#   target-client/
#
# Both sides are built from the one source tree this script lives in, and that
# tree must stay where it is afterwards. Cargo records absolute paths in its
# unit fingerprints, so moving or duplicating the source tree invalidates every
# artifact here; giving each side its own copy of the *source* would rebuild
# from scratch and make these warm target dirs worthless. Two artifact trees
# over one source tree is what keeps the working side incremental.
#
# baseline/ is built once: an instrumented pivot binary, a profiling run over
# the small subset (the client driving the instrumented server), the merged
# profile, then the profile-use build. working/ then starts as a byte-identical
# copy (cp -a preserves mtimes, so cargo sees every unit as fresh), which is
# what makes the first build after an edit recompile only the crates that
# changed.
#
# Re-running is cheap: the cargo builds are no-ops when nothing changed, and
# the profiling run - the one expensive non-cargo step - is skipped when a
# merged profile already exists. Pass --regen-profile to force it.
#
# Meant to be launched detached and polled from a short-lived ssh session, so
# it writes its PID and emits a sentinel on every exit path.
#
# Usage:
#   just setup-bench --pgo-source ~/hits-pgo-subset
#   just setup-bench --pgo-source ~/hits-pgo-subset --root ~/bench --suite tpch-flat
#   just setup-bench --pgo-source ~/hits-pgo-subset --profile release

set -uo pipefail

pid_file="${PID_FILE:-/tmp/setup-bench.pid}"
echo $$ >"$pid_file"
# Fire on every exit path carrying the real exit code, so a poller watching the
# log never hangs on a silent failure.
trap 'echo "=== SETUP-BENCH COMPLETE exit=$? ==="' EXIT
set -e

crate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

root="$HOME/bench"
pgo_source=""
suite="clickbench"
iterations=2
regen_profile=0
force=0
queries=""
cargo_profile="profiling"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)          root="$2"; shift 2 ;;
        # Small dataset the instrumented server runs over to produce the
        # profile. Never a measurement dataset: instrumented code is ~80x
        # slower, and the numbers that matter come off the full one.
        --pgo-source)    pgo_source="$2"; shift 2 ;;
        --suite)         suite="$2"; shift 2 ;;
        --iterations)    iterations="$2"; shift 2 ;;
        # Queries to profile over. Default is every query the suite ships.
        --query)         queries="$2"; shift 2 ;;
        # Cargo profile for the measured server. The default, profiling,
        # inherits release and adds debug info, so perf can attribute to source
        # lines and resolve inlined frames. Codegen is unchanged, so timings are
        # release timings. Pass the same profile to bench-build, or the two
        # sides land in different artifact directories and are not comparable.
        --profile)       cargo_profile="$2"; shift 2 ;;
        # Re-run the profiling workload even when a merged profile exists.
        --regen-profile) regen_profile=1; shift ;;
        # Overwrite an existing working tree. Withheld by default because the
        # copy discards whatever the working side was built from.
        --force)         force=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$pgo_source" ]]; then
    echo "error: --pgo-source is required" >&2
    exit 2
fi

# Paths may arrive with a leading ~ from a launcher that could not expand it
# against this box's home, so expand it here.
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
pgo_source="$(expand_tilde "$pgo_source")"

[[ -d "$pgo_source" ]] || { echo "error: --pgo-source $pgo_source does not exist" >&2; exit 2; }
[[ -d "$crate_dir/$suite" ]] || { echo "error: no suite directory $crate_dir/$suite" >&2; exit 2; }

# cargo, just and llvm-profdata live under the login home but not on the
# non-interactive ssh PATH, so add them explicitly.
export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin"
export NO_COLOR=1

command -v ld.lld >/dev/null || {
    echo "error: ld.lld is not installed, and the instrumented build links with it" >&2
    echo "       (instrumentation grows the text past the aarch64 128MB branch" >&2
    echo "       range, which GNU ld fails). Install it: sudo apt-get install -y lld" >&2
    exit 2
}

baseline="$root/baseline"
working="$root/working"
profdata="$baseline/pgo/merged.profdata"
# The profile the compiler actually reads. Both sides name this one path, since
# the path is part of RUSTFLAGS and giving each side its own would change the
# flags hash and rebuild every crate. Each side's own profile is kept under its
# pgo/ directory and copied into this slot when it should be the one in force.
active="$root/pgo/active.profdata"
host_target="$(rustc -vV | sed -n 's/^host: //p')"
# The instrumented server is always release: the profile records which branches
# are hot, and debug info neither helps that nor survives into the final build.
instrumented="$baseline/target-pgogen/$host_target/release/pivot"
client="$root/target-client/release/pivot-bench"
# Cargo names most profiles' directories after the profile, but not the two
# built-in ones that predate named profiles.
case "$cargo_profile" in
    dev|test) profile_dir="debug" ;;
    bench)    profile_dir="release" ;;
    *)        profile_dir="$cargo_profile" ;;
esac

if [[ -e "$working" && $force -eq 0 ]]; then
    echo "error: $working already exists; pass --force to replace it" >&2
    exit 2
fi

# Report each phase's wall time, so a run that got slower is visible without
# instrumenting anything by hand.
phase_start=0
begin() { phase_start=$SECONDS; echo; echo ">>> $*"; }
elapsed() { echo "<<< $1 took $((SECONDS - phase_start))s"; }

echo "=== setup-bench ==="
echo "source tree : $crate_dir"
echo "root        : $root"
echo "suite       : $suite"
echo "profile     : $cargo_profile"
echo "pgo source  : $pgo_source"
echo "host target : $host_target"

mkdir -p "$root/pgo" "$baseline/pgo" "$baseline/target-pgogen" "$baseline/target-pgouse"

export PGO_DIR="$baseline/pgo"
export PGO_GEN_TARGET_DIR="$baseline/target-pgogen"
export PGO_USE_TARGET_DIR="$baseline/target-pgouse"

begin "building instrumented pivot server"
( cd "$crate_dir" && just pgo-gen-build build --release -p bin --bin pivot )
elapsed "instrumented build"

begin "building the client (plain release, no profile flags)"
( cd "$crate_dir" && CARGO_TARGET_DIR="$root/target-client" \
    cargo build --release -p benchmarks --bin pivot-bench )
elapsed "client build"

if [[ -f "$profdata" && $regen_profile -eq 0 ]]; then
    echo
    echo ">>> reusing existing profile $profdata (pass --regen-profile to rebuild it)"
else
    begin "profiling run over $pgo_source (instrumented server, expect it to crawl)"
    rm -f "$baseline"/pgo/*.profraw "$profdata"
    query_args=()
    [[ -n "$queries" ]] && query_args=(--query "$queries")
    # PIVOT_SPIN_LIMIT=0 parks idle workers immediately instead of spinning,
    # so the profile records the wait-heavy control-flow mix that cold runs
    # on the full dataset execute, deterministically rather than as a
    # timing-dependent draw per build. Both variables reach the instrumented
    # server through the environment pivot-bench spawns it with; the client
    # itself is not instrumented and writes nothing.
    LLVM_PROFILE_FILE="$baseline/pgo/%m-%p.profraw" PIVOT_SPIN_LIMIT=0 \
        "$client" --suite "$suite" --server-bin "$instrumented" \
        --source "$pgo_source" --iterations "$iterations" --skip-check "${query_args[@]}"
    elapsed "profiling run"

    begin "merging profiles"
    # The shared recipe fails loudly when the run produced no .profraw files,
    # so a build can never quietly proceed un-PGOed.
    ( cd "$crate_dir" && just pgo-merge "$profdata" "$baseline/pgo" )
    elapsed "merge"
fi

# Put baseline's profile in force. A fresh mtime here is correct: baseline is
# genuinely being built against a new profile.
cp "$profdata" "$active"

begin "building profile-use pivot server (--profile $cargo_profile)"
( cd "$crate_dir" && just pgo-use-with "$active" build --profile "$cargo_profile" -p bin --bin pivot )
elapsed "profile-use build"

# Tripwire: fail loudly if the profile did not actually apply to the build.
( cd "$crate_dir" && just pgo-verify "$baseline/target-pgouse/$host_target/$profile_dir/pivot" "$active" )

baseline_kb="$(du -sk "$baseline" | cut -f1)"
free_kb="$(df -Pk "$root" | awk 'NR == 2 { print $4 }')"
if (( free_kb < baseline_kb + baseline_kb / 10 )); then
    echo "error: cloning baseline needs $((baseline_kb / 1024))MB but only $((free_kb / 1024))MB is free on $root" >&2
    exit 1
fi

begin "cloning baseline -> working ($((baseline_kb / 1024))MB)"
rm -rf "$working"
# -a preserves mtimes, which is what lets cargo treat the copied units as fresh
# instead of rebuilding all of them at the new path.
cp -a "$baseline" "$working"
elapsed "clone"

echo
echo "=== ready ==="
echo "client binary   : $client"
echo "baseline server : $baseline/target-pgouse/$host_target/$profile_dir/pivot"
echo "working  server : $working/target-pgouse/$host_target/$profile_dir/pivot"
echo "baseline profile: $profdata"
echo "profile in force: $active"
