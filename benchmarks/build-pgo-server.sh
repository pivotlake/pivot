#!/usr/bin/env bash
#
# build-pgo-server.sh - build one PGO-optimized pivotdb-server.
#
# Instruments the server itself, drives it through each requested suite with
# pivot-bench (a pure pgwire client carrying no engine code and no
# instrumentation), merges the profiles the server process wrote, builds the
# optimized server from them, and verifies the profile actually applied.
# Profiling the artifact that ships is what makes the profile's symbol names
# match by construction; PGO matches records to functions by exact mangled
# name, and a mismatch silently disables it. Every build goes through the
# pgo.just recipes rather than repeating their RUSTFLAGS, so it cannot drift
# from them.
#
# Prints `SERVER_BIN=<path>` on stdout, ready to be sourced; all build and
# benchmark chatter goes to stderr.
#
# Usage:
#   build-pgo-server.sh --pgo-dir ~/pgo \
#       --suite clickbench=~/hits --suite tpch=~/tpch-sf100
#
#   --suite <name>=<source>  a suite directory under benchmarks/ and the
#                            --source data it reads; repeatable, run in the
#                            order given. At least one is required.
#   --pgo-dir <dir>          where the .profraw files and the merged profile
#                            land; wiped at the start of the run. Required.
#   --iterations <n>         pivot-bench iterations per suite (default 2:
#                            one cold pass, one warm).
#   --tree <repo-root>       repository tree to build; defaults to the tree
#                            this script sits in. Lets one checkout drive
#                            builds of several trees (see build-ab-servers.sh).
#
# Environment knobs, all optional:
#   PGO_TARGET_CPU           reaches pgo.just: default native for bench boxes;
#                            deploy.yml sets it empty so shipped binaries stay
#                            portable.
#   PGO_GEN_TARGET_DIR, PGO_USE_TARGET_DIR, PIVOT_BENCH_TARGET_DIR
#                            per-flavor cargo target dirs, relative to the
#                            tree's benchmarks/ dir unless absolute. Defaults:
#                            target-pgogen, target-pgouse, target-client.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pgo_dir=""
iterations=2
tree="$(cd "$script_dir/.." && pwd)"
suites=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pgo-dir)    pgo_dir="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --tree)       tree="$2"; shift 2 ;;
        --suite)      suites+=("$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$pgo_dir" ]] || { echo "error: --pgo-dir is required" >&2; exit 2; }
[[ ${#suites[@]} -gt 0 ]] || { echo "error: at least one --suite <name>=<source> is required" >&2; exit 2; }

# Paths may arrive with a leading ~ from a launcher that could not expand it.
expand_tilde() {
    # The "~" patterns match a literal leading tilde in the argument.
    # shellcheck disable=SC2088
    case "$1" in
        "~")   printf '%s' "$HOME" ;;
        "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
        *)     printf '%s' "$1" ;;
    esac
}
pgo_dir="$(expand_tilde "$pgo_dir")"
tree="$(expand_tilde "$tree")"

export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin"
export NO_COLOR=1

command -v ld.lld >/dev/null || {
    echo "error: ld.lld is not installed, and the instrumented build links with it" >&2
    echo "       (instrumentation grows the text past the aarch64 128MB branch" >&2
    echo "       range, which GNU ld fails). Install it: sudo apt-get install -y lld" >&2
    exit 2
}

host_target="$(rustc -vV | sed -n 's/^host: //p')"
llvm_profdata="$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata"
[[ -x "$llvm_profdata" ]] || {
    echo "error: llvm-profdata is not at $llvm_profdata" >&2
    echo "       install it: rustup component add llvm-tools" >&2
    exit 2
}

gen_target_dir="${PGO_GEN_TARGET_DIR:-target-pgogen}"
use_target_dir="${PGO_USE_TARGET_DIR:-target-pgouse}"
client_target_dir="${PIVOT_BENCH_TARGET_DIR:-target-client}"

# Validate every suite spec up front, before spending minutes on builds.
for spec in "${suites[@]}"; do
    name="${spec%%=*}"
    source_dir="$(expand_tilde "${spec#*=}")"
    [[ "$spec" == *=* && -n "$name" && -n "$source_dir" ]] || {
        echo "error: --suite wants <name>=<source>, got '$spec'" >&2; exit 2; }
    [[ -d "$tree/benchmarks/$name" ]] || {
        echo "error: no suite directory at $tree/benchmarks/$name" >&2; exit 2; }
    [[ -d "$source_dir" ]] || {
        echo "error: suite $name's source directory does not exist: $source_dir" >&2; exit 2; }
done

(
    cd "$tree/benchmarks"
    rm -rf "$pgo_dir"
    mkdir -p "$pgo_dir"

    echo ">>> building the instrumented server" >&2
    PGO_DIR="$pgo_dir" just pgo-gen-build build --release -p server --bin pivotdb-server >&2

    # The client is a plain build in its own target dir: it takes no profile
    # flags, and sharing a flagged dir would rebuild it for nothing on every
    # flavor switch.
    CARGO_TARGET_DIR="$client_target_dir" RUSTC_WRAPPER= \
        cargo build --release -p benchmarks --bin pivot-bench >&2

    for spec in "${suites[@]}"; do
        name="${spec%%=*}"
        source_dir="$(expand_tilde "${spec#*=}")"
        echo ">>> profiling run: suite $name over $source_dir" >&2
        # CREATE TABLE over pre-existing parquet leaves a _delta_log inside
        # the source directory; one left by an earlier run, or by a previous
        # suite sharing this directory, must not be adopted. Clear them the
        # way bench-ab.sh does before its runs. Depth 2 also reaches the
        # per-table layouts (<source>/<table>/_delta_log).
        find "$source_dir" -maxdepth 2 -type d -name _delta_log -prune -exec rm -rf {} + 2>/dev/null || true
        # LLVM_PROFILE_FILE reaches the instrumented server through the
        # environment pivot-bench spawns it with; the client itself is not
        # instrumented and writes nothing. --skip-check because training data
        # need not match the committed expected outputs, and a profiling run
        # only wants the queries executed, not graded.
        LLVM_PROFILE_FILE="$pgo_dir/$name-%m-%p.profraw" \
            "$client_target_dir/release/pivot-bench" \
            --suite "$name" \
            --server-bin "$gen_target_dir/$host_target/release/pivotdb-server" \
            --source "$source_dir" --iterations "$iterations" --skip-check >&2
    done

    "$llvm_profdata" merge -o "$pgo_dir/merged.profdata" "$pgo_dir"/*.profraw >&2

    echo ">>> building the optimized server" >&2
    just pgo-use-with "$pgo_dir/merged.profdata" build --release -p server --bin pivotdb-server >&2
)

case "$use_target_dir" in
    /*) server="$use_target_dir/$host_target/release/pivotdb-server" ;;
    *)  server="$tree/benchmarks/$use_target_dir/$host_target/release/pivotdb-server" ;;
esac
[[ -x "$server" ]] || { echo "error: expected a server at $server" >&2; exit 1; }

# Tripwire for the profile applying at all: the decode family's
# monomorphization hashes in the built server must appear in the profile it
# was compiled against. Zero overlap means the server was built outside the
# profiled symbol universe and is effectively un-PGOed, which is silent at
# compile time and shows up only as a mystery regression.
family="RleDecoder4read"
binary_hashes="$( (nm "$server" | grep "$family" | grep -oE '17h[0-9a-f]+E' | sort -u) || true)"
if [[ -z "$binary_hashes" ]]; then
    echo "error: no $family symbols in the built server; the tripwire that" >&2
    echo "       proves the profile applied has nothing to check. If the decode" >&2
    echo "       family was renamed, update \$family in this script." >&2
    exit 1
fi
profile_hashes="$( ("$llvm_profdata" show -all-functions "$pgo_dir/merged.profdata" \
    2>/dev/null | grep "$family" | grep -oE '17h[0-9a-f]+E' | sort -u) || true)"
covered=$(comm -12 <(printf '%s\n' "$binary_hashes") \
                   <(printf '%s\n' "$profile_hashes") | grep -c . || true)
if [[ "$covered" -eq 0 ]]; then
    echo "error: the server shares no $family symbols with its profile;" >&2
    echo "       the profile did not apply and the binary is effectively un-PGOed" >&2
    exit 1
fi
echo "server: $covered $family monomorphizations carry profile records" >&2

printf 'SERVER_BIN=%s\n' "$server"
