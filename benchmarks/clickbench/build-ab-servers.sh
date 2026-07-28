#!/usr/bin/env bash
#
# build-ab-servers.sh - build one PGO pivotdb-server per side, for bench-ab.sh.
#
# bench-ab.sh measures prebuilt binaries and no longer builds anything; this is
# what builds them. Each side gets its own profile directory, so neither has to
# wipe a shared one and the two profiles cannot be confused for each other. The
# build goes through the pgo.just recipes rather than repeating their RUSTFLAGS,
# so it cannot drift from them.
#
# Prints `BEFORE_BIN=<path>` and `AFTER_BIN=<path>` on stdout, ready to be
# sourced; all build chatter goes to stderr.
#
# Usage:
#   build-ab-servers.sh --before-dir ~/perf-ab/before --after-dir ~/perf-ab/after \
#       --pgo-subset ~/hits-pgo-subset

set -euo pipefail

before_dir=""
after_dir=""
pgo_subset=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --before-dir)  before_dir="$2"; shift 2 ;;
        --after-dir)   after_dir="$2"; shift 2 ;;
        --pgo-subset)  pgo_subset="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for req in before_dir after_dir pgo_subset; do
    [[ -n "${!req}" ]] || { echo "error: --${req//_/-} is required" >&2; exit 2; }
done

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
before_dir="$(expand_tilde "$before_dir")"
after_dir="$(expand_tilde "$after_dir")"
pgo_subset="$(expand_tilde "$pgo_subset")"

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

# Build the instrumented binary, run it over the small subset to produce this
# side's profile, then build the server against that profile. The profiles land
# where LLVM_PROFILE_FILE says at run time, so -Cprofile-generate can keep
# naming one fixed directory and the build stays cacheable run over run.
build_side() {
    local tree="$1" side="$2"
    # Beside the tree, never inside it: the launcher rsyncs each tree with
    # --delete and only excludes the target dirs, so a profile directory in
    # there would be deleted out from under the build.
    local pgo="$(dirname "$tree")/pgo-$side"
    if ! (
        cd "$tree/benchmarks"
        rm -rf "$pgo"
        mkdir -p "$pgo"
        PGO_DIR="$pgo" PGO_GEN_TARGET_DIR=target-pgogen \
            just pgo-gen-build build --release --bin pivot-bench
        LLVM_PROFILE_FILE="$pgo/%m-%p.profraw" \
            "target-pgogen/$host_target/release/pivot-bench" \
            --source "$pgo_subset" --iterations 2 --skip-check >/dev/null
        "$llvm_profdata" merge -o "$pgo/merged.profdata" "$pgo"/*.profraw
        PGO_USE_TARGET_DIR=target-pgouse \
            just pgo-use-with "$pgo/merged.profdata" build --release -p server --bin pivotdb-server
    ) >&2; then
        # Without this the subshell's failure is swallowed by the printf below,
        # and the run only trips at the final existence check, which then names
        # whichever side is checked first rather than the one that broke.
        echo "error: building the $side server from $tree failed" >&2
        return 1
    fi
    printf '%s' "$tree/benchmarks/target-pgouse/$host_target/release/pivotdb-server"
}

echo ">>> building BEFORE server" >&2
before_bin="$(build_side "$before_dir" before)"
echo ">>> building AFTER server" >&2
after_bin="$(build_side "$after_dir" after)"

for built in "$before_bin" "$after_bin"; do
    [[ -x "$built" ]] || { echo "error: expected a server at $built" >&2; exit 1; }
done

printf 'BEFORE_BIN=%s\n' "$before_bin"
printf 'AFTER_BIN=%s\n' "$after_bin"
