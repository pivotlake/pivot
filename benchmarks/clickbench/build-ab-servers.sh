#!/usr/bin/env bash
#
# build-ab-servers.sh - build one PGO pivotdb-server per side, for bench-ab.sh.
#
# bench-ab.sh measures prebuilt binaries and no longer builds anything; this is
# what builds them. Each side gets its own profile directory, so neither has to
# wipe a shared one and the two profiles cannot be confused for each other. The
# instrument/train/optimize/verify pipeline itself is build-pgo-server.sh
# (which in turn goes through the pgo.just recipes), so the A/B and deploy
# builds cannot drift from each other.
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The delegate may sit beside this script (perf-ab.yml ships both standalone
# to /tmp on the box, staged from the run's commit so an older 'before' tree
# can't miss them) or one directory up in the repo layout
# (benchmarks/clickbench/ -> benchmarks/).
if [[ -x "$script_dir/build-pgo-server.sh" ]]; then
    build_pgo_server="$script_dir/build-pgo-server.sh"
elif [[ -x "$script_dir/../build-pgo-server.sh" ]]; then
    build_pgo_server="$script_dir/../build-pgo-server.sh"
else
    echo "error: build-pgo-server.sh not found beside $script_dir or one directory up" >&2
    exit 2
fi

# Instrument the server itself, drive it through pivot-bench over the hits
# subset, and build the optimized server from the profile the server process
# wrote - all delegated to build-pgo-server.sh, which also verifies the
# profile applied. This launcher's copy of that script builds both trees, so
# the two sides go through identical logic even when the trees differ in age.
build_side() {
    local tree="$1" side="$2"
    # Beside the tree, never inside it: the launcher rsyncs each tree with
    # --delete and only excludes the target dirs, so a profile directory in
    # there would be deleted out from under the build.
    local pgo="$(dirname "$tree")/pgo-$side"
    local out
    if ! out="$("$build_pgo_server" \
            --tree "$tree" --pgo-dir "$pgo" --iterations 2 \
            --suite "clickbench=$pgo_subset")"; then
        echo "error: building the $side server from $tree failed" >&2
        return 1
    fi
    printf '%s' "${out#SERVER_BIN=}"
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
