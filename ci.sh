#!/usr/bin/env bash
#
# Single source of truth for CI checks — used by both developers and CI.
#
#   ./ci.sh                  run every check on every crate (local dev)
#   ./ci.sh <check> <crate>  run one cell (what each CI matrix job calls)
#   ./ci.sh --matrix         emit the check×crate matrix as JSON (for CI)
#
# CI (.github/workflows/ci.yml) generates its matrix from `--matrix` and runs
# one job per cell calling `./ci.sh <check> <crate>`, so every check shows up
# separately in the GitHub UI. The actual commands live here, once, so local
# runs and CI can never drift.
#
# Run from the repo root.
set -uo pipefail

# The crates CI checks and the checks it runs. Edit these in one place; both the
# local run-all and the CI matrix follow.
CRATES=(dispatch planner duckdb-planner catalog datastore-delta metastore-disk bin)
CHECKS=(fmt clippy test doc)

export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"

# An env RUSTFLAGS REPLACES the [target.*] rustflags in .cargo/config.toml, so
# the -D warnings default must restate the host's instruction-set floor or CI
# would silently build (and test) below what we ship. Keep these in step with
# .cargo/config.toml.
floor=""
case "$(uname -m)" in
    x86_64) floor=" -Ctarget-cpu=x86-64-v3" ;;
    aarch64) [[ "$(uname -s)" == "Linux" ]] &&
        floor=" -Ctarget-feature=+lse,+crc,+rdm,+dpb,+neon,+aes,+sha2,+dotprod,+ssbs,+rcpc,+bf16" ;;
esac
export RUSTFLAGS="${RUSTFLAGS:--D warnings$floor}"
export RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}"

# Run a single check against a single crate. Exit status is the check's status.
run_cell() {
    local check="$1" crate="$2"
    (
        cd "$crate" || exit 2
        case "$check" in
            fmt)    cargo fmt --check ;;
            clippy) cargo clippy --all-targets ;;
            test)   cargo test ;;
            # Both the public-docs and internal-docs builds CI gates on.
            doc)    cargo doc --no-deps &&
                    cargo doc --document-private-items --no-deps ;;
            *) echo "unknown check '$check' (expected: ${CHECKS[*]})" >&2; exit 2 ;;
        esac
    )
}

# Emit the matrix as a single-line JSON array of {check, crate} objects, e.g.
# [{"check":"fmt","crate":"dispatch"},...]. Consumed by CI via fromJSON.
emit_matrix() {
    local first=1 out="["
    for check in "${CHECKS[@]}"; do
        for crate in "${CRATES[@]}"; do
            [[ $first -eq 1 ]] || out+=","
            first=0
            out+="{\"check\":\"$check\",\"crate\":\"$crate\"}"
        done
    done
    out+="]"
    printf '%s\n' "$out"
}

case "${1:-}" in
    --matrix)
        emit_matrix
        ;;
    "")
        # Local dev: run the whole matrix, keep going on failure, summarise.
        fail=0
        failed=()
        for check in "${CHECKS[@]}"; do
            for crate in "${CRATES[@]}"; do
                printf '\n\033[1m=== %s / %s ===\033[0m\n' "$check" "$crate"
                if run_cell "$check" "$crate"; then
                    printf '  \033[32m✓ %s %s\033[0m\n' "$check" "$crate"
                else
                    printf '  \033[31m✗ %s %s\033[0m\n' "$check" "$crate"
                    fail=1
                    failed+=("$check/$crate")
                fi
            done
        done
        echo
        if [[ $fail -eq 0 ]]; then
            printf '\033[32mALL CHECKS PASSED\033[0m\n'
        else
            printf '\033[31mFAILED: %s\033[0m\n' "${failed[*]}"
        fi
        exit $fail
        ;;
    *)
        # CI cell (or targeted local run): exactly one check on one crate.
        if [[ $# -ne 2 ]]; then
            echo "usage: $0 [--matrix | <check> <crate>]   (no args = run all)" >&2
            exit 2
        fi
        run_cell "$1" "$2"
        ;;
esac
