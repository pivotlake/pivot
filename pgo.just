# pgo.just — shared PGO recipes, imported by per-crate justfiles.
# The importing justfile must define `pgo_dir`.
#
# Each flavor gets its own target dir (target-pgogen/, target-pgouse/): RUSTFLAGS
# are part of cargo's fingerprint, so sharing target/ between instrumented, plain,
# and profile-use builds forces a full rebuild on every flavor switch (~5 min each).
# With separate dirs each flavor rebuilds incrementally (~10 s for a leaf-crate
# change). Costs disk for the extra target dirs.

# Run a cargo command instrumented for PGO profile generation, then merge the profiles.
#
#   just pgo-gen bench --bench clickbench
#   just pgo-gen test --test integration
alias pgo-generate := pgo-gen
pgo-gen *args:
    mkdir -p {{pgo_dir}}
    RUSTFLAGS="-Cprofile-generate={{pgo_dir}} -Ctarget-cpu=native" \
        CARGO_TARGET_DIR=target-pgogen cargo {{args}}
    $(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata merge -o {{pgo_dir}}/merged.profdata {{pgo_dir}}

# Run a cargo command using the merged PGO profile.
#
#   just pgo-use bench --bench clickbench
pgo-use *args:
    RUSTFLAGS="-Cprofile-use={{pgo_dir}}/merged.profdata -Ctarget-cpu=native" \
        CARGO_TARGET_DIR=target-pgouse cargo {{args}}


# Remove all PGO profiles and merged data.
pgo-clean:
    rm -rf {{pgo_dir}}
