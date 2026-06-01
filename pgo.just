# pgo.just — shared PGO recipes, imported by per-crate justfiles.
# The importing justfile must define `pgo_dir`.

# Run a cargo command instrumented for PGO profile generation, then merge the profiles.
#
#   just pgo-gen bench --bench clickbench
#   just pgo-gen test --test integration
alias pgo-generate := pgo-gen
pgo-gen *args:
    mkdir -p {{pgo_dir}}
    RUSTFLAGS="-Cprofile-generate={{pgo_dir}} -Ctarget-cpu=native" \
        cargo {{args}}
    $(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata merge -o {{pgo_dir}}/merged.profdata {{pgo_dir}}

# Run a cargo command using the merged PGO profile.
#
#   just pgo-use bench --bench clickbench
pgo-use *args:
    RUSTFLAGS="-Cprofile-use={{pgo_dir}}/merged.profdata -Ctarget-cpu=native" \
        cargo {{args}}


# Remove all PGO profiles and merged data.
pgo-clean:
    rm -rf {{pgo_dir}}
