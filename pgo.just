# pgo.just — shared PGO recipes, imported by per-crate justfiles.
# The importing justfile must define `pgo_dir`.
#
# Each flavor gets its own target dir (target-pgogen/, target-pgouse/): RUSTFLAGS
# are part of cargo's fingerprint, so sharing target/ between instrumented, plain,
# and profile-use builds forces a full rebuild on every flavor switch (~5 min each).
# With separate dirs each flavor rebuilds incrementally (~10 s for a leaf-crate
# change). Costs disk for the extra target dirs.
#
# Both target dirs can be pointed elsewhere. setup-bench.sh gives the baseline
# and working sides of an A/B their own pair, so each side keeps its own warm
# artifacts and neither invalidates the other.
pgogen_target_dir := env_var_or_default("PGO_GEN_TARGET_DIR", "target-pgogen")
pgouse_target_dir := env_var_or_default("PGO_USE_TARGET_DIR", "target-pgouse")

# Run a cargo command instrumented for PGO profile generation, then merge the profiles.
#
#   just pgo-gen bench --bench clickbench
#   just pgo-gen test --test integration
alias pgo-generate := pgo-gen
pgo-gen *args:
    mkdir -p {{pgo_dir}}
    # RUSTC_WRAPPER= bypasses the kache rustc wrapper (.cargo/config.toml): these
    # optimized PGO builds are compiled fresh, never restored from cache.
    # PIVOT_SPIN_LIMIT=0 parks idle workers immediately during the profiled
    # workload, so the profile records the wait-heavy control-flow mix that
    # cold runs on the full dataset execute, deterministically rather than
    # as a timing-dependent draw per build.
    RUSTC_WRAPPER= RUSTFLAGS="-Cprofile-generate={{pgo_dir}} -Ctarget-cpu=native" \
        PIVOT_SPIN_LIMIT=0 \
        CARGO_TARGET_DIR={{pgogen_target_dir}} cargo {{args}}
    $(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata merge -o {{pgo_dir}}/merged.profdata {{pgo_dir}}

# Run a cargo command using the merged PGO profile.
#
#   just pgo-use bench --bench clickbench
pgo-use *args:
    RUSTC_WRAPPER= RUSTFLAGS="-Cprofile-use={{pgo_dir}}/merged.profdata -Ctarget-cpu=native" \
        CARGO_TARGET_DIR={{pgouse_target_dir}} cargo {{args}}


# Host triple, for the A/B recipes below. Passing an explicit --target (even
# the host's own triple) makes cargo stop applying RUSTFLAGS to host units
# (build scripts, proc macros): instrumentation never reaches them, and a
# profile-use flag change no longer invalidates their outputs, chiefly the
# DuckDB C++ cmake build, which no profile flag affects anyway.
host_target := `rustc -vV | sed -n 's/^host: //p'`

# Build instrumented for profile generation, without running or merging:
# for A/B automation that runs the workload itself, redirects each side's
# profiles via LLVM_PROFILE_FILE, and merges per side. Binaries land under
# target-pgogen/{{host_target}}/.
# lld because instrumentation can grow the text past the 128MB aarch64
# branch range, which GNU ld fails with relocation overflows.
pgo-gen-build subcmd *args:
    mkdir -p {{pgo_dir}}
    RUSTC_WRAPPER= RUSTFLAGS="-Cprofile-generate={{pgo_dir}} -Ctarget-cpu=native -Clink-arg=-fuse-ld=lld" \
        CARGO_TARGET_DIR={{pgogen_target_dir}} cargo {{subcmd}} --target {{host_target}} {{args}}

# Like pgo-use but with a caller-supplied profdata path. rustc records the
# profile file in dep-info, so even at a constant path a regenerated profile
# rebuilds every Rust unit; passing a per-profile path makes that rebuild a
# consequence of the flags hash too, immune to mtime comparisons against
# artifacts restored from another machine. Build-script outputs (unflagged
# thanks to the explicit --target) stay cached either way. Binaries land
# under target-pgouse/{{host_target}}/.
pgo-use-with profdata subcmd *args:
    RUSTC_WRAPPER= RUSTFLAGS="-Cprofile-use={{profdata}} -Ctarget-cpu=native" \
        CARGO_TARGET_DIR={{pgouse_target_dir}} cargo {{subcmd}} --target {{host_target}} {{args}}

# Remove all PGO profiles and merged data.
pgo-clean:
    rm -rf {{pgo_dir}}
