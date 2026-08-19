//! Boot-time check that this CPU implements the instruction-set extensions the
//! binary was compiled to use.
//!
//! Linux aarch64 builds compile against a floor above the bare target default
//! (see `.cargo/config.toml`), and PGO builds go further with
//! `-Ctarget-cpu=native`. A CPU lacking one of those extensions dies with
//! `SIGILL` at whichever vectorised loop reaches the instruction first: no
//! message, and a stack pointing at arithmetic rather than the cause.
//! [`Dispatch::spin_up`](crate::Dispatch::spin_up) calls this before spawning a
//! worker, so every binary fails there with a named reason instead.
//!
//! Each check pairs "did we build with this" (`cfg!(target_feature = ...)`) with
//! "does this CPU have it" (a bit in `AT_HWCAP`). Only extensions compiled in are
//! required, so one list covers the floor and a native build alike.
//!
//! `std::arch::is_aarch64_feature_detected!` cannot serve as the runtime half: it
//! short-circuits to a compile-time `true` whenever the feature is already enabled
//! statically, exactly the case here, making the pairing a tautology that reports
//! nothing on any CPU. On Linux, the auxiliary vector is the way to learn what
//! the hardware advertises from inside a binary compiled for it.
//!
//! A diagnostic, not a guarantee: an illegal instruction in a static initialiser
//! or above the call site still faults first. It covers the case that matters,
//! where the instruction is in a hot loop reached long after startup.

/// Extensions this process was compiled to use that the CPU does not advertise.
/// Empty on every supported CPU.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub fn missing_cpu_features() -> Vec<&'static str> {
    // `AT_HWCAP2` bits, which the libc crate still has commented out. Values
    // from the kernel's arch/arm64/include/uapi/asm/hwcap.h; the assignment is
    // append-only ABI, so these cannot drift.
    const HWCAP2_DCPODP: u64 = 1 << 0;
    const HWCAP2_SVE2: u64 = 1 << 1;
    const HWCAP2_I8MM: u64 = 1 << 13;
    const HWCAP2_BF16: u64 = 1 << 14;

    // SAFETY: `getauxval` reads this process's own auxiliary vector. It takes an
    // integer and returns one, touching no memory we own, and answers 0 for a
    // type the kernel did not supply.
    let (hwcap, hwcap2) = unsafe {
        (
            libc::getauxval(libc::AT_HWCAP) as u64,
            libc::getauxval(libc::AT_HWCAP2) as u64,
        )
    };

    let mut missing = Vec::new();

    // Each arm: if we compiled with the extension, the CPU has to advertise it.
    // The two spellings differ on purpose - rustc's `target_feature` name, then
    // the kernel's capability bit, which is often a different word for the same
    // thing (`neon` is `ASIMD`, `lse` is `ATOMICS`, `dotprod` is `ASIMDDP`).
    macro_rules! require {
        ($feature:literal, $word:expr, $bit:expr) => {
            if cfg!(target_feature = $feature) && $word & ($bit as u64) == 0 {
                missing.push($feature);
            }
        };
    }

    // The shipped floor: Armv8.2-A (as its individually named features) plus
    // SIMD, crypto, dot product, RCpc and BFloat16.
    //
    // `ssbs` is in the floor's compile flags but is deliberately absent here. It
    // controls a PSTATE speculation bit rather than any instruction a compiler
    // emits, and Linux does not advertise `HWCAP_SSBS` on every core that
    // implements it - Neoverse-V2 is one that does not - so requiring it would
    // refuse to boot on hardware that runs the code perfectly well.
    require!("lse", hwcap, libc::HWCAP_ATOMICS);
    require!("crc", hwcap, libc::HWCAP_CRC32);
    require!("rdm", hwcap, libc::HWCAP_ASIMDRDM);
    require!("dpb", hwcap, libc::HWCAP_DCPOP);
    require!("neon", hwcap, libc::HWCAP_ASIMD);
    require!("aes", hwcap, libc::HWCAP_AES);
    require!("sha2", hwcap, libc::HWCAP_SHA2);
    require!("dotprod", hwcap, libc::HWCAP_ASIMDDP);
    require!("rcpc", hwcap, libc::HWCAP_LRCPC);
    require!("bf16", hwcap2, HWCAP2_BF16);

    // Above the floor: what `-Ctarget-cpu=native` adds on the Neoverse cores the
    // benchmark and PGO builds are tuned for. Unset in a floor build, so these
    // arms cost nothing there.
    require!("lse2", hwcap, libc::HWCAP_USCAT);
    require!("rcpc2", hwcap, libc::HWCAP_ILRCPC);
    require!("fp16", hwcap, libc::HWCAP_ASIMDHP);
    require!("sha3", hwcap, libc::HWCAP_SHA3);
    require!("sve", hwcap, libc::HWCAP_SVE);
    require!("sve2", hwcap2, HWCAP2_SVE2);
    require!("i8mm", hwcap2, HWCAP2_I8MM);
    require!("dpb2", hwcap2, HWCAP2_DCPODP);

    missing
}

/// Nothing to check away from Linux aarch64: no other target sets an
/// instruction-set floor above its default.
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
pub fn missing_cpu_features() -> Vec<&'static str> {
    Vec::new()
}
