//! Startup check for CPU features enabled at compile time.
//!
//! Linux aarch64 and x86_64 builds set an instruction floor above the target
//! default, while PGO may add `-Ctarget-cpu=native`. Unsupported instructions
//! would otherwise cause a late `SIGILL` in hot code.
//! [`Dispatch::spin_up`](crate::Dispatch::spin_up) checks first and reports
//! the missing feature before workers start.
//!
//! For each `cfg!(target_feature = ...)`, hardware support comes from
//! `AT_HWCAP` on aarch64 and CPUID on x86_64. The standard feature-detection
//! macros cannot do this check because statically enabled features report
//! `true` without probing the CPU.
//!
//! This is diagnostic, not a guarantee. An illegal instruction before the call
//! site can still fault first.

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

/// Extensions this process was compiled to use that the CPU does not advertise.
/// Empty on every supported CPU.
#[cfg(target_arch = "x86_64")]
pub fn missing_cpu_features() -> Vec<&'static str> {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    // Unsupported CPUID leaves return unrelated data, so gate optional leaves.
    let max_leaf = __cpuid(0).eax;
    let leaf1_ecx = __cpuid(1).ecx;
    let (leaf7_ebx, leaf7_ecx) = if max_leaf >= 7 {
        let leaf7 = __cpuid_count(7, 0);
        (leaf7.ebx, leaf7.ecx)
    } else {
        (0, 0)
    };
    let max_ext_leaf = __cpuid(0x8000_0000).eax;
    let ext1_ecx = if max_ext_leaf >= 0x8000_0001 {
        __cpuid(0x8000_0001).ecx
    } else {
        0
    };

    let mut missing = Vec::new();

    // Require each compiled feature. CPUID bit assignments are stable ABI.
    macro_rules! require {
        ($feature:literal, $register:expr, $bit:expr) => {
            if cfg!(target_feature = $feature) && $register & (1u32 << $bit) == 0 {
                missing.push($feature);
            }
        };
    }

    // Shipped x86-64-v3 floor: the v2 level from CPUID leaf 1 ECX, then the
    // AVX2 generation split across leaves. Linux enables XSAVE with
    // AVX-capable hardware; this check remains diagnostic.
    require!("sse3", leaf1_ecx, 0);
    require!("ssse3", leaf1_ecx, 9);
    require!("cmpxchg16b", leaf1_ecx, 13);
    require!("sse4.1", leaf1_ecx, 19);
    require!("sse4.2", leaf1_ecx, 20);
    require!("popcnt", leaf1_ecx, 23);
    require!("fma", leaf1_ecx, 12);
    require!("movbe", leaf1_ecx, 22);
    require!("avx", leaf1_ecx, 28);
    require!("f16c", leaf1_ecx, 29);
    require!("lzcnt", ext1_ecx, 5);
    require!("bmi1", leaf7_ebx, 3);
    require!("avx2", leaf7_ebx, 5);
    require!("bmi2", leaf7_ebx, 8);

    // Native PGO builds may enable these.
    require!("avx512f", leaf7_ebx, 16);
    require!("avx512dq", leaf7_ebx, 17);
    require!("avx512ifma", leaf7_ebx, 21);
    require!("avx512cd", leaf7_ebx, 28);
    require!("avx512bw", leaf7_ebx, 30);
    require!("avx512vl", leaf7_ebx, 31);
    require!("avx512vbmi", leaf7_ecx, 1);
    require!("avx512vbmi2", leaf7_ecx, 6);
    require!("gfni", leaf7_ecx, 8);
    require!("vaes", leaf7_ecx, 9);
    require!("vpclmulqdq", leaf7_ecx, 10);
    require!("avx512vnni", leaf7_ecx, 11);
    require!("avx512bitalg", leaf7_ecx, 12);
    require!("avx512vpopcntdq", leaf7_ecx, 14);

    missing
}

/// Nothing to check away from Linux aarch64 and x86_64: no other target sets an
/// instruction-set floor above its default.
#[cfg(not(any(
    all(target_arch = "aarch64", target_os = "linux"),
    target_arch = "x86_64"
)))]
pub fn missing_cpu_features() -> Vec<&'static str> {
    Vec::new()
}

/// Runtime check for x86-64-v4 target-feature clones: the base AVX-512 set of
/// Skylake-SP and Cascade Lake. Keep this feature list aligned with their
/// `#[target_feature]` attributes.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn supports_v4_kernels() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512cd")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

/// Runtime check for Ice Lake target-feature clones, also supported by AMD
/// Zen 4 and later. Keep this feature list aligned with their attributes.
///
/// Statically enabled features report `true`, which is correct here because
/// the whole binary already requires them.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn supports_icelake_kernels() -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512cd")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vbmi")
            && std::arch::is_x86_feature_detected!("avx512vbmi2")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512bitalg")
            && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

/// Defines a shared kernel body, `#[target_feature]` clones for each x86 target
/// tier (Ice Lake and x86-64-v4), and a runtime dispatcher that picks the
/// widest tier this CPU supports.
///
/// The body must stay `#[inline(always)]` so each tier compiles it with its own
/// features. Keep the clones' feature lists aligned with
/// [`supports_icelake_kernels`] and [`supports_v4_kernels`].
///
/// Generics and `where` bounds use square brackets:
///
/// ```ignore
/// multitarget_kernel! {
///     fn example[A, T](values: &[T], seed: A) -> A
///     where [T: Copy + Into<A>]
///     { ... }
/// }
/// ```
///
/// All generic parameters must be inferred from arguments; free const
/// parameters and different tier shapes need manual wrappers.
macro_rules! multitarget_kernel {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $([$($gen:tt)*])? ( $($arg:ident : $argty:ty),* $(,)? )
        $(-> $ret:ty)? $(where [$($wh:tt)*])?
        $body:block
    ) => {
        ::paste::paste! {
            $(#[$meta])*
            #[inline(always)]
            $vis fn $name $(<$($gen)*>)? ( $($arg : $argty),* ) $(-> $ret)? $(where $($wh)*)? {
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: each feature check covers every feature on its clone.
                    if $crate::cpu_features::supports_icelake_kernels() {
                        return unsafe { [<$name _icelake>]($($arg),*) };
                    }
                    if $crate::cpu_features::supports_v4_kernels() {
                        return unsafe { [<$name _v4>]($($arg),*) };
                    }
                }
                [<$name _body>]($($arg),*)
            }

            /// Ice Lake target-feature clone.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(
                enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,bmi1,bmi2,lzcnt,movbe,fma"
            )]
            fn [<$name _icelake>] $(<$($gen)*>)? ( $($arg : $argty),* ) $(-> $ret)? $(where $($wh)*)? {
                [<$name _body>]($($arg),*)
            }

            /// x86-64-v4 target-feature clone (Skylake-SP / Cascade Lake).
            #[cfg(target_arch = "x86_64")]
            #[target_feature(
                enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,bmi1,bmi2,lzcnt,movbe,fma"
            )]
            fn [<$name _v4>] $(<$($gen)*>)? ( $($arg : $argty),* ) $(-> $ret)? $(where $($wh)*)? {
                [<$name _body>]($($arg),*)
            }

            #[inline(always)]
            fn [<$name _body>] $(<$($gen)*>)? ( $($arg : $argty),* ) $(-> $ret)? $(where $($wh)*)?
            $body
        }
    };
    // Capture `$slf` so `self` keeps invocation-site macro hygiene.
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(&mut $slf:ident $(, $arg:ident : $argty:ty)* $(,)?)
        $(-> $ret:ty)?
        $body:block
    ) => {
        ::paste::paste! {
            $(#[$meta])*
            #[inline(always)]
            $vis fn $name(&mut $slf $(, $arg : $argty)*) $(-> $ret)? {
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: each feature check covers every feature on its clone.
                    if $crate::cpu_features::supports_icelake_kernels() {
                        return unsafe { $slf.[<$name _icelake>]($($arg),*) };
                    }
                    if $crate::cpu_features::supports_v4_kernels() {
                        return unsafe { $slf.[<$name _v4>]($($arg),*) };
                    }
                }
                $slf.[<$name _body>]($($arg),*)
            }

            /// Ice Lake target-feature clone.
            #[cfg(target_arch = "x86_64")]
            #[target_feature(
                enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,bmi1,bmi2,lzcnt,movbe,fma"
            )]
            fn [<$name _icelake>](&mut $slf $(, $arg : $argty)*) $(-> $ret)? {
                $slf.[<$name _body>]($($arg),*)
            }

            /// x86-64-v4 target-feature clone (Skylake-SP / Cascade Lake).
            #[cfg(target_arch = "x86_64")]
            #[target_feature(
                enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,bmi1,bmi2,lzcnt,movbe,fma"
            )]
            fn [<$name _v4>](&mut $slf $(, $arg : $argty)*) $(-> $ret)? {
                $slf.[<$name _body>]($($arg),*)
            }

            #[inline(always)]
            fn [<$name _body>](&mut $slf $(, $arg : $argty)*) $(-> $ret)?
            $body
        }
    };
}
pub(crate) use multitarget_kernel;
