//! Decode the `PERF_SAMPLE_DATA_SRC` u64 bitfield.
//!
//! The layout is defined in `<linux/perf_event.h>` as `union perf_mem_data_src`.
//! Modern x86 (and AMD with IBS) populates `mem_lvl_num` (cache-level enum)
//! plus `mem_remote` and `mem_hops` rather than the legacy `mem_lvl` bitmask;
//! we honour `mem_lvl_num` first and fall back to `mem_lvl` so older perf
//! captures still decode.
//!
//! Field bit positions (matching the kernel header verbatim):
//!
//! ```text
//!   mem_op       :  5  bits 0..5     PERF_MEM_OP_*
//!   mem_lvl      : 14  bits 5..19    PERF_MEM_LVL_* (legacy bitmask)
//!   mem_snoop    :  5  bits 19..24   PERF_MEM_SNOOP_*
//!   mem_lock     :  2  bits 24..26   PERF_MEM_LOCK_*
//!   mem_dtlb     :  7  bits 26..33   PERF_MEM_TLB_*
//!   mem_lvl_num  :  4  bits 33..37   PERF_MEM_LVLNUM_* (modern)
//!   mem_remote   :  1  bit  37       PERF_MEM_REMOTE
//!   mem_snoopx   :  2  bits 38..40   PERF_MEM_SNOOPX_*
//!   mem_blk      :  3  bits 40..43   PERF_MEM_BLK_*
//!   mem_hops     :  3  bits 43..46   PERF_MEM_HOPS_*
//!   mem_rsvd     : 18  bits 46..64
//! ```
//!
//! Some constants below aren't consumed by the decoder yet — they document
//! the bitfield's full layout for the next reader.
#![allow(dead_code)]

use crate::model::{CacheLevel, OpType, SnoopStatus, TlbLevel};

// PERF_MEM_OP_* (5 bits, 0..5)
const PERF_MEM_OP_NA:     u64 = 1 << 0;
const PERF_MEM_OP_LOAD:   u64 = 1 << 1;
const PERF_MEM_OP_STORE:  u64 = 1 << 2;
const PERF_MEM_OP_PFETCH: u64 = 1 << 3;
const PERF_MEM_OP_EXEC:   u64 = 1 << 4;

// PERF_MEM_LVL_* (legacy 14-bit bitmask, 5..19)
const PERF_MEM_LVL_NA:      u64 = 1 << 0;
const PERF_MEM_LVL_HIT:     u64 = 1 << 1;
const PERF_MEM_LVL_MISS:    u64 = 1 << 2;
const PERF_MEM_LVL_L1:      u64 = 1 << 3;
const PERF_MEM_LVL_LFB:     u64 = 1 << 4;
const PERF_MEM_LVL_L2:      u64 = 1 << 5;
const PERF_MEM_LVL_L3:      u64 = 1 << 6;
const PERF_MEM_LVL_LOC_RAM: u64 = 1 << 7;
const PERF_MEM_LVL_REM_RAM1: u64 = 1 << 8;
const PERF_MEM_LVL_REM_RAM2: u64 = 1 << 9;
const PERF_MEM_LVL_REM_CCE1: u64 = 1 << 10;
const PERF_MEM_LVL_REM_CCE2: u64 = 1 << 11;
const PERF_MEM_LVL_IO:       u64 = 1 << 12;
const PERF_MEM_LVL_UNC:      u64 = 1 << 13;

// PERF_MEM_LVLNUM_* (modern, 4 bits, 33..37) — value is encoded directly.
const PERF_MEM_LVLNUM_L1:        u64 = 0x01;
const PERF_MEM_LVLNUM_L2:        u64 = 0x02;
const PERF_MEM_LVLNUM_L3:        u64 = 0x03;
const PERF_MEM_LVLNUM_L4:        u64 = 0x04;
const PERF_MEM_LVLNUM_ANY_CACHE: u64 = 0x0b;
const PERF_MEM_LVLNUM_LFB:       u64 = 0x0c;
const PERF_MEM_LVLNUM_RAM:       u64 = 0x0d;
const PERF_MEM_LVLNUM_PMEM:      u64 = 0x0e;
const PERF_MEM_LVLNUM_NA:        u64 = 0x0f;

// PERF_MEM_SNOOP_* (5 bits, 19..24)
const PERF_MEM_SNOOP_NA:   u64 = 1 << 0;
const PERF_MEM_SNOOP_NONE: u64 = 1 << 1;
const PERF_MEM_SNOOP_HIT:  u64 = 1 << 2;
const PERF_MEM_SNOOP_MISS: u64 = 1 << 3;
const PERF_MEM_SNOOP_HITM: u64 = 1 << 4;

// PERF_MEM_LOCK_* (2 bits, 24..26)
const PERF_MEM_LOCK_NA:     u64 = 1 << 0;
const PERF_MEM_LOCK_LOCKED: u64 = 1 << 1;

// PERF_MEM_TLB_* (7 bits, 26..33) — bitmask
const PERF_MEM_TLB_NA:   u64 = 1 << 0;
const PERF_MEM_TLB_HIT:  u64 = 1 << 1;
const PERF_MEM_TLB_MISS: u64 = 1 << 2;
const PERF_MEM_TLB_L1:   u64 = 1 << 3;
const PERF_MEM_TLB_L2:   u64 = 1 << 4;
const PERF_MEM_TLB_WK:   u64 = 1 << 5;
const PERF_MEM_TLB_OS:   u64 = 1 << 6;

// PERF_MEM_REMOTE / PERF_MEM_HOPS — used to disambiguate local vs remote
// DRAM/cache when `mem_lvl_num` doesn't carry the info itself.
const PERF_MEM_HOPS_LOCAL: u64 = 0;

#[inline]
fn bits(val: u64, lo: u32, n: u32) -> u64 {
    (val >> lo) & ((1u64 << n) - 1)
}

/// Decoded `perf_mem_data_src` — same five fields the Python parser pulled
/// out of `|OP|LVL|TLB|LCK|SNP|`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DataSrc {
    pub op: OpType,
    pub cache: Option<CacheLevel>,
    pub tlb: TlbLevel,
    pub snoop: SnoopStatus,
    pub locked: bool,
}

impl Default for OpType {
    fn default() -> Self { OpType::NA }
}
impl Default for TlbLevel {
    fn default() -> Self { TlbLevel::NA }
}
impl Default for SnoopStatus {
    fn default() -> Self { SnoopStatus::NA }
}

impl DataSrc {
    /// Decode the raw u64. Returns a fully-defaulted struct if `val == 0`
    /// (perf records 0 for samples with no memory-access info).
    pub fn from_raw(val: u64) -> Self {
        if val == 0 {
            return DataSrc::default();
        }

        let mem_op       = bits(val, 0, 5);
        let mem_lvl      = bits(val, 5, 14);
        let mem_snoop    = bits(val, 19, 5);
        let mem_lock     = bits(val, 24, 2);
        let mem_dtlb     = bits(val, 26, 7);
        let mem_lvl_num  = bits(val, 33, 4);
        let mem_remote   = bits(val, 37, 1);
        let mem_hops     = bits(val, 43, 3);

        let op = decode_op(mem_op);
        let cache = decode_cache(mem_lvl_num, mem_remote, mem_hops, mem_lvl);
        let tlb = decode_tlb(mem_dtlb);
        let snoop = decode_snoop(mem_snoop);
        let locked = (mem_lock & PERF_MEM_LOCK_LOCKED) != 0;

        DataSrc { op, cache, tlb, snoop, locked }
    }
}

fn decode_op(mem_op: u64) -> OpType {
    if mem_op == 0 || (mem_op & PERF_MEM_OP_NA) != 0 {
        OpType::NA
    } else if (mem_op & PERF_MEM_OP_LOAD) != 0 || (mem_op & PERF_MEM_OP_PFETCH) != 0 {
        OpType::Load
    } else if (mem_op & PERF_MEM_OP_STORE) != 0 {
        OpType::Store
    } else {
        // EXEC and any unrecognised flavour map to N/A — they aren't
        // load-or-store memory ops.
        OpType::NA
    }
}

/// Resolve the cache level. We trust `mem_lvl_num` first (modern), then fall
/// back to the legacy `mem_lvl` bitmask. The remote/hops bits promote a hit
/// at L3 or RAM to REMOTE / DRAM appropriately.
fn decode_cache(mem_lvl_num: u64, mem_remote: u64, mem_hops: u64, mem_lvl: u64) -> Option<CacheLevel> {
    let remote = mem_remote != 0 || mem_hops != PERF_MEM_HOPS_LOCAL;

    if mem_lvl_num != 0 {
        return match mem_lvl_num {
            PERF_MEM_LVLNUM_L1 => Some(CacheLevel::L1),
            PERF_MEM_LVLNUM_L2 => Some(CacheLevel::L2),
            PERF_MEM_LVLNUM_L3 => {
                if remote { Some(CacheLevel::REMOTE) } else { Some(CacheLevel::L3) }
            }
            PERF_MEM_LVLNUM_L4 => Some(CacheLevel::L3),
            PERF_MEM_LVLNUM_LFB => Some(CacheLevel::LFB),
            PERF_MEM_LVLNUM_ANY_CACHE => Some(CacheLevel::L3),
            PERF_MEM_LVLNUM_RAM | PERF_MEM_LVLNUM_PMEM => {
                if remote { Some(CacheLevel::REMOTE) } else { Some(CacheLevel::DRAM) }
            }
            PERF_MEM_LVLNUM_NA => None,
            _ => None,
        };
    }

    // Legacy mem_lvl bitmask — IO / Uncached / NA all fall through to None.
    if mem_lvl == 0 || (mem_lvl & PERF_MEM_LVL_NA) != 0 {
        return None;
    }
    if (mem_lvl & PERF_MEM_LVL_L1) != 0      { return Some(CacheLevel::L1); }
    if (mem_lvl & PERF_MEM_LVL_LFB) != 0     { return Some(CacheLevel::LFB); }
    if (mem_lvl & PERF_MEM_LVL_L2) != 0      { return Some(CacheLevel::L2); }
    if (mem_lvl & PERF_MEM_LVL_L3) != 0 {
        return if remote { Some(CacheLevel::REMOTE) } else { Some(CacheLevel::L3) };
    }
    if (mem_lvl & PERF_MEM_LVL_LOC_RAM) != 0 { return Some(CacheLevel::DRAM); }
    if (mem_lvl & (PERF_MEM_LVL_REM_RAM1 | PERF_MEM_LVL_REM_RAM2)) != 0 {
        return Some(CacheLevel::REMOTE);
    }
    if (mem_lvl & (PERF_MEM_LVL_REM_CCE1 | PERF_MEM_LVL_REM_CCE2)) != 0 {
        return Some(CacheLevel::REMOTE);
    }
    if (mem_lvl & (PERF_MEM_LVL_IO | PERF_MEM_LVL_UNC)) != 0 {
        return None;
    }
    None
}

fn decode_tlb(mem_dtlb: u64) -> TlbLevel {
    if mem_dtlb == 0 || (mem_dtlb & PERF_MEM_TLB_NA) != 0 {
        return TlbLevel::NA;
    }
    if (mem_dtlb & PERF_MEM_TLB_MISS) != 0 {
        return TlbLevel::Miss;
    }
    if (mem_dtlb & PERF_MEM_TLB_HIT) != 0 {
        if (mem_dtlb & PERF_MEM_TLB_L1) != 0 { return TlbLevel::L1Hit; }
        if (mem_dtlb & PERF_MEM_TLB_L2) != 0 { return TlbLevel::L2Hit; }
        return TlbLevel::L1Hit; // generic hit → assume L1 (rare on modern hw)
    }
    TlbLevel::NA
}

fn decode_snoop(mem_snoop: u64) -> SnoopStatus {
    if mem_snoop == 0 || (mem_snoop & PERF_MEM_SNOOP_NA) != 0 {
        return SnoopStatus::NA;
    }
    if (mem_snoop & PERF_MEM_SNOOP_HITM) != 0 { return SnoopStatus::HitM; }
    if (mem_snoop & PERF_MEM_SNOOP_HIT) != 0  { return SnoopStatus::Hit; }
    if (mem_snoop & PERF_MEM_SNOOP_MISS) != 0 { return SnoopStatus::Miss; }
    if (mem_snoop & PERF_MEM_SNOOP_NONE) != 0 { return SnoopStatus::None; }
    SnoopStatus::NA
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(mem_op: u64, mem_lvl: u64, mem_snoop: u64, mem_lock: u64, mem_dtlb: u64,
            mem_lvl_num: u64, mem_remote: u64, mem_hops: u64) -> u64 {
        (mem_op & 0x1f)
            | ((mem_lvl & 0x3fff) << 5)
            | ((mem_snoop & 0x1f) << 19)
            | ((mem_lock & 0x3) << 24)
            | ((mem_dtlb & 0x7f) << 26)
            | ((mem_lvl_num & 0xf) << 33)
            | ((mem_remote & 0x1) << 37)
            | ((mem_hops & 0x7) << 43)
    }

    #[test]
    fn decode_load_l1_hit() {
        let raw = make(
            PERF_MEM_OP_LOAD,
            PERF_MEM_LVL_HIT | PERF_MEM_LVL_L1,
            PERF_MEM_SNOOP_NONE,
            PERF_MEM_LOCK_NA,
            PERF_MEM_TLB_HIT | PERF_MEM_TLB_L1,
            PERF_MEM_LVLNUM_L1,
            0,
            PERF_MEM_HOPS_LOCAL,
        );
        let d = DataSrc::from_raw(raw);
        assert_eq!(d.op, OpType::Load);
        assert_eq!(d.cache, Some(CacheLevel::L1));
        assert_eq!(d.tlb, TlbLevel::L1Hit);
        assert_eq!(d.snoop, SnoopStatus::None);
        assert!(!d.locked);
    }

    #[test]
    fn decode_load_local_dram() {
        let raw = make(
            PERF_MEM_OP_LOAD,
            PERF_MEM_LVL_HIT | PERF_MEM_LVL_LOC_RAM,
            PERF_MEM_SNOOP_NONE,
            PERF_MEM_LOCK_NA,
            PERF_MEM_TLB_HIT | PERF_MEM_TLB_L2,
            PERF_MEM_LVLNUM_RAM,
            0,
            PERF_MEM_HOPS_LOCAL,
        );
        let d = DataSrc::from_raw(raw);
        assert_eq!(d.cache, Some(CacheLevel::DRAM));
        assert_eq!(d.tlb, TlbLevel::L2Hit);
    }

    #[test]
    fn decode_load_remote_dram() {
        let raw = make(
            PERF_MEM_OP_LOAD,
            PERF_MEM_LVL_HIT | PERF_MEM_LVL_REM_RAM1,
            PERF_MEM_SNOOP_NONE,
            PERF_MEM_LOCK_NA,
            0,
            PERF_MEM_LVLNUM_RAM,
            1, // mem_remote
            2, // mem_hops > 0
        );
        let d = DataSrc::from_raw(raw);
        assert_eq!(d.cache, Some(CacheLevel::REMOTE));
    }

    #[test]
    fn decode_store_l2_with_locked() {
        let raw = make(
            PERF_MEM_OP_STORE,
            PERF_MEM_LVL_HIT | PERF_MEM_LVL_L2,
            PERF_MEM_SNOOP_HITM,
            PERF_MEM_LOCK_LOCKED,
            PERF_MEM_TLB_HIT | PERF_MEM_TLB_L1,
            PERF_MEM_LVLNUM_L2,
            0,
            PERF_MEM_HOPS_LOCAL,
        );
        let d = DataSrc::from_raw(raw);
        assert_eq!(d.op, OpType::Store);
        assert_eq!(d.cache, Some(CacheLevel::L2));
        assert_eq!(d.snoop, SnoopStatus::HitM);
        assert!(d.locked);
    }

    #[test]
    fn decode_zero_is_default() {
        let d = DataSrc::from_raw(0);
        assert_eq!(d.op, OpType::NA);
        assert_eq!(d.cache, None);
        assert_eq!(d.tlb, TlbLevel::NA);
        assert_eq!(d.snoop, SnoopStatus::NA);
    }

    #[test]
    fn decode_tlb_miss() {
        let raw = make(
            PERF_MEM_OP_LOAD,
            PERF_MEM_LVL_HIT | PERF_MEM_LVL_L1,
            PERF_MEM_SNOOP_NONE,
            PERF_MEM_LOCK_NA,
            PERF_MEM_TLB_MISS | PERF_MEM_TLB_L2,
            PERF_MEM_LVLNUM_L1,
            0,
            PERF_MEM_HOPS_LOCAL,
        );
        assert_eq!(DataSrc::from_raw(raw).tlb, TlbLevel::Miss);
    }
}
