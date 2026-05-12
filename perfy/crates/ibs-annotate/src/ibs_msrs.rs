//! Decode AMD IBS MSR records out of a `PERF_SAMPLE_RAW` payload.
//!
//! For an `ibs_op` event the kernel emits its in-memory buffer starting at
//! the `caps` field (the `size` field is consumed by the outer
//! `PERF_SAMPLE_RAW` u32 size header that `linux-perf-event-reader` already
//! strips). What we receive in `raw` is therefore:
//!
//! ```text
//!   u32 caps                     ← byte 0..4
//!   u64 regs[N]                  ← byte 4..; regs in MSR order, packed
//! ```
//!
//! The first emipirical evidence: dump a real `ibs_op` payload — at offset 4
//! you'll see a u64 that decodes as the CTL register (`En 1 Val 1` low
//! bits), at offset 12 a u64 in the userspace RIP range (≈ `0x5a…`), and so
//! on. Treating the header as 8 bytes (caps + 4 bytes "padding") shifts
//! every reg by half a u64 and produces nonsense (TagToRetCtr always 0,
//! mabs_max always 0, etc.).
//!
//! The reg order for `ibs_op` (from `arch/x86/events/amd/ibs.c`):
//!
//! ```text
//!   regs[0] = IBS_OP_CTL          (skip)
//!   regs[1] = IBS_OP_RIP          (skip)
//!   regs[2] = IBS_OP_DATA
//!   regs[3] = IBS_OP_DATA2        (skip)
//!   regs[4] = IBS_OP_DATA3
//!   regs[5] = IBS_DC_LIN_ADDR     (optional, dictated by `caps`)
//!   regs[6] = IBS_DC_PHYS_ADDR
//!   regs[7] = IBS_OP_DATA4
//!   regs[8] = IBS_OP_BRN_TARGET
//! ```
//!
//! The AMD PPR (and `arch/x86/include/asm/amd-ibs.h`) gives the bit layouts
//! for `ibs_op_data` and `ibs_op_data3` — we mirror those structs.

use byteorder::{ByteOrder, NativeEndian};

use crate::model::IBSRaw;

const HEADER_BYTES: usize = 4; // u32 caps (no padding before regs)
const MIN_REG_COUNT: usize = 5; // CTL, RIP, DATA, DATA2, DATA3

/// Decoded IBS_OP_DATA (MSRC001_1035).
#[derive(Debug, Clone, Copy, Default)]
pub struct IbsOpData {
    pub comp_to_ret_ctr: u16,
    pub tag_to_ret_ctr: u16,
    pub op_return: bool,
    pub op_brn_taken: bool,
    pub op_brn_misp: bool,
    pub op_brn_ret: bool,
    pub op_rip_invalid: bool,
    pub op_brn_fuse: bool,
    pub op_microcode: bool,
}

impl IbsOpData {
    pub fn from_u64(val: u64) -> Self {
        Self {
            comp_to_ret_ctr: (val & 0xFFFF) as u16,
            tag_to_ret_ctr: ((val >> 16) & 0xFFFF) as u16,
            // bits 32..34 reserved
            op_return:      (val >> 34) & 1 != 0,
            op_brn_taken:   (val >> 35) & 1 != 0,
            op_brn_misp:    (val >> 36) & 1 != 0,
            op_brn_ret:     (val >> 37) & 1 != 0,
            op_rip_invalid: (val >> 38) & 1 != 0,
            op_brn_fuse:    (val >> 39) & 1 != 0,
            op_microcode:   (val >> 40) & 1 != 0,
        }
    }
}

/// Decoded IBS_OP_DATA3 (MSRC001_1037).
#[derive(Debug, Clone, Copy, Default)]
pub struct IbsOpData3 {
    pub ld_op: bool,
    pub st_op: bool,
    pub dc_l1tlb_miss: bool,
    pub dc_l2tlb_miss: bool,
    pub dc_l1tlb_hit_2m: bool,
    pub dc_l1tlb_hit_1g: bool,
    pub dc_l2tlb_hit_2m: bool,
    pub dc_miss: bool,
    pub dc_mis_acc: bool,
    pub dc_wc_mem_acc: bool,
    pub dc_uc_mem_acc: bool,
    pub dc_locked_op: bool,
    pub dc_miss_no_mab_alloc: bool,
    pub dc_lin_addr_valid: bool,
    pub dc_phy_addr_valid: bool,
    pub dc_l2_tlb_hit_1g: bool,
    pub l2_miss: bool,
    pub sw_pf: bool,
    pub op_mem_width: u8,           // 4 bits
    pub op_dc_miss_open_mem_reqs: u8, // 6 bits
    pub dc_miss_lat: u16,           // 16 bits
    pub tlb_refill_lat: u16,        // 16 bits
}

impl IbsOpData3 {
    pub fn from_u64(val: u64) -> Self {
        Self {
            ld_op:                      (val >> 0)  & 1 != 0,
            st_op:                      (val >> 1)  & 1 != 0,
            dc_l1tlb_miss:              (val >> 2)  & 1 != 0,
            dc_l2tlb_miss:              (val >> 3)  & 1 != 0,
            dc_l1tlb_hit_2m:            (val >> 4)  & 1 != 0,
            dc_l1tlb_hit_1g:            (val >> 5)  & 1 != 0,
            dc_l2tlb_hit_2m:            (val >> 6)  & 1 != 0,
            dc_miss:                    (val >> 7)  & 1 != 0,
            dc_mis_acc:                 (val >> 8)  & 1 != 0,
            // bits 9..13 reserved
            dc_wc_mem_acc:              (val >> 13) & 1 != 0,
            dc_uc_mem_acc:              (val >> 14) & 1 != 0,
            dc_locked_op:               (val >> 15) & 1 != 0,
            dc_miss_no_mab_alloc:       (val >> 16) & 1 != 0,
            dc_lin_addr_valid:          (val >> 17) & 1 != 0,
            dc_phy_addr_valid:          (val >> 18) & 1 != 0,
            dc_l2_tlb_hit_1g:           (val >> 19) & 1 != 0,
            l2_miss:                    (val >> 20) & 1 != 0,
            sw_pf:                      (val >> 21) & 1 != 0,
            op_mem_width:               ((val >> 22) & 0xF) as u8,
            op_dc_miss_open_mem_reqs:   ((val >> 26) & 0x3F) as u8,
            dc_miss_lat:                ((val >> 32) & 0xFFFF) as u16,
            tlb_refill_lat:             ((val >> 48) & 0xFFFF) as u16,
        }
    }
}

/// Both halves we care about plus the raw u64s in case callers need them.
#[derive(Debug, Clone, Copy, Default)]
pub struct IbsOpRecord {
    pub data: IbsOpData,
    pub data3: IbsOpData3,
    pub data_raw: u64,
    pub data3_raw: u64,
}

/// Walk a `PERF_SAMPLE_RAW` payload for an `ibs_op` event and decode the two
/// MSR fields we need. Returns `None` if the payload is too short — perf
/// hardware older than Zen2 emits fewer regs, but we always need at least
/// CTL/RIP/DATA/DATA2/DATA3.
pub fn decode_ibs_op(raw: &[u8]) -> Option<IbsOpRecord> {
    if raw.len() < HEADER_BYTES + MIN_REG_COUNT * 8 {
        return None;
    }
    // Skip u32 caps; regs follow packed (no padding).
    let regs = &raw[HEADER_BYTES..];
    let data_raw = NativeEndian::read_u64(&regs[16..24]);     // regs[2]
    let data3_raw = NativeEndian::read_u64(&regs[32..40]);    // regs[4]
    Some(IbsOpRecord {
        data: IbsOpData::from_u64(data_raw),
        data3: IbsOpData3::from_u64(data3_raw),
        data_raw,
        data3_raw,
    })
}

/// Fold an `IbsOpRecord` into the rolling [`IBSRaw`] for an instruction.
///
/// Tag-to-retire / Comp-to-retire and the branch counters apply to every
/// sampled µop, so they accumulate unconditionally (gated only by `> 0`).
/// The memory-related fields (DC miss latency, TLB refill latency, MAB
/// occupancy, memory access width, DC/L2 miss bits, prefetch and TLB-miss
/// bits) are *only* meaningful when the sample is a load or store: on
/// Zen 4 the OP_DATA3 register isn't zeroed between samples, so feeding
/// stale memory bits from a prior load into a non-memory sample's average
/// would skew everything. The `is_mem_op` gate below mirrors what
/// ibs_annotate gets for free from `perf script -D` (which prints those
/// fields as 0 for non-memory ops).
pub fn accumulate_into(stats: &mut IBSRaw, rec: &IbsOpRecord) {
    stats.sample_count += 1;

    let comp_to_ret = rec.data.comp_to_ret_ctr as u64;
    let tag_to_ret = rec.data.tag_to_ret_ctr as u64;
    if comp_to_ret > 0 {
        stats.comp_to_ret_sum += comp_to_ret;
        stats.comp_to_ret_count += 1;
    }
    if tag_to_ret > 0 {
        stats.tag_to_ret_sum += tag_to_ret;
        stats.tag_to_ret_count += 1;
    }

    let is_mem_op = rec.data3.ld_op || rec.data3.st_op;
    if is_mem_op {
        stats.mem_op_count += 1;
        let dc_miss_lat = rec.data3.dc_miss_lat as u64;
        let tlb_refill_lat = rec.data3.tlb_refill_lat as u64;
        let mabs = rec.data3.op_dc_miss_open_mem_reqs as u64;
        let mem_width = rec.data3.op_mem_width as u64;

        if dc_miss_lat > 0 {
            stats.dc_miss_lat_sum += dc_miss_lat;
            stats.dc_miss_lat_count += 1;
        }
        if tlb_refill_lat > 0 {
            stats.tlb_refill_lat_sum += tlb_refill_lat;
            stats.tlb_refill_lat_count += 1;
        }
        if mabs > 0 {
            stats.mabs_sum += mabs;
            stats.mabs_count += 1;
            if mabs > stats.mabs_max {
                stats.mabs_max = mabs;
                stats.mabs_max_count = 1;
            } else if mabs == stats.mabs_max {
                stats.mabs_max_count += 1;
            }
        }
        if mem_width > 0 {
            *stats.mem_width_counts.entry(mem_width).or_default() += 1;
        }
        if rec.data3.sw_pf { stats.sw_pf_count += 1; }
        if rec.data3.dc_mis_acc { stats.misaligned_count += 1; }
        if rec.data3.dc_miss_no_mab_alloc { stats.dc_miss_no_mab_count += 1; }
        if rec.data3.dc_l1tlb_miss { stats.dc_l1_tlb_miss_count += 1; }
        if rec.data3.dc_l2tlb_miss { stats.dc_l2_tlb_miss_count += 1; }
        if rec.data3.dc_miss { stats.dc_miss_count += 1; }
        if rec.data3.l2_miss { stats.l2_miss_count += 1; }
    }

    if rec.data.op_brn_ret {
        stats.brn_ret_count += 1;
        if rec.data.op_brn_misp { stats.brn_misp_count += 1; }
        if rec.data.op_brn_taken { stats.brn_taken_count += 1; }
        if rec.data.op_return { stats.brn_return_count += 1; }
    }
    if rec.data.op_brn_fuse { stats.brn_fuse_count += 1; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_payload(op_data: u64, op_data3: u64) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER_BYTES + MIN_REG_COUNT * 8];
        // skip u32 size + u32 caps (left as zero)
        let regs = &mut buf[HEADER_BYTES..];
        // CTL, RIP — leave zero
        NativeEndian::write_u64(&mut regs[16..24], op_data);
        NativeEndian::write_u64(&mut regs[32..40], op_data3);
        buf
    }

    #[test]
    fn decode_brn_ret_with_misp_and_taken() {
        // op_brn_ret bit 37, op_brn_misp bit 36, op_brn_taken bit 35
        let op_data = (1u64 << 37) | (1u64 << 36) | (1u64 << 35);
        // CompToRetCtr=200 (low 16), TagToRetCtr=100 (next 16)
        let op_data = op_data | 200 | (100u64 << 16);
        let payload = build_payload(op_data, 0);
        let rec = decode_ibs_op(&payload).expect("decode");
        assert_eq!(rec.data.comp_to_ret_ctr, 200);
        assert_eq!(rec.data.tag_to_ret_ctr, 100);
        assert!(rec.data.op_brn_ret);
        assert!(rec.data.op_brn_misp);
        assert!(rec.data.op_brn_taken);
    }

    #[test]
    fn decode_op_data3_load_with_miss() {
        // ld_op=1, dc_miss=1 (bit 7), op_mem_width=8 (bits 22-25),
        // dc_miss_lat=42 (bits 32..48)
        let mut v: u64 = 0;
        v |= 1 << 0;            // ld_op
        v |= 1 << 7;            // dc_miss
        v |= 8u64 << 22;        // op_mem_width
        v |= 42u64 << 32;       // dc_miss_lat
        let payload = build_payload(0, v);
        let rec = decode_ibs_op(&payload).expect("decode");
        assert!(rec.data3.ld_op);
        assert!(!rec.data3.st_op);
        assert!(rec.data3.dc_miss);
        assert_eq!(rec.data3.op_mem_width, 8);
        assert_eq!(rec.data3.dc_miss_lat, 42);
    }

    #[test]
    fn accumulate_matches_text_parser_semantics() {
        let mut payload = build_payload(0, 0);
        // Sample 1: ld_op + dc_miss + dc_miss_lat=10 + mabs=2
        let s1_data3: u64 = (1<<0) | (1<<7) | (2u64<<26) | (10u64<<32);
        NativeEndian::write_u64(&mut payload[HEADER_BYTES + 32..HEADER_BYTES + 40], s1_data3);
        let r1 = decode_ibs_op(&payload).unwrap();

        // Sample 2: same with mabs=2 again (so mabs_max_count == 2)
        let r2 = decode_ibs_op(&payload).unwrap();

        let mut stats = IBSRaw::default();
        accumulate_into(&mut stats, &r1);
        accumulate_into(&mut stats, &r2);

        assert_eq!(stats.sample_count, 2);
        assert_eq!(stats.dc_miss_count, 2);
        assert_eq!(stats.mem_op_count, 2);
        assert_eq!(stats.dc_miss_lat_sum, 20);
        assert_eq!(stats.dc_miss_lat_count, 2);
        assert_eq!(stats.mabs_max, 2);
        assert_eq!(stats.mabs_max_count, 2);
    }

    #[test]
    fn payload_too_short_returns_none() {
        let buf = vec![0u8; 16];
        assert!(decode_ibs_op(&buf).is_none());
    }
}
