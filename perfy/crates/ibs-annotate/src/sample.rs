//! Custom `PERF_RECORD_SAMPLE` decoder that exposes `data_src` and `weight`.
//!
//! `linux-perf-event-reader` 0.10 (the version `linux-perf-data` 0.11 pins)
//! reads these fields from the wire but discards them. We re-implement the
//! parse function here using its public byte cursor (`RawData`) and the
//! `RecordParseInfo` it surfaces, walking fields in the canonical kernel
//! order documented in `<linux/perf_event.h>`. The parse stops as soon as
//! every field we need is in hand, so we don't have to model the long tail
//! (CGROUP, AUX, …).

use byteorder::ByteOrder;
use linux_perf_event_reader::{RawData, RawDataU64, RecordParseInfo, SampleFormat};

#[derive(Debug, Clone)]
pub struct FullSample<'a> {
    pub id: Option<u64>,
    pub ip: Option<u64>,
    pub pid: Option<i32>,
    pub tid: Option<i32>,
    pub time: Option<u64>,
    pub cpu: Option<u32>,
    pub period: Option<u64>,
    pub callchain: Option<RawDataU64<'a>>,
    pub raw: Option<RawData<'a>>,
    pub data_src: Option<u64>,
    pub weight: Option<u64>,
}

impl<'a> FullSample<'a> {
    pub fn parse<T: ByteOrder>(
        data: RawData<'a>,
        info: &RecordParseInfo,
    ) -> std::io::Result<Self> {
        let sf = info.sample_format;
        let mut cur = data;

        // PERF_SAMPLE_IDENTIFIER
        let identifier = if sf.contains(SampleFormat::IDENTIFIER) {
            Some(cur.read_u64::<T>()?)
        } else {
            None
        };

        // PERF_SAMPLE_IP
        let ip = if sf.contains(SampleFormat::IP) {
            Some(cur.read_u64::<T>()?)
        } else {
            None
        };

        // PERF_SAMPLE_TID
        let (pid, tid) = if sf.contains(SampleFormat::TID) {
            (Some(cur.read_i32::<T>()?), Some(cur.read_i32::<T>()?))
        } else {
            (None, None)
        };

        // PERF_SAMPLE_TIME
        let time = if sf.contains(SampleFormat::TIME) {
            Some(cur.read_u64::<T>()?)
        } else {
            None
        };

        // PERF_SAMPLE_ADDR — skip
        if sf.contains(SampleFormat::ADDR) {
            let _addr = cur.read_u64::<T>()?;
        }

        // PERF_SAMPLE_ID
        let id = if sf.contains(SampleFormat::ID) {
            Some(cur.read_u64::<T>()?)
        } else {
            None
        };
        let id = identifier.or(id);

        // PERF_SAMPLE_STREAM_ID — skip
        if sf.contains(SampleFormat::STREAM_ID) {
            let _stream_id = cur.read_u64::<T>()?;
        }

        // PERF_SAMPLE_CPU — { cpu, res }
        let cpu = if sf.contains(SampleFormat::CPU) {
            let cpu = cur.read_u32::<T>()?;
            let _res = cur.read_u32::<T>()?;
            Some(cpu)
        } else {
            None
        };

        // PERF_SAMPLE_PERIOD
        let period = if sf.contains(SampleFormat::PERIOD) {
            Some(cur.read_u64::<T>()?)
        } else {
            None
        };

        // PERF_SAMPLE_READ — variable-size struct, skip via read_format width.
        if sf.contains(SampleFormat::READ) {
            // We don't currently consume read groups; skip the whole thing.
            // For non-group reads the size is fixed (1-3 u64s); for group
            // reads it's `nr` plus per-event values. We delegate to the
            // RawEventRecord parser by NOT supporting profiles that combine
            // PERF_SAMPLE_READ with the fields we need — perf record doesn't
            // mix them in practice. Bail out cleanly.
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "PERF_SAMPLE_READ in record stream is not supported",
            ));
        }

        // PERF_SAMPLE_CALLCHAIN — { nr; ips[nr] }
        let callchain = if sf.contains(SampleFormat::CALLCHAIN) {
            let nr = cur.read_u64::<T>()?;
            let bytes = (nr as usize).checked_mul(8).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "callchain too large")
            })?;
            let raw = cur.split_off_prefix(bytes)?;
            Some(RawDataU64::from_raw_data::<T>(raw))
        } else {
            None
        };

        // PERF_SAMPLE_RAW — { u32 size; bytes[size] }
        let raw = if sf.contains(SampleFormat::RAW) {
            let size = cur.read_u32::<T>()?;
            Some(cur.split_off_prefix(size as usize)?)
        } else {
            None
        };

        // PERF_SAMPLE_BRANCH_STACK — skip
        if sf.contains(SampleFormat::BRANCH_STACK) {
            let bnr = cur.read_u64::<T>()?;
            // each branch_entry is currently 24 bytes (from / to / flags)
            let bytes = (bnr as usize).checked_mul(24).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "branch_stack too large")
            })?;
            let _ = cur.split_off_prefix(bytes)?;
        }

        // PERF_SAMPLE_REGS_USER — { abi; regs[count(mask)] }
        if sf.contains(SampleFormat::REGS_USER) {
            let abi = cur.read_u64::<T>()?;
            if abi != 0 {
                let bytes = (info.user_regs_count as usize) * 8;
                let _ = cur.split_off_prefix(bytes)?;
            }
        }

        // PERF_SAMPLE_STACK_USER — { size; data[size]; dyn_size if size!=0 }
        if sf.contains(SampleFormat::STACK_USER) {
            let size = cur.read_u64::<T>()?;
            if size > 0 {
                let _ = cur.split_off_prefix(size as usize)?;
                let _dyn_size = cur.read_u64::<T>()?;
            }
        }

        // PERF_SAMPLE_WEIGHT or WEIGHT_STRUCT
        let weight = if sf.contains(SampleFormat::WEIGHT) {
            Some(cur.read_u64::<T>()?)
        } else if sf.contains(SampleFormat::WEIGHT_STRUCT) {
            // 1 u64 var, then 2 more u64 var values.
            let w = cur.read_u64::<T>()?;
            let _ = cur.read_u64::<T>()?;
            let _ = cur.read_u64::<T>()?;
            Some(w)
        } else {
            None
        };

        // PERF_SAMPLE_DATA_SRC — the prize.
        let data_src = if sf.contains(SampleFormat::DATA_SRC) {
            Some(cur.read_u64::<T>()?)
        } else {
            None
        };

        // We don't need anything past DATA_SRC for our pipeline, so stop here.

        Ok(FullSample {
            id,
            ip,
            pid,
            tid,
            time,
            cpu,
            period,
            callchain,
            raw,
            data_src,
            weight,
        })
    }
}
