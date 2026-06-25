//! Drive `linux-perf-data` to read a `perf.data` file end-to-end.
//!
//! Output is a single [`Profile`] containing:
//!   - per-sample timeline records (time, cpu, ip, callchain ips, classified
//!     event class — Cycles / L1 / L2 / L3 / DRAM / Pmc / Other),
//!   - a [`SymbolCache`] + [`AddressSpaces`] you can query with
//!     `(pid, ip) → (sym, offset)`,
//!   - per-instruction [`InsnStats`] keyed by `"<sym>\0<offset>"` (compatible
//!     with the existing source/asm view),
//!   - the perf event attribute table so the caller can resolve sample.id
//!     back to a human-readable event name.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use ahash::AHashMap;
use byteorder::NativeEndian;
use linux_perf_data::{
    linux_perf_event_reader::{EventRecord, RecordType},
    AttributeDescription, PerfFileReader, PerfFileRecord,
};

use crate::data_src::DataSrc;
use crate::ibs_msrs;
use crate::model::{CacheLevel, InsnStats};
use crate::sample::FullSample;
use crate::symbols::{AddressSpaces, Mapping, SymbolCache};

/// Coarse classification of a sample's event — kept in the reader because
/// downstream code (perfy-server) wants different display tracks per class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventClass {
    Cycles,
    /// `ibs_op//` sample. The decoded data_src cache level is in the
    /// per-sample record.
    IbsOp,
    /// Any PMC counter / tracepoint / software event we don't categorise.
    Other,
}

#[derive(Debug, Clone)]
pub struct EventDesc {
    pub name: String,
    pub class: EventClass,
}

/// One sample on the timeline.
#[derive(Debug, Clone)]
pub struct ParsedSample {
    pub time_ns: u64,
    pub cpu: u32,
    pub pid: i32,
    pub tid: i32,
    pub ip: u64,
    pub callchain: Vec<u64>,
    pub event_idx: u32,
    /// Decoded `perf_mem_data_src` (if the event captures it).
    pub cache: Option<CacheLevel>,
    pub period: u64,
}

#[derive(Debug)]
pub struct Profile {
    pub samples: Vec<ParsedSample>,
    pub events: Vec<EventDesc>,
    pub address_spaces: AddressSpaces,
    pub symbol_cache: SymbolCache,
    /// Per-instruction stats keyed `"<sym>\0<offset>"`. Compatible with the
    /// existing `InsnStats` consumer (the source/asm view).
    pub insn_stats: AHashMap<String, InsnStats>,
    /// runtime IP → "<sym>\0<offset>" — same shape as the Python
    /// `ip_to_key` map.
    pub ip_to_key: AHashMap<u64, String>,
    pub time_start_ns: u64,
    pub time_end_ns: u64,
    /// Wall-clock (CLOCK_REALTIME) anchor: at the instant
    /// `clock_id_time_ns` (CLOCK_MONOTONIC ns), the system wall-clock
    /// was `wall_clock_ns` (ns since the Unix epoch). Recorded by perf
    /// when it opens the session (the HEADER_CLOCK_DATA feature). Used
    /// to align perf-stat's seconds-precision wall-clock to the
    /// profile's CLOCK_MONOTONIC sample timestamps. `None` for older
    /// perf versions (pre-5.7) that don't emit the feature.
    pub clock_anchor: Option<ClockAnchor>,
}

#[derive(Debug, Clone, Copy)]
pub struct ClockAnchor {
    pub wall_clock_ns: u64,
    pub clock_id_time_ns: u64,
}

impl Profile {
    /// Best-effort: pick the most-used user-space binary (the one that owns
    /// the largest number of resolved sample IPs). Used by the source/asm
    /// view as the default `--binary`.
    pub fn primary_binary(&self) -> Option<String> {
        let mut counts: AHashMap<String, u64> = AHashMap::new();
        for s in &self.samples {
            if let Some(m) = self.address_spaces.lookup(s.pid, s.ip) {
                if !is_kernel_path(&m.binary) && !is_pseudo_path(&m.binary) {
                    *counts.entry(m.binary.clone()).or_default() += 1;
                }
            }
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(p, _)| p)
    }
}

fn is_kernel_path(p: &str) -> bool {
    p.starts_with("[kernel") || p.contains("vmlinux") || p.starts_with("/proc/kcore")
}
fn is_pseudo_path(p: &str) -> bool {
    p.starts_with('[') || p.starts_with("anon_inode")
}

fn classify_event(name: &str) -> EventClass {
    let n = name.to_ascii_lowercase();
    if n.starts_with("cycles") || n.starts_with("cpu-cycles") || n.starts_with("cpu-clock") {
        EventClass::Cycles
    } else if n.starts_with("ibs_op") {
        EventClass::IbsOp
    } else {
        EventClass::Other
    }
}

fn event_name(attr: &AttributeDescription) -> String {
    attr.name().map(|s| s.to_string()).unwrap_or_else(|| "unknown".to_string())
}

pub fn read_perf_data(path: &Path) -> anyhow::Result<Profile> {
    let t_open = Instant::now();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let PerfFileReader {
        mut perf_file,
        mut record_iter,
    } = PerfFileReader::parse_file(reader)?;
    eprintln!(
        "  [timing] perf.data header parsed in {:.2}s",
        t_open.elapsed().as_secs_f64()
    );

    // Build the event table once; map id → event_idx so each sample finds
    // its event without re-scanning the attrs.
    let attrs: Vec<AttributeDescription> = perf_file.event_attributes().to_vec();
    let mut events: Vec<EventDesc> = Vec::with_capacity(attrs.len());
    let mut id_to_idx: AHashMap<u64, u32> = AHashMap::new();
    for (idx, attr) in attrs.iter().enumerate() {
        let name = event_name(attr);
        let class = classify_event(&name);
        events.push(EventDesc { name, class });
        for &id in attr.ids() {
            id_to_idx.insert(id, idx as u32);
        }
    }

    let mut samples: Vec<ParsedSample> = Vec::new();
    let mut address_spaces = AddressSpaces::default();
    let mut insn_stats: AHashMap<String, InsnStats> = AHashMap::new();
    let mut ip_to_key: AHashMap<u64, String> = AHashMap::new();
    let symbol_cache = SymbolCache::new();

    let mut sample_count: u64 = 0;
    let mut mmap_count: u64 = 0;
    let mut other_count: u64 = 0;

    let t_records = Instant::now();
    while let Some(record) = record_iter.next_record(&mut perf_file)? {
        match record {
            PerfFileRecord::EventRecord { attr_index, record } => {
                if record.record_type == RecordType::SAMPLE {
                    sample_count += 1;
                    let info = record.parse_info;
                    let sample =
                        FullSample::parse::<NativeEndian>(record.data.clone(), &info)?;

                    let (pid, tid) = (
                        sample.pid.unwrap_or(0),
                        sample.tid.unwrap_or(0),
                    );

                    // Resolve event_idx: prefer sample.id, fall back to attr_index.
                    let event_idx = sample
                        .id
                        .and_then(|id| id_to_idx.get(&id).copied())
                        .unwrap_or(attr_index as u32);
                    let class = events
                        .get(event_idx as usize)
                        .map(|e| e.class)
                        .unwrap_or(EventClass::Other);

                    let ds = sample.data_src.map(DataSrc::from_raw).unwrap_or_default();

                    // Per-instruction stats — only if we can resolve the
                    // leaf IP to a symbol. We do this lazily here because
                    // the source/asm view wants the same keying as the
                    // Python tool.
                    if let Some(ip) = sample.ip {
                        if let Some(mapping) = address_spaces.lookup(pid, ip) {
                            let binary_base = address_spaces
                                .binary_base(pid, &mapping.binary)
                                .unwrap_or(mapping.addr_lo - mapping.page_offset);
                            if let Some((sym, off)) =
                                symbol_cache.resolve_with_base(mapping, ip, binary_base)
                            {
                                let key = format!("{sym}\0{off}");
                                ip_to_key.insert(ip, key.clone());
                                let entry = insn_stats.entry(key).or_default();
                                match class {
                                    EventClass::Cycles => {
                                        entry.cycles += 1;
                                    }
                                    EventClass::IbsOp => {
                                        // Treat OP=N/A samples as the
                                        // NonMemory bucket (matching
                                        // ibs_annotate's CacheLevel.NON_MEMORY
                                        // mapping). This keeps timing-only
                                        // fields like tag-to-retire and
                                        // comp-to-retire counted across
                                        // every ibs_op sample at this IP.
                                        let bucket = ds
                                            .cache
                                            .unwrap_or(CacheLevel::NonMemory);
                                        entry.add(
                                            bucket, ds.tlb, ds.op, ds.snoop, ds.locked, 0,
                                        );
                                        // Accumulate the raw IBS registers
                                        // for every sample. Memory-only
                                        // fields (dc_miss_lat,
                                        // tlb_refill_lat, mabs, dc_miss bit,
                                        // …) are guarded by `> 0` or by
                                        // `data3.ld_op | data3.st_op` inside
                                        // accumulate_into, so non-memory
                                        // samples don't pollute them. This
                                        // mirrors ibs_annotate's text
                                        // parser: it accumulates per-IP for
                                        // every sample whose IP was ever
                                        // seen as memory.
                                        if let Some(raw) = sample.raw.as_ref() {
                                            let bytes = raw.as_slice();
                                            if let Some(rec) =
                                                ibs_msrs::decode_ibs_op(&bytes)
                                            {
                                                ibs_msrs::accumulate_into(
                                                    &mut entry.ibs,
                                                    &rec,
                                                );
                                            }
                                        }
                                    }
                                    EventClass::Other => {
                                        // PMC events are deliberately not
                                        // double-counted in this branch.
                                    }
                                }
                            }
                        }
                    }

                    let callchain: Vec<u64> = sample
                        .callchain
                        .as_ref()
                        .map(|c| (0..c.len()).filter_map(|i| c.get(i)).collect())
                        .unwrap_or_default();

                    samples.push(ParsedSample {
                        time_ns: sample.time.unwrap_or(0),
                        cpu: sample.cpu.unwrap_or(0),
                        pid,
                        tid,
                        ip: sample.ip.unwrap_or(0),
                        callchain,
                        event_idx,
                        cache: ds.cache,
                        period: sample.period.unwrap_or(0),
                    });
                } else if record.record_type == RecordType::MMAP2 {
                    mmap_count += 1;
                    let parsed = record.parse()?;
                    if let EventRecord::Mmap2(m) = parsed {
                        // Only register file-backed mappings — anonymous
                        // ones (heap, stack, [vdso] aliases) won't have a
                        // resolvable binary on disk.
                        let path = m.path.as_slice();
                        let path = std::str::from_utf8(&path)
                            .unwrap_or("")
                            .trim_end_matches('\0')
                            .to_string();
                        if !path.is_empty() && !path.starts_with('[') {
                            address_spaces.add_mapping(
                                m.pid,
                                Mapping {
                                    addr_lo: m.address,
                                    addr_hi: m.address + m.length,
                                    page_offset: m.page_offset,
                                    binary: path,
                                },
                            );
                        }
                    }
                } else if record.record_type == RecordType::MMAP {
                    mmap_count += 1;
                    let parsed = record.parse()?;
                    if let EventRecord::Mmap(m) = parsed {
                        let path = m.path.as_slice();
                        let path = std::str::from_utf8(&path)
                            .unwrap_or("")
                            .trim_end_matches('\0')
                            .to_string();
                        if !path.is_empty() && !path.starts_with('[') {
                            address_spaces.add_mapping(
                                m.pid,
                                Mapping {
                                    addr_lo: m.address,
                                    addr_hi: m.address + m.length,
                                    page_offset: m.page_offset,
                                    binary: path,
                                },
                            );
                        }
                    }
                } else {
                    other_count += 1;
                }
            }
            PerfFileRecord::UserRecord(_) => {
                // perf-internal records (build-ids, cpu topology, etc.)
                // We don't need them for the runtime profile.
            }
        }
    }

    eprintln!(
        "  [timing] record stream processed in {:.2}s ({sample_count} samples, {mmap_count} mmaps, {other_count} other)",
        t_records.elapsed().as_secs_f64()
    );
    let t_sort = Instant::now();
    samples.sort_by_key(|s| s.time_ns);
    eprintln!(
        "  [timing] samples sorted in {:.2}s",
        t_sort.elapsed().as_secs_f64()
    );
    let time_start_ns = samples.first().map(|s| s.time_ns).unwrap_or(0);
    let time_end_ns = samples.last().map(|s| s.time_ns).unwrap_or(0);

    // CLOCK_DATA feature (perf >= 5.7): a single (wall, clockid) pair
    // captured the instant perf-record opened the session. Lets us
    // convert any sample timestamp back to wall-clock, and (paired
    // with stat.csv's `# started on …` header) align perf-stat
    // intervals to the profile's CLOCK_MONOTONIC axis.
    let clock_anchor = perf_file
        .clock_data()
        .ok()
        .flatten()
        .map(|cd| ClockAnchor {
            wall_clock_ns: cd.wall_clock_ns,
            clock_id_time_ns: cd.clockid_time_ns,
        });
    if let Some(a) = &clock_anchor {
        eprintln!(
            "  perf.data: clock anchor wall={}.{:09} mono={}.{:09}",
            a.wall_clock_ns / 1_000_000_000,
            a.wall_clock_ns % 1_000_000_000,
            a.clock_id_time_ns / 1_000_000_000,
            a.clock_id_time_ns % 1_000_000_000,
        );
    }

    eprintln!(
        "  perf.data: {sample_count} samples, {mmap_count} mmaps, {other_count} other; \
         resolved {} insn-stat keys, {} events ({:?})",
        insn_stats.len(),
        events.len(),
        events.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
    );

    Ok(Profile {
        samples,
        events,
        address_spaces,
        symbol_cache,
        insn_stats,
        ip_to_key,
        clock_anchor,
        time_start_ns,
        time_end_ns,
    })
}
