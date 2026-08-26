//! Periodic memory visibility: process RSS, jemalloc arena statistics, and
//! on-demand jemalloc heap profile dumps.
//!
//! The buffer-pool ring is one anonymous `mmap` sized at startup, so its
//! resident footprint is constant. Everything that can grow without bound lives
//! on the jemalloc heap beside it. Logging both, and dumping heap profiles that
//! `jeprof` can diff, attributes growth to the call stack that allocated it.
//!
//! Nothing here runs unless `PIVOT_MEMWATCH_DIR` is set, and heap dumps
//! additionally require the binary to be built with jemalloc profiling and
//! started with `prof:true` in `_RJEM_MALLOC_CONF`.

use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use tracing::{info, warn};

/// Reads a `/proc/self/status` field in kB.
fn read_status_kb(field: &str) -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            return rest
                .trim_start_matches(':')
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

/// One mapping's resident footprint, with the `/proc/self/maps` description
/// line that identifies it.
struct Mapping {
    rss: u64,
    description: String,
}

/// Parses `/proc/self/smaps` into per-mapping resident sizes, largest first.
///
/// jemalloc is linked with prefixed symbols, so C and C++ allocations (the
/// DuckDB planner, the compression and TLS libraries) go to glibc `malloc` and
/// never appear in a jemalloc heap profile. Mapping-level totals are what shows
/// those: glibc arenas are distinctive 64 MB heaps, and a large `malloc` is its
/// own anonymous mapping.
fn read_mappings() -> Vec<Mapping> {
    let Ok(smaps) = fs::read_to_string("/proc/self/smaps") else {
        return Vec::new();
    };
    let mut mappings = Vec::new();
    let mut description = String::new();
    for line in smaps.lines() {
        if let Some(rest) = line.strip_prefix("Rss:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
            mappings.push(Mapping {
                rss: kb * 1024,
                description: std::mem::take(&mut description),
            });
        } else if !line.starts_with(|c: char| c.is_ascii_uppercase())
            && line.contains('-')
            && line.contains(' ')
        {
            // A header line: "start-end perms offset dev inode  path".
            description = line.to_string();
        }
    }
    mappings.sort_by(|a, b| b.rss.cmp(&a.rss));
    mappings
}

/// Reads a `/proc/meminfo` field in kB.
fn read_meminfo_kb(field: &str) -> u64 {
    let Ok(meminfo) = fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            return rest
                .trim_start_matches(':')
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

fn mallctl_read_usize(name: &[u8]) -> u64 {
    match unsafe { tikv_jemalloc_ctl::raw::read::<usize>(name) } {
        Ok(value) => value as u64,
        Err(_) => 0,
    }
}

fn profiling_active() -> bool {
    matches!(
        unsafe { tikv_jemalloc_ctl::raw::read::<bool>(b"opt.prof\0") },
        Ok(true)
    )
}

fn dump_heap_profile(path: &PathBuf) {
    let Ok(c_path) = CString::new(path.to_string_lossy().as_bytes()) else {
        return;
    };
    // `prof.dump` takes the destination filename as a NUL-terminated string.
    match unsafe { tikv_jemalloc_ctl::raw::write(b"prof.dump\0", c_path.as_ptr()) } {
        Ok(()) => info!(path = %path.display(), "wrote jemalloc heap profile"),
        Err(error) => warn!(%error, "jemalloc heap profile dump failed"),
    }
}

/// Spawns the memory-watch thread when `PIVOT_MEMWATCH_DIR` is set.
pub fn spawn() {
    let Ok(dir) = std::env::var("PIVOT_MEMWATCH_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    if let Err(error) = fs::create_dir_all(&dir) {
        warn!(%error, dir = %dir.display(), "memwatch directory unusable");
        return;
    }
    let interval: u64 = std::env::var("PIVOT_MEMWATCH_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    // Emit a full line only every Nth sample, so a short interval can track a
    // spike without burying the server's own log.
    let log_every: u64 = std::env::var("PIVOT_MEMWATCH_LOG_EVERY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12);
    // A heap profile is dumped whenever live bytes set a new high-water mark
    // this many MiB above the last dump. The peak is what runs the machine out
    // of memory, so the newest high-water dump is the one worth reading.
    let dump_step_mb: u64 = std::env::var("PIVOT_MEMWATCH_DUMP_STEP_MB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);

    // A profile is also dumped whenever the machine's available memory falls
    // under this many MiB, so the spike that gets the process killed is
    // captured even when it never clears the next high-water step.
    let low_available_mb: u64 = std::env::var("PIVOT_MEMWATCH_LOW_AVAILABLE_MB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1024);

    let prof = profiling_active();
    info!(
        dir = %dir.display(),
        interval_secs = interval,
        heap_profiling = prof,
        "memwatch enabled"
    );

    std::thread::Builder::new()
        .name("memwatch".to_string())
        .spawn(move || {
            let mut tick: u64 = 0;
            let mut high_water_mb: u64 = 0;
            let mut next_low_dump_tick: u64 = 0;
            loop {
                std::thread::sleep(Duration::from_secs(interval));
                tick += 1;

                // jemalloc's counters are cached; advancing the epoch refreshes them.
                let _ = tikv_jemalloc_ctl::epoch::advance();
                let allocated = mallctl_read_usize(b"stats.allocated\0");
                let active = mallctl_read_usize(b"stats.active\0");
                let metadata = mallctl_read_usize(b"stats.metadata\0");
                let resident = mallctl_read_usize(b"stats.resident\0");
                let mapped = mallctl_read_usize(b"stats.mapped\0");
                let retained = mallctl_read_usize(b"stats.retained\0");

                let allocated_mb = allocated / 1_048_576;
                let new_peak = allocated_mb >= high_water_mb + dump_step_mb;

                let rss = read_status_kb("VmRSS") * 1024;
                let vm_size = read_status_kb("VmSize") * 1024;
                let mappings = read_mappings();
                let smaps_rss: u64 = mappings.iter().map(|mapping| mapping.rss).sum();
                // The ring is the one giant mapping; the remainder is everything else.
                let largest_mapping = mappings.first().map_or(0, |mapping| mapping.rss);
                let rss_outside_ring = smaps_rss.saturating_sub(largest_mapping);
                // What neither the ring nor the jemalloc heap accounts for: glibc
                // malloc, thread stacks, io_uring rings, mapped files.
                let unaccounted = rss_outside_ring.saturating_sub(resident);

                if tick % log_every == 0 || new_peak {
                    info!(
                        tick,
                        peak = new_peak,
                        rss_mb = rss / 1_048_576,
                        vm_size_mb = vm_size / 1_048_576,
                        ring_mapping_mb = largest_mapping / 1_048_576,
                        rss_outside_ring_mb = rss_outside_ring / 1_048_576,
                        je_allocated_mb = allocated_mb,
                        je_active_mb = active / 1_048_576,
                        je_metadata_mb = metadata / 1_048_576,
                        je_resident_mb = resident / 1_048_576,
                        je_mapped_mb = mapped / 1_048_576,
                        je_retained_mb = retained / 1_048_576,
                        mapping_count = mappings.len(),
                        unaccounted_mb = unaccounted / 1_048_576,
                        "memwatch"
                    );

                    // The biggest mappings after the ring, so growth that is not
                    // on the jemalloc heap still has somewhere to show up.
                    for mapping in mappings.iter().skip(1).take(8) {
                        if mapping.rss < 256 * 1_048_576 {
                            break;
                        }
                        info!(
                            tick,
                            rss_mb = mapping.rss / 1_048_576,
                            mapping = %mapping.description,
                            "memwatch mapping"
                        );
                    }
                }

                if prof && new_peak {
                    high_water_mb = allocated_mb;
                    let path = dir.join(format!("peak.{allocated_mb:06}mb.tick{tick:06}.heap"));
                    dump_heap_profile(&path);
                }
                // An operator can ask for a dump at any moment by creating this
                // file; it is consumed so the next request is a fresh one.
                let request = dir.join("dump-now");
                if prof && request.exists() {
                    let _ = fs::remove_file(&request);
                    let path = dir.join(format!("ondemand.{allocated_mb:06}mb.tick{tick:06}.heap"));
                    dump_heap_profile(&path);
                }
                let available_mb = read_meminfo_kb("MemAvailable") / 1024;
                if prof && available_mb < low_available_mb && tick >= next_low_dump_tick {
                    // At most one low-memory dump per ten ticks, so a machine
                    // that stays low does not spend its last memory on dumps.
                    next_low_dump_tick = tick + 10;
                    info!(
                        tick,
                        available_mb, allocated_mb, "memwatch low available memory"
                    );
                    let path = dir.join(format!("low.{available_mb:06}mb.tick{tick:06}.heap"));
                    dump_heap_profile(&path);
                }
            }
        })
        .expect("spawn memwatch thread");
}
