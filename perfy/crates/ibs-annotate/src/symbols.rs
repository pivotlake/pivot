//! Symbol resolution for sample IPs, backed by [`wholesym`].
//!
//! `wholesym` (the symbolicator from the Firefox profiler) handles every
//! finicky bit we don't want to: build-id matching, separate `.debug` files,
//! `.gnu_debuglink` / `.gnu_debugaltlink` chains, system debug-info paths
//! (`/usr/lib/debug/.build-id/…`), DWARF fallback for stripped binaries,
//! and Rust + C++ + Swift demangling.
//!
//! The flow:
//!
//! 1. We track per-pid mmap2 regions in [`AddressSpaces`].
//! 2. On lookup, find the [`Mapping`] containing the runtime IP.
//! 3. Compute its **file offset** inside the binary via
//!    `page_offset + (ip - addr_lo)` and pass that as
//!    `LookupAddress::Relative` to `wholesym` (RVA == file offset for the
//!    Linux ELFs we care about — see comment at [`SymbolCache::resolve`]).
//! 4. wholesym returns the enclosing function's name + relative base; we
//!    return `(name, offset_within_function)`.
//!
//! The first lookup against a given binary path triggers an async load
//! through [`wholesym::SymbolManager`]; we drive that on a small tokio
//! runtime owned by the cache so callers can stay synchronous.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use ahash::AHashMap;
use wholesym::{LookupAddress, SymbolManager, SymbolManagerConfig, SymbolMap};

/// One mapped region within a single process. We keep what we need to map an
/// IP back to the originating ELF file at the right file offset.
#[derive(Debug, Clone)]
pub struct Mapping {
    pub addr_lo: u64,
    pub addr_hi: u64,        // exclusive
    pub page_offset: u64,    // bytes into the binary at which this region starts
    pub binary: String,      // filesystem path
}

/// Per-pid sorted mapping list.
#[derive(Default, Debug, Clone)]
pub struct ProcessMappings {
    mappings: Vec<Mapping>,
}

impl ProcessMappings {
    pub fn add(&mut self, m: Mapping) {
        self.mappings.push(m);
    }

    /// Find the mapping containing `ip`, preferring the most-recent overlay.
    pub fn find(&self, ip: u64) -> Option<&Mapping> {
        self.mappings
            .iter()
            .rev()
            .find(|m| m.addr_lo <= ip && ip < m.addr_hi)
    }

    /// Compute the binary's load base — the runtime address where vaddr 0
    /// (the first byte of the ELF file) would be. For a typical PIE binary
    /// the first LOAD segment has `p_offset == 0` and is mapped at
    /// `addr_lo`; we take the minimum of `addr_lo - page_offset` across all
    /// mmaps for this binary, which is robust against the kernel's
    /// page-aligned segment mappings (the alignment gap shifts later
    /// segments' derived base by a few KB).
    pub fn binary_base(&self, binary: &str) -> Option<u64> {
        self.mappings
            .iter()
            .filter(|m| m.binary == binary)
            .map(|m| m.addr_lo.saturating_sub(m.page_offset))
            .min()
    }
}

/// Address-space tracker: per-pid `ProcessMappings`. Kernel mappings
/// (`pid <= 0`) are stored under bucket 0 and consulted as a fallback for
/// kernel IPs.
#[derive(Default, Debug)]
pub struct AddressSpaces {
    per_pid: AHashMap<i32, ProcessMappings>,
}

impl AddressSpaces {
    pub fn add_mapping(&mut self, pid: i32, mapping: Mapping) {
        let bucket = if pid <= 0 { 0 } else { pid };
        self.per_pid.entry(bucket).or_default().add(mapping);
    }

    pub fn lookup(&self, pid: i32, ip: u64) -> Option<&Mapping> {
        if let Some(maps) = self.per_pid.get(&pid) {
            if let Some(m) = maps.find(ip) {
                return Some(m);
            }
        }
        self.per_pid.get(&0).and_then(|m| m.find(ip))
    }

    /// Resolve a binary's load base for the given pid (falls back to pid 0).
    pub fn binary_base(&self, pid: i32, binary: &str) -> Option<u64> {
        if let Some(maps) = self.per_pid.get(&pid) {
            if let Some(b) = maps.binary_base(binary) {
                return Some(b);
            }
        }
        self.per_pid.get(&0).and_then(|m| m.binary_base(binary))
    }

    /// Iterate every mapping across every pid.
    pub fn all_mappings_iter(&self) -> impl Iterator<Item = &Mapping> {
        self.per_pid.values().flat_map(|p| p.mappings.iter())
    }
}

/// Keyed cache entry: symbol map for a binary, or `None` if the load failed
/// (we cache failures to avoid retrying every IP).
type Entry = Option<Arc<SymbolMap>>;

/// Send-side of a request for the worker thread to load a binary's symbols.
struct LoadRequest {
    path: PathBuf,
    response: std::sync::mpsc::Sender<Option<Arc<SymbolMap>>>,
}

pub struct SymbolCache {
    cache: Mutex<AHashMap<String, Entry>>,
    /// Channel into a dedicated worker thread that owns wholesym's async
    /// runtime. Cross-runtime `block_on` would panic when we're called from
    /// the axum handler (already inside a runtime), and re-creating a
    /// runtime per load would mean dropping it from inside the axum
    /// runtime. A long-lived worker thread sidesteps both problems.
    request_tx: std::sync::mpsc::Sender<LoadRequest>,
    /// Stats — exposed via [`SymbolCache::stats`].
    stats: Mutex<CacheStats>,
}

#[derive(Default, Debug, Clone)]
pub struct CacheStats {
    pub binaries_loaded: u64,
    pub binaries_failed: u64,
    pub lookup_hits: u64,
    pub lookup_misses: u64,
}

impl Default for SymbolCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SymbolCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats.lock().map(|s| s.clone()).unwrap_or_default();
        f.debug_struct("SymbolCache")
            .field("stats", &stats)
            .finish()
    }
}

impl SymbolCache {
    pub fn new() -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<LoadRequest>();
        // Dedicated worker thread + runtime — never dropped from within the
        // axum runtime, never re-created per request.
        std::thread::Builder::new()
            .name("symbols-loader".into())
            .spawn(move || worker_loop(request_rx))
            .expect("spawn symbols-loader thread");
        Self {
            cache: Mutex::new(AHashMap::new()),
            request_tx,
            stats: Mutex::new(CacheStats::default()),
        }
    }

    fn load(&self, path: &str) -> Option<Arc<SymbolMap>> {
        {
            let guard = self.cache.lock().unwrap();
            if let Some(entry) = guard.get(path) {
                return entry.clone();
            }
        }
        let pb = PathBuf::from(path);
        if !pb.exists() {
            eprintln!("  symbols: {path}: not on disk; skipping");
            self.cache.lock().unwrap().insert(path.to_string(), None);
            self.stats.lock().unwrap().binaries_failed += 1;
            return None;
        }

        // Hand off to the worker thread and wait for the result.
        let (tx, rx) = std::sync::mpsc::channel();
        if self
            .request_tx
            .send(LoadRequest { path: pb, response: tx })
            .is_err()
        {
            eprintln!("  symbols: {path}: loader thread is gone");
            return None;
        }
        let result = rx.recv().ok().flatten();
        match &result {
            Some(arc) => {
                eprintln!("  symbols: {path}: {} symbols loaded", arc.symbol_count());
                self.stats.lock().unwrap().binaries_loaded += 1;
            }
            None => {
                eprintln!("  symbols: {path}: load failed");
                self.stats.lock().unwrap().binaries_failed += 1;
            }
        }
        self.cache
            .lock()
            .unwrap()
            .insert(path.to_string(), result.clone());
        result
    }

    /// Resolve `ip` to `(symbol_name, offset_within_function)` using a
    /// caller-supplied binary base (so the same RVA convention is used for
    /// both lookup input and offset arithmetic).
    pub fn resolve_with_base(
        &self,
        mapping: &Mapping,
        ip: u64,
        binary_base: u64,
    ) -> Option<(String, u64)> {
        let map = self.load(&mapping.binary)?;
        if ip < binary_base {
            return None;
        }
        let rva: u32 = u32::try_from(ip - binary_base).ok()?;
        let info = map.lookup_sync(LookupAddress::Relative(rva))?;
        let offset = (rva as u64).saturating_sub(info.symbol.address as u64);
        self.stats.lock().unwrap().lookup_hits += 1;
        Some((info.symbol.name, offset))
    }

    /// Diagnostic helper that mirrors the production [`resolve_with_base`]
    /// path so its trace lines up with what's actually stored.
    pub fn resolve_verbose(
        &self,
        mapping: &Mapping,
        ip: u64,
        binary_base: u64,
    ) -> ResolveTrace {
        let mut trace = ResolveTrace {
            binary: mapping.binary.clone(),
            ip,
            mapping_addr_lo: mapping.addr_lo,
            mapping_addr_hi: mapping.addr_hi,
            mapping_page_offset: mapping.page_offset,
            file_off: ip.saturating_sub(binary_base),
            sym: None,
            sym_address: 0,
            sym_size: None,
            offset_within_sym: 0,
            map_loaded: false,
            map_symbol_count: 0,
        };
        let Some(map) = self.load(&mapping.binary) else { return trace };
        trace.map_loaded = true;
        trace.map_symbol_count = map.symbol_count();
        if ip < binary_base {
            return trace;
        }
        let rva = match u32::try_from(ip - binary_base) {
            Ok(v) => v,
            Err(_) => return trace,
        };
        if let Some(info) = map.lookup_sync(LookupAddress::Relative(rva)) {
            trace.sym = Some(info.symbol.name);
            trace.sym_address = info.symbol.address as u64;
            trace.sym_size = info.symbol.size;
            trace.offset_within_sym = (rva as u64).saturating_sub(info.symbol.address as u64);
        }
        trace
    }

    pub fn note_miss(&self) {
        self.stats.lock().unwrap().lookup_misses += 1;
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }
}

fn worker_loop(rx: std::sync::mpsc::Receiver<LoadRequest>) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("symbols-loader: tokio runtime build failed: {e}");
            return;
        }
    };
    let manager = SymbolManager::with_config(SymbolManagerConfig::default());
    while let Ok(req) = rx.recv() {
        let result = runtime
            .block_on(manager.load_symbol_map_for_binary_at_path(&req.path, None));
        let mapped = match result {
            Ok(m) => Some(Arc::new(m)),
            Err(_) => None,
        };
        let _ = req.response.send(mapped);
    }
}

/// Verbose resolve trace — used by `perfy diagnose` to inspect what's
/// happening during symbol lookups.
#[derive(Debug, Clone)]
pub struct ResolveTrace {
    pub binary: String,
    pub ip: u64,
    pub mapping_addr_lo: u64,
    pub mapping_addr_hi: u64,
    pub mapping_page_offset: u64,
    /// Now reused as the runtime IP minus the binary base (= RVA). Kept the
    /// name `file_off` for compatibility with existing diagnostic output.
    pub file_off: u64,
    pub sym: Option<String>,
    pub sym_address: u64,
    pub sym_size: Option<u32>,
    pub offset_within_sym: u64,
    pub map_loaded: bool,
    pub map_symbol_count: usize,
}

// Used by reader.rs to ensure the SymbolCache stays a consistent type across
// the public API (re-exported from lib.rs).
pub(crate) static _MARKER: OnceLock<()> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_mappings_picks_newest() {
        let mut p = ProcessMappings::default();
        p.add(Mapping {
            addr_lo: 0x1000,
            addr_hi: 0x2000,
            page_offset: 0,
            binary: "/lib/old".into(),
        });
        p.add(Mapping {
            addr_lo: 0x1000,
            addr_hi: 0x2000,
            page_offset: 0,
            binary: "/lib/new".into(),
        });
        assert_eq!(p.find(0x1500).unwrap().binary, "/lib/new");
    }

    #[test]
    fn address_spaces_falls_back_to_kernel_pid() {
        let mut a = AddressSpaces::default();
        a.add_mapping(
            0,
            Mapping {
                addr_lo: 0xffff_0000,
                addr_hi: 0xffff_1000,
                page_offset: 0,
                binary: "/boot/vmlinux".into(),
            },
        );
        a.add_mapping(
            12345,
            Mapping {
                addr_lo: 0x1000,
                addr_hi: 0x2000,
                page_offset: 0,
                binary: "/usr/bin/foo".into(),
            },
        );
        assert!(a.lookup(12345, 0x1500).is_some());
        assert_eq!(
            a.lookup(12345, 0xffff_0500).unwrap().binary,
            "/boot/vmlinux"
        );
    }
}
