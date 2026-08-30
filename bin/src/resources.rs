//! Machine resource budgets shared by the server and the shell: memory
//! readings, the buffer pool pre-flight check, and the disk cache.

use std::path::PathBuf;
use std::sync::Arc;

use dispatch::io::DiskCache;
use metastore_disk::ByteSize;
use tracing::info;

/// The disk cache's byte budget when none is given.
pub(crate) const DEFAULT_DISK_CACHE_SIZE: ByteSize = ByteSize::from_bytes(64 * 1024 * 1024 * 1024);

/// The disk cache's object-count limit when none is given.
pub(crate) const DEFAULT_DISK_CACHE_MAX_OBJECTS: usize = 65536;

/// Bytes in a mebibyte, the unit memory budgets are reported in.
const MIB: usize = 1024 * 1024;

/// The buffer pool budget does not fit in the machine's available memory.
#[derive(Debug, thiserror::Error)]
#[error(
    "the buffer pool needs {} MiB but the machine only has {} MiB available right now; every \
     pool slot is faulted in at startup, so starting would be killed by the OOM killer part \
     way through. Free memory on the machine, or lower the memory budget (the server's \
     `server.memory` config key or PIVOT_MEMORY_PCT environment variable, the shell's \
     --memory flag)",
    .requested_bytes / MIB,
    .available_bytes / MIB,
)]
pub struct InsufficientMemory {
    pub requested_bytes: usize,
    pub available_bytes: usize,
}

/// The configured disk cache could not be opened.
#[derive(Debug, thiserror::Error)]
#[error("failed to open the disk cache at {dir}: {source}")]
pub struct DiskCacheUnavailable {
    pub dir: String,
    #[source]
    pub source: std::io::Error,
}

/// Return the machine's total physical memory in bytes.
pub(crate) fn total_memory_bytes() -> usize {
    let bytes = read_memory().total_memory();
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

/// Return how many bytes a new allocation on this machine can actually get
/// hold of: free pages plus what the kernel can reclaim without swapping.
pub(crate) fn available_memory_bytes() -> usize {
    let bytes = read_memory().available_memory();
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

/// One reading of the machine's memory, as the OS reports it now.
fn read_memory() -> sysinfo::System {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
}

/// Refuse a buffer pool budget the machine cannot back right now. Every pool
/// slot is faulted in while the workers start, so a budget beyond what is free
/// is an OOM kill part way through boot, not a slower instance. Returns the
/// available bytes the budget was checked against.
pub(crate) fn check_pool_fits(requested_bytes: usize) -> Result<usize, InsufficientMemory> {
    let available_bytes = available_memory_bytes();
    if requested_bytes > available_bytes {
        return Err(InsufficientMemory {
            requested_bytes,
            available_bytes,
        });
    }
    Ok(available_bytes)
}

/// Open the disk cache, raising the process's open-file limit first: the cache
/// holds one open descriptor per cached object.
pub(crate) fn open_disk_cache(
    dir: PathBuf,
    size_bytes: u64,
    max_objects: usize,
) -> Result<Arc<DiskCache>, DiskCacheUnavailable> {
    crate::server::raise_open_file_limit();
    let cache = DiskCache::open(dir.clone(), size_bytes, max_objects).map_err(|source| {
        DiskCacheUnavailable {
            dir: dir.display().to_string(),
            source,
        }
    })?;
    info!(
        dir = %dir.display(),
        bytes = size_bytes,
        max_objects,
        "disk cache enabled"
    );
    Ok(Arc::new(cache))
}
