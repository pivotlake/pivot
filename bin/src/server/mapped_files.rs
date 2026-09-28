//! Reading the process's own file-backed mappings into memory at startup.

/// Fault in every readable file-backed mapping of this process: the executable's
/// code and read-only data, and those of the shared libraries it links.
///
/// The kernel maps an executable lazily, so a page of code is read from disk the
/// first time any thread runs it. When the page cache has been emptied since the
/// binary last ran, the first query to reach each part of the engine blocks its
/// workers on those reads one page at a time, queued behind the query's own
/// column reads on the same disk. Reading the mappings in before accepting
/// connections moves that cost to startup.
///
/// Failure is logged, not fatal: `MADV_POPULATE_READ` needs Linux 5.14, and a
/// server that cannot populate still runs correctly, only with slower first
/// queries.
#[cfg(target_os = "linux")]
pub fn populate_mapped_files() {
    use std::ptr::NonNull;

    use nix::sys::mman::{MmapAdvise, madvise};
    use tracing::{info, warn};

    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => maps,
        Err(error) => {
            warn!(%error, "cannot read /proc/self/maps; leaving mapped files unpopulated");
            return;
        }
    };
    let mut populated_bytes = 0usize;
    for mapping in maps.lines().filter_map(parse_readable_file_mapping) {
        let Some(start) = NonNull::new(mapping.start as *mut std::ffi::c_void) else {
            continue;
        };
        let len = mapping.end - mapping.start;
        // SAFETY: the range is one whole mapping of this process, and
        // MADV_POPULATE_READ only faults pages in; it never changes contents.
        match unsafe { madvise(start, len, MmapAdvise::MADV_POPULATE_READ) } {
            Ok(()) => populated_bytes += len,
            Err(errno) => {
                warn!(%errno, path = mapping.path, "cannot populate a mapped file");
                return;
            }
        }
    }
    info!(populated_bytes, "populated mapped files");
}

#[cfg(not(target_os = "linux"))]
pub fn populate_mapped_files() {}

/// One line of `/proc/self/maps`, restricted to what population needs.
#[cfg(target_os = "linux")]
struct FileMapping<'a> {
    start: usize,
    end: usize,
    path: &'a str,
}

/// Parse a `/proc/self/maps` line (`start-end perms offset dev inode path`),
/// keeping it only when it is readable and backed by a file on disk. Anonymous
/// mappings and pseudo-files such as `[stack]` or `[vdso]` have no path starting
/// with `/`.
#[cfg(target_os = "linux")]
fn parse_readable_file_mapping(line: &str) -> Option<FileMapping<'_>> {
    let mut fields = line.split_whitespace();
    let address_range = fields.next()?;
    let perms = fields.next()?;
    let _offset = fields.next()?;
    let _device = fields.next()?;
    let _inode = fields.next()?;
    let path = fields.next()?;
    if !perms.starts_with('r') || !path.starts_with('/') {
        return None;
    }
    let (start, end) = address_range.split_once('-')?;
    Some(FileMapping {
        start: usize::from_str_radix(start, 16).ok()?,
        end: usize::from_str_radix(end, 16).ok()?,
        path,
    })
}
