//! Runtime detection of direct I/O alignment requirements.
//!
//! Direct I/O (O_DIRECT) requires reads to be aligned to a device-specific
//! boundary. On Linux this is detected via `statx(STATX_DIOALIGN)` (kernel 6.1+)
//! or falls back to `fstat.st_blksize`. macOS and other platforms default to 4096.

use std::sync::LazyLock;

/// Lazily detected DIO alignment for the current filesystem.
pub static DIO_ALIGNMENT: LazyLock<usize> = LazyLock::new(detect_dio_alignment);

#[cfg(target_os = "linux")]
fn detect_dio_alignment() -> usize {
    use std::fs::File;

    let test_paths = ["/tmp", "/", "."];

    for path in test_paths {
        if let Ok(file) = File::open(path) {
            if let Some(align) = try_statx_alignment(&file) {
                return align;
            }
            if let Some(align) = try_fstat_alignment(&file) {
                return align;
            }
        }
    }

    4096
}

#[cfg(target_os = "linux")]
fn try_statx_alignment(file: &std::fs::File) -> Option<usize> {
    use std::os::unix::io::AsRawFd;

    // STATX_DIOALIGN was added in Linux 6.1
    const STATX_DIOALIGN: libc::c_uint = 0x00040000;

    unsafe {
        let mut statx_buf: libc::statx = std::mem::zeroed();
        let result = libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            STATX_DIOALIGN,
            &mut statx_buf,
        );

        if result == 0 && statx_buf.stx_dio_mem_align > 0 {
            return Some(statx_buf.stx_dio_mem_align as usize);
        }
    }

    None
}

#[cfg(target_os = "linux")]
fn try_fstat_alignment(file: &std::fs::File) -> Option<usize> {
    use std::os::unix::io::AsRawFd;

    unsafe {
        let mut stat_buf: libc::stat = std::mem::zeroed();
        if libc::fstat(file.as_raw_fd(), &mut stat_buf) == 0 && stat_buf.st_blksize > 0 {
            return Some((stat_buf.st_blksize as usize).max(512));
        }
    }

    None
}

#[cfg(target_os = "macos")]
fn detect_dio_alignment() -> usize {
    // macOS uses F_NOCACHE instead of O_DIRECT — no strict alignment needed,
    // but 4096 matches typical page/block size.
    4096
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_dio_alignment() -> usize {
    4096
}
