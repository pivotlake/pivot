use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};
use std::slice;
use std::sync::LazyLock;

/// Cached direct I/O alignment requirement for the system
pub static DIO_ALIGNMENT: LazyLock<usize> = LazyLock::new(detect_dio_alignment);

#[cfg(target_os = "linux")]
fn detect_dio_alignment() -> usize {
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    // Try to detect from a common mount point
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

    // Fallback: 4096 covers most modern systems
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
            b"\0".as_ptr() as *const libc::c_char,
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
    // macOS uses F_NOCACHE instead of O_DIRECT.
    // There's no strict alignment requirement, but 4096 is safe
    // and matches typical page size / SSD block size.
    4096
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_dio_alignment() -> usize {
    // Conservative default for unknown platforms
    4096
}

/// A buffer aligned for O_DIRECT I/O operations.
pub struct DiskBuffer {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
}

impl DiskBuffer {
    /// Creates a new buffer aligned for O_DIRECT I/O.
    ///
    /// Size is rounded up to the system's required alignment.
    pub fn new(size: usize) -> std::io::Result<Self> {
        let alignment = *DIO_ALIGNMENT;
        let aligned_size = size.checked_add(alignment - 1).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "size overflow")
        })? & !(alignment - 1);

        // Ensure we allocate at least one block
        let aligned_size = aligned_size.max(alignment);

        let layout = Layout::from_size_align(aligned_size, alignment)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "failed to allocate aligned memory",
            ));
        }

        unsafe {
            std::ptr::write_bytes(ptr, 0, aligned_size);
        }

        Ok(Self {
            ptr,
            size: aligned_size,
            layout,
        })
    }

    /// Returns the system's detected O_DIRECT alignment
    pub fn alignment() -> usize {
        *DIO_ALIGNMENT
    }

    /// Returns the buffer size (rounded up to alignment)
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Deref for DiskBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }
}

impl DerefMut for DiskBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for DiskBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

unsafe impl Send for DiskBuffer {}
unsafe impl Sync for DiskBuffer {}

impl AsRef<[u8]> for DiskBuffer {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl AsMut<[u8]> for DiskBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}
