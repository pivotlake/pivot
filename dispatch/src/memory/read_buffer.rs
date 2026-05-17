use crate::memory::ring::Ring;
use crate::memory::{BUFFER_SIZE, WriteBuffer};
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use crate::memory_ctx;

/// Shared, immutable handle to a 2MB slot in the ring buffer. While held, the slot cannot be used
/// for writing (similar to a Read lock in a Rwlock)
///
/// The buffer can be used as `&[u8]` through `Deref`.
///
/// On drop, atomically decrements the slot's `used` count. When the last reader drops
/// (count reaches 0), the slot becomes available for reuse by the free pool.
pub struct ReadBuffer {
    /// Raw pointer to the start of this slot's 2MB region within the mmap'd ring.
    pub ptr: *const u8,
    /// Index of the slot in the ring.
    pub slot_idx: usize,
}

unsafe impl Send for ReadBuffer {}

unsafe impl Sync for ReadBuffer {}

impl ReadBuffer {
    /// Returns the ring slot index for this buffer.
    pub fn ring_idx(&self) -> usize {
        self.slot_idx
    }

    /// Returns the buffer contents as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, BUFFER_SIZE) }
    }
}

impl AsRef<[u8]> for ReadBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Deref for ReadBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Decrements the slot's reader count. When the last `ReadBuffer` for a slot is dropped,
/// the count reaches 0 and the slot becomes reclaimable.
impl Drop for ReadBuffer {
    fn drop(&mut self) {
        memory_ctx().ring().slots[self.slot_idx]
            .used
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// Converts a `WriteBuffer` into a `ReadBuffer`, transitioning the slot from
/// exclusive write mode to shared read mode.
///
/// The `WriteBuffer` is forgotten (its `Drop` is skipped) and the slot's `used`
/// count is set to 1 (one reader). The pointer is reused as-is — no copy occurs.
impl From<WriteBuffer> for ReadBuffer {
    fn from(value: WriteBuffer) -> Self {
        let ptr = value.ptr;
        let slot_idx = value.slot_idx;
        std::mem::forget(value);
        memory_ctx().ring().set_slot_used(slot_idx, 1, Ordering::Release);
        Self {
            ptr,
            slot_idx,
        }
    }
}
