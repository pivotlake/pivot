use crate::memory::free_pool::push_free_idx;
use crate::memory::{BUFFER_SIZE, RING};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering;

/// Exclusive, mutable handle to a 2MB slot in the ring buffer.
///
/// Acquired via `Ring::try_write`, which atomically marks the slot as being written.
/// Only one `WriteBuffer` can exist per slot at a time — the ring's CAS on the `used`
/// field enforces this.
///
/// The buffer can be used as `&[u8]` / `&mut [u8]` through `Deref`/`DerefMut`.
///
/// When done writing, the buffer can either be:
/// - Dropped — returns the slot to the free pool as dirty (not zeroed).
/// - Converted into a [`super::ReadBuffer`] via `From` — transitions the slot from exclusive
///   write mode to shared read mode without releasing it.
/// - Consumed via [`zero_out`](Self::zero_out) — zeroes the memory and returns the slot
///   to the free pool as zeroed.
pub struct WriteBuffer {
    /// Raw pointer to the start of this slot's 2MB region within the mmap'd ring.
    pub ptr: *mut u8,
    /// Index of the slot in the ring.
    pub slot_idx: usize,
    /// Whether the memory was zeroed when this buffer was acquired.
    /// Callers can check this to skip redundant zeroing.
    pub zeroed: bool,
}

impl WriteBuffer {
    /// Returns the buffer contents as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, BUFFER_SIZE) }
    }

    /// Returns the buffer contents as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, BUFFER_SIZE) }
    }

    /// Zeroes the entire 2MB buffer and returns the slot to the free pool as zeroed.
    pub fn zero_out(self) {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, BUFFER_SIZE);
        }
        RING.set_slot_used(self.slot_idx, 0, Ordering::Release);
        RING.set_slot_zeroed(self.slot_idx, true);
        push_free_idx(self.slot_idx, true);
        mem::forget(self);
    }
}

unsafe impl Send for WriteBuffer {}

unsafe impl Sync for WriteBuffer {}

impl Deref for WriteBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for WriteBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for WriteBuffer {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl AsMut<[u8]> for WriteBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

/// Marks the slot as unused (dirty) and returns it to the free pool.
impl Drop for WriteBuffer {
    fn drop(&mut self) {
        RING.set_slot_zeroed(self.slot_idx, false);
        RING.set_slot_used(self.slot_idx, 0, Ordering::Release);
        push_free_idx(self.slot_idx, false);
    }
}
