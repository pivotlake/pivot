use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::LazyLock;
use crate::env::{get_env_var_with_default, get_total_memory};
use crate::io::backend::IOBackend;
use crate::memory::arena::Arena;

pub static BUFFER_CACHE: LazyLock<BufferCache> = LazyLock::new(|| {
    let slot_size = get_env_var_with_default("SLOT_SIZE", 4 * 1024 * 1024);
    let buffer_memory = get_env_var_with_default("BUFFER_MEMORY", get_total_memory() / 2);
    let slots = buffer_memory / slot_size;
    println!("Creating arena of size {:?} ....", buffer_memory);
    let arena = Arena::new(buffer_memory).expect("Unable to allocate arena");
    BufferCache {
        arena,
        slots: (0..slots).map(|_| Slot {
            ptr: unsafe { arena.ptr().add(slot_size)  },
            ref_bit: Default::default(),
            pin_count: Default::default(),
        }).collect(),
        hand: Default::default(),
    }
});

struct ContiguousBuffer {
    arena: &'static Arena,
    ptr: *mut u8,
    size: usize
}

impl Drop for ContiguousBuffer {
    fn drop(&mut self) {
        todo!()
    }
}

#[derive(Copy, Clone)]
pub struct GenerationalBuffer {
    pub(crate) generation: usize,
    buffer: Arena
}

pub struct ReadBufferGuard {
    pub(crate) buffer: GenerationalBuffer,
    pub(crate) slot_idx: usize,
    pub(crate) cache: &'static BufferCache
}

impl Drop for ReadBufferGuard {
    fn drop(&mut self) {
        let slot = &self.cache.slots[self.slot_idx];
        slot.pin_count.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct WriteBufferGuard {
    pub(crate) buffer: GenerationalBuffer,
    pub(crate) slot_idx: usize,
    converted_to_read_guard: bool,
    cache: &'static BufferCache
}

impl From<WriteBufferGuard> for ReadBufferGuard {
    fn from(mut guard: WriteBufferGuard) -> Self {
        let slot = &guard.cache.slots[guard.slot_idx];
        unsafe { *slot.buffer.get() = guard.buffer; }
        slot.ref_bit.store(true, Ordering::Relaxed);
        slot.pin_count.store(1, Ordering::Release);
        guard.converted_to_read_guard = true;
        ReadBufferGuard {
            buffer: guard.buffer,
            slot_idx: guard.slot_idx,
            cache: guard.cache,
        }
    }
}

impl Drop for WriteBufferGuard {
    fn drop(&mut self) {
        if !self.converted_to_read_guard {
            let slot = &self.cache.slots[self.slot_idx];
            slot.ref_bit.store(false, Ordering::Relaxed);
            unsafe { *slot.buffer.get() = self.buffer; }
            slot.pin_count.store(0, Ordering::Release);
        }
    }
}


pub const WRITING: u32 = 1 << 31;

pub struct Slot {
    ptr: *mut u8,
    pub(crate) ref_bit: AtomicBool,
    pub(crate) pin_count: AtomicU32,
}

pub struct BufferCache {
    arena: Arena,
    pub(crate) slots: Vec<Slot>,
    hand: AtomicUsize
}

unsafe impl Send for BufferCache {}
unsafe impl Sync for BufferCache {}

impl BufferCache {

    pub fn register_buffers_to_uring(&self, backend: &IOBackend) {
        backend.register_buffers(self.slots.iter().map(|s| {
            let buffer = unsafe {
                s.buffer.get().as_ref()
            }.unwrap();
            (buffer.buffer.ptr() as *mut libc::c_void, buffer.buffer.len())
        }))
    }

    pub fn take_for_writing(&'static self) -> WriteBufferGuard {
        loop {
            let slot_idx = self.hand.fetch_add(1, Ordering::Relaxed) % self.slots.len();
            let slot = &self.slots[slot_idx];
            if slot.pin_count.load(Ordering::Relaxed) > 0 {
                continue;
            }

            if slot.ref_bit.swap(false, Ordering::Relaxed) {
                continue;
            }

            // Try to take this buffer - if the previous was 0, we know that we are the ones
            // evicting, and no one is using this buffer in read.
            // We set the flag so that no one will attempt to read the buffer while we are writing
            // to it. Once we return the buffer, we'll reset the previous
            if slot.pin_count.compare_exchange(
                0,
                WRITING,
                Ordering::AcqRel,
                Ordering::Relaxed
            ).is_ok() {
                // Won exclusively
                let mut buffer = unsafe { *slot.buffer.get() };
                buffer.generation += 1;
                return WriteBufferGuard {
                    buffer,
                    slot_idx,
                    converted_to_read_guard: false,
                    cache: self,
                };
            }
        }
    }
}
