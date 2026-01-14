// use std::sync::atomic::{AtomicUsize, Ordering};
// use crate::io::backend::IOBackend;
// use crate::memory::arena::Arena;
// use crate::memory::buffer_cache::{WriteBufferGuard, WRITING};
//
// struct Slot {
//
//     ref_bit: bool
// }
//
//
// pub struct BufferCache {
//     arena: Arena,
//     pub(crate) slots: Vec<Slot>,
//     hand: AtomicUsize
// }
//
// unsafe impl Send for BufferCache {}
// unsafe impl Sync for BufferCache {}
//
// impl BufferCache {
//
//     pub fn register_buffers_to_uring(&self, backend: &IOBackend) {
//         backend.register_buffers(self.slots.iter().map(|s| {
//             let buffer = unsafe {
//                 s.buffer.get().as_ref()
//             }.unwrap();
//             (buffer.buffer.ptr() as *mut libc::c_void, buffer.buffer.len())
//         }))
//     }
//
//     pub fn take_for_writing(&'static self) -> WriteBufferGuard {
//         loop {
//             let slot_idx = self.hand.fetch_add(1, Ordering::Relaxed) % self.slots.len();
//             let slot = &self.slots[slot_idx];
//             if slot.pin_count.load(Ordering::Relaxed) > 0 {
//                 continue;
//             }
//
//             if slot.ref_bit.swap(false, Ordering::Relaxed) {
//                 continue;
//             }
//
//             // Try to take this buffer - if the previous was 0, we know that we are the ones
//             // evicting, and no one is using this buffer in read.
//             // We set the flag so that no one will attempt to read the buffer while we are writing
//             // to it. Once we return the buffer, we'll reset the previous
//             if slot.pin_count.compare_exchange(
//                 0,
//                 WRITING,
//                 Ordering::AcqRel,
//                 Ordering::Relaxed
//             ).is_ok() {
//                 // Won exclusively
//                 let mut buffer = unsafe { *slot.buffer.get() };
//                 buffer.generation += 1;
//                 return WriteBufferGuard {
//                     buffer,
//                     slot_idx,
//                     converted_to_read_guard: false,
//                     cache: self,
//                 };
//             }
//         }
//     }
// }
