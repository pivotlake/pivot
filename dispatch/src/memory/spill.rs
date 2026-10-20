//! Spilling: writing a buffer's bytes to disk early, so that its slot can be given back
//! the moment the ring runs out of memory.
//!
//! An operator that may hold a lot of memory carves it out of [`SpillBuffer`]s. While the
//! worker's node is under the spill threshold, nothing happens. Over it, the worker asks the
//! operator's [`Allocator`] to spill one full buffer: the buffer is pushed to a [`Spiller`],
//! which writes its bytes to disk and marks it written. The bytes stay in memory and the
//! operator keeps reading them, so a query that spills but never runs out of memory loses
//! nothing. Only when the ring has no free slot and nothing cached left to evict does it take
//! a written buffer back from the allocator (see [`Allocator::evict`]); from then on those
//! bytes exist on disk only.
//!
//! A written buffer still pins its slot, but can be taken back at any moment, so the ring
//! counts such slots apart from the rest of the pinned memory (see [`Ring::spilled_slots`]).

use crate::memory::{BUFFER_SIZE, Ring, WriteBuffer, memory_ctx};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A ring buffer that chunks of an operator's data are carved from, whose bytes may be
/// written to disk and later taken out of memory.
///
/// Shared by everything holding a chunk of it. The bytes stay in memory until
/// [`evict`](Self::evict) takes them, which it does only once they are written and nobody
/// has them pinned (see [`pin`](Self::pin)).
pub struct SpillBuffer {
    /// Start of the bytes in the ring, fixed for the buffer's lifetime. See
    /// [`ptr`](Self::ptr) for when it may be read through.
    start: *mut u8,
    slot_idx: usize,
    in_memory: Mutex<InMemory>,
    written: AtomicBool,
    /// Where the bytes start in the spiller's file, once written.
    file_offset: AtomicU64,
    ring: Arc<Ring>,
}

/// The bytes while they are in memory.
struct InMemory {
    /// The ring buffer, until it is evicted.
    buffer: Option<WriteBuffer>,
    /// How many readers need the bytes to stay in memory right now.
    pinned_by: usize,
}

// SAFETY: the pointer is into the ring, which lives as long as the process. Who may read
// through it is governed by `ptr` and `pin`, and the rest of the state is behind a lock or
// atomic.
unsafe impl Send for SpillBuffer {}
unsafe impl Sync for SpillBuffer {}

impl SpillBuffer {
    pub fn new(buffer: WriteBuffer) -> Arc<Self> {
        Arc::new(Self {
            start: buffer.ptr,
            slot_idx: buffer.slot_idx,
            in_memory: Mutex::new(InMemory {
                buffer: Some(buffer),
                pinned_by: 0,
            }),
            written: AtomicBool::new(false),
            file_offset: AtomicU64::new(0),
            ring: memory_ctx().shared_ring(),
        })
    }

    /// Start of the bytes. Valid while they are in memory, which is certain until the
    /// buffer is written, since only a written buffer can be evicted. A reader that may run
    /// while the buffer is being evicted [pins](Self::pin) the bytes instead.
    pub fn ptr(&self) -> *mut u8 {
        self.start
    }

    /// The bytes to write to disk. For the spiller, which reads them before it marks the
    /// buffer written.
    pub fn bytes(&self) -> &[u8] {
        assert!(
            !self.is_written(),
            "a written buffer may already be evicted"
        );
        unsafe { std::slice::from_raw_parts(self.start, BUFFER_SIZE) }
    }

    /// Records that every byte is on disk, starting at `file_offset` in the spiller's file.
    /// Called by the spiller, after which it must not touch the bytes again.
    pub fn mark_written(&self, file_offset: u64) {
        self.file_offset.store(file_offset, Ordering::Relaxed);
        self.written.store(true, Ordering::Release);
        self.ring.mark_spilled(self.slot_idx);
    }

    pub fn is_written(&self) -> bool {
        self.written.load(Ordering::Acquire)
    }

    /// Where the bytes start in the spiller's file.
    pub fn file_offset(&self) -> u64 {
        assert!(
            self.is_written(),
            "only a written buffer has a place on disk"
        );
        self.file_offset.load(Ordering::Relaxed)
    }

    /// Keeps the bytes in memory for as long as the returned guard lives, or returns `None`
    /// when they were already evicted.
    pub fn pin(&self) -> Option<Pinned<'_>> {
        let mut in_memory = self.in_memory.lock().unwrap();
        in_memory.buffer.as_ref()?;
        in_memory.pinned_by += 1;
        Some(Pinned { buffer: self })
    }

    /// Takes the bytes out of memory, giving the ring buffer back, when they are written
    /// and nobody has them pinned. Returns `None` otherwise, or when they were already
    /// taken.
    pub fn evict(&self) -> Option<WriteBuffer> {
        if !self.is_written() {
            return None;
        }
        let mut in_memory = self.in_memory.lock().unwrap();
        if in_memory.pinned_by > 0 {
            return None;
        }
        let mut buffer = in_memory.buffer.take()?;
        self.ring.unmark_spilled(self.slot_idx);
        buffer.zeroed = false;
        Some(buffer)
    }
}

impl Drop for SpillBuffer {
    fn drop(&mut self) {
        let still_in_memory = self.in_memory.lock().unwrap().buffer.is_some();
        if still_in_memory && self.is_written() {
            self.ring.unmark_spilled(self.slot_idx);
        }
    }
}

/// Keeps a spill buffer's bytes in memory for as long as it lives.
pub struct Pinned<'a> {
    buffer: &'a SpillBuffer,
}

impl Pinned<'_> {
    /// Start of the bytes.
    pub fn ptr(&self) -> *mut u8 {
        self.buffer.start
    }
}

impl Drop for Pinned<'_> {
    fn drop(&mut self) {
        self.buffer.in_memory.lock().unwrap().pinned_by -= 1;
    }
}

/// Writes spill buffers to disk.
pub trait Spiller {
    /// Writes the buffer's bytes to disk and marks it written, with where the bytes start
    /// in the spiller's file.
    fn push(&mut self, buffer: Arc<SpillBuffer>) -> io::Result<()>;
}

/// Appends every spill buffer to one file per worker, writing it before returning.
///
/// Nothing reads the file back yet, so the file is only truncated when the worker starts
/// and grows with every spill.
pub struct FileSpiller {
    file: File,
    /// Bytes written so far, which is where the next buffer goes.
    len: u64,
}

impl FileSpiller {
    /// Creates (or truncates) `directory/worker-<worker>.spill`.
    pub fn create(directory: &Path, worker: usize) -> io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        let file = File::create(directory.join(format!("worker-{worker}.spill")))?;
        Ok(Self { file, len: 0 })
    }
}

impl Spiller for FileSpiller {
    fn push(&mut self, buffer: Arc<SpillBuffer>) -> io::Result<()> {
        self.file.write_all(buffer.bytes())?;
        let file_offset = self.len;
        self.len += BUFFER_SIZE as u64;
        buffer.mark_written(file_offset);
        Ok(())
    }
}

/// Memory an operator holds in spill buffers, as seen by the worker that spills it and the
/// ring that takes it back.
///
/// Reached through the operator's node rather than the operator itself, because the ring
/// asks for an eviction from inside an allocation, while the operator making that
/// allocation is borrowed. An implementation must therefore keep whatever `evict` touches
/// apart from what the operator borrows while it allocates.
pub trait Allocator {
    /// Writes one buffer this allocator can do without in memory to `spiller`, while keeping
    /// its bytes in memory and in use until they are evicted. Returns whether there was a
    /// buffer to spill.
    fn spill(&self, spiller: &mut dyn Spiller) -> io::Result<bool>;

    /// Takes one written, unpinned spilled buffer out of memory, if there is one.
    fn evict(&self) -> Option<WriteBuffer>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Marks buffers written without writing anything anywhere.
    pub(crate) struct InstantSpiller;

    impl Spiller for InstantSpiller {
        fn push(&mut self, buffer: Arc<SpillBuffer>) -> io::Result<()> {
            buffer.mark_written(0);
            Ok(())
        }
    }

    /// Holds spill buffers the way an operator would: `spill` hands over the next one not
    /// yet spilled, `evict` gives back the first one that can be.
    pub(crate) struct HeldBuffers {
        unspilled: RefCell<Vec<Arc<SpillBuffer>>>,
        spilled: RefCell<Vec<Arc<SpillBuffer>>>,
    }

    impl HeldBuffers {
        /// Takes `count` buffers from the ring.
        pub(crate) fn holding(count: usize) -> Rc<Self> {
            let unspilled = (0..count)
                .map(|_| SpillBuffer::new(memory_ctx().get_write_buffer(false)))
                .collect();
            Rc::new(Self {
                unspilled: RefCell::new(unspilled),
                spilled: RefCell::new(Vec::new()),
            })
        }

        pub(crate) fn spilled_slots(&self) -> Vec<usize> {
            self.spilled
                .borrow()
                .iter()
                .map(|buffer| buffer.slot_idx)
                .collect()
        }
    }

    impl Allocator for HeldBuffers {
        fn spill(&self, spiller: &mut dyn Spiller) -> io::Result<bool> {
            let Some(buffer) = self.unspilled.borrow_mut().pop() else {
                return Ok(false);
            };
            spiller.push(buffer.clone())?;
            self.spilled.borrow_mut().push(buffer);
            Ok(true)
        }

        fn evict(&self) -> Option<WriteBuffer> {
            let mut spilled = self.spilled.borrow_mut();
            for i in 0..spilled.len() {
                if let Some(buffer) = spilled[i].evict() {
                    spilled.swap_remove(i);
                    return Some(buffer);
                }
            }
            None
        }
    }

    #[test]
    fn a_written_buffer_counts_as_spilled_until_evicted() {
        init_test_free_pool(4);
        let spilled = SpillBuffer::new(memory_ctx().get_write_buffer(false));
        let ring = memory_ctx().ring();

        InstantSpiller.push(spilled.clone()).unwrap();
        let spilled_while_held = ring.spilled_slots(0);
        let buffer = spilled.evict().expect("written and unpinned");

        assert_eq!(spilled_while_held, 1);
        assert_eq!(ring.spilled_slots(0), 0);
        assert_eq!(ring.pinned_slots(0), 1);
        drop(buffer);
        assert_eq!(ring.pinned_slots(0), 0);
    }

    #[test]
    fn an_unwritten_buffer_stays_in_memory() {
        init_test_free_pool(4);
        let spilled = SpillBuffer::new(memory_ctx().get_write_buffer(false));

        let evicted = spilled.evict();

        assert!(evicted.is_none());
        assert!(spilled.pin().is_some());
    }

    #[test]
    fn a_pinned_buffer_is_evicted_only_after_the_pin_is_dropped() {
        init_test_free_pool(4);
        let spilled = SpillBuffer::new(memory_ctx().get_write_buffer(false));
        InstantSpiller.push(spilled.clone()).unwrap();
        let pinned = spilled.pin().unwrap();

        let while_pinned = spilled.evict();
        drop(pinned);
        let after = spilled.evict();

        assert!(while_pinned.is_none());
        assert!(after.is_some());
        assert!(spilled.pin().is_none());
    }

    #[test]
    fn dropping_a_written_buffer_settles_the_count() {
        init_test_free_pool(4);
        let spilled = SpillBuffer::new(memory_ctx().get_write_buffer(false));
        InstantSpiller.push(spilled.clone()).unwrap();

        drop(spilled);

        assert_eq!(memory_ctx().ring().spilled_slots(0), 0);
        assert_eq!(memory_ctx().ring().pinned_slots(0), 0);
    }

    #[test]
    fn file_spiller_appends_buffers_and_records_where_each_went() {
        init_test_free_pool(4);
        let directory = tempfile::tempdir().unwrap();
        let mut spiller = FileSpiller::create(directory.path(), 7).unwrap();
        let mut first = memory_ctx().get_write_buffer(false);
        first.as_mut_slice().fill(0xAB);
        let first = SpillBuffer::new(first);
        let second = SpillBuffer::new(memory_ctx().get_write_buffer(false));

        spiller.push(first.clone()).unwrap();
        spiller.push(second.clone()).unwrap();

        let written = std::fs::read(directory.path().join("worker-7.spill")).unwrap();
        assert_eq!(written.len(), 2 * BUFFER_SIZE);
        assert!(written[..BUFFER_SIZE].iter().all(|&byte| byte == 0xAB));
        assert_eq!(first.file_offset(), 0);
        assert_eq!(second.file_offset(), BUFFER_SIZE as u64);
    }
}
