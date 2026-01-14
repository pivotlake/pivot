use crate::operations::unary::group::ArenaKey;
use std::mem::MaybeUninit;
use std::ptr;

const START_BUFFER_SIZE: usize = 4096;

/// A byte arena for efficient bulk string allocation.
///
/// # Memory Layout
///
/// ```text
/// ByteArena
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │  cursor: usize          (write position in active buffer)               │
/// │  active_buffer: Box<[u8]>  ──────────────────────┐                      │
/// │  buffers: Vec<Box<[u8]>>                         │                      │
/// └──────────────────────────────────────────────────│──────────────────────┘
///                                                    ▼
///     buffers (full, immutable)              active_buffer (being filled)
///     ┌─────────────────────┐                ┌─────────────────────────────┐
///     │ Box<[u8]> (4KB)     │──►["foo","bar",...]                          │
///     ├─────────────────────┤                │"baz"|"qux"|     ◄── cursor  │
///     │ Box<[u8]> (8KB)     │──►["...","...",...]                          │
///     ├─────────────────────┤                │                (free space) │
///     │ Box<[u8]> (16KB)    │──►["...","...",...]                          │
///     └─────────────────────┘                └─────────────────────────────┘
/// ```
///
/// # How It Works
///
/// 1. **Push**: Strings are copied contiguously into `active_buffer` at `cursor`.
///    Returns an `ArenaKey` (raw pointer + length) to the copied bytes.
///
/// 2. **Overflow**: When a string doesn't fit, the full buffer moves to `buffers`
///    and a new `active_buffer` is allocated at **2x the previous size**.
///
/// 3. **Growth**: 4KB → 8KB → 16KB → 32KB → ... (exponential, amortized O(1) push)
///
/// # Why Pointers Stay Valid
///
/// Unlike `Vec::push` which may `realloc` and invalidate pointers, this arena
/// **never moves existing data**:
/// - Full buffers are moved to `buffers` Vec, but the `Box<[u8]>` heap allocation
///   doesn't move—only the Box (pointer) itself is moved into the Vec
/// - New allocations go to a fresh buffer, leaving old pointers untouched
///
/// This makes `ArenaKey` (a raw pointer) safe to hold as long as the arena lives.
///
/// # Performance
/// This is much more performant than simply reallocating (and having an index into arena) because
/// reallocating can potentially cause movement of data if the vec can't grow in place. This saves
/// the need to move data - once data is written, it will never be moved, and new buffers will
/// be created for additional data if a buffer passes its capacity
///
/// # Safety
/// It is *extremely* important to be aware that the arena demands the user be responsible for not
/// accessing `ArenaKey`s after the ByteArena is dropped
pub struct ByteArena {
    cursor: usize,
    active_buffer: Box<[MaybeUninit<u8>]>,
    buffers: Vec<Box<[MaybeUninit<u8>]>>,
}

fn create_buffer(size: usize) -> Box<[MaybeUninit<u8>]> {
    let mut v: Vec<MaybeUninit<u8>> = Vec::with_capacity(size);
    unsafe {
        v.set_len(size);
    }
    v.into_boxed_slice()
}

impl ByteArena {
    pub(crate) fn new() -> Self {
        Self {
            cursor: 0,
            active_buffer: create_buffer(START_BUFFER_SIZE),
            buffers: vec![],
        }
    }

    #[cold]
    fn add_buffer(&mut self) {
        let current_capacity = self.active_buffer.len();
        let full_buffer =
            std::mem::replace(&mut self.active_buffer, create_buffer(current_capacity * 2));
        self.buffers.push(full_buffer);
        self.cursor = 0;
    }

    /// Push in a new &str, and return a pointer to it. *Note that it is the responsibility of the
    /// user to not dereference the key if the ByteArena has been dropped*. This is obviously not
    /// ideal and should have had a lifetime, but given issues with Rusts self-referential structs,
    /// we're going with this for now.
    #[inline]
    pub(crate) fn push(&mut self, s: &str) -> ArenaKey {
        if self.cursor + s.len() > self.active_buffer.len() {
            self.add_buffer();
        }

        let end = self.cursor + s.len();

        unsafe {
            ptr::copy_nonoverlapping(
                s.as_ptr(),
                self.active_buffer.as_mut_ptr().add(self.cursor) as *mut u8,
                s.len(),
            );
        }

        let ptr = unsafe { self.active_buffer.as_ptr().add(self.cursor) as *const u8 };

        self.cursor = end;
        ArenaKey::new(ptr, s.len() as u32)
    }
}
