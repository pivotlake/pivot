use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{LiveKey, PersistedKey};
use arrow_array::builder::make_view;
use std::hash::{Hash, Hasher};

#[cfg(not(target_endian = "little"))]
compile_error!("ArenaKey layout assumes little-endian");

/// A key stored as a u128 with the same layout as Arrow's StringView.
///
/// - Strings ≤ 12 bytes are **inlined**: `[len: u32, data: [u8; 12]]`
/// - Strings > 12 bytes are **views**: `[len: u32, prefix: [u8; 4], buffer_index: u32, offset: u32]`
///
/// To get the full bytes of a non-inline key, pass the [`SharedArena`] that owns the buffers.
#[derive(Copy, Clone)]
#[repr(transparent)]
#[derive(Default)]
pub struct ArenaKey(u128);

impl ArenaKey {
    const INLINE_MAX: usize = 12;

    /// Create an inline key from data that is ≤ 12 bytes.
    pub fn inline(data: &[u8]) -> Self {
        debug_assert!(data.len() <= Self::INLINE_MAX);
        Self(make_view(data, 0, 0))
    }

    /// Create a view key for data > 12 bytes, referencing an arena buffer.
    pub fn view(data: &[u8], buffer_index: u32, offset: u32) -> Self {
        debug_assert!(data.len() > Self::INLINE_MAX);
        Self(make_view(data, buffer_index, offset))
    }

    #[inline]
    /// Returns `true` if the key data is stored inline (≤ 12 bytes).
    pub fn is_inline(&self) -> bool {
        self.len() as usize <= Self::INLINE_MAX
    }

    #[inline]
    /// Length of the key data in bytes.
    pub fn len(&self) -> u32 {
        self.0 as u32
    }

    /// Pointer to the inline bytes inside this ArenaKey's own u128 storage (LE layout).
    #[inline]
    fn inline_data(&self) -> &[u8] {
        let ptr = self as *const Self as *const u8;
        unsafe { std::slice::from_raw_parts(ptr.add(4), self.len() as usize) }
    }

    #[inline]
    fn buffer_index(&self) -> u32 {
        (self.0 >> 64) as u32
    }

    #[inline]
    fn offset(&self) -> u32 {
        (self.0 >> 96) as u32
    }

    /// Resolve to the underlying bytes.
    ///
    /// Inline keys (≤ 12 bytes) borrow from `self`. Non-inline keys borrow from
    /// the arena's buffers. Both `self` and `arena` must outlive `'a`.
    #[inline]
    pub fn resolve<'a>(&'a self, arena: &'a SharedArena) -> &'a [u8] {
        if self.is_inline() {
            self.inline_data()
        } else {
            arena.resolve(self.buffer_index(), self.offset(), self.len())
        }
    }

    /// The raw u128 view — same layout as a StringViewArray view entry.
    #[inline]
    pub fn as_u128(&self) -> u128 {
        self.0
    }
}

impl PersistedKey for ArenaKey {}

/// A live string key holding the raw `&str` and a mutable reference to the
/// worker arena. `eq_persisted` uses the shared arena for comparison;
/// `persist` pushes the string into the arena only when a new entry is needed.
pub struct StringKey<'a, 'b> {
    arena: &'a mut WorkerArena,
    value: &'b str,
}

impl<'a, 'b> StringKey<'a, 'b> {
    pub fn new(arena: &'a mut WorkerArena, value: &'b str) -> Self {
        Self { arena, value }
    }
}

impl Hash for StringKey<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl LiveKey for StringKey<'_, '_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.value.as_bytes() == other.resolve(self.arena.shared())
    }

    #[inline(always)]
    fn persist(self) -> ArenaKey {
        self.arena.push(self.value)
    }
}

/// A previously-persisted [`ArenaKey`] paired with the arena needed to
/// resolve it. Used during the merge phase to re-insert entries from
/// source tables into the target table without re-hashing the raw bytes.
pub struct ResolvedKey<'a> {
    pub key: ArenaKey,
    pub arena: &'a SharedArena,
}

impl Hash for ResolvedKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.resolve(self.arena).hash(state);
    }
}

impl LiveKey for ResolvedKey<'_> {
    type Persisted = ArenaKey;

    #[inline(always)]
    fn eq_persisted(&self, other: &ArenaKey) -> bool {
        self.key.resolve(self.arena) == other.resolve(self.arena)
    }

    fn persist(self) -> ArenaKey {
        self.key
    }
}
