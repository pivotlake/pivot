use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::PersistedKey;
use arrow_array::builder::make_view;

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

    /// Reconstruct a key from the raw `u128` bit pattern returned by
    /// [`as_u128`](Self::as_u128) (e.g. when read back out of a `SlabColumn`).
    #[inline]
    pub fn from_raw(raw: u128) -> Self {
        Self(raw)
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

    /// Arena buffer index of a non-inline key (meaningless for inline keys).
    #[inline]
    pub fn buffer_index(&self) -> u32 {
        (self.0 >> 64) as u32
    }

    /// Byte offset within the arena buffer of a non-inline key.
    #[inline]
    pub fn offset(&self) -> u32 {
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

impl PersistedKey for ArenaKey {
    const HAS_BLOB: bool = true;

    #[inline(always)]
    fn prefetch_blob(&self, arena: &SharedArena) {
        if !self.is_inline() {
            arena.prefetch(self.buffer_index(), self.offset());
        }
    }
}
