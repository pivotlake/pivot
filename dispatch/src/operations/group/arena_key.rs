use crate::operations::group::arena::ByteArena;
use crate::operations::group::hashtable::{LiveKey, PersistedKey};
use std::hash::{Hash, Hasher};
use std::{ptr, slice};

/// A live string key, referencing both the str now and the ByteArena where it will be persisted
/// (see `impl LiveKey`)
pub struct StringKey<'a, 'b> {
    arena: &'a mut ByteArena,
    value: &'b str,
}

impl<'a, 'b> StringKey<'a, 'b> {
    pub fn new(arena: &'a mut ByteArena, value: &'b str) -> Self {
        Self { arena, value }
    }
}

impl<'a, 'b> PartialEq<Self> for StringKey<'a, 'b> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<'a, 'b> Hash for StringKey<'a, 'b> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<'a, 'b> PartialEq<ArenaKey> for StringKey<'a, 'b> {
    fn eq(&self, other: &ArenaKey) -> bool {
        self.value.as_bytes() == other.as_ref()
    }
}

impl<'a, 'b> LiveKey for StringKey<'a, 'b> {
    type Persisted = ArenaKey;

    #[inline]
    fn persist(self) -> Self::Persisted {
        self.arena.push(self.value)
    }
}

/// A pointer to a key with an arena.
///
/// # Safety
/// See comments on `Arena`- the user needs to be aware that if the underlying `Arena` is
/// deallocated, nothing prevents the accessing of deallocated memory from the `ArenaKey`.
#[derive(Copy, Clone)]
pub struct ArenaKey {
    ptr: *const u8,
    length: u32,
}

unsafe impl Send for ArenaKey {}
unsafe impl Sync for ArenaKey {}

impl ArenaKey {
    pub fn new(ptr: *const u8, length: u32) -> Self {
        Self { ptr, length }
    }
}

impl PartialEq for ArenaKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl AsRef<[u8]> for ArenaKey {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.length as usize) }
    }
}

impl<'a, 'b> PartialEq<StringKey<'a, 'b>> for ArenaKey {
    fn eq(&self, other: &StringKey<'a, 'b>) -> bool {
        self.as_ref() == other.value.as_bytes()
    }
}

impl Default for ArenaKey {
    fn default() -> Self {
        ArenaKey {
            ptr: ptr::null(),
            length: 0,
        }
    }
}

impl PersistedKey for ArenaKey {}
