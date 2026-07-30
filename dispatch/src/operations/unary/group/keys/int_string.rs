//! `GROUP BY (integer, string)`: one integer key column followed by one
//! `Utf8View` key column.
//!
//! The dedicated alternative to the generic [`RowKeyExtractor`] for this common
//! two-column shape. Rather than byte-encoding the tuple into one row blob and
//! hashing the blob, the persisted key is the native integer sitting beside the
//! string's [`ArenaKey`]: the integer is compared and emitted directly, and the
//! string persists into / resolves from the shared arena exactly as
//! [`StringKeyExtractor`] does. `eq_persisted` checks the (cheap, `Copy`) integer
//! first and short-circuits before touching the string.
//!
//! Because the key carries an out-of-line string, it radix-scatters exactly like
//! [`StringKeyExtractor`] (`RADIX_ABANDON` is `true`):
//! on overflow the active table is abandoned, draining one deduplicated entry per
//! distinct key, so each string is persisted once instead of re-copied for every
//! occurrence.
//!
//! [`RowKeyExtractor`]: super::RowKeyExtractor
//! [`StringKeyExtractor`]: super::StringKeyExtractor

use crate::arrays::{ArrayBuilder, PrimitiveBuilder, SlabColumn};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{LiveKey, PersistedKey};
use crate::operations::unary::group::keys::string::ArenaKey;
use crate::operations::unary::group::keys::{KeyColumnBuilder, KeyExtractor, StoredKey};
use ahash::RandomState;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

/// The persisted `(integer, string)` key: the native integer beside the string's
/// arena handle. `Copy`, so it lives inline in a hash-table entry; it carries no
/// `Hash`/`Eq` of its own because the table keys on the precomputed hash and
/// confirms a match with [`LiveKey::eq_persisted`].
///
/// The string handle is the [`ArenaKey`]'s `u128` stored as two `u64`s rather than
/// an `ArenaKey` field directly: a `u128` is 16-byte aligned, which would pad
/// `{ int, ArenaKey }` out to 32 bytes (8 int + 8 pad + 16 handle). Splitting it
/// keeps the struct 8-byte aligned and packs it to 24, shrinking every hash-table
/// `Entry`, so the table probe and merge (the dominant cost of this group-by)
/// touch fewer cache lines per entry.
pub struct IntStrKey<N> {
    int: N,
    string: [u64; 2],
}

impl<N: Copy> Copy for IntStrKey<N> {}
impl<N: Copy> Clone for IntStrKey<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<N: Default> Default for IntStrKey<N> {
    fn default() -> Self {
        Self::from_parts(N::default(), ArenaKey::default())
    }
}
impl<N: Copy + Default + Send + Sync + 'static> PersistedKey for IntStrKey<N> {
    const HAS_BLOB: bool = true;

    #[inline(always)]
    fn prefetch_blob(&self, arena: &SharedArena) {
        let key = self.string_key();
        if !key.is_inline() {
            arena.prefetch(key.buffer_index(), key.offset());
        }
    }
}

impl<N> IntStrKey<N> {
    #[inline(always)]
    fn from_parts(int: N, string: ArenaKey) -> Self {
        let raw = string.as_u128();
        Self {
            int,
            string: [raw as u64, (raw >> 64) as u64],
        }
    }

    /// Reassemble the string's [`ArenaKey`] from the split halves.
    #[inline(always)]
    fn string_key(&self) -> ArenaKey {
        ArenaKey::from_raw((self.string[1] as u128) << 64 | self.string[0] as u128)
    }
}

/// A live `(int, string)` key holding the integer by value and the `&str`
/// borrowed from the input batch. Persists the string into the worker arena only
/// when a new entry is created.
pub struct IntStrLiveKey<'a, 'b, N> {
    arena: &'a mut WorkerArena,
    int: N,
    string: &'b str,
}

impl<N: Copy + Default + PartialEq + Send + Sync + 'static> LiveKey for IntStrLiveKey<'_, '_, N> {
    type Persisted = IntStrKey<N>;

    #[inline(always)]
    fn eq_persisted(&self, other: &IntStrKey<N>) -> bool {
        self.int == other.int
            && self.string.as_bytes() == other.string_key().resolve(self.arena.shared())
    }

    #[inline(always)]
    fn persist(self) -> IntStrKey<N> {
        IntStrKey::from_parts(self.int, self.arena.push(self.string))
    }
}

/// A previously-persisted [`IntStrKey`] paired with the shared arena needed to
/// resolve its string. Used during the partition merge to re-insert entries from
/// one table into another without re-hashing the raw bytes.
pub struct IntStrResolvedKey<'a, N> {
    int: N,
    string: ArenaKey,
    arena: &'a SharedArena,
}

impl<N: Copy + Default + PartialEq + Send + Sync + 'static> LiveKey for IntStrResolvedKey<'_, N> {
    type Persisted = IntStrKey<N>;

    #[inline(always)]
    fn eq_persisted(&self, other: &IntStrKey<N>) -> bool {
        self.int == other.int
            && self.string.resolve(self.arena) == other.string_key().resolve(self.arena)
    }

    #[inline(always)]
    fn persist(self) -> IntStrKey<N> {
        IntStrKey::from_parts(self.int, self.string)
    }
}

/// Per-batch reader: the two downcast key columns.
pub struct IntStrReader<'b, T: ArrowPrimitiveType> {
    ints: &'b PrimitiveArray<T>,
    strings: &'b StringViewArray,
}

/// `GROUP BY` over one integer column and one string column, in either order.
/// `STR_FIRST` selects which `key_col` is which and the emitted column order:
/// `false` = `(integer, string)`, `true` = `(string, integer)` (the order a
/// `COUNT(DISTINCT)` lowering produces for its inner key). The persisted key,
/// hash, and equality are order-independent; only the column the integer/string
/// is read from and the output column order change.
pub struct IntStrKeyExtractor<T: ArrowPrimitiveType, const STR_FIRST: bool = false>(PhantomData<T>);

// `PhantomData<T>` is only a type tag; the extractor holds no `T` value.
unsafe impl<T: ArrowPrimitiveType, const STR_FIRST: bool> Send
    for IntStrKeyExtractor<T, STR_FIRST>
{
}

impl<T: ArrowPrimitiveType + Send + 'static, const STR_FIRST: bool> KeyExtractor
    for IntStrKeyExtractor<T, STR_FIRST>
where
    T::Native: Copy + Default + Hash + Eq + Send + Sync,
{
    // Equivalent to `StringKeyExtractor`: the key owns an out-of-line string, so
    // abandon (dedup during the scan) persists each string once rather than
    // re-copying it for every occurrence the way raw scatter would.
    const RADIX_ABANDON: bool = true;
    type Config = ();
    type Persisted = IntStrKey<T::Native>;
    type LiveKey<'a, 'b> = IntStrLiveKey<'a, 'b, T::Native>;
    type Stored = IntStrStored<T::Native>;
    type Reader<'b> = IntStrReader<'b, T>;
    type ColumnBuilder = IntStrKeyColumnBuilder<T, STR_FIRST>;
    type Scratch = ();

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _config: &(),
        _scratch: &'b mut (),
    ) -> Self::Reader<'b> {
        // `STR_FIRST` ⇒ the string is `key_cols[0]` and the integer `key_cols[1]`.
        let (int_col, str_col) = if STR_FIRST {
            (key_cols[1], key_cols[0])
        } else {
            (key_cols[0], key_cols[1])
        };
        IntStrReader {
            ints: batch
                .column(int_col)
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .expect("int key column type mismatch"),
            strings: batch
                .column(str_col)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("string key column type mismatch"),
        }
    }

    #[inline(always)]
    fn prepare_and_hash(reader: &mut Self::Reader<'_>, state: &RandomState, hashes: &mut [u64]) {
        // The hash placed here is the only hash the table uses: it locates the
        // slot on insert and probe, and the merge carries it across (it never
        // re-hashes). Equal `(int, string)` rows hash identically; `eq_persisted`
        // resolves the rare collision. Hashing the integer then the `&str` mirrors
        // the int and string extractors composed.
        for (i, h) in hashes.iter_mut().enumerate() {
            let int = unsafe { reader.ints.value_unchecked(i) };
            let string = unsafe { reader.strings.value_unchecked(i) };
            *h = state.hash_one((int, string));
        }
    }

    #[inline(always)]
    fn live_key<'a, 'r>(
        reader: &'r Self::Reader<'_>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r> {
        IntStrLiveKey {
            arena,
            int: unsafe { reader.ints.value_unchecked(idx) },
            string: unsafe { reader.strings.value_unchecked(idx) },
        }
    }
}

/// Emits the two key columns. The integer is built as its primitive type, the
/// string as a zero-copy `StringViewArray` whose views point into the shared
/// arena's ring buffers. `STR_FIRST` selects the emit order so the leading column
/// matches the GROUP BY order (`k0` is whichever key came first).
pub struct IntStrKeyColumnBuilder<T: ArrowPrimitiveType, const STR_FIRST: bool> {
    ints: PrimitiveBuilder<T>,
    views: SlabColumn<u128>,
}

impl<T: ArrowPrimitiveType, const STR_FIRST: bool> KeyColumnBuilder
    for IntStrKeyColumnBuilder<T, STR_FIRST>
{
    type Key = IntStrKey<T::Native>;
    type Config = ();

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _config: &()) -> Self {
        Self {
            ints: PrimitiveBuilder::<T>::with_capacity(allocator, rows),
            views: SlabColumn::with_capacity(allocator, rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &IntStrKey<T::Native>) {
        self.ints.push(&key.int, 1);
        self.views.push(key.string_key().as_u128());
    }

    fn finish(
        self,
        _arena: &Arc<SharedArena>,
        output_buffers: &Arc<[Buffer]>,
        _allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let len = self.views.len();
        let views = ScalarBuffer::<u128>::new(self.views.into_buffer(), 0, len);
        // Share the one buffer list built for this output phase: cloning the
        // `Arc<[Buffer]>` is a single bump, vs. re-wrapping every arena buffer per
        // batch. Safe: views were built from valid ArenaKeys, and the shared
        // buffers (via their `Arc<SharedArena>`) keep the ring memory alive.
        let strings: ArrayRef = Arc::new(unsafe {
            StringViewArray::new_unchecked(views, output_buffers.clone(), None)
        });
        let ints: ArrayRef = self.ints.into_array(None);
        let int_type = T::DATA_TYPE;
        // Emit in GROUP BY order, naming columns positionally (`k0`, `k1`) like the
        // other multi-key extractors so the leading column is the first group key.
        let ((k0_type, k0), (k1_type, k1)) = if STR_FIRST {
            ((DataType::Utf8View, strings), (int_type, ints))
        } else {
            ((int_type, ints), (DataType::Utf8View, strings))
        };
        let fields = vec![
            Field::new("k0", k0_type, false),
            Field::new("k1", k1_type, false),
        ];
        (fields, vec![k0, k1])
    }
}

/// Keys stored as an integer plus an arena string blob. Parameterised on the
/// integer's width only: the two columns' order in the GROUP BY changes how the
/// result columns are emitted, not how the key is stored, so both orders share
/// this one implementation.
pub struct IntStrStored<N>(std::marker::PhantomData<N>);

impl<N: Copy + Default + PartialEq + Send + Sync + 'static> StoredKey for IntStrStored<N> {
    type Persisted = IntStrKey<N>;
    type PersistedLiveKey<'a> = IntStrResolvedKey<'a, N>;

    #[inline(always)]
    fn resolve_persisted(arena: &SharedArena, persisted: IntStrKey<N>) -> IntStrResolvedKey<'_, N> {
        IntStrResolvedKey {
            int: persisted.int,
            string: persisted.string_key(),
            arena,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The persisted key must stay packed: an `ArenaKey` field (16-byte aligned
    // `u128`) would pad `{ i64, handle }` to 32 bytes; splitting the handle into
    // two `u64`s keeps it 8-byte aligned and 24 bytes, so hash-table entries stay
    // small. Guard against a regression that reintroduces the padding.
    #[test]
    fn persisted_key_is_packed() {
        assert_eq!(std::mem::size_of::<IntStrKey<i64>>(), 24);
        assert_eq!(std::mem::align_of::<IntStrKey<i64>>(), 8);
    }
}
