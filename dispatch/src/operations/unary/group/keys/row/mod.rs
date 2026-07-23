//! Row-encoded multi-column GROUP BY keys.
//!
//! The general-purpose key extractor: it covers every shape the specialised
//! extractors don't — three or more keys, strings mixed with integers, and
//! integer pairs outside the packed [`IntPairKeyExtractor`](super::IntPairKeyExtractor)
//! set. Each row's key columns are serialised, in GROUP BY order, into one
//! contiguous byte string:
//!
//! - fixed-width integers (and a decimal's raw unscaled `i64`/`i128`) as their
//!   little-endian bytes,
//! - strings as a `u32` length prefix followed by the raw bytes — except a
//!   *trailing* string, whose bytes simply run to the end of the blob (its
//!   length is the blob's remaining length, so the prefix is redundant). This
//!   shaves 4 bytes off every such row and, more importantly, keeps many more
//!   tuples within the 12-byte inline budget of [`ArenaKey`].
//!
//! Because the layout is canonical, key-tuple equality is exactly byte equality
//! and one hash covers the whole tuple — so the hash table needs no per-shape
//! logic. The persisted form is an [`ArenaKey`], identical to a single string
//! key: a tuple of ≤ 12 encoded bytes inlines into the table entry, a longer one
//! lives in the shared arena. Output decodes the blobs back into typed arrow
//! columns, emitting string sub-keys as zero-copy views into the same arena
//! bytes (nothing is copied on the string output path).
//!
//! The encoding is driven by a [`RowKeySchema`] (the key columns' arrow types),
//! which the planner supplies as the extractor's `Config`. It is the one thing
//! neither the per-batch reader nor the output decode can recover from the data
//! alone — the reader needs it to know each column's width, and the decode needs
//! it to rebuild typed columns.
//!
//! ## Module layout
//!
//! - [`schema`] — [`RowKeySchema`], the per-query key types.
//! - [`reader`] — the encode side: [`RowReader`]/[`RowScratch`] turn a batch's
//!   key columns into hashes + a contiguous blob buffer.
//! - [`live_key`] — [`RowKey`], the transient key probed against the table.
//! - [`columns`] — the decode side: [`RowKeyColumns`] rebuilds typed output
//!   columns from the persisted blobs.

/// The fixed-width integer types a row key can hold, listed **once**. Both the
/// encode side ([`reader`]) and the decode side ([`columns`]) build a parallel
/// enum + per-type match arms over exactly this set, so spelling it out in each
/// place is how they would silently drift (one supporting a width the other
/// panics on).
///
/// This is a *callback* macro: `int_key_types!(some_macro)` expands to
/// `some_macro! { (I8, Int8, Int8Type, i8), … }`, handing the list to a macro in
/// the consuming module that stamps out that module's enum and match arms. Each
/// row is `(enum variant, arrow DataType, arrow primitive type, native int)`.
/// The `Utf8View` string case is deliberately *not* here: it is genuinely
/// special on both sides (length-prefixing, zero-copy arena views), so each
/// module spells that one arm out explicitly. So are `Decimal64` and
/// `Decimal128`: their arrow `DataType`s carry a precision/scale the
/// unit-variant pattern can't name, so each module spells out those arms
/// beside the string one.
///
/// Defined before the `mod` declarations below so both child modules see it by
/// bare name (macro_rules textual scope); it is not used anywhere else.
macro_rules! int_key_types {
    ($callback:ident) => {
        $callback! {
            (I8,  Int8,   Int8Type,   i8),
            (I16, Int16,  Int16Type,  i16),
            (I32, Int32,  Int32Type,  i32),
            (I64, Int64,  Int64Type,  i64),
            (U8,  UInt8,  UInt8Type,  u8),
            (U16, UInt16, UInt16Type, u16),
            (U32, UInt32, UInt32Type, u32),
            (U64, UInt64, UInt64Type, u64),
        }
    };
}

mod columns;
mod live_key;
mod reader;
mod schema;

pub use columns::RowKeyColumns;
pub use live_key::RowKey;
pub use reader::{RowReader, RowScratch};
pub use schema::RowKeySchema;

use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::keys::string::ResolvedKey;
use crate::operations::unary::group::keys::{ArenaKey, KeyExtractor};
use ahash::RandomState;
use arrow_array::RecordBatch;

/// `GROUP BY (k0, k1, …)` over any mix of integer and string key columns.
pub struct RowKeyExtractor;

impl KeyExtractor for RowKeyExtractor {
    // Radix-abandon like the string and (int, string) extractors: the persisted
    // key is an out-of-line `ArenaKey`, so on overflow the active table is
    // abandoned (draining one deduplicated entry per distinct key) rather than
    // scattering raw rows. Abandon persists each row blob once instead of once
    // per occurrence, sidestepping the arena blow-up that raw scatter would cause
    // at high cardinality while still radix-partitioning the overflow.
    const RADIX_ABANDON: bool = true;
    type Config = RowKeySchema;
    type Persisted = ArenaKey;
    type LiveKey<'a, 'b> = RowKey<'a, 'b>;
    type PersistedLiveKey<'a> = ResolvedKey<'a>;
    type Reader<'b> = RowReader<'b>;
    type Columns = RowKeyColumns;
    type Scratch = RowScratch;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        config: &RowKeySchema,
        scratch: &'b mut RowScratch,
    ) -> RowReader<'b> {
        RowReader::new(batch, key_cols, config, scratch)
    }

    fn prepare_and_hash(reader: &mut RowReader<'_>, state: &RandomState, hashes: &mut [u64]) {
        reader.encode_and_hash(state, hashes);
    }

    #[inline(always)]
    fn live_key<'a, 'r>(
        reader: &'r Self::Reader<'_>,
        idx: usize,
        arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r> {
        // The row slice borrows the reader's scratch for exactly `'r`, the live
        // key's own lifetime — so no transmute is needed.
        RowKey::new(arena, reader.row(idx))
    }

    fn resolve_persisted(arena: &SharedArena, persisted: ArenaKey) -> ResolvedKey<'_> {
        ResolvedKey::new(persisted, arena)
    }
}
