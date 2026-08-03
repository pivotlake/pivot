//! How a join reads, hashes, stores, and re-verifies its equi-join key.
//!
//! A [`JoinKey`] is the join's one point of contact with the key columns: the
//! build side reads and hashes each row's key through it and stores the
//! [`Stored`](JoinKey::Stored) value in the key arena, and the probe side
//! hashes its rows the same way and compares candidates' stored values during
//! the match walk. Everything else in the join (the directory, the prefetch
//! pipeline, the output accumulators) never sees a key column.
//!
//! A stored value need not determine the key exactly: a shape whose stored
//! value is lossy (a hash of the full key tuple) confirms each stored-equal
//! candidate through [`verify`](JoinKey::verify), which can read the actual
//! key columns of both sides. Shapes whose stored equality is already exact
//! use a `()` verifier and a `verify` that constant-folds to `true`, leaving
//! the match loop untouched.

use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, PrimitiveArray, RecordBatch};
use std::hash::Hash;
use std::marker::PhantomData;

/// One key shape of a hash equi-join. Implementations are zero-sized markers;
/// every method is associated, taking the per-batch state they build.
pub trait JoinKey: 'static {
    /// The `Copy` value stored in each build tuple and the key arena, compared
    /// against the probe row's value during the match walk.
    type Stored: Copy + Eq + Send;
    /// Per-batch accessor bound to one side's key columns.
    type Reader<'a>;
    /// Probe-side state for exact verification of a candidate whose stored
    /// value compared equal, bound to the build payload once the table
    /// publishes. `()` when stored equality is already exact.
    type Verifier<'a>;

    /// Bind `batch`'s key columns. Cheap: downcasts only.
    fn make_reader<'a>(batch: &'a RecordBatch, key_columns: &[usize]) -> Self::Reader<'a>;

    /// Read row `idx`'s stored key. The row is in bounds, and its key columns
    /// are only null on an outer build's kept rows, which the caller screens
    /// with [`is_null`](Self::is_null) first.
    fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> Self::Stored;

    /// Hash row `idx`'s key. Both sides of a join hash through the same shape
    /// and state, so equal keys land in the same directory slot.
    fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64;

    /// Whether any key column is null at row `idx`. Only an outer build's
    /// tuple generation asks; the other paths filter null-keyed rows out of
    /// the batch up front.
    fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool;

    /// Bind the build payload's key columns for [`verify`](Self::verify).
    fn make_verifier<'a>(
        build_rows: &'a RecordBatch,
        build_key_columns: &[usize],
    ) -> Self::Verifier<'a>;

    /// Whether probe row `probe_idx` (read through `reader`) and build payload
    /// row `build_row` (read through `verifier`) hold equal keys, given their
    /// stored values already compared equal.
    fn verify(
        reader: &Self::Reader<'_>,
        verifier: &Self::Verifier<'_>,
        probe_idx: usize,
        build_row: u32,
    ) -> bool;
}

/// A join keyed on one primitive column of type `T`: the stored value is the
/// native value itself, so no verification is needed.
///
/// A wrapper rather than an impl on `T` directly because a blanket impl over
/// every `ArrowPrimitiveType` would forbid, by coherence, every other
/// [`JoinKey`] impl.
pub struct SingleColumnKey<T>(PhantomData<fn() -> T>);

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> JoinKey for SingleColumnKey<T> {
    type Stored = T::Native;
    type Reader<'a> = &'a PrimitiveArray<T>;
    type Verifier<'a> = ();

    fn make_reader<'a>(batch: &'a RecordBatch, key_columns: &[usize]) -> Self::Reader<'a> {
        let [key_column] = key_columns else {
            panic!("a single-column join key reads exactly one column");
        };
        batch.column(*key_column).as_primitive::<T>()
    }

    #[inline(always)]
    fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> T::Native {
        unsafe { reader.value_unchecked(idx) }
    }

    #[inline(always)]
    fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(Self::read_stored(reader, idx))
    }

    #[inline(always)]
    fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool {
        reader.is_null(idx)
    }

    fn make_verifier<'a>(_build_rows: &'a RecordBatch, _build_key_columns: &[usize]) {}

    #[inline(always)]
    fn verify(
        _reader: &Self::Reader<'_>,
        _verifier: &(),
        _probe_idx: usize,
        _build_row: u32,
    ) -> bool {
        true
    }
}
