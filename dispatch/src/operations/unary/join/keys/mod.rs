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
//!
//! One implementation per shape, each in its own module:
//! [`SingleColumnKey`] for one primitive column, [`PackedKey`] for tuples of
//! integer-natured columns, and [`DynamicRowKey`] for everything else.

mod dynamic;
mod packed;
mod single_column;

pub use dynamic::DynamicRowKey;
pub use packed::PackedKey;
pub use single_column::SingleColumnKey;

use ahash::RandomState;
use arrow::compute::filter_record_batch;
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_buffer::NullBuffer;

/// One key shape of a hash equi-join. Implementations are zero-sized markers;
/// every method is associated, taking the per-batch state they build.
pub trait JoinKey: 'static {
    /// The `Copy` value stored in each build tuple and the key arena, compared
    /// against the probe row's value during the match walk.
    type Stored: Copy + Eq + Send;
    /// Per-batch accessor bound to one side's key columns.
    type Reader<'a>;
    /// Probe-side state for exact verification of a candidate whose stored
    /// value compared equal, bound to the stored build rows once the table
    /// publishes. `()` when stored equality is already exact.
    type Verifier<'a>;

    /// Bind `batch`'s key columns. Cheap: downcasts only. `state` is the
    /// join's shared hash state, for a shape whose stored value is itself a
    /// hash and so must carry the state into
    /// [`read_stored`](Self::read_stored).
    fn make_reader<'a>(
        batch: &'a RecordBatch,
        key_columns: &[usize],
        state: &RandomState,
    ) -> Self::Reader<'a>;

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

    /// Bind the stored build row batches' key columns for
    /// [`verify`](Self::verify).
    fn make_verifier<'a>(
        build_row_batches: &'a [RecordBatch],
        build_key_columns: &[usize],
    ) -> Self::Verifier<'a>;

    /// Whether probe row `probe_idx` (read through `reader`) and the build row
    /// at id `build_row_id` (read through `verifier`) hold equal keys, given
    /// their stored values already compared equal.
    fn verify(
        reader: &Self::Reader<'_>,
        verifier: &Self::Verifier<'_>,
        probe_idx: usize,
        build_row_id: u64,
    ) -> bool;
}

/// The validity shared by `key_columns`: a row is valid when every one of the
/// columns is valid at it. `None` when no key column holds a null.
pub(crate) fn combined_key_validity(
    batch: &RecordBatch,
    key_columns: &[usize],
) -> Option<NullBuffer> {
    let mut combined: Option<NullBuffer> = None;
    for &key_column in key_columns {
        let column = batch.column(key_column);
        if column.null_count() == 0 {
            continue;
        }
        let nulls = column.nulls().unwrap();
        combined = Some(match combined {
            None => nulls.clone(),
            Some(previous) => NullBuffer::new(previous.inner() & nulls.inner()),
        });
    }
    combined
}

/// Drop rows any of whose join key columns is null: an equi-join can never
/// match them, and the probe's hot loops read key values without validity
/// checks. Only the probe side filters; the build side keeps its batches
/// whole (they become the stored build row batches) and skips null-keyed rows when
/// generating tuples.
pub(crate) fn filter_null_keys(batch: RecordBatch, key_columns: &[usize]) -> RecordBatch {
    let Some(combined_validity) = combined_key_validity(&batch, key_columns) else {
        return batch;
    };
    let mask = BooleanArray::new(combined_validity.inner().clone(), None);
    filter_record_batch(&batch, &mask).expect("null-key filter mask matches batch length")
}
