//! The implicit unit key used to execute a Cartesian product as a hash join.

use ahash::RandomState;
use arrow_array::RecordBatch;

use super::JoinKey;

/// Every row has this one logical key, so every probe row matches every build
/// row. A byte is stored for each build row because the join's slab buffers do
/// not support zero-sized values.
pub(crate) struct CrossJoinKey;

impl JoinKey for CrossJoinKey {
    type Stored = u8;
    type Reader<'a> = ();
    type Verifier<'a> = ();

    fn make_reader<'a>(
        _batch: &'a RecordBatch,
        key_columns: &[usize],
        _state: &RandomState,
    ) -> Self::Reader<'a> {
        assert!(key_columns.is_empty(), "a cross join has no key columns");
    }

    #[inline(always)]
    fn read_stored(_reader: &(), _idx: usize) -> u8 {
        0
    }

    #[inline(always)]
    fn hash_row(_reader: &(), _idx: usize, _state: &RandomState) -> u64 {
        0
    }

    #[inline(always)]
    fn is_null(_reader: &(), _idx: usize) -> bool {
        false
    }

    fn make_verifier<'a>(
        _build_row_batches: &'a [RecordBatch],
        build_key_columns: &[usize],
    ) -> Self::Verifier<'a> {
        assert!(
            build_key_columns.is_empty(),
            "a cross join has no key columns"
        );
    }

    #[inline(always)]
    fn verify(_reader: &(), _verifier: &(), _probe_idx: usize, _build_row_id: u32) -> bool {
        true
    }
}
