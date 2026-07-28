//! Runtime layout of a hash-table entry or strided scatter row.

use crate::memory::BUFFER_SIZE;
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// Runtime metadata, byte offsets, stride, and alignment for one
/// `(hash, key, state)` entry.
///
/// Fields are placed in descending alignment order. Equal alignments retain
/// their logical hash, key, state order. This minimizes padding and preserves
/// the previous layout for fixed-size aggregation states.
pub(super) struct EntryLayout<K, V: AggregationValue> {
    pub(super) state_meta: V::EntryStateMeta,
    pub(super) hash_offset: usize,
    pub(super) key_offset: usize,
    pub(super) state_offset: usize,
    pub(super) stride: usize,
    pub(super) align: usize,
    _marker: PhantomData<(K, V)>,
}

impl<K, V: AggregationValue> From<&V::Context> for EntryLayout<K, V> {
    fn from(context: &V::Context) -> Self {
        let state_meta = V::entry_state_meta(context);

        // Each tuple is (alignment, size), in logical hash, key, state order.
        let fields = [
            (align_of::<u64>(), size_of::<u64>()),
            (align_of::<K>(), size_of::<K>()),
            (V::entry_state_align(), V::entry_state_size(state_meta)),
        ];
        let mut order = [0usize, 1, 2];
        order.sort_by_key(|&field| std::cmp::Reverse(fields[field].0));

        let mut offsets = [0usize; 3];
        let mut cursor = 0;
        for &field in &order {
            cursor = align_up(cursor, fields[field].0);
            offsets[field] = cursor;
            cursor += fields[field].1;
        }

        let align = fields.iter().map(|&(align, _)| align).max().unwrap();
        let stride = align_up(cursor, align);
        assert!(stride <= BUFFER_SIZE, "entry stride exceeds one slab");

        Self {
            state_meta,
            hash_offset: offsets[0],
            key_offset: offsets[1],
            state_offset: offsets[2],
            stride,
            align,
            _marker: PhantomData,
        }
    }
}

/// The exact padded bytes occupied by one entry for `K` and `V`.
pub(crate) fn entry_stride<K, V: AggregationValue>(context: &V::Context) -> usize {
    EntryLayout::<K, V>::from(context).stride
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}
