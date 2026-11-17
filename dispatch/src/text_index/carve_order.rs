//! [`CarveOrder`]: which vec each chunk of a piece belongs to.
//!
//! A piece is several [`SlabVec`]s that grow together and take their chunks
//! from one source, so the chunks come out interleaved:
//!
//! ```text
//!   carved, in order:   │ vec 0 │ vec 1 │ vec 1 │ vec 0 │ vec 2 │ vec 1 │ …
//!   carve order:          0       1       1       0       2       1
//! ```
//!
//! A cache keeps a piece as nothing but the slabs its build carved, in that
//! order, and hands them back the same way on a hit. The carve order, kept
//! beside them, is what tells the slabs of one vec from those of another, so
//! each vec can be put back together over its own.

use crate::memory::{FrozenSlab, SlabSource, SlabVec};

/// For each chunk a piece carved, in order, the number of the vec it went to.
/// Every append to a vec of the piece goes through here, which is how the
/// order gets recorded.
#[derive(Default)]
pub(super) struct CarveOrder {
    vec_of_chunk: Vec<u8>,
}

impl CarveOrder {
    /// Push `value` to `vec`, the piece's vec number `vec_number`.
    pub fn push<T: Copy>(
        &mut self,
        vec_number: u8,
        vec: &mut SlabVec<T>,
        source: &mut impl SlabSource,
        value: T,
    ) {
        let chunks_before = vec.count_chunks();
        vec.push(source, value);
        self.record_chunks(vec_number, vec.count_chunks() - chunks_before);
    }

    /// Append `values` to `vec`, the piece's vec number `vec_number`.
    pub fn extend_from_slice<T: Copy>(
        &mut self,
        vec_number: u8,
        vec: &mut SlabVec<T>,
        source: &mut impl SlabSource,
        values: &[T],
    ) {
        let chunks_before = vec.count_chunks();
        vec.extend_from_slice(source, values);
        self.record_chunks(vec_number, vec.count_chunks() - chunks_before);
    }

    fn record_chunks(&mut self, vec_number: u8, chunk_count: usize) {
        self.vec_of_chunk
            .extend(std::iter::repeat_n(vec_number, chunk_count));
    }

    /// Deal `slabs`, the slabs a piece carved in order, out to its
    /// `VEC_COUNT` vecs: the chunks of vec 0, of vec 1, and so on, each in
    /// order.
    pub fn deal_slabs<const VEC_COUNT: usize>(
        &self,
        slabs: Vec<FrozenSlab>,
    ) -> [Vec<FrozenSlab>; VEC_COUNT] {
        assert_eq!(
            slabs.len(),
            self.vec_of_chunk.len(),
            "the piece was built into another number of slabs"
        );
        let mut slabs_of_vecs = [const { Vec::new() }; VEC_COUNT];
        for (slab, &vec_number) in slabs.into_iter().zip(&self.vec_of_chunk) {
            slabs_of_vecs[vec_number as usize].push(slab);
        }
        slabs_of_vecs
    }
}
