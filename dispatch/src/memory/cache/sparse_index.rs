//! The sparse index family: a single decoded form ([`SparseIndex`]) with no
//! compressed twin. A sparse index is small and always read first, so it is
//! given the heaviest policy of any cache and rebuilt from the file on a miss.
//!
//! The decoded form is a placeholder for now, like [`DictIndex`](super::DictIndex).

use crate::memory::cache::cached::Cached;
use crate::memory::cache::derived_cache::Cache;
use crate::memory::slab::FrozenSlab;
use std::sync::Arc;

/// A decoded sparse index, pinned for as long as this is held.
pub struct SparseIndex {
    slabs: Vec<FrozenSlab>,
}

impl SparseIndex {
    /// The slabs the index was built into, in build order.
    pub fn slabs(&self) -> &[FrozenSlab] {
        &self.slabs
    }
}

/// What the cache keeps beside a sparse index's bytes. Empty until the
/// decoded form needs anything the bytes do not say.
pub struct SparseIndexMeta;

impl Cached for SparseIndex {
    type Meta = SparseIndexMeta;

    fn from_slabs(_meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        Self { slabs }
    }
}

/// The sparse index family's caches.
#[derive(Clone)]
pub struct SparseIndexCaches {
    pub decoded: Arc<Cache<SparseIndex>>,
}
