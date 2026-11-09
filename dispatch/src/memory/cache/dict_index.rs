//! The dictionary index family: the index's file bytes as read
//! ([`CompressedCache`]) and decoded into its in-memory form ([`DictIndex`]).
//!
//! The decoded form is a placeholder for now: it holds the slabs its builder
//! carved and nothing else. The real structure will be built through
//! [`SlabSource`](crate::memory::slab::SlabSource) into aligned slabs and
//! read back as typed slab buffers; only [`Cached::from_slabs`] and the meta
//! change when it lands.

use crate::memory::cache::cached::Cached;
use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::derived_cache::Cache;
use crate::memory::slab::FrozenSlab;
use std::sync::Arc;

/// A decoded dictionary index, pinned for as long as this is held.
pub struct DictIndex {
    slabs: Vec<FrozenSlab>,
}

impl DictIndex {
    /// The slabs the index was built into, in build order.
    pub fn slabs(&self) -> &[FrozenSlab] {
        &self.slabs
    }
}

/// What the cache keeps beside a dictionary index's bytes. Empty until the
/// decoded form needs anything the bytes do not say.
pub struct DictIndexMeta;

impl Cached for DictIndex {
    type Meta = DictIndexMeta;

    fn from_slabs(_meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        Self { slabs }
    }
}

/// The dictionary index family's caches.
#[derive(Clone)]
pub struct DictIndexCaches {
    pub compressed: Arc<CompressedCache>,
    pub decoded: Arc<Cache<DictIndex>>,
}
