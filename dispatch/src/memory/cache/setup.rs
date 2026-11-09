//! The caches this engine runs with, declared in one place.
//!
//! ```text
//!   family          member                        lives (insert / touch / reinforce)   share
//!   ──────          ──────                        ──────────────────────────────────   ─────
//!   pages           compressed_cache              1 / 1 / 6                            30 %
//!                   decompressed_cache            1 / 1 / 6                            70 %
//!   dict index      dict_index_compressed_cache   2 / 3 / 8                            30 %
//!                   dict_index_cache              2 / 3 / 8                            70 %
//!   sparse index    sparse_index_cache            3 / 4 / 4                           100 %
//! ```
//!
//! Pages are the bulk of the ring and carry the lightest policy. The index
//! caches are small and each entry saves far more reading than it occupies, so
//! they get more lives at equal recency, the sparse index most of all since it
//! is read before anything else. The sparse index has no sibling, so its
//! reinforce value never applies and equals its touch value.

use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::decompressed_page::PageCaches;
use crate::memory::cache::derived_cache::Cache;
use crate::memory::cache::dict_index::{DictIndex, DictIndexCaches};
use crate::memory::cache::id::CachePolicy;
use crate::memory::cache::registry::{Caches, CachesBuilder};
use crate::memory::cache::sparse_index::{SparseIndex, SparseIndexCaches};
use std::sync::Arc;

/// Every cache, as the registry the evictor uses and as the typed handles the
/// readers use. The handles hold the same `Arc`s the registry does.
#[derive(Clone)]
pub struct CacheSetup {
    pub caches: Arc<Caches>,
    pub pages: PageCaches,
    pub dict_index: DictIndexCaches,
    pub sparse_index: SparseIndexCaches,
}

impl CacheSetup {
    /// The standard set over a ring of `slot_count` slots.
    pub fn standard(slot_count: usize) -> Self {
        let mut builder = CachesBuilder::new();

        let pages_compressed = builder.declare(CachePolicy {
            name: "compressed_cache",
            lives_at_insert: 1,
            lives_per_touch: 1,
            lives_at_reinforce: 6,
        });
        let pages_decompressed = builder.declare(CachePolicy {
            name: "decompressed_cache",
            lives_at_insert: 1,
            lives_per_touch: 1,
            lives_at_reinforce: 6,
        });
        let pages = PageCaches {
            compressed: Arc::new(CompressedCache::new(pages_compressed, slot_count)),
            decompressed: Arc::new(Cache::new(
                pages_decompressed,
                slot_count,
                "PIVOT_DECOMPRESSED_CACHE",
            )),
        };
        builder.install_compressed(pages.compressed.clone());
        builder.install(pages.decompressed.clone());
        builder.family([(pages_compressed, 30), (pages_decompressed, 70)]);

        let dict_compressed = builder.declare(CachePolicy {
            name: "dict_index_compressed_cache",
            lives_at_insert: 2,
            lives_per_touch: 3,
            lives_at_reinforce: 8,
        });
        let dict_decoded = builder.declare(CachePolicy {
            name: "dict_index_cache",
            lives_at_insert: 2,
            lives_per_touch: 3,
            lives_at_reinforce: 8,
        });
        let dict_index = DictIndexCaches {
            compressed: Arc::new(CompressedCache::new(dict_compressed, slot_count)),
            decoded: Arc::new(Cache::<DictIndex>::new(
                dict_decoded,
                slot_count,
                "PIVOT_DICT_INDEX_CACHE",
            )),
        };
        builder.install_compressed(dict_index.compressed.clone());
        builder.install(dict_index.decoded.clone());
        builder.family([(dict_compressed, 30), (dict_decoded, 70)]);

        let sparse_decoded = builder.declare(CachePolicy {
            name: "sparse_index_cache",
            lives_at_insert: 3,
            lives_per_touch: 4,
            lives_at_reinforce: 4,
        });
        let sparse_index = SparseIndexCaches {
            decoded: Arc::new(Cache::<SparseIndex>::new(
                sparse_decoded,
                slot_count,
                "PIVOT_SPARSE_INDEX_CACHE",
            )),
        };
        builder.install(sparse_index.decoded.clone());
        builder.family([(sparse_decoded, 100)]);

        Self {
            caches: Arc::new(builder.build()),
            pages,
            dict_index,
            sparse_index,
        }
    }
}
