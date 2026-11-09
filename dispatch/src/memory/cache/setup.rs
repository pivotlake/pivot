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

use crate::memory::cache::decompressed_page::PageCaches;
use crate::memory::cache::dict_index::DictIndexCaches;
use crate::memory::cache::id::CachePolicy;
use crate::memory::cache::registry::{Caches, CachesBuilder};
use crate::memory::cache::sparse_index::SparseIndexCaches;
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
        let mut builder = CachesBuilder::new(slot_count);

        let pages = builder.add_family(|family| PageCaches {
            compressed: family.add_compressed(
                CachePolicy {
                    name: "compressed_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce: 6,
                },
                30,
            ),
            decompressed: family.add_derived(
                CachePolicy {
                    name: "decompressed_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce: 6,
                },
                70,
                "PIVOT_DECOMPRESSED_CACHE",
            ),
        });

        let dict_index = builder.add_family(|family| DictIndexCaches {
            compressed: family.add_compressed(
                CachePolicy {
                    name: "dict_index_compressed_cache",
                    lives_at_insert: 2,
                    lives_per_touch: 3,
                    lives_at_reinforce: 8,
                },
                30,
            ),
            decoded: family.add_derived(
                CachePolicy {
                    name: "dict_index_cache",
                    lives_at_insert: 2,
                    lives_per_touch: 3,
                    lives_at_reinforce: 8,
                },
                70,
                "PIVOT_DICT_INDEX_CACHE",
            ),
        });

        let sparse_index = builder.add_family(|family| SparseIndexCaches {
            decoded: family.add_derived(
                CachePolicy {
                    name: "sparse_index_cache",
                    lives_at_insert: 3,
                    lives_per_touch: 4,
                    lives_at_reinforce: 4,
                },
                100,
                "PIVOT_SPARSE_INDEX_CACHE",
            ),
        });

        Self {
            caches: Arc::new(builder.build()),
            pages,
            dict_index,
            sparse_index,
        }
    }
}
