//! The caches this engine runs with, declared in one place.
//!
//! ```text
//!   family        member                      lives (insert / touch / reinforce)   share
//!   ──────        ──────                      ──────────────────────────────────   ─────
//!   pages         compressed_cache            1 / 1 / 6                            30 %
//!                 decompressed_cache          1 / 1 / 6                            70 %
//!   text index    text_index_file_cache       1 / 1 / 6                             0 %
//!                 text_index_sparse_cache     3 / 4 / 8                           100 %
//!                 text_index_dict_cache       2 / 3 / 8                             0 %
//!                 text_index_postings_cache   1 / 2 / 6                             0 %
//! ```
//!
//! Pages are the bulk of the ring and carry the lightest policy.
//!
//! The text index's shares are not a balance but an order. With the whole
//! target on the sparse index it can never be over its share, and with none
//! on the others each of them is over as soon as it holds a slot. So the
//! sparse index is protected exactly while any other member holds anything:
//! a file's sparse index is never given up before its dictionary blocks,
//! which could not be found again without it. The three unprotected members
//! then go by weight: file bytes first, since every decoded form can be
//! rebuilt from the file; posting lists next, which are the bulk; dictionary
//! blocks last, since each one serves every token in it.

use crate::memory::cache::decompressed_page::PageCaches;
use crate::memory::cache::id::CachePolicy;
use crate::memory::cache::registry::{Caches, CachesBuilder};
use crate::memory::cache::text_index::TextIndexCaches;
use std::sync::Arc;

/// Every cache, as the registry the evictor uses and as the typed handles the
/// readers use. The handles hold the same `Arc`s the registry does.
#[derive(Clone)]
pub struct CacheSetup {
    pub caches: Arc<Caches>,
    pub pages: PageCaches,
    pub text_index: TextIndexCaches,
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

        let text_index = builder.add_family(|family| TextIndexCaches {
            file_bytes: family.add_compressed(
                CachePolicy {
                    name: "text_index_file_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce: 6,
                },
                0,
            ),
            sparse_index: family.add_derived(
                CachePolicy {
                    name: "text_index_sparse_cache",
                    lives_at_insert: 3,
                    lives_per_touch: 4,
                    lives_at_reinforce: 8,
                },
                100,
                "PIVOT_TEXT_INDEX_SPARSE_CACHE",
            ),
            dict_blocks: family.add_derived(
                CachePolicy {
                    name: "text_index_dict_cache",
                    lives_at_insert: 2,
                    lives_per_touch: 3,
                    lives_at_reinforce: 8,
                },
                0,
                "PIVOT_TEXT_INDEX_DICT_CACHE",
            ),
            postings: family.add_derived(
                CachePolicy {
                    name: "text_index_postings_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 2,
                    lives_at_reinforce: 6,
                },
                0,
                "PIVOT_TEXT_INDEX_POSTINGS_CACHE",
            ),
        });

        Self {
            caches: Arc::new(builder.build()),
            pages,
            text_index,
        }
    }
}
