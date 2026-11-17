//! The caches this engine runs with, declared in one place.
//!
//! ```text
//!   family       member                      lives (insert / touch / reinforce)   share
//!   ──────       ──────                      ──────────────────────────────────   ─────
//!   pages        compressed_cache            1 / 1 / 6                             30 %
//!                decompressed_cache          1 / 1 / 6                             70 %
//!   text index   text_index_sparse_cache     3 / 3 / 3                            100 %
//!                text_index_postings_cache   1 / 1 / 1                              0 %
//!                text_index_dict_cache       2 / 2 / 2                              0 %
//! ```
//!
//! The compressed share and the reinforce value of the pages family are the
//! defaults of the environment variables below.
//!
//! In the text index family the sparse indexes have the whole share, so they
//! are never over it, and the postings and the dict indexes have none, so
//! each is over it while it holds a single slot. The evictor takes from the
//! first member over its share, in the order the members were added, which
//! makes the family give its slots strictly in this order:
//!
//! ```text
//!   1. postings         while any of them can be taken
//!   2. dict indexes     once no postings can
//!   3. sparse indexes   once nothing else in the family can
//! ```
//!
//! The lives follow the same ranking, most for a sparse index and fewest for
//! postings. They decide how each member holds up against the other families
//! and which of a member's own slots go first. The three members hold
//! different parts of an index file, so a slot one of them drops has no copy
//! in another to reinforce, and reinforcing is given the same lives as a
//! touch.

use crate::env::get_env_var_with_default;
use crate::memory::cache::decompressed_page::PageCaches;
use crate::memory::cache::id::CachePolicy;
use crate::memory::cache::registry::{Caches, CachesBuilder};
use crate::memory::cache::text_index::TextIndexCaches;
use std::sync::Arc;

/// Reads the target share of the pages family's slots the compressed cache
/// should hold, in percent (env `PIVOT_COMPRESSED_CACHE_PCT`, default 30). The
/// decompressed cache's target is the rest.
fn get_compressed_share_pct() -> u8 {
    let compressed_share_pct = get_env_var_with_default("PIVOT_COMPRESSED_CACHE_PCT", 30u8);
    assert!(
        compressed_share_pct <= 100,
        "PIVOT_COMPRESSED_CACHE_PCT is a percentage, got {compressed_share_pct}"
    );
    compressed_share_pct
}

/// Reads the lives a page slot gains per read once its sibling dropped its
/// copy of the same bytes (env `PIVOT_REINFORCE_BUMP`, default 6).
fn get_lives_at_reinforce() -> u8 {
    get_env_var_with_default("PIVOT_REINFORCE_BUMP", 6)
}

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
        let compressed_share_pct = get_compressed_share_pct();
        let lives_at_reinforce = get_lives_at_reinforce();
        let mut builder = CachesBuilder::new(slot_count);

        let pages = builder.add_family(|family| PageCaches {
            compressed: family.add_compressed(
                CachePolicy {
                    name: "compressed_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce,
                },
                compressed_share_pct,
            ),
            decompressed: family.add_derived(
                CachePolicy {
                    name: "decompressed_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce,
                },
                100 - compressed_share_pct,
                "PIVOT_DECOMPRESSED_CACHE",
            ),
        });

        let text_index = builder.add_family(|family| {
            let sparse_indexes = family.add_derived(
                CachePolicy {
                    name: "text_index_sparse_cache",
                    lives_at_insert: 3,
                    lives_per_touch: 3,
                    lives_at_reinforce: 3,
                },
                100,
                "PIVOT_TEXT_INDEX_SPARSE_CACHE",
            );
            let postings = family.add_derived(
                CachePolicy {
                    name: "text_index_postings_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce: 1,
                },
                0,
                "PIVOT_TEXT_INDEX_POSTINGS_CACHE",
            );
            let dict_indexes = family.add_derived(
                CachePolicy {
                    name: "text_index_dict_cache",
                    lives_at_insert: 2,
                    lives_per_touch: 2,
                    lives_at_reinforce: 2,
                },
                0,
                "PIVOT_TEXT_INDEX_DICT_CACHE",
            );
            TextIndexCaches {
                sparse_indexes,
                dict_indexes,
                postings,
            }
        });

        Self {
            caches: Arc::new(builder.build()),
            pages,
            text_index,
        }
    }
}
