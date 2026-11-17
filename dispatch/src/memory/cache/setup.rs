//! The caches this engine runs with, declared in one place.
//!
//! ```text
//!   family        member                      lives (insert / touch / reinforce)   share
//!   ──────        ──────                      ──────────────────────────────────   ─────
//!   pages         compressed_cache            1 / 1 / 6                            30 %
//!                 decompressed_cache          1 / 1 / 6                            70 %
//!   text index    text_index_sparse_cache     3 / 4 / 8                           100 %
//!                 text_index_file_cache       1 / 1 / 6                             0 %
//!                 text_index_postings_cache   1 / 2 / 6                             0 %
//!                 text_index_dict_cache       2 / 3 / 8                             0 %
//! ```
//!
//! The compressed share and the reinforce value of the pages family are the
//! defaults of the environment variables below.
//!
//! The text index's shares are not a balance but an order. A family gives
//! from its first member holding more than its target share, and moves on to
//! the next member, in the order they were added and round again, only when
//! that one has nothing to give. With the whole target on the sparse index it
//! is never over its share, and with none on the others each of them is over
//! as soon as it holds a slot. So the members give in the order they are
//! listed after the sparse index, each only once those before it hold nothing
//! the evictor can take, and the sparse index, added first, is reached last:
//!
//! ```text
//!   file bytes  ─▶  posting lists  ─▶  dictionary blocks  ─▶  sparse index
//!   every form      the bulk           each serves every      without it no
//!   above is                           token in it            dictionary block
//!   built from it                                             can be found
//! ```
//!
//! A file's sparse index is therefore never given up before a dictionary
//! block the evictor could have taken instead. The lives decide which slots
//! of one member go first, and how the family fares against the pages.

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
            let sparse_index = family.add_derived(
                CachePolicy {
                    name: "text_index_sparse_cache",
                    lives_at_insert: 3,
                    lives_per_touch: 4,
                    lives_at_reinforce: 8,
                },
                100,
                "PIVOT_TEXT_INDEX_SPARSE_CACHE",
            );
            let file_bytes = family.add_compressed(
                CachePolicy {
                    name: "text_index_file_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce: 6,
                },
                0,
            );
            let postings = family.add_derived(
                CachePolicy {
                    name: "text_index_postings_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 2,
                    lives_at_reinforce: 6,
                },
                0,
                "PIVOT_TEXT_INDEX_POSTINGS_CACHE",
            );
            let dict_blocks = family.add_derived(
                CachePolicy {
                    name: "text_index_dict_cache",
                    lives_at_insert: 2,
                    lives_per_touch: 3,
                    lives_at_reinforce: 8,
                },
                0,
                "PIVOT_TEXT_INDEX_DICT_CACHE",
            );
            TextIndexCaches {
                file_bytes,
                sparse_index,
                dict_blocks,
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
