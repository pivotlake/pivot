//! Caching on the ring: every form of every kind of data the engine keeps
//! resident, and how those forms are balanced against each other.
//!
//! ```text
//!                              Caches (registry)
//!                ┌─────────────────────┴─────────────────────┐
//!                │ caches[CacheId] ──▶ dyn RingCache          │
//!                │ families ──▶ [(CacheId, share)] …          │
//!                └─────────────────────┬─────────────────────┘
//!          reclaim(id, slot) / reinforce siblings / invalidate / clear
//!                                      │
//!        ┌─────────────────────────────┼──────────────────────────────┐
//!        ▼                             ▼                              ▼
//!   CompressedCache              Cache<DecompressedPage>        Cache<SparseIndex>, …
//!   raw file bytes,              bytes decompressed from        forms decoded from an
//!   4 KB blocks, async fills     a compressed twin              index file's raw bytes
//!        │                             │                              │
//!        └───────── family "pages" ────┘                    family "text index", with
//!                                                           a CompressedCache of its own
//!
//!   shared underneath:  FileMap (what is cached, per file)
//!                       SlotTenants (what is in a slot)
//!                       FillCursor + CacheFill (where this worker packs the next entry)
//!                       Clock (one hand per region; policies and shares decide who goes)
//! ```
//!
//! Two kinds of cache. [`CompressedCache`] holds file bytes as read, published
//! before they land so concurrent readers follow one fill. [`Cache<V>`] holds a
//! form derived from them, built straight into ring slabs and published only
//! once complete. Both implement [`RingCache`], which is all the evictor sees.
//!
//! A *family* groups the caches holding the same bytes in different forms.
//! Within a family the caches are balanced by target share and reinforce each
//! other when one drops a copy; across families, slots compete on recency and
//! each cache's [`CachePolicy`] weight alone. See [`family`] and [`clock`](crate::memory::clock).
//!
//! [`setup`] declares the families this engine runs with.

mod cached;
pub mod compressed;
mod decompressed_page;
mod derived_cache;
mod family;
mod file_map;
mod fill;
mod fill_cursor;
mod id;
mod registry;
mod ring_cache;
mod setup;
mod slot_tenants;
mod text_index;

pub use cached::Cached;
pub use compressed::{CacheLookup, CompressedCache, MissingExtent};
pub use decompressed_page::{DecompressedPage, DecompressedPageMeta, PageCaches};
pub use derived_cache::{Cache, CacheKey, Segment};
pub use family::FamilySpec;
pub use fill::CacheFill;
pub use fill_cursor::FillCursor;
pub use id::{CacheId, CachePolicy, CacheSet};
pub use registry::Caches;
pub use ring_cache::{Reclaimed, RingCache, SourceRange};
pub use setup::CacheSetup;
pub use text_index::TextIndexCaches;

#[cfg(test)]
mod tests;
