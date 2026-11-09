//! [`Caches`]: every cache, indexed by [`CacheId`], and the families that
//! group them.
//!
//! ```text
//!   Caches
//!   ├── caches[id]    Arc<dyn RingCache>      what the evictor reclaims from
//!   ├── policies[id]  CachePolicy             name, lives at insert, lives per touch
//!   ├── compressed[id] Option<Arc<CompressedCache>>   typed handle, for the IO path
//!   ├── families[f]   FamilySpec              members and their target shares
//!   └── family_of[id] f                       which family a cache belongs to
//! ```
//!
//! Built once at startup through [`CachesBuilder`] and shared by every worker's
//! memory context. The hot path through here is
//! [`reclaim`](Caches::reclaim): reclaim from the owning cache, then reinforce
//! its siblings over whatever it dropped.

use crate::io::OpenFile;
use crate::memory::WriteBuffer;
use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::family::FamilySpec;
use crate::memory::cache::id::{CacheId, CachePolicy, CacheSet};
use crate::memory::cache::ring_cache::RingCache;
use std::sync::Arc;

pub struct Caches {
    caches: Vec<Arc<dyn RingCache>>,
    policies: Vec<CachePolicy>,
    compressed: Vec<Option<Arc<CompressedCache>>>,
    families: Vec<FamilySpec>,
    family_of: Vec<usize>,
}

impl Caches {
    pub fn len(&self) -> usize {
        self.caches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = CacheId> {
        (0..self.caches.len()).map(|index| CacheId::new(index as u8))
    }

    pub fn policies(&self) -> &[CachePolicy] {
        &self.policies
    }

    pub fn policy(&self, id: CacheId) -> &CachePolicy {
        &self.policies[id.index()]
    }

    pub fn name(&self, id: CacheId) -> &'static str {
        self.policy(id).name
    }

    pub fn cache(&self, id: CacheId) -> &dyn RingCache {
        self.caches[id.index()].as_ref()
    }

    /// The compressed cache registered as `id`. Panics for an id that names a
    /// derived cache: only reads of raw file bytes come through here, and
    /// they always name a compressed cache.
    pub fn compressed(&self, id: CacheId) -> &CompressedCache {
        self.compressed[id.index()]
            .as_deref()
            .unwrap_or_else(|| panic!("cache {} is not a compressed cache", self.name(id)))
    }

    pub fn family_of(&self, id: CacheId) -> &FamilySpec {
        &self.families[self.family_of[id.index()]]
    }

    /// The caches the evictor should step over in a region whose members hold
    /// `slots_of(id)` slots each: the union of every family's protected members.
    pub fn protected_members(&self, slots_of: impl Fn(CacheId) -> usize) -> CacheSet {
        let mut protected = CacheSet::EMPTY;
        for family in &self.families {
            let family_protected = family.protected_members(&slots_of);
            for member in family.members() {
                if family_protected.contains(member) {
                    protected.insert(member);
                }
            }
        }
        protected
    }

    /// Reclaim `slot` from cache `id`, then reinforce its siblings over every
    /// range that died with it. The reinforcement runs after the cache has
    /// released its locks (its `reclaim` contract), so no cache ever holds a
    /// lock while another is entered: there is no lock order between caches
    /// to get wrong.
    pub fn reclaim(&self, id: CacheId, slot: usize) -> Option<WriteBuffer> {
        let reclaimed = self.cache(id).reclaim(slot)?;
        let family = self.family_of(id);
        for range in &reclaimed.dropped {
            for sibling in family.siblings_of(id) {
                self.cache(sibling).reinforce_range(range);
            }
        }
        Some(reclaimed.buffer)
    }

    /// A local descriptor was (re)opened: it may now name a different file
    /// than an equal, earlier descriptor did, so nothing cached under it may
    /// serve a later read. A remote object is keyed by its own identity and
    /// keeps what it cached.
    pub fn file_reopened(&self, file: &OpenFile) {
        if !matches!(file, OpenFile::Local(_)) {
            return;
        }
        for cache in &self.caches {
            cache.invalidate(file);
        }
    }

    /// Drop everything from every cache; the total entries dropped.
    pub fn clear_all(&self) -> usize {
        self.caches.iter().map(|cache| cache.clear()).sum()
    }
}

/// Assembles a [`Caches`]. Ids are handed out before the caches exist, since
/// each cache is built knowing its own id:
///
/// ```text
///   let id = builder.declare(policy);             // reserve the id
///   let cache = Arc::new(Cache::<V>::new(id, ..)); // build the cache with it
///   builder.install(cache);                        // register it under that id
///   builder.family([(id, 100)]);                   // say who it shares bytes with
/// ```
#[derive(Default)]
pub struct CachesBuilder {
    policies: Vec<CachePolicy>,
    caches: Vec<Option<Arc<dyn RingCache>>>,
    compressed: Vec<Option<Arc<CompressedCache>>>,
    families: Vec<FamilySpec>,
}

impl CachesBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve the id the next cache will be built with.
    pub fn declare(&mut self, policy: CachePolicy) -> CacheId {
        assert!(
            self.policies.len() < CacheId::MAX_CACHES,
            "too many caches declared"
        );
        let id = CacheId::new(self.policies.len() as u8);
        self.policies.push(policy);
        self.caches.push(None);
        self.compressed.push(None);
        id
    }

    /// Register a derived cache under the id it was built with.
    pub fn install(&mut self, cache: Arc<dyn RingCache>) {
        let index = cache.id().index();
        assert!(
            self.caches[index].is_none(),
            "cache {index} installed twice"
        );
        self.caches[index] = Some(cache);
    }

    /// Register a compressed cache, keeping a typed handle for the IO path.
    pub fn install_compressed(&mut self, cache: Arc<CompressedCache>) {
        self.compressed[cache.id().index()] = Some(cache.clone());
        self.install(cache);
    }

    pub fn family(&mut self, members: impl IntoIterator<Item = (CacheId, u8)>) {
        self.families.push(FamilySpec::new(members));
    }

    /// Every declared id must be installed and belong to exactly one family.
    pub fn build(self) -> Caches {
        let caches: Vec<Arc<dyn RingCache>> = self
            .caches
            .into_iter()
            .enumerate()
            .map(|(index, cache)| cache.unwrap_or_else(|| panic!("cache {index} never installed")))
            .collect();
        let mut family_of = vec![usize::MAX; caches.len()];
        for (family_index, family) in self.families.iter().enumerate() {
            for member in family.members() {
                assert!(
                    member.index() < caches.len(),
                    "family names an undeclared cache"
                );
                assert_eq!(
                    family_of[member.index()],
                    usize::MAX,
                    "cache {} is in two families",
                    self.policies[member.index()].name
                );
                family_of[member.index()] = family_index;
            }
        }
        for (index, family) in family_of.iter().enumerate() {
            assert_ne!(
                *family,
                usize::MAX,
                "cache {} is in no family",
                self.policies[index].name
            );
        }
        Caches {
            caches,
            policies: self.policies,
            compressed: self.compressed,
            families: self.families,
            family_of,
        }
    }
}
