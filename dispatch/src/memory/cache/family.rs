//! A family: caches holding the same bytes in different forms.
//!
//! ```text
//!   family "pages"       family "dict index"    family "sparse index"
//!   ┌──────────────┐     ┌──────────────┐       ┌──────────────┐
//!   │ compressed   │     │ compressed   │       │ decoded      │
//!   │ decompressed │     │ decoded      │       └──────────────┘
//!   └──────────────┘     └──────────────┘
//! ```
//!
//! Two things follow from being in one family, and nothing else does:
//!
//! - **Reinforcement.** When a member drops an entry, every sibling's entry
//!   over the same file range is reinforced: with one copy gone the others are
//!   closer to being the last, and losing the last turns the next read into IO.
//! - **Weakening.** When a member's entry is read, every sibling's entry over
//!   the same file range is weakened: the copy being read is staying, so the
//!   others are the redundant ones.
//!
//! Between them these decide which form of each piece of data stays, one
//! piece at a time; how much of the ring each member ends up holding is
//! whatever that adds up to. A family with a single member has neither; it is
//! just a cache with a weight.

use crate::memory::cache::id::CacheId;

/// The caches that hold one kind of data in its different forms.
#[derive(Clone, Debug)]
pub struct FamilySpec {
    members: Vec<CacheId>,
}

impl FamilySpec {
    pub fn new(members: impl IntoIterator<Item = CacheId>) -> Self {
        let members: Vec<CacheId> = members.into_iter().collect();
        assert!(!members.is_empty(), "a family needs at least one cache");
        Self { members }
    }

    pub fn members(&self) -> impl Iterator<Item = CacheId> + '_ {
        self.members.iter().copied()
    }

    pub fn contains(&self, id: CacheId) -> bool {
        self.members().any(|member| member == id)
    }

    /// Every member but `id`.
    pub fn siblings_of(&self, id: CacheId) -> impl Iterator<Item = CacheId> + '_ {
        self.members().filter(move |member| *member != id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPRESSED: CacheId = CacheId::new(0);
    const DECOMPRESSED: CacheId = CacheId::new(1);

    #[test]
    #[should_panic(expected = "at least one cache")]
    fn a_family_without_members_is_rejected() {
        FamilySpec::new([]);
    }

    #[test]
    fn siblings_exclude_the_cache_itself() {
        let family = FamilySpec::new([COMPRESSED, DECOMPRESSED]);

        let siblings: Vec<_> = family.siblings_of(COMPRESSED).collect();

        assert_eq!(siblings, vec![DECOMPRESSED]);
    }

    #[test]
    fn a_single_member_has_no_siblings() {
        let family = FamilySpec::new([COMPRESSED]);

        assert_eq!(family.siblings_of(COMPRESSED).count(), 0);
    }
}
