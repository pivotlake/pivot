//! A family: caches holding the same bytes in different forms.
//!
//! ```text
//!   family "pages"              family "dict index"          family "sparse index"
//!   ┌──────────────┬─────────┐  ┌──────────────┬─────────┐   ┌──────────────┐
//!   │ compressed   │ 30 %    │  │ compressed   │ 30 %    │   │ decoded 100% │
//!   │ decompressed │ 70 %    │  │ decoded      │ 70 %    │   └──────────────┘
//!   └──────────────┴─────────┘  └──────────────┴─────────┘
//! ```
//!
//! Two things follow from being in one family, and nothing else does:
//!
//! - **Share protection.** A member holding less than its target share of the
//!   family's slots is spared by the evictor while a sibling holds more than
//!   its target. Shares are judged within the family only; the family's total
//!   against the rest of the ring is left to recency and weight.
//! - **Reinforcement.** When a member drops an entry, every sibling's entry
//!   over the same file range is reinforced: with one copy gone the others are
//!   closer to being the last, and losing the last turns the next read into IO.
//!
//! A family with a single member has neither; it is just a cache with a weight.

use crate::memory::cache::id::{CacheId, CacheSet};

/// One member of a family and the share of the family's slots it should hold.
#[derive(Clone, Copy, Debug)]
pub struct FamilyMember {
    pub cache: CacheId,
    pub target_share_pct: u8,
}

/// The caches that hold one kind of data in its different forms.
#[derive(Clone, Debug)]
pub struct FamilySpec {
    members: Vec<FamilyMember>,
}

impl FamilySpec {
    /// `members` are `(cache, target share percent)`; the shares must sum to
    /// 100.
    pub fn new(members: impl IntoIterator<Item = (CacheId, u8)>) -> Self {
        let members: Vec<FamilyMember> = members
            .into_iter()
            .map(|(cache, target_share_pct)| FamilyMember {
                cache,
                target_share_pct,
            })
            .collect();
        assert!(!members.is_empty(), "a family needs at least one cache");
        let total: u32 = members
            .iter()
            .map(|member| member.target_share_pct as u32)
            .sum();
        assert_eq!(total, 100, "a family's target shares must sum to 100");
        Self { members }
    }

    pub fn members(&self) -> impl Iterator<Item = CacheId> + '_ {
        self.members.iter().map(|member| member.cache)
    }

    pub fn contains(&self, id: CacheId) -> bool {
        self.members().any(|member| member == id)
    }

    /// Every member but `id`.
    pub fn siblings_of(&self, id: CacheId) -> impl Iterator<Item = CacheId> + '_ {
        self.members().filter(move |member| *member != id)
    }

    /// The members the evictor should step over right now: those not above
    /// their target share while some sibling is strictly above its own.
    /// `slots_of` reports how many slots each member holds in the region
    /// being evicted from; it is read once per member, so the answer describes
    /// one instant and is as approximate as the counts it read.
    ///
    /// The strictness means that at exactly the target nobody is protected
    /// and whichever slot drains first goes, so the shares cannot oscillate
    /// around the boundary.
    pub fn protected_members(&self, slots_of: impl Fn(CacheId) -> usize) -> CacheSet {
        let held: Vec<usize> = self.members.iter().map(|m| slots_of(m.cache)).collect();
        let total: usize = held.iter().sum();
        let is_over_target =
            |i: usize| held[i] * 100 > self.members[i].target_share_pct as usize * total;
        let someone_is_over = (0..held.len()).any(is_over_target);
        let mut protected = CacheSet::EMPTY;
        if !someone_is_over {
            return protected;
        }
        for (i, member) in self.members.iter().enumerate() {
            if !is_over_target(i) {
                protected.insert(member.cache);
            }
        }
        protected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPRESSED: CacheId = CacheId::new(0);
    const DECOMPRESSED: CacheId = CacheId::new(1);
    const DECODED: CacheId = CacheId::new(2);

    fn pages() -> FamilySpec {
        FamilySpec::new([(COMPRESSED, 30), (DECOMPRESSED, 70)])
    }

    fn held(compressed: usize, decompressed: usize) -> impl Fn(CacheId) -> usize {
        move |id| {
            if id == COMPRESSED {
                compressed
            } else {
                decompressed
            }
        }
    }

    #[test]
    #[should_panic(expected = "sum to 100")]
    fn shares_must_sum_to_one_hundred() {
        FamilySpec::new([(COMPRESSED, 30), (DECOMPRESSED, 60)]);
    }

    #[test]
    fn the_member_under_target_is_protected_while_a_sibling_is_over() {
        let family = pages();

        let protected = family.protected_members(held(1, 10)); // compressed 9%

        assert!(protected.contains(COMPRESSED));
        assert!(!protected.contains(DECOMPRESSED));
    }

    #[test]
    fn the_member_over_target_is_never_protected() {
        let family = pages();

        let protected = family.protected_members(held(1, 1)); // compressed 50%

        assert!(!protected.contains(COMPRESSED));
        assert!(protected.contains(DECOMPRESSED));
    }

    #[test]
    fn exactly_on_target_protects_nobody() {
        let family = pages();

        let protected = family.protected_members(held(3, 7));

        assert!(protected.is_empty());
    }

    #[test]
    fn an_empty_family_protects_nobody() {
        let family = pages();

        assert!(family.protected_members(held(0, 0)).is_empty());
    }

    #[test]
    fn a_single_member_family_protects_nobody() {
        let family = FamilySpec::new([(DECODED, 100)]);

        assert!(family.protected_members(|_| 5).is_empty());
    }

    #[test]
    fn with_three_members_every_non_over_member_is_protected() {
        let family = FamilySpec::new([(COMPRESSED, 20), (DECOMPRESSED, 50), (DECODED, 30)]);
        let slots = |id: CacheId| match id {
            COMPRESSED => 2,
            DECOMPRESSED => 1,
            _ => 1,
        }; // compressed 50% of 4

        let protected = family.protected_members(slots);

        assert!(!protected.contains(COMPRESSED));
        assert!(protected.contains(DECOMPRESSED));
        assert!(protected.contains(DECODED));
    }

    #[test]
    fn siblings_exclude_the_cache_itself() {
        let family = pages();

        let siblings: Vec<_> = family.siblings_of(COMPRESSED).collect();

        assert_eq!(siblings, vec![DECOMPRESSED]);
    }
}
