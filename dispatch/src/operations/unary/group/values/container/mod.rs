//! GROUP BY aggregation containers.
//!
//! - [`Compiled`] stores a fixed tuple of numeric slots.
//! - [`Dynamic`] stores any runtime number and mix of supported slots.
//!
//! Both are generic over a [`SeenMask`], the per-group record of which slots
//! have folded a non-NULL value. The planner instantiates `u8` when an
//! aggregated column can hold NULLs and the zero-sized `()` when every one is
//! provably NULL-free, so a NULL-free query's entries and fold loops are
//! bit-identical to a mask-less build.

mod compiled;
mod dynamic;

pub use compiled::{Compiled, CountSlot, CountValidSlot, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use dynamic::Dynamic;

use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use arrow_buffer::{BooleanBuffer, NullBuffer};

/// Per-group storage of the per-slot seen bits (bit `s` set = slot `s` folded a
/// non-NULL value; `u8` suffices since a [`Compiled`] signature holds at most 8
/// slots, and a tracked [`Dynamic`] keeps one whole mask cell per group). A
/// slot whose bit never sets renders as SQL NULL; its cell holds the fold's
/// identity, so the numeric fold loops stay branch-free.
///
/// The `()` form is zero-sized and reports [`TRACKING`](Self::TRACKING)
/// `false`, which const-folds every per-row validity check away and keeps the
/// group entry layout identical to a mask-less build, so a NULL-free query pays
/// nothing. `false` is a planner promise that no aggregated column holds NULLs.
pub trait SeenMask: Copy + Default + Send + Sync + 'static {
    /// Whether this storage records anything. `false` promises the planner
    /// proved every aggregated column NULL-free, so rows need no validity check.
    const TRACKING: bool;
    /// Record slot `slot`'s bit; `seen` ORs in (a bit never clears).
    fn record(&mut self, slot: usize, seen: bool);
    /// Whether slot `slot` has folded a non-NULL value.
    fn is_seen(&self, slot: usize) -> bool;
    /// Combine two partials' masks.
    fn union(self, other: Self) -> Self;
    /// The mask as bits (all set when untracked).
    fn as_bits(&self) -> u8;
}

impl SeenMask for () {
    const TRACKING: bool = false;
    #[inline(always)]
    fn record(&mut self, _slot: usize, _seen: bool) {}
    #[inline(always)]
    fn is_seen(&self, _slot: usize) -> bool {
        true
    }
    #[inline(always)]
    fn union(self, _other: Self) -> Self {}
    #[inline(always)]
    fn as_bits(&self) -> u8 {
        u8::MAX
    }
}

impl SeenMask for u8 {
    const TRACKING: bool = true;
    #[inline(always)]
    fn record(&mut self, slot: usize, seen: bool) {
        *self |= (seen as u8) << slot;
    }
    #[inline(always)]
    fn is_seen(&self, slot: usize) -> bool {
        self & (1 << slot) != 0
    }
    #[inline(always)]
    fn union(self, other: Self) -> Self {
        self | other
    }
    #[inline(always)]
    fn as_bits(&self) -> u8 {
        *self
    }
}

/// The output-side companion to a [`SeenMask`]: buffers each pushed group's
/// mask so `finish` can build per-slot null buffers, tracking the AND of all
/// masks to skip the pass for slots no group left NULL. The `()` mask stores
/// nothing and its column allocates nothing.
pub struct SeenMaskColumn<S: SeenMask> {
    masks: Option<SlabColumn<u8>>,
    all_seen: u8,
    marker: std::marker::PhantomData<S>,
}

impl<S: SeenMask> SeenMaskColumn<S> {
    pub fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            masks: S::TRACKING.then(|| SlabColumn::with_capacity(allocator, rows)),
            all_seen: u8::MAX,
            marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, bits: u8) {
        if S::TRACKING {
            self.masks
                .as_mut()
                .expect("a tracking mask column buffers every pushed mask")
                .push(bits);
            self.all_seen &= bits;
        }
    }

    /// The null buffer for `slot`'s output column: NULL where the group's seen
    /// bit is unset. `all_seen` short-circuits the pass when the slot's bit is
    /// set in every pushed mask (no group is NULL).
    pub fn nulls_for_slot(&self, slot: usize) -> Option<NullBuffer> {
        if self.all_seen & (1 << slot) != 0 {
            return None;
        }
        let masks = self
            .masks
            .as_ref()
            .expect("a cleared all_seen bit implies a tracking mask column")
            .as_slice();
        Some(NullBuffer::new(BooleanBuffer::collect_bool(
            masks.len(),
            |i| masks[i] & (1 << slot) != 0,
        )))
    }
}
