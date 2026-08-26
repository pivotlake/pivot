//! A shared scalar boundary that lets one operator prune another operator's
//! scan while the query is running.
//!
//! Some useful filter values do not exist when a query is planned. They emerge
//! only after execution starts:
//!
//! - a Top-N learns its current Nth-best leading sort key as rows arrive;
//! - a hash join learns the minimum and maximum build keys only after every
//!   build worker has finished.
//!
//! A [`BoundarySlot`] is the small piece of shared state that carries such a
//! value from the operator that discovers it (the **producer**) to scans that
//! can use it (the **consumers**). The plan compiler gives both ends the same
//! `Arc<BoundarySlot>`:
//!
//! ```text
//!                         one shared Arc
//!
//!     producer  ───────►  BoundarySlot  ───────►  consumer scan(s)
//!     Top-N or join       Option<Scalar>          row-group pruning
//! ```
//!
//! The slot carries only the value. It deliberately does not carry a column or
//! comparison operator: those belong to each consumer. This lets the same type
//! represent all of these relationships:
//!
//! ```text
//!     Top-N boundary ───────────────► scan compares < or >
//!
//!                         ┌─ min ───► scan compares >=
//!     sealed join build ──┤
//!                         └─ max ───► scan compares <=
//! ```
//!
//! # Lifecycle and correctness
//!
//! A slot starts **unarmed** (`None`). An unarmed consumer prunes nothing, so
//! publication is always an optimization rather than a correctness event.
//!
//! Top-N boundaries are published repeatedly, but only when the new value is
//! tighter than the old one. Consequently, a concurrent scan that observes an
//! older value merely prunes less work. Join bounds are published once, after
//! the build seals and the extrema are final; publishing a partial build bound
//! would be unsafe because it could discard a probe row that matches a later
//! build row.
//!
//! ```text
//!     unarmed ──► first safe boundary ──► tighter boundary ──► ...
//!       None          Some(value)            Some(value)
//!       │                 │                       │
//!       └─ no pruning     └─ safe pruning        └─ more pruning
//! ```
//!
//! The `RwLock` makes publication and snapshot reads thread-safe. Reading
//! clones the Arrow scalar's reference-counted array, not its underlying data.

use arrow::compute::kernels::cmp;
use arrow_array::{Array, ArrayRef, Datum, Scalar};
use std::sync::RwLock;

/// The producer-to-consumer cell for one runtime boundary value.
///
/// A logical producer may have many worker instances, and any number of scans
/// may consume the result. Producers publish through [`Self::publish_value`]
/// or [`Self::publish_if_tighter`]; consumers call [`Self::boundary`] whenever
/// they are about to decide whether work can be skipped.
///
/// The slot does not coordinate operator scheduling. In particular, the hash
/// join's separate build-ready gate prevents probing before its table is
/// complete. This type only publishes the scalar used for pruning.
#[derive(Debug, Default)]
pub struct BoundarySlot {
    /// The currently published one-value Arrow scalar. `None` means consumers
    /// must proceed without boundary pruning.
    boundary: RwLock<Option<Scalar<ArrayRef>>>,
}

impl BoundarySlot {
    /// Create an unarmed slot. Consumers initially observe `None` and prune
    /// nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a snapshot of the current boundary, or `None` if no safe value
    /// has been published yet.
    ///
    /// The returned scalar owns a clone of the underlying `ArrayRef`, so the
    /// lock is released before the consumer evaluates its comparison.
    pub fn boundary(&self) -> Option<Scalar<ArrayRef>> {
        self.boundary
            .read()
            .expect("boundary slot poisoned")
            .clone()
    }

    /// Publish a final boundary unconditionally.
    ///
    /// `boundary` must be a one-element array. This path is for a producer that
    /// computes the exact value over complete input—for example, a hash join
    /// publishing its build-key minimum or maximum after the final gather
    /// arrival. Callers must not publish a partial bound that could later widen.
    pub fn publish_value(&self, boundary: ArrayRef) {
        debug_assert_eq!(boundary.len(), 1);
        *self.boundary.write().expect("boundary slot poisoned") = Some(Scalar::new(boundary));
    }

    /// Publish `new_boundary` only if it prunes at least as safely and more
    /// tightly than the current value.
    ///
    /// Top-N uses this path because its boundary improves as workers see more
    /// rows. For an ascending Top-N, a smaller value is tighter; for a
    /// descending Top-N, a larger value is tighter. Equal, null, incomparable,
    /// or type-mismatched candidates leave the current boundary unchanged.
    ///
    /// The sort direction here is used only to decide whether the value became
    /// tighter. Each scan still owns the actual comparison it applies.
    pub(crate) fn publish_if_tighter(&self, new_boundary: Scalar<ArrayRef>, descending: bool) {
        let mut guard = self.boundary.write().expect("boundary slot poisoned");
        let kernel = if descending { cmp::gt } else { cmp::lt };
        let tighter = match guard.as_ref() {
            None => true,
            Some(current) => match kernel(&new_boundary as &dyn Datum, current as &dyn Datum) {
                Ok(verdict) => verdict.is_valid(0) && verdict.value(0),
                Err(_) => false,
            },
        };
        if tighter {
            *guard = Some(new_boundary);
        }
    }
}
