//! Builders that accumulate values on slab memory and finish into an Arrow
//! array with zero copies. [`ArrayBuilder`] is the interface the parquet page
//! decoders and the GROUP BY output write through; [`PrimitiveBuilder`] builds
//! fixed-width primitive arrays, [`BooleanBuilder`] boolean arrays and
//! [`ViewBuilder`] byte-view arrays, and [`BitmapBuilder`] a bitmap: the null
//! bitmap that goes with any of them, under the name [`ValidityBuilder`], and
//! the values of a boolean array. [`BitSink`] is what the decoders append
//! bits through, whichever bitmap they are building.

mod bitmap_builder;
mod boolean_builder;
mod primitive_builder;
mod view_builder;

use arrow_array::ArrayRef;
use arrow_buffer::Buffer;
pub use bitmap_builder::{BitSink, BitmapBuilder, append_bits, set_bit, set_run};
pub use boolean_builder::BooleanBuilder;
pub use primitive_builder::PrimitiveBuilder;
pub use view_builder::ViewBuilder;

/// The null bitmap of a nullable column: `1` marks a present row.
pub type ValidityBuilder = BitmapBuilder;

use crate::memory::SlabAllocator;

/// Accumulates values into engine memory and produces a finished Arrow array.
pub trait ArrayBuilder {
    /// `Default` is the element written under null slots: the slab is not
    /// zeroed on allocation, and downstream kernels (and unsafe array
    /// constructors) may touch masked slots, so they must hold a valid value.
    type Element: Copy + Default;

    /// Creates a builder pre-allocated for `capacity` elements.
    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self;

    /// Number of elements pushed so far.
    fn len(&self) -> usize;

    /// Whether no elements have been pushed yet.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends `element` repeated `amount` times (used by RLE runs).
    fn push(&mut self, element: &Self::Element, amount: usize);

    /// Returns a mutable slice of `count` uninitialised slots at the end of
    /// the buffer, advancing the length. Callers must fill every slot.
    fn spare_mut(&mut self, count: usize) -> &mut [Self::Element];

    /// Consumes the builder and returns the finished Arrow array.
    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef;
}
