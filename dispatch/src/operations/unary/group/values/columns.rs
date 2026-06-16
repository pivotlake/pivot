//! The result value columns, shared by every accumulator-row value type.
//!
//! Output is identical across the value types — one column per slot (`v0`, `v1`,
//! …) of the accumulator's Arrow type (`Int64` for `i64`, `Decimal128(38, 0)` for
//! `i128`) — so it lives here once. A value pushes its own row in via
//! [`push_row`](RowColumns::push_row).

use super::cell::Cell;
use super::row::AggregationRow;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use arrow_array::ArrayRef;
use arrow_schema::Field;

/// One engine-slab column builder per slot.
pub struct RowColumns<const N: usize, A: Cell = i64> {
    cols: [PrimitiveBuilder<A::Arrow>; N],
}

impl<const N: usize, A: Cell> RowColumns<N, A> {
    pub fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            cols: std::array::from_fn(|_| PrimitiveBuilder::with_capacity(allocator, rows)),
        }
    }

    #[inline(always)]
    pub fn push_row(&mut self, row: &AggregationRow<N, A>) {
        for s in 0..N {
            self.cols[s].push(&row.0[s], 1);
        }
    }

    pub fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(N);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(N);
        for (s, c) in self.cols.into_iter().enumerate() {
            let array = A::finalize(c.into_array(None));
            fields.push(Field::new(
                format!("v{s}"),
                array.data_type().clone(),
                false,
            ));
            columns.push(array);
        }
        (fields, columns)
    }
}
