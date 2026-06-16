//! [`Str`] — a row's string, persisted into the value arena.
//!
//! Reads the string-view column, pushes the bytes into the worker's *value*
//! arena, and yields the resulting [`ArenaKey`](crate::operations::unary::group::ArenaKey)
//! bits as the `u128` cell. Paired with [`StrMin`](super::super::fold::StrMin)/
//! [`StrMax`](super::super::fold::StrMax) this is a string `MIN`/`MAX`.

use super::Read;
use crate::operations::unary::group::arena::WorkerArena;
use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, StringViewArray};

/// Persists a row's string into the value arena, cell = its `ArenaKey` bits.
pub struct Str;

impl Read<u128> for Str {
    type Reader<'b> = &'b StringViewArray;
    #[inline(always)]
    fn make_reader(batch: &RecordBatch, column: usize) -> &StringViewArray {
        batch.column(column).as_string_view()
    }
    #[inline(always)]
    fn read(reader: &&StringViewArray, idx: usize, arena: &mut WorkerArena) -> u128 {
        arena.push(reader.value(idx)).as_u128()
    }
}
