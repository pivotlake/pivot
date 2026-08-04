//! Columns split across many batches, downcast once for gathering by row id.
//!
//! A join stores its build rows as a list of batches and addresses a row with
//! one `u32`: the high bits pick the batch, the low bits the row within it.
//! Gathering such rows is two indexes, but a batch's column arrives as
//! `Arc<dyn Array>`, which cannot be read without resolving its concrete
//! type, and the batch changes from row to row. The batches are immutable
//! once handed over, so each column of each batch is resolved exactly once,
//! here, and the per-row work in the accumulators is plain indexing.
//!
//! Everything stored is an owned, refcounted handle (buffers, arrays), so
//! this holds no raw pointers and keeps its sources alive by itself.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::{Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::DataType;

/// The columns of a chunked row store, each downcast once. Built by
/// [`prepare`](ChunkedColumns::prepare); read through
/// [`append_chunked_by_ids`](super::BatchAccumulator::append_chunked_by_ids).
pub struct ChunkedColumns {
    pub(super) columns: Vec<ChunkedColumn>,
    /// How a row id splits: `id >> shift` picks the batch,
    /// `id & ((1 << shift) - 1)` the row within it.
    pub(super) shift: u32,
}

/// One column, as its per-batch downcast results. The variants mirror the
/// accumulator kinds, and a column always reaches the accumulator built for
/// its type, so each accumulator sees its own variant.
pub(super) enum ChunkedColumn {
    FixedWidth {
        width: usize,
        chunks: Vec<FixedWidthChunk>,
    },
    View {
        chunks: Vec<ViewChunk>,
    },
    Struct {
        nulls: Vec<Option<NullBuffer>>,
        children: Vec<ChunkedColumn>,
    },
    /// A type without a raw fast path (bit-packed booleans, lists), gathered
    /// through the arrays themselves.
    Whole {
        chunks: Vec<ArrayRef>,
    },
}

/// One batch of a fixed-width column: its values buffer and the element
/// offset the array starts at within it.
pub(super) struct FixedWidthChunk {
    pub(super) values: Buffer,
    pub(super) offset: usize,
    pub(super) nulls: Option<NullBuffer>,
}

/// One batch of a view column: the 16-byte views and the data buffers the
/// non-inline values point into.
pub(super) struct ViewChunk {
    pub(super) views: ScalarBuffer<u128>,
    pub(super) data_buffers: Vec<Buffer>,
    pub(super) nulls: Option<NullBuffer>,
}

impl ChunkedColumns {
    /// Downcast every column of every batch, once. The batches must share a
    /// schema and each hold at most `1 << shift` rows.
    pub fn prepare(batches: &[RecordBatch], shift: u32) -> Self {
        let column_count = batches
            .first()
            .map(|batch| batch.num_columns())
            .unwrap_or(0);
        Self {
            columns: (0..column_count)
                .map(|column| {
                    prepare_column(
                        batches
                            .iter()
                            .map(|batch| batch.column(column).clone())
                            .collect(),
                    )
                })
                .collect(),
            shift,
        }
    }
}

fn prepare_column(arrays: Vec<ArrayRef>) -> ChunkedColumn {
    let data_type = arrays[0].data_type().clone();
    match &data_type {
        DataType::Utf8View | DataType::BinaryView => ChunkedColumn::View {
            chunks: arrays
                .iter()
                .map(|array| {
                    let (views, data_buffers) = match &data_type {
                        DataType::Utf8View => {
                            let array = array.as_string_view();
                            (array.views().clone(), array.data_buffers().to_vec())
                        }
                        _ => {
                            let array = array.as_binary_view();
                            (array.views().clone(), array.data_buffers().to_vec())
                        }
                    };
                    ViewChunk {
                        views,
                        data_buffers,
                        nulls: nulls_of(array),
                    }
                })
                .collect(),
        },
        DataType::Struct(fields) => ChunkedColumn::Struct {
            nulls: arrays.iter().map(nulls_of).collect(),
            children: (0..fields.len())
                .map(|child| {
                    prepare_column(
                        arrays
                            .iter()
                            .map(|array| array.as_struct().column(child).clone())
                            .collect(),
                    )
                })
                .collect(),
        },
        other => match other.primitive_width() {
            Some(width @ (1 | 2 | 4 | 8 | 16)) => ChunkedColumn::FixedWidth {
                width,
                chunks: arrays
                    .iter()
                    .map(|array| {
                        let data = array.to_data();
                        FixedWidthChunk {
                            values: data.buffers()[0].clone(),
                            offset: data.offset(),
                            nulls: nulls_of(array),
                        }
                    })
                    .collect(),
            },
            _ => ChunkedColumn::Whole { chunks: arrays },
        },
    }
}

/// A batch's validity, normalized to `None` when it holds no null so the
/// per-id loops keep their null-free fast path.
fn nulls_of(array: &ArrayRef) -> Option<NullBuffer> {
    array
        .nulls()
        .filter(|nulls| nulls.null_count() > 0)
        .cloned()
}
