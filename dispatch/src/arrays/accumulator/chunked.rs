//! A gather source over many at-most-fixed-size chunks, read by encoded ids.
//!
//! A join's build payload is a list of chunk batches, each holding at most
//! `1 << shift` rows, and a build row is addressed as `chunk << shift | row`
//! (a short chunk simply leaves a gap in the id space). Resolving a column
//! through `RecordBatch` accessors and a type downcast on every gathered row
//! would dominate the gather, so this prepares each column once: per chunk, the
//! raw values pointer (or view slice) and the validity, behind one enum tag per
//! column rather than per row per column.
//!
//! The prepared pointers stay valid because the source keeps the chunk batches
//! alive for its own lifetime.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_schema::DataType;

/// One column of a [`ChunkedGatherSource`], prepared for per-id reads. The
/// variants mirror the column accumulator kinds, and a column always reaches
/// the accumulator built for its type, so each accumulator sees its own
/// variant.
pub(super) enum PreparedColumn {
    FixedWidth {
        width: usize,
        /// Per chunk, the first value's address (the chunk's offset applied).
        values: Vec<*const u8>,
        nulls: Vec<Option<NullBuffer>>,
    },
    View {
        chunks: Vec<ViewChunk>,
        nulls: Vec<Option<NullBuffer>>,
    },
    Struct {
        nulls: Vec<Option<NullBuffer>>,
        children: Vec<PreparedColumn>,
    },
    /// A type without a raw fast path (bit-packed booleans, lists). Gathered
    /// through the plain arrays, which the concatenated accumulator slices.
    Whole { arrays: Vec<ArrayRef> },
}

/// One chunk of a view column: the views (offset applied) and the data
/// buffers the non-inline views point into.
pub(super) struct ViewChunk {
    pub(super) views: *const u128,
    pub(super) data_buffers: Vec<Buffer>,
}

/// A join build payload prepared for gathering output rows by encoded id.
pub struct ChunkedGatherSource {
    pub(super) columns: Vec<PreparedColumn>,
    pub(super) shift: u32,
    /// Keeps every prepared pointer alive.
    _chunks: Vec<RecordBatch>,
}

// The prepared pointers address immutable Arc-backed buffers the source keeps
// alive, so reading them from another thread is sound.
unsafe impl Send for ChunkedGatherSource {}

impl ChunkedGatherSource {
    /// Prepare `chunks` (all of one schema, each at most `1 << shift` rows)
    /// for gathering by `chunk << shift | row` ids.
    pub fn prepare(chunks: &[RecordBatch], shift: u32) -> Self {
        let column_count = chunks.first().map(|chunk| chunk.num_columns()).unwrap_or(0);
        let columns = (0..column_count)
            .map(|column| {
                prepare_column(
                    chunks
                        .iter()
                        .map(|chunk| chunk.column(column).clone())
                        .collect(),
                )
            })
            .collect();
        Self {
            columns,
            shift,
            _chunks: chunks.to_vec(),
        }
    }
}

fn prepare_column(chunk_arrays: Vec<ArrayRef>) -> PreparedColumn {
    let data_type = chunk_arrays[0].data_type().clone();
    match &data_type {
        DataType::Utf8View | DataType::BinaryView => PreparedColumn::View {
            nulls: prepare_nulls(&chunk_arrays),
            chunks: chunk_arrays
                .iter()
                .map(|array| {
                    let (views, data_buffers) = match &data_type {
                        DataType::Utf8View => {
                            let array = array.as_string_view();
                            (array.views().as_ptr(), array.data_buffers().to_vec())
                        }
                        _ => {
                            let array = array.as_binary_view();
                            (array.views().as_ptr(), array.data_buffers().to_vec())
                        }
                    };
                    ViewChunk {
                        views,
                        data_buffers,
                    }
                })
                .collect(),
        },
        DataType::Struct(fields) => PreparedColumn::Struct {
            nulls: prepare_nulls(&chunk_arrays),
            children: (0..fields.len())
                .map(|child| {
                    prepare_column(
                        chunk_arrays
                            .iter()
                            .map(|array| array.as_struct().column(child).clone())
                            .collect(),
                    )
                })
                .collect(),
        },
        other => match other.primitive_width() {
            Some(width @ (1 | 2 | 4 | 8 | 16)) => PreparedColumn::FixedWidth {
                width,
                nulls: prepare_nulls(&chunk_arrays),
                values: chunk_arrays
                    .iter()
                    .map(|array| {
                        let data = array.to_data();
                        // The pointer outlives the temporary ArrayData: it
                        // addresses the buffer allocation the array itself
                        // holds, which the source keeps alive.
                        unsafe { data.buffers()[0].as_ptr().add(data.offset() * width) }
                    })
                    .collect(),
            },
            _ => PreparedColumn::Whole {
                arrays: chunk_arrays,
            },
        },
    }
}

/// Each chunk's validity, normalized to `None` when the chunk holds no null so
/// the per-id loops keep their null-free fast path.
fn prepare_nulls(chunk_arrays: &[ArrayRef]) -> Vec<Option<NullBuffer>> {
    chunk_arrays
        .iter()
        .map(|array| {
            array
                .nulls()
                .filter(|nulls| nulls.null_count() > 0)
                .cloned()
        })
        .collect()
}
