//! The encoder stage: encode one column chunk into Parquet column chunks —
//! [`dictionary`]-encoded where it pays, else [`plain`].
//!
//! A parallel 1→1 map (one [`ColumnChunkJob`] in, one [`EncodedColumnChunk`]
//! out), parallel across chunks; finished chunks route back to the worker that
//! owns their row group (`return_to_worker`, keyed by
//! [`EncodedColumnChunk::worker_id`](super::types::EncodedColumnChunk)) so a file
//! assembles in one place.
//!
//! Parquet stores a chunk per *leaf*, so the job's column is first flattened
//! into its leaves ([`leaves`]) and each is encoded on its own. A flat column is
//! its own single leaf and takes the same path it always did; a shredded variant
//! yields one leaf per primitive under it. Flattening here rather than in the
//! upstream [`indexer`](super::indexer) keeps that stage cheap and serial —
//! the levels and the encoding are computed on the parallel side.
//!
//! The two strategies live in [`plain`] and [`dictionary`]; both frame their
//! pages with [`pages`] (which also cuts a leaf into pages) — see those modules
//! for the on-the-wire format.

pub(crate) mod delta;
mod dictionary;
mod leaves;
mod pages;
mod plain;
mod rle;

use arrow_array::ArrayRef;
use arrow_schema::Field;
use dispatch::memory::SlabAllocator;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};
use thriftparquet::footer::Statistics;
use thriftparquet::general::Encoding;

use super::compression::Compression;
use super::error::{WriteError, WriteResult};
use super::stats;
use super::types::{ColumnChunkJob, EncodedColumnChunk, EncodedLeaf};
use dispatch::arrays::take::concat_chunks;
use leaves::Leaf;

pub(super) type ColumnEncoderFactory = DefaultUnaryFactory<ColumnEncoder>;

pub(super) fn factories(worker_count: usize) -> Vec<ColumnEncoderFactory> {
    (0..worker_count)
        .map(|_| DefaultUnaryFactory::new())
        .collect()
}

#[derive(Default)]
pub(super) struct ColumnEncoder {
    /// The ring memory this worker's pages are written into, taken on the first
    /// page rather than when the operator is built, so a worker that encodes
    /// nothing holds no buffer.
    allocator: Option<SlabAllocator>,
}

impl Unary<ColumnChunkJob, EncodedColumnChunk> for ColumnEncoder {
    fn consume(
        &mut self,
        job: ColumnChunkJob,
        sender: &mut dyn Sender<EncodedColumnChunk>,
    ) -> UnaryResult<()> {
        let field = job.header.schema.field(job.column);
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        // The row group's rows arrive as the chunks they concatenate from:
        // materializing them is per-column work, so it parallelizes with the
        // encoding, and the stitched values are encoded while hot.
        let values = concat_chunks(allocator, &job.chunks).map_err(WriteError::from)?;
        let values = match &job.shredding {
            Some(shredding) => super::shredding::shred_gathered_column(&values, shredding)?,
            None => values,
        };
        let leaves = encode_column_chunk(field, &values, allocator)?;
        sender.send(EncodedColumnChunk {
            header: job.header,
            column: job.column,
            leaves,
        })?;
        Ok(())
    }
}

/// Encode one column's values for a row group: flatten it into leaves and encode
/// each, in the depth-first order Parquet numbers them.
pub(in crate::parquet::writing) fn encode_column_chunk(
    field: &Field,
    values: &ArrayRef,
    allocator: &mut SlabAllocator,
) -> WriteResult<Vec<EncodedLeaf>> {
    leaves::flatten(field, values)?
        .into_iter()
        .map(|leaf| encode_leaf(leaf, allocator))
        .collect()
}

/// Encode one leaf, preferring a dictionary, then a delta form, then PLAIN.
///
/// The order follows what each one costs the reader. A dictionary is best where
/// it fits: the values are stored once and the column becomes small integers,
/// which also lets a reader prune a row group by comparing against the
/// dictionary alone. Where it does not fit, the leaf used to fall to PLAIN and
/// pay the full width per value; a delta form instead packs the differences to
/// the width they need, which is most of the size of a key column. Floats, and
/// decimals too wide to store as an integer, have no delta form and still take
/// PLAIN.
fn encode_leaf(leaf: Leaf, allocator: &mut SlabAllocator) -> WriteResult<EncodedLeaf> {
    let physical_type = crate::parquet::arrow_to_parquet_physical(leaf.values.data_type())?;
    let compression = Compression::for_leaf(leaf.values.data_type());
    let statistics = leaf_statistics(&leaf);
    let (dictionary_page, data_page_encoding, data_pages) =
        match dictionary::try_encode(&leaf, compression, allocator)? {
            Some((dictionary_page, index_page)) => (
                Some(dictionary_page),
                Encoding::RLE_DICTIONARY,
                vec![index_page],
            ),
            None => match delta::try_encode_chunk(&leaf, compression, allocator)? {
                Some((encoding, pages)) => (None, encoding, pages),
                None => (
                    None,
                    Encoding::PLAIN,
                    plain::encode_chunk(&leaf, compression, allocator)?,
                ),
            },
        };
    Ok(EncodedLeaf {
        path: leaf.path,
        physical_type,
        statistics,
        dictionary_page,
        data_page_encoding,
        data_pages,
        compression,
    })
}

/// This leaf's footer statistics: the range of the values it stores, and how many
/// of its rows store none at all.
///
/// Every leaf gets them, which is what lets a reader prune row groups by any
/// column instead of only by the sort key. The null count is recorded even for a
/// leaf with no range to give — a leaf absent on every row has no values to take
/// one from — because it prunes on its own account: a shredded path's typed leaf
/// may only be trusted when every `value` fallback beside it is all-null, and the
/// null count is how a reader establishes that.
fn leaf_statistics(leaf: &Leaf) -> Statistics {
    let (min_value, max_value) = match stats::column_min_max(&leaf.values) {
        Some((min, max)) => (stats::stat_bytes(&min), stats::stat_bytes(&max)),
        None => (None, None),
    };
    Statistics {
        // The deprecated pair, superseded by `min_value`/`max_value`.
        min: None,
        max: None,
        // Absent rows: the leaf spans every row, but stores only the present.
        null_count: Some((leaf.rows() - leaf.values.len()) as i64),
        distinct_count: None,
        min_value,
        max_value,
    }
}
