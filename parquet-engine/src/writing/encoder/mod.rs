//! Encodes one row-group leaf at a time as Parquet pages.
//!
//! Each [`LeafChunkJob`] becomes one [`EncodedLeafChunk`]. Jobs run in
//! parallel, then the output channel routes each result to its file's assembly
//! worker.
//!
//! The [`shredder`](super::shredder) has already split every column into its
//! primitive leaves, so the encoder's input is one leaf's values in row-ordered
//! chunks. It materializes them into ring memory and encodes the result.
//!
//! Encoding prefers a dictionary when it is beneficial, then a supported delta
//! encoding, and finally PLAIN. [`pages`] frames and compresses the output.

pub(crate) mod delta;
mod dictionary;
mod pages;
mod plain;
mod rle;

use std::sync::Arc;

use crate::thrift::footer::Statistics;
use crate::thrift::general::Encoding;
use dispatch::memory::SlabAllocator;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};

use super::compression;
use super::error::{WriteError, WriteResult};
use super::leaves::Leaf;
use super::stats;
use super::types::{EncodedLeaf, EncodedLeafChunk, LeafChunkJob};
use dispatch::arrays::concat_chunks;

pub(super) type LeafEncoderFactory = DefaultUnaryFactory<LeafEncoder>;

pub(super) fn factories(worker_count: usize) -> Vec<LeafEncoderFactory> {
    DefaultUnaryFactory::create_for_workers(worker_count)
}

#[derive(Default)]
pub(super) struct LeafEncoder {
    /// Initialized on first use so an inactive encoder holds no ring buffer.
    allocator: Option<SlabAllocator>,
}

impl Unary<LeafChunkJob, EncodedLeafChunk> for LeafEncoder {
    fn consume(
        &mut self,
        job: LeafChunkJob,
        sender: &mut dyn Sender<EncodedLeafChunk>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let values = concat_chunks(allocator, &job.value_chunks).map_err(WriteError::from)?;
        let leaf = Leaf {
            path: job.path,
            values,
            def_levels: job.def_levels.map(Arc::from),
            max_def_level: job.max_def_level,
        };
        let encoded = encode_leaf(leaf, allocator)?;
        sender.send(EncodedLeafChunk {
            context: job.context,
            leaf_index: job.leaf_index,
            leaf: encoded,
        })?;
        Ok(())
    }
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
pub(in crate::writing) fn encode_leaf(
    leaf: Leaf,
    allocator: &mut SlabAllocator,
) -> WriteResult<EncodedLeaf> {
    let physical_type = crate::arrow_to_parquet_physical(leaf.values.data_type())?;
    let codec = compression::page_codec();
    let statistics = leaf_statistics(&leaf);
    let (dictionary_page, data_page_encoding, data_pages) =
        match dictionary::try_encode(&leaf, codec, allocator)? {
            Some((dictionary_page, index_page)) => (
                Some(dictionary_page),
                Encoding::RLE_DICTIONARY,
                vec![index_page],
            ),
            None => match delta::try_encode_chunk(&leaf, codec, allocator)? {
                Some((encoding, pages)) => (None, encoding, pages),
                None => (
                    None,
                    Encoding::PLAIN,
                    plain::encode_chunk(&leaf, codec, allocator)?,
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
        codec,
    })
}

/// This leaf's footer statistics: the range of the values it stores, and how many
/// of its rows store none at all.
///
/// Every leaf gets them, which is what lets a reader prune row groups by any
/// column instead of only by the sort key. The null count is recorded even for a
/// leaf with no range to give (a leaf absent on every row has no values to take
/// one from), because it prunes on its own account: a shredded path's typed leaf
/// may only be trusted when every `value` fallback beside it is all-null, and the
/// null count is how a reader establishes that. A float leaf counts its NaNs
/// and keeps them out of its range, as the format asks.
fn leaf_statistics(leaf: &Leaf) -> Statistics {
    let (nan_count, bounds) = match stats::floating_point_statistics(&leaf.values) {
        Some((nan_count, bounds)) => (Some(nan_count), bounds),
        None => (None, stats::column_min_max(&leaf.values)),
    };
    let (min_value, max_value) = match bounds {
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
        nan_count,
    }
}
