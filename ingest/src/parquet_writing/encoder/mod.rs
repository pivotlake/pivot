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
//! upstream [`partition`](super::partition) breaker keeps that stage cheap and
//! serial — the levels and the encoding are computed on the parallel side.
//!
//! The two strategies live in [`plain`] and [`dictionary`]; both frame their
//! pages with [`pages`] (which also cuts a leaf into pages) — see those modules
//! for the on-the-wire format.

mod dictionary;
mod leaves;
mod pages;
mod plain;
mod rle;

use arrow_array::ArrayRef;
use arrow_schema::Field;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};
use thriftparquet::footer::Statistics;

use super::error::WriteResult;
use super::stats;
use super::types::{ColumnChunkJob, EncodedColumnChunk, EncodedLeaf};
use leaves::Leaf;

pub(super) type ColumnEncoderFactory = DefaultUnaryFactory<ColumnEncoder>;

pub(super) fn factories(worker_count: usize) -> Vec<ColumnEncoderFactory> {
    (0..worker_count)
        .map(|_| DefaultUnaryFactory::new())
        .collect()
}

#[derive(Default)]
pub(super) struct ColumnEncoder;

impl Unary<ColumnChunkJob, EncodedColumnChunk> for ColumnEncoder {
    fn consume<S: Sender<EncodedColumnChunk>>(
        &mut self,
        job: ColumnChunkJob,
        sender: &mut S,
    ) -> UnaryResult<()> {
        let field = job.header.schema.field(job.column);
        let leaves = encode_column_chunk(field, &job.values)?;
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
pub(in crate::parquet_writing) fn encode_column_chunk(
    field: &Field,
    values: &ArrayRef,
) -> WriteResult<Vec<EncodedLeaf>> {
    leaves::flatten(field, values)?
        .into_iter()
        .map(encode_leaf)
        .collect()
}

/// Encode one leaf, preferring a dictionary and falling back to PLAIN.
fn encode_leaf(leaf: Leaf) -> WriteResult<EncodedLeaf> {
    let physical_type = catalog::parquet::arrow_to_parquet_physical(leaf.values.data_type())?;
    let statistics = leaf_statistics(&leaf);
    let (dictionary_page, data_pages) = match dictionary::try_encode(&leaf)? {
        Some((dictionary_page, index_page)) => (Some(dictionary_page), vec![index_page]),
        None => (None, plain::encode_chunk(&leaf)?),
    };
    Ok(EncodedLeaf {
        path: leaf.path,
        physical_type,
        statistics,
        dictionary_page,
        data_pages,
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
