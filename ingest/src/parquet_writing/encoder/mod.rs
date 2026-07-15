//! The encoder stage: encode one column chunk into a Parquet column chunk —
//! [`dictionary`]-encoded where it pays, else [`plain`].
//!
//! A parallel 1→1 map (one [`ColumnChunkJob`] in, one [`EncodedColumnChunk`]
//! out), parallel across chunks; finished chunks route back to the worker that
//! owns their row group (`return_to_worker`, keyed by
//! [`EncodedColumnChunk::worker_id`](super::types::EncodedColumnChunk)) so a file
//! assembles in one place. The two strategies live in [`plain`] and
//! [`dictionary`]; both frame their pages with [`pages`] (which also cuts a chunk
//! into pages) — see those modules for the on-the-wire format.

mod dictionary;
mod pages;
mod plain;
mod rle;

use arrow_array::ArrayRef;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};

use super::error::WriteResult;
use super::types::{ColumnChunkJob, DefinitionLevels, EncodedColumnChunk, EncodedPage};

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
        let (dictionary_page, data_pages) = encode_values(&job.values, &job.levels)?;
        sender.send(EncodedColumnChunk {
            header: job.header,
            leaf: job.leaf,
            column: job.column,
            field: job.field,
            path: job.path,
            dictionary_page,
            data_pages,
        })?;
        Ok(())
    }
}

/// Encode one primitive leaf, preferring a dictionary and falling back to
/// PLAIN. Definition levels describe its logical null rows separately.
pub(in crate::parquet_writing) fn encode_values(
    values: &ArrayRef,
    levels: &DefinitionLevels,
) -> WriteResult<(Option<EncodedPage>, Vec<EncodedPage>)> {
    match dictionary::try_encode(values, levels)? {
        Some((dictionary_page, index_page)) => Ok((Some(dictionary_page), vec![index_page])),
        None => Ok((None, plain::encode_chunk(values, levels)?)),
    }
}
