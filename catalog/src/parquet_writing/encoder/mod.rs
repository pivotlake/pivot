//! The encoder stage: encode one column chunk into a Parquet column chunk -
//! [`dictionary`]-encoded where it pays, else [`plain`].
//!
//! A parallel 1→1 map (one [`ColumnChunkJob`] in, one [`EncodedColumnChunk`]
//! out), parallel across chunks; finished chunks route back to the worker that
//! owns their row group (`return_to_worker`, keyed by
//! [`EncodedColumnChunk::worker_id`](super::types::EncodedColumnChunk)) so a file
//! assembles in one place. The two strategies live in [`plain`] and
//! [`dictionary`]; both frame their pages with [`pages`] (which also cuts a chunk
//! into pages) - see those modules for the on-the-wire format.

mod dictionary;
mod pages;
mod plain;
mod rle;

use arrow_array::ArrayRef;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};

use super::error::{WriteError, WriteResult};
use super::types::{ColumnChunkJob, EncodedColumnChunk, EncodedPage};

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
        let (dictionary_page, data_pages) = encode_column_chunk(&job.values)?;
        sender.send(EncodedColumnChunk {
            header: job.header,
            column: job.column,
            dictionary_page,
            data_pages,
        })?;
        Ok(())
    }
}

/// Encode a column chunk, preferring a dictionary and falling back to PLAIN.
/// Returns the optional dictionary page and the data pages.
pub(in crate::parquet_writing) fn encode_column_chunk(
    values: &ArrayRef,
) -> WriteResult<(Option<EncodedPage>, Vec<EncodedPage>)> {
    // Reject nulls before choosing an encoding: the dictionary cast would
    // silently fold them into arbitrary keys, and PLAIN's own check only sees
    // its input (the dictionary path hands it the null-free distinct values).
    if values.null_count() > 0 {
        return Err(WriteError::NullsInRequiredColumn {
            nulls: values.null_count(),
        });
    }
    match dictionary::try_encode(values)? {
        Some((dictionary_page, index_page)) => Ok((Some(dictionary_page), vec![index_page])),
        None => Ok((None, plain::encode_chunk(values)?)),
    }
}
