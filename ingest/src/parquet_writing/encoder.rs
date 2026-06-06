//! PLAIN-encodes and snappy-compresses one page.
//!
//! A parallel 1→1 map: encode a page's column slices into one buffer,
//! snappy-compress it, and prepend the thrift page header — producing a
//! self-contained [`EncodedPage`]. The PLAIN encode is the single copy of the
//! data, and it is parallel. The stage's output channel routes each page back
//! to the worker that owns its row group (`return_to_worker`, keyed by
//! [`PipeEncodedPage::worker_id`](super::types::PipeEncodedPage)), so a row
//! group's pages reassemble in one place.
//!
//! Format: DATA_PAGE v1, PLAIN encoding, SNAPPY compression, **required**
//! columns only (no def/rep levels) — a page is just the concatenated PLAIN
//! values: fixed-width values little-endian, BYTE_ARRAY values a 4-byte LE
//! length prefix followed by the bytes.

use arrow_array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, StringViewArray,
};
use arrow_schema::DataType;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};
use snap::raw::Encoder;
use thriftparquet::general::{Encoding, PageType};
use thriftparquet::headers::{DataPageHeader, PageHeader};

use super::types::{EncodedPage, PageJob, PipeEncodedPage, PipePageJob};
use super::{to_arrow, write_thrift};

pub(super) type PageEncoderFactory = DefaultUnaryFactory<PageEncoder>;

pub(super) fn factories(worker_count: usize) -> Vec<PageEncoderFactory> {
    (0..worker_count)
        .map(|_| DefaultUnaryFactory::new())
        .collect()
}

#[derive(Default)]
pub(super) struct PageEncoder;

impl Unary<PipePageJob, PipeEncodedPage> for PageEncoder {
    fn consume<S: Sender<PipeEncodedPage>>(
        &mut self,
        pj: PipePageJob,
        sender: &mut S,
    ) -> UnaryResult<()> {
        let page = encode_page(pj.job).map_err(to_arrow)?;
        sender.send(PipeEncodedPage {
            rg_id: pj.rg_id,
            dest_worker: pj.dest_worker,
            n_pages: pj.n_pages,
            schema: pj.schema,
            page,
        })?;
        Ok(())
    }
}

/// Encode one page: PLAIN-encode its slices into one buffer, snappy-compress,
/// prepend the page header.
pub(super) fn encode_page(job: PageJob) -> Result<EncodedPage, String> {
    let mut raw = Vec::new();
    for piece in &job.pieces {
        encode_plain_into(piece.as_ref(), &mut raw)?;
    }
    let compressed = Encoder::new()
        .compress_vec(&raw)
        .map_err(|e| format!("snappy compress: {e}"))?;

    let header = PageHeader {
        r#type: PageType::DATA_PAGE,
        uncompressed_page_size: raw.len() as i32,
        compressed_page_size: compressed.len() as i32,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: job.num_rows as i32,
            encoding: Encoding::PLAIN,
            // Unused for required columns (the reader skips levels when
            // max_def_level == 0), but the fields are required.
            definition_level_encoding: Encoding::RLE,
            repetition_level_encoding: Encoding::RLE,
            statistics: None,
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    };

    let mut bytes = Vec::with_capacity(compressed.len() + 32);
    write_thrift(&header, &mut bytes)?;
    let header_len = bytes.len();
    bytes.extend_from_slice(&compressed);

    Ok(EncodedPage {
        column: job.column,
        page_index: job.page_index,
        num_rows: job.num_rows,
        uncompressed_size: raw.len(),
        header_len,
        bytes,
    })
}

/// Append a required column slice's PLAIN-encoded values to `out`.
fn encode_plain_into(array: &dyn Array, out: &mut Vec<u8>) -> Result<(), String> {
    if array.null_count() > 0 {
        return Err(format!(
            "parquet write: column has {} nulls but only required columns are supported",
            array.null_count()
        ));
    }
    let len = array.len();
    match array.data_type() {
        DataType::Int32 => {
            let a = downcast::<Int32Array>(array)?;
            out.reserve(len * 4);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Int64 => {
            let a = downcast::<Int64Array>(array)?;
            out.reserve(len * 8);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Float32 => {
            let a = downcast::<Float32Array>(array)?;
            out.reserve(len * 4);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Float64 => {
            let a = downcast::<Float64Array>(array)?;
            out.reserve(len * 8);
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }
        DataType::Utf8 => {
            let a = downcast::<StringArray>(array)?;
            (0..len).for_each(|i| encode_byte_array(a.value(i).as_bytes(), out));
        }
        DataType::Utf8View => {
            let a = downcast::<StringViewArray>(array)?;
            (0..len).for_each(|i| encode_byte_array(a.value(i).as_bytes(), out));
        }
        other => {
            return Err(format!(
                "unsupported column type for parquet write: {other:?}"
            ));
        }
    }
    Ok(())
}

fn encode_byte_array(bytes: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn downcast<A: 'static>(array: &dyn Array) -> Result<&A, String> {
    array.as_any().downcast_ref::<A>().ok_or_else(|| {
        format!(
            "parquet write: array downcast to {} failed",
            std::any::type_name::<A>()
        )
    })
}
