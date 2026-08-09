//! Which codec a page body is compressed with.
//!
//! A codec is a bet about the bytes it is given, and a file's columns are not
//! alike. Text still holds the repetition a compressor lives on, so it pays
//! there. A column of packed differences or dictionary indices has had its
//! repetition encoded away already and comes back barely smaller, which is why
//! the codec is chosen per column rather than per file.

use std::sync::OnceLock;

use arrow_schema::DataType;
use thriftparquet::general::CompressionCodec;

use super::error::WriteResult;

/// Whether byte-array columns are written LZ4 instead of snappy
/// (`PIVOT_LZ4_TEXT_PAGES`, default off). LZ4 decodes faster than snappy on
/// bytes that compress, and no better on bytes that do not, so it is offered
/// where the values themselves are stored rather than everywhere.
fn lz4_text_pages() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| dispatch::env::get_env_var_with_default("PIVOT_LZ4_TEXT_PAGES", false))
}

/// How one leaf's pages are compressed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Compression {
    Snappy,
    /// The raw LZ4 block format, one block per page.
    Lz4Raw,
}

impl Compression {
    /// The codec a leaf of this type takes.
    ///
    /// A byte array holds the values themselves whichever encoding it took: a
    /// dictionary's distinct values, or the bytes behind packed lengths.
    /// Everything else reaching here is a number.
    pub(crate) fn for_leaf(data_type: &DataType) -> Self {
        match data_type {
            DataType::Utf8 | DataType::Utf8View | DataType::BinaryView if lz4_text_pages() => {
                Self::Lz4Raw
            }
            _ => Self::Snappy,
        }
    }

    /// The codec a column chunk's footer entry records.
    pub(crate) fn codec(self) -> CompressionCodec {
        match self {
            Self::Snappy => CompressionCodec::SNAPPY,
            Self::Lz4Raw => CompressionCodec::LZ4_RAW,
        }
    }

    /// Compress one page body.
    pub(crate) fn compress(self, raw: &[u8]) -> WriteResult<Vec<u8>> {
        Ok(match self {
            Self::Snappy => snap::raw::Encoder::new().compress_vec(raw)?,
            Self::Lz4Raw => lz4_flex::block::compress(raw),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off by default, so a table's files are snappy until someone asks
    /// otherwise. The flag is read once per process, so this covers the
    /// default rather than toggling it.
    #[test]
    fn text_takes_lz4_only_when_the_flag_is_on() {
        let text = Compression::for_leaf(&DataType::Utf8);

        assert_eq!(
            text,
            if lz4_text_pages() {
                Compression::Lz4Raw
            } else {
                Compression::Snappy
            }
        );
    }

    /// Whatever the flag says, a column of numbers stays snappy: its pages are
    /// packed differences or dictionary indices, which hold nothing for a
    /// compressor to find.
    #[test]
    fn numbers_stay_snappy() {
        assert_eq!(Compression::for_leaf(&DataType::Int64), Compression::Snappy);
        assert_eq!(
            Compression::for_leaf(&DataType::Date32),
            Compression::Snappy
        );
        assert_eq!(
            Compression::for_leaf(&DataType::Float64),
            Compression::Snappy
        );
    }

    /// Each codec's output is what its own decoder reads back, which is what
    /// the reader will be handed.
    #[test]
    fn each_codec_round_trips_a_page() {
        let raw: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();

        for compression in [Compression::Snappy, Compression::Lz4Raw] {
            let compressed = compression.compress(&raw).unwrap();

            let read = match compression {
                Compression::Snappy => snap::raw::Decoder::new().decompress_vec(&compressed),
                Compression::Lz4Raw => lz4_flex::block::decompress(&compressed, raw.len())
                    .map_err(|_| snap::Error::Header),
            };
            assert_eq!(read.unwrap(), raw, "{compression:?}");
        }
    }
}
