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

/// The writer's codec policy (`PIVOT_PAGE_CODEC`, default `snappy`).
///
/// `lz4-text` spends LZ4 on byte arrays and leaves everything else snappy:
/// LZ4 decodes faster than snappy on bytes that compress, and no better on
/// bytes that do not. `zstd` compresses every leaf: its entropy coding pays
/// even on packed differences and dictionary indices, trading decode CPU for
/// a smaller file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CodecPolicy {
    Snappy,
    Lz4Text,
    Zstd,
}

fn codec_policy() -> CodecPolicy {
    static POLICY: OnceLock<CodecPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| match std::env::var("PIVOT_PAGE_CODEC") {
        Err(std::env::VarError::NotPresent) => CodecPolicy::Snappy,
        Ok(value) => match value.as_str() {
            "snappy" => CodecPolicy::Snappy,
            "lz4-text" => CodecPolicy::Lz4Text,
            "zstd" => CodecPolicy::Zstd,
            other => panic!("PIVOT_PAGE_CODEC must be snappy, lz4-text or zstd, got {other:?}"),
        },
        Err(err) => panic!("PIVOT_PAGE_CODEC: {err}"),
    })
}

/// The level passed to the zstd encoder (`PIVOT_ZSTD_LEVEL`, default the
/// library's own default, 3).
fn zstd_level() -> i32 {
    static LEVEL: OnceLock<i32> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        dispatch::env::get_env_var_with_default("PIVOT_ZSTD_LEVEL", zstd::DEFAULT_COMPRESSION_LEVEL)
    })
}

/// How one leaf's pages are compressed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Compression {
    Snappy,
    /// The raw LZ4 block format, one block per page.
    Lz4Raw,
    /// One zstd frame per page.
    Zstd,
}

impl Compression {
    /// The codec a leaf of this type takes under the process's policy.
    ///
    /// A byte array holds the values themselves whichever encoding it took: a
    /// dictionary's distinct values, or the bytes behind packed lengths.
    /// Everything else reaching here is a number.
    pub(crate) fn for_leaf(data_type: &DataType) -> Self {
        match codec_policy() {
            CodecPolicy::Snappy => Self::Snappy,
            CodecPolicy::Zstd => Self::Zstd,
            CodecPolicy::Lz4Text => match data_type {
                DataType::Utf8 | DataType::Utf8View | DataType::BinaryView => Self::Lz4Raw,
                _ => Self::Snappy,
            },
        }
    }

    /// The codec a column chunk's footer entry records.
    pub(crate) fn codec(self) -> CompressionCodec {
        match self {
            Self::Snappy => CompressionCodec::SNAPPY,
            Self::Lz4Raw => CompressionCodec::LZ4_RAW,
            Self::Zstd => CompressionCodec::ZSTD,
        }
    }

    /// Compress one page body.
    pub(crate) fn compress(self, raw: &[u8]) -> WriteResult<Vec<u8>> {
        Ok(match self {
            Self::Snappy => snap::raw::Encoder::new().compress_vec(raw)?,
            Self::Lz4Raw => lz4_flex::block::compress(raw),
            Self::Zstd => zstd::bulk::compress(raw, zstd_level())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The policy is read once per process, so this covers whichever policy
    /// the process was started with; a plain `cargo test` covers the snappy
    /// default. Under `lz4-text` only byte arrays take LZ4, since a column of
    /// numbers holds packed differences or dictionary indices, which have
    /// nothing left for LZ4 to find; zstd's entropy coding still pays there,
    /// so `zstd` takes every leaf.
    #[test]
    fn leaf_codecs_follow_the_policy() {
        let text = Compression::for_leaf(&DataType::Utf8);
        let number = Compression::for_leaf(&DataType::Int64);

        let (expected_text, expected_number) = match codec_policy() {
            CodecPolicy::Snappy => (Compression::Snappy, Compression::Snappy),
            CodecPolicy::Lz4Text => (Compression::Lz4Raw, Compression::Snappy),
            CodecPolicy::Zstd => (Compression::Zstd, Compression::Zstd),
        };
        assert_eq!(text, expected_text);
        assert_eq!(number, expected_number);
    }

    /// Each codec's output is what its own decoder reads back, which is what
    /// the reader will be handed.
    #[test]
    fn each_codec_round_trips_a_page() {
        let raw: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();

        for compression in [Compression::Snappy, Compression::Lz4Raw, Compression::Zstd] {
            let compressed = compression.compress(&raw).unwrap();

            let read = match compression {
                Compression::Snappy => snap::raw::Decoder::new().decompress_vec(&compressed),
                Compression::Lz4Raw => lz4_flex::block::decompress(&compressed, raw.len())
                    .map_err(|_| snap::Error::Header),
                Compression::Zstd => {
                    zstd::bulk::decompress(&compressed, raw.len()).map_err(|_| snap::Error::Header)
                }
            };
            assert_eq!(read.unwrap(), raw, "{compression:?}");
        }
    }
}
