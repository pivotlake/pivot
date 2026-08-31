//! Which codec page bodies are compressed with.

use std::sync::OnceLock;

use crate::thrift::general::CompressionCodec;

use super::error::WriteResult;

/// The codec every written page takes (`PIVOT_PAGE_CODEC`: `snappy`, `lz4` or
/// `zstd`, default `snappy`). An unrecognized value panics rather than
/// silently writing snappy.
pub(super) fn page_codec() -> CompressionCodec {
    static CODEC: OnceLock<CompressionCodec> = OnceLock::new();
    *CODEC.get_or_init(|| match std::env::var("PIVOT_PAGE_CODEC") {
        Err(std::env::VarError::NotPresent) => CompressionCodec::SNAPPY,
        Ok(value) => match value.as_str() {
            "snappy" => CompressionCodec::SNAPPY,
            "lz4" => CompressionCodec::LZ4_RAW,
            "zstd" => CompressionCodec::ZSTD,
            other => panic!("PIVOT_PAGE_CODEC must be snappy, lz4 or zstd, got {other:?}"),
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

/// Compress one page body: LZ4 is the raw block format, one block per page;
/// zstd is one frame per page. Only reachable with a codec [`page_codec`]
/// hands out.
pub(super) fn compress(codec: CompressionCodec, raw: &[u8]) -> WriteResult<Vec<u8>> {
    Ok(match codec {
        CompressionCodec::SNAPPY => snap::raw::Encoder::new().compress_vec(raw)?,
        CompressionCodec::LZ4_RAW => lz4_flex::block::compress(raw),
        CompressionCodec::ZSTD => zstd::bulk::compress(raw, zstd_level())?,
        other => panic!("the writer has no encoder for {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The knob is read once per process, and a plain `cargo test` starts
    /// without it, so this pins the default; a run that sets the knob skips
    /// the assertion rather than restating the parser.
    #[test]
    fn page_codec_defaults_to_snappy() {
        if std::env::var_os("PIVOT_PAGE_CODEC").is_some() {
            return;
        }

        assert_eq!(page_codec(), CompressionCodec::SNAPPY);
    }

    /// Each codec's output is what its own decoder reads back, which is what
    /// the reader will be handed.
    #[test]
    fn each_codec_round_trips_a_page() {
        let raw: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();

        for codec in [
            CompressionCodec::SNAPPY,
            CompressionCodec::LZ4_RAW,
            CompressionCodec::ZSTD,
        ] {
            let compressed = compress(codec, &raw).unwrap();

            let read = match codec {
                CompressionCodec::SNAPPY => snap::raw::Decoder::new().decompress_vec(&compressed),
                CompressionCodec::LZ4_RAW => lz4_flex::block::decompress(&compressed, raw.len())
                    .map_err(|_| snap::Error::Header),
                CompressionCodec::ZSTD => {
                    zstd::bulk::decompress(&compressed, raw.len()).map_err(|_| snap::Error::Header)
                }
                other => panic!("untested codec {other}"),
            };
            assert_eq!(read.unwrap(), raw, "{codec}");
        }
    }
}
