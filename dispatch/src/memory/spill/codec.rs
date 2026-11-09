//! Compressing rows on their way to a spill file.
//!
//! Each slab is compressed by itself, so that it can be read back and
//! decompressed without the slabs written beside it: a merge loads only the
//! slabs whose buffers were evicted. What a write gains from carrying several
//! slabs is that they go out packed together as one block-aligned stretch.

use std::cell::RefCell;
use std::sync::OnceLock;

/// What spilled rows are compressed with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillCodec {
    Lz4,
    Snappy,
    Zstd,
}

/// The codec `PIVOT_SPILL_COMPRESSION` names (`lz4`, `snappy` or `zstd`), or
/// `None` when it is unset or `none` and rows are spilled as they are. Any
/// other value stops the server instead of spilling uncompressed.
pub fn get_spill_codec() -> Option<SpillCodec> {
    static CODEC: OnceLock<Option<SpillCodec>> = OnceLock::new();
    *CODEC.get_or_init(|| {
        let codec = match std::env::var("PIVOT_SPILL_COMPRESSION") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) => match value.as_str() {
                "none" => None,
                "lz4" => Some(SpillCodec::Lz4),
                "snappy" => Some(SpillCodec::Snappy),
                "zstd" => Some(SpillCodec::Zstd),
                other => {
                    panic!(
                        "PIVOT_SPILL_COMPRESSION must be none, lz4, snappy or zstd, got {other:?}"
                    )
                }
            },
            Err(error) => panic!("PIVOT_SPILL_COMPRESSION: {error}"),
        };
        tracing::info!(?codec, "spill compression");
        codec
    })
}

/// The zstd level spilled rows are compressed at (`PIVOT_SPILL_ZSTD_LEVEL`).
/// The default is the fastest ordinary level: spilling is paid for in the
/// time the disk takes, and a level that compresses a little better for a lot
/// more CPU gives that time back.
fn get_zstd_level() -> i32 {
    static LEVEL: OnceLock<usize> = OnceLock::new();
    crate::env::get_tunable(&LEVEL, "PIVOT_SPILL_ZSTD_LEVEL", 1, |level| {
        (1..=19).contains(&level)
    }) as i32
}

thread_local! {
    static ZSTD_COMPRESSOR: RefCell<zstd::bulk::Compressor<'static>> = RefCell::new(
        zstd::bulk::Compressor::new(get_zstd_level()).expect("creating a zstd compressor"),
    );
    static ZSTD_DECOMPRESSOR: RefCell<zstd::bulk::Decompressor<'static>> = RefCell::new(
        zstd::bulk::Decompressor::new().expect("creating a zstd decompressor"),
    );
}

impl SpillCodec {
    /// The most bytes `len` bytes of rows can compress to, which is what
    /// [`compress`](Self::compress) needs room for.
    pub fn get_most_compressed_len(self, len: usize) -> usize {
        match self {
            SpillCodec::Lz4 => lz4_flex::block::get_maximum_output_size(len),
            SpillCodec::Snappy => snap::raw::max_compress_len(len),
            SpillCodec::Zstd => zstd::zstd_safe::compress_bound(len),
        }
    }

    /// Compress `rows` into the front of `compressed` and return how many
    /// bytes that took. `compressed` must have room for
    /// [`get_most_compressed_len`](Self::get_most_compressed_len) of them.
    pub fn compress(self, rows: &[u8], compressed: &mut [u8]) -> usize {
        match self {
            SpillCodec::Lz4 => lz4_flex::block::compress_into(rows, compressed)
                .expect("the output has room for the most lz4 can produce"),
            SpillCodec::Snappy => snap::raw::Encoder::new()
                .compress(rows, compressed)
                .expect("the output has room for the most snappy can produce"),
            SpillCodec::Zstd => ZSTD_COMPRESSOR.with(|compressor| {
                compressor
                    .borrow_mut()
                    .compress_to_buffer(rows, compressed)
                    .expect("the output has room for the most zstd can produce")
            }),
        }
    }

    /// Decompress `compressed`, which [`compress`](Self::compress) produced
    /// from exactly `rows.len()` bytes, into `rows`. Bytes that do not
    /// decompress to that are not what was written, and that is a panic.
    pub fn decompress(self, compressed: &[u8], rows: &mut [u8]) {
        let decompressed_len = match self {
            SpillCodec::Lz4 => lz4_flex::block::decompress_into(compressed, rows)
                .expect("spilled rows decompress with lz4 as they were compressed"),
            SpillCodec::Snappy => snap::raw::Decoder::new()
                .decompress(compressed, rows)
                .expect("spilled rows decompress with snappy as they were compressed"),
            SpillCodec::Zstd => ZSTD_DECOMPRESSOR.with(|decompressor| {
                decompressor
                    .borrow_mut()
                    .decompress_to_buffer(compressed, rows)
                    .expect("spilled rows decompress with zstd as they were compressed")
            }),
        };
        assert_eq!(
            decompressed_len,
            rows.len(),
            "spilled rows decompressed to a different length than was compressed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_gives_back_the_rows_it_was_given() {
        let rows: Vec<u8> = (0..70_000u32)
            .flat_map(|i| (i % 300).to_le_bytes())
            .collect();

        for codec in [SpillCodec::Lz4, SpillCodec::Snappy, SpillCodec::Zstd] {
            let mut compressed = vec![0u8; codec.get_most_compressed_len(rows.len())];
            let compressed_len = codec.compress(&rows, &mut compressed);
            let mut read_back = vec![0u8; rows.len()];
            codec.decompress(&compressed[..compressed_len], &mut read_back);

            assert!(compressed_len < rows.len(), "{codec:?}");
            assert_eq!(read_back, rows, "{codec:?}");
        }
    }
}
