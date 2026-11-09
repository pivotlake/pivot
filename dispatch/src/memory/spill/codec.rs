//! Compressing rows on their way to a spill file.
//!
//! Each slab is compressed by itself, so that it can be read back and
//! decompressed without the slabs written beside it: a merge loads only the
//! slabs whose buffers were evicted. What a write gains from carrying several
//! slabs is that they go out packed together as one block-aligned stretch.

use std::cell::RefCell;

/// The zstd level spilled rows are compressed at, the fastest ordinary one.
/// Spilling is paid for in the time the disk takes, and a level that
/// compresses a little better for a lot more CPU gives that time back.
const ZSTD_LEVEL: i32 = 1;

thread_local! {
    static ZSTD_COMPRESSOR: RefCell<zstd::bulk::Compressor<'static>> = RefCell::new(
        zstd::bulk::Compressor::new(ZSTD_LEVEL).expect("creating a zstd compressor"),
    );
    static ZSTD_DECOMPRESSOR: RefCell<zstd::bulk::Decompressor<'static>> = RefCell::new(
        zstd::bulk::Decompressor::new().expect("creating a zstd decompressor"),
    );
}

/// The most bytes `len` bytes of rows can compress to, which is what
/// [`compress_spilled_rows`] needs room for.
pub fn get_most_compressed_len(len: usize) -> usize {
    zstd::zstd_safe::compress_bound(len)
}

/// Compress `rows` into the front of `compressed` and return how many bytes
/// that took. `compressed` must have room for [`get_most_compressed_len`] of
/// them.
pub fn compress_spilled_rows(rows: &[u8], compressed: &mut [u8]) -> usize {
    ZSTD_COMPRESSOR.with(|compressor| {
        compressor
            .borrow_mut()
            .compress_to_buffer(rows, compressed)
            .expect("the output has room for the most zstd can produce")
    })
}

/// Decompress `compressed`, which [`compress_spilled_rows`] produced from
/// exactly `rows.len()` bytes, into `rows`. Bytes that do not decompress to
/// that are not what was written, and that is a panic.
pub fn decompress_spilled_rows(compressed: &[u8], rows: &mut [u8]) {
    let decompressed_len = ZSTD_DECOMPRESSOR.with(|decompressor| {
        decompressor
            .borrow_mut()
            .decompress_to_buffer(compressed, rows)
            .expect("spilled rows decompress as they were compressed")
    });
    assert_eq!(
        decompressed_len,
        rows.len(),
        "spilled rows decompressed to a different length than was compressed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_rows_decompress_to_the_rows_they_were() {
        let rows: Vec<u8> = (0..70_000u32)
            .flat_map(|i| (i % 300).to_le_bytes())
            .collect();
        let mut compressed = vec![0u8; get_most_compressed_len(rows.len())];

        let compressed_len = compress_spilled_rows(&rows, &mut compressed);
        let mut read_back = vec![0u8; rows.len()];
        decompress_spilled_rows(&compressed[..compressed_len], &mut read_back);

        assert!(compressed_len < rows.len());
        assert_eq!(read_back, rows);
    }
}
