use super::parquet_thrift::*;
use crate::{general_err, thrift_enum, thrift_union_all_empty};
use std::fmt;
use std::io::Write;

thrift_enum!(
/// Types supported by Parquet.
///
/// These physical types are intended to be used in combination with the encodings to
/// control the on disk storage format.
enum Type {
  BOOLEAN = 0;
  INT32 = 1;
  INT64 = 2;
  INT96 = 3;
  FLOAT = 4;
  DOUBLE = 5;
  BYTE_ARRAY = 6;
  FIXED_LEN_BYTE_ARRAY = 7;
}
);

thrift_enum!(
/// Available data pages for Parquet file format.
/// Note that some of the page types may not be supported.
enum PageType {
  DATA_PAGE = 0;
  INDEX_PAGE = 1;
  DICTIONARY_PAGE = 2;
  DATA_PAGE_V2 = 3;
}
);

thrift_enum!(
/// Compression codecs supported by the Parquet format. Every standard codec is
/// listed so any well-formed footer parses; which of them the engine can
/// actually decompress is decided by the reader.
enum CompressionCodec {
  UNCOMPRESSED = 0;
  SNAPPY = 1;
  GZIP = 2;
  LZO = 3;
  BROTLI = 4;
  /// Deprecated Hadoop-framed LZ4 (undocumented framing, writer-dependent).
  LZ4 = 5;
  ZSTD = 6;
  /// Raw LZ4 block format, one block per page.
  LZ4_RAW = 7;
}
);

thrift_union_all_empty!(
/// Time unit for `Time` and `Timestamp` logical types.
union TimeUnit {
  1: MilliSeconds MILLIS
  2: MicroSeconds MICROS
  3: NanoSeconds NANOS
}
);

thrift_enum!(
/// Encodings supported by Parquet.
/// Not all encodings are valid for all types.
enum Encoding {
  /// Default encoding.
  /// - BOOLEAN: 1 bit per value, LSB first
  /// - INT32/INT64/INT96/FLOAT/DOUBLE: plain encoding
  /// - BYTE_ARRAY/FIXED_LEN_BYTE_ARRAY: length-prefixed
  PLAIN = 0;
  /// Group VarInt encoding (deprecated, not implemented)
  GROUP_VAR_INT = 1;
  /// Deprecated: dictionary encoding (same as RLE_DICTIONARY)
  PLAIN_DICTIONARY = 2;
  /// Run-length / bit-packing hybrid encoding
  RLE = 3;
  /// Bit-packing encoding (deprecated, superseded by RLE)
  BIT_PACKED = 4;
  /// Delta encoding for integers
  DELTA_BINARY_PACKED = 5;
  /// Delta encoding for byte arrays (lengths)
  DELTA_LENGTH_BYTE_ARRAY = 6;
  /// Delta encoding for byte arrays
  DELTA_BYTE_ARRAY = 7;
  /// Dictionary encoding with RLE-encoded indices
  RLE_DICTIONARY = 8;
  /// Byte stream split encoding for floating-point data
  BYTE_STREAM_SPLIT = 9;
}
);
