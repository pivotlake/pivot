//! Minimal Thrift + Parquet metadata layer, shared by the Parquet reader and
//! the Parquet writer in this crate.
//!
//! This is the in-house alternative to pulling in the full `parquet` crate: it
//! carries just the compact-protocol codec (read and write) and the subset of
//! Parquet metadata structures (file footer, row groups, column chunks, schema
//! elements, page headers) needed to parse and emit Parquet files. The structs
//! derive both [`ReadThrift`](parquet_thrift::ReadThrift) and
//! [`WriteThrift`](parquet_thrift::WriteThrift) via the `thrift_struct!` macro,
//! so the same definitions serve both directions.
//!
//! - [`parquet_thrift`]: compact-protocol primitives:
//!   [`ReadThrift`](parquet_thrift::ReadThrift),
//!   [`WriteThrift`](parquet_thrift::WriteThrift),
//!   [`ThriftSliceInputProtocol`](parquet_thrift::ThriftSliceInputProtocol),
//!   [`ThriftCompactOutputProtocol`](parquet_thrift::ThriftCompactOutputProtocol).
//! - [`footer`]: `FileMetaData`, `RowGroup`, `ColumnChunk`, `ColumnMetaData`,
//!   `SchemaElement`, `Statistics`, `LogicalType`.
//! - [`headers`]: `PageHeader`, `DataPageHeader`, `DictionaryPageHeader`.
//! - [`general`]: shared enums (`Type`, `Encoding`, `PageType`, `Compression`, ...).
//! - `macros`: `thrift_struct!` / `thrift_enum!` / `thrift_union*!` (exported at
//!   the crate root) used to define the structs above.

pub mod footer;
#[allow(clippy::upper_case_acronyms)]
pub mod general;
pub mod headers;
mod macros;
pub mod parquet_thrift;
