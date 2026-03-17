//! Minimal Thrift deserialization for Parquet metadata.
//!
//! The structs and decoding logic here are largely adapted from the
//! `arrow-rs` / `parquet` crate, trimmed down to only the subset that
//! dispatch needs (footer metadata, page headers, and the compact-protocol
//! reader). This avoids pulling in the full `parquet` dependency while still
//! being able to parse file footers and page headers directly.
//!
//! - [`parquet_thrift`] — compact-protocol primitives (`ReadThrift`,
//!   `ThriftSliceInputProtocol`).
//! - [`footer`] — `FileMetaData`, `RowGroup`, `ColumnChunk`, and schema
//!   element types parsed from the file trailer.
//! - [`headers`] — `PageHeader`, `DataPageHeader`, `DictionaryPageHeader`.
//! - [`general`] — shared / general-purpose Thrift helpers.
//! - [`macros`] — internal macros used to reduce boilerplate in the Thrift
//!   struct definitions.

pub(crate) mod footer;
#[allow(clippy::upper_case_acronyms)]
pub(crate) mod general;
pub(crate) mod headers;
mod macros;
pub(crate) mod parquet_thrift;
