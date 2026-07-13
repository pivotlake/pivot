//! The error type for the Parquet write pipeline.
//!
//! It converts into dispatch's `UnaryError` (see the [`From`] impl below), so a
//! helper's `?` lifts straight onto a stage's `unary` error channel with the
//! typed cause preserved rather than stringified.

use arrow_schema::{ArrowError, DataType};
use thriftparquet::parquet_thrift::ParquetError;

#[derive(Debug, thiserror::Error)]
pub(super) enum WriteError {
    /// An Arrow kernel or schema operation failed (sort, take, concat, min/max,
    /// `index_of`, building a typed scalar map, or encoding a row key).
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Serializing a page header or the file footer.
    #[error("encoding Parquet thrift: {0}")]
    Thrift(#[from] ParquetError),
    /// Snappy-compressing a page body.
    #[error("snappy-compressing a page: {0}")]
    Snappy(#[from] snap::Error),
    /// A column whose Arrow type the encoder doesn't handle.
    #[error("unsupported column type for Parquet encoding: {0:?}")]
    UnsupportedType(DataType),
    /// A catalog parquet type-mapping error, e.g. an arrow type with no Parquet
    /// physical type (from [`catalog::parquet::arrow_to_parquet_physical`]).
    #[error(transparent)]
    Catalog(#[from] crate::parquet::ParquetTableError),
    /// A row group whose column yielded no pages — an internal invariant break.
    #[error("column {column} of a row group produced no pages")]
    MissingPages { column: usize },
    /// Only required (non-null) columns are supported; this one has nulls.
    #[error("column has {nulls} null(s); only required (non-null) columns are supported")]
    NullsInRequiredColumn { nulls: usize },
    /// An array did not have the Arrow type its column's schema declared.
    #[error("array downcast to {expected} failed")]
    Downcast { expected: &'static str },
}

pub(super) type WriteResult<T> = Result<T, WriteError>;

/// Carry a `WriteError` out of a stage on dispatch's `unary` error channel as the
/// typed cause, so a helper's `?` lifts straight into a `UnaryResult` (mirrors
/// `catalog`'s reader). Uses the generic `Operator` variant — the one for
/// operators living outside dispatch.
impl From<WriteError> for dispatch::UnaryError {
    fn from(e: WriteError) -> Self {
        dispatch::UnaryError::Operator(Box::new(e))
    }
}
