//! The **reading** pipeline — the per-query scan, as a dispatch dataflow:
//!
//! - [`fetching`] — the source: injects a table's row groups and reads each
//!   projected column chunk over the io_uring ring (disk or presigned HTTP).
//! - [`indexer`] — splits the fetched chunks into compressed pages.
//! - [`decompressor`] — decompresses each page.
//! - [`decoding`] — decodes pages into Arrow arrays.
//! - [`materializer`] — late materialization: re-reads surviving rows by row id.
//! - [`staged`] — staged scans: evaluates a pushed row filter on the fetched
//!   filter columns and requests the remaining columns only where rows survive.
//! - [`scan`] — the builders that chain the stages into one spec.
//!
//! It scans the row groups the [`metadata`](super::metadata) (table-load)
//! pipeline produced at `CREATE`/`ATTACH` time.

mod fetching;
pub use fetching::{RowGroupFetcherFactory, RowGroupInjectorFactory};

mod indexer;
pub use indexer::IndexerFactory;

mod decompressor;
pub use decompressor::{DecompressorFactory, Error as DecompressorError};

mod decoding;
pub use decoding::{DecoderFactory, RowGroupDecoderError, ScanEqualityPredicate};

mod materializer;
pub use materializer::MaterializerFactory;

mod staged;
pub use staged::StagingDecoderFactory;

mod record_batch_metadata;

mod scan;
pub use scan::{
    materialize, table_input, table_input_with_filter, table_input_with_filter_and_eq_predicates,
};
