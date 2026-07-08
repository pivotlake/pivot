//! The Parquet engine: two separate dataflow pipelines over the same files.
//!
//! - [`reading`] — the per-query scan: fetch a table's projected column chunks
//!   over the ring, index, decompress, and decode them into record batches.
//! - [`metadata`] — the table load, run once at `CREATE`/`ATTACH`: read every
//!   file's *footer* into the [`ParquetTable`] whose row groups all later scans
//!   reuse.
//!
//! [`types`] (the table, row-group metadata, pages, requests) and
//! [`row_group_stats`] (min/max pruning, shared by the catalog's pushdown and
//! scan-time dynamic filters) are common to both.

/// Map a Parquet-stage error (thrift parse / decompress / decode) onto
/// dispatch's generic operator error, so an operator's `?` converts into
/// `dispatch::UnaryResult` now that the pipeline lives outside dispatch.
pub(crate) fn op_err(e: impl std::error::Error + Send + Sync + 'static) -> dispatch::UnaryError {
    dispatch::UnaryError::Operator(Box::new(e))
}

/// Remote read-ahead depth: how many HTTP block reads a fetcher keeps in flight
/// per worker before admitting the next row group. The effective scan concurrency
/// against object storage is this times the worker count, so for a table of many
/// small files (round-trip bound) raising it can hide more latency. Tunable via
/// `PIVOT_HTTP_READAHEAD`; resolved once.
pub(crate) fn http_readahead() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| dispatch::env::get_env_var_with_default("PIVOT_HTTP_READAHEAD", 64))
}

mod request_tracker;

mod reading;
pub use reading::{
    DecoderFactory, DecompressorError, DecompressorFactory, IndexerFactory, MaterializerFactory,
    RowGroupDecoderError, RowGroupFetcherFactory, RowGroupInjectorFactory, ScanEqualityPredicate,
    materialize, table_input, table_input_with_filter, table_input_with_filter_and_eq_predicates,
};

mod metadata;
pub(crate) use metadata::{fetch_table_files_spec, load_table_files};

mod row_group_stats;
pub use row_group_stats::{
    RowGroupFilter, ScanOrder, row_group_eliminated, row_group_filter_from, scan_order_from,
};

pub(crate) mod types;
pub use types::arrow_map::arrow_to_parquet_physical;
pub use types::metadata::{ColumnStatistics, RowGroupMetadata};
pub use types::page::{CompressedPage, DecompressedPage};
pub use types::requests::{RowGroupBuffer, RowGroupRequest};
pub use types::table::{Error as ParquetTableError, ParquetTable};

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::parquet::types::metadata::{QueryRowGroupMetadata, RowGroupMetadata};
    use crate::parquet::types::table::ParquetTable;
    use arrow_schema::Schema;
    use dispatch::io::FileLocation;
    use std::sync::Arc;

    pub fn dummy_row_group() -> Arc<RowGroupMetadata> {
        Arc::new(RowGroupMetadata {
            location: FileLocation::Local(Arc::new(std::fs::File::open("/dev/null").unwrap())),
            schema: Arc::new(Schema::empty()),
            columns: vec![],
            num_rows: 0,
            file_row_group_idx: 0,
        })
    }

    pub fn dummy_metadata(filtered_indices: Option<Vec<u32>>) -> QueryRowGroupMetadata {
        let table = ParquetTable::new(vec![dummy_row_group()]);
        QueryRowGroupMetadata::new(&table, 0, filtered_indices)
    }
}
