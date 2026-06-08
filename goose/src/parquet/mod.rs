/// Map a Parquet-stage error (thrift parse / decompress / decode) onto
/// dispatch's generic operator error, so an operator's `?` converts into
/// `dispatch::UnaryResult` now that the pipeline lives outside dispatch.
pub(crate) fn op_err(e: impl std::error::Error + Send + Sync + 'static) -> dispatch::UnaryError {
    dispatch::UnaryError::Operator(Box::new(e))
}

mod record_batch_metadata;

mod row_group_stats;
pub use row_group_stats::{
    RowGroupFilter, ScanOrder, row_group_eliminated, row_group_filter_from, scan_order_from,
};

mod indexer;
pub use indexer::IndexerFactory;

mod decompressor;
pub use decompressor::{DecompressorFactory, Error as DecompressorError};

mod decoding;
pub use decoding::{DecoderFactory, RowGroupDecoderError, ScanEqualityPredicate};

mod fetching;
pub use fetching::{
    ParquetSource, RowGroupFetcherFactory, RowGroupInjectorFactory, materialize_metadata,
};

pub(crate) mod types;

mod materializer;
pub use materializer::MaterializerFactory;

mod scan;
pub use scan::{
    materialize, table_input, table_input_with_filter, table_input_with_filter_and_eq_predicates,
};

pub use types::metadata::{ColumnStatistics, RowGroupMetadata};
pub use types::page::{CompressedPage, DecompressedPage};
pub use types::requests::{RowGroupBuffer, RowGroupRequest};
pub use types::table::{DataFileLocation, Error as ParquetTableError, ParquetTable};

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::parquet::types::metadata::{FileSource, QueryRowGroupMetadata, RowGroupMetadata};
    use crate::parquet::types::table::ParquetTable;
    use arrow_schema::Schema;
    use std::sync::Arc;

    pub fn dummy_row_group() -> Arc<RowGroupMetadata> {
        Arc::new(RowGroupMetadata {
            source: FileSource::Local(Arc::new(std::fs::File::open("/dev/null").unwrap())),
            schema: Arc::new(Schema::empty()),
            columns: vec![],
            num_rows: 0,
            file_row_group_idx: 0,
            global_row_group_idx: 0,
        })
    }

    pub fn dummy_metadata(filtered_indices: Option<Vec<u32>>) -> QueryRowGroupMetadata {
        let table = ParquetTable::new(vec![dummy_row_group()]);
        QueryRowGroupMetadata::new(&table, 0, filtered_indices)
    }
}
