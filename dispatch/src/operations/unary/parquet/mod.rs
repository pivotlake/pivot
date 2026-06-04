mod indexer;
pub use indexer::IndexerFactory;

mod decompressor;
pub use decompressor::{DecompressorFactory, Error as DecompressorError};

mod decoding;
pub use decoding::{DecoderFactory, RowGroupDecoderError, ScanEqualityPredicate};

mod fetching;
pub use fetching::{RowGroupFetcherFactory, RowGroupFilter, RowGroupInjectorFactory};

pub(crate) mod types;

mod materializer;
pub use materializer::MaterializerFactory;

pub use types::page::{CompressedPage, DecompressedPage};
pub use types::requests::{RowGroupBuffer, RowGroupRequest};
pub use types::table::ParquetTable;

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::operations::unary::parquet::types::metadata::{
        QueryRowGroupMetadata, RowGroupMetadata,
    };
    use crate::operations::unary::parquet::types::table::ParquetTable;
    use arrow_schema::Schema;
    use std::sync::Arc;

    pub fn dummy_row_group() -> Arc<RowGroupMetadata> {
        Arc::new(RowGroupMetadata {
            file: Arc::new(std::fs::File::open("/dev/null").unwrap()),
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
