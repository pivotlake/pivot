use crate::operations::channels::Sender;
use crate::operations::unary::{Unary, UnaryFactory};
use crate::table::input::RowGroupBuffer;
use crate::table::{Projection, RowGroupMetadataHandle};
use crate::worker::WORKER_IDX;
use arrow_array::BooleanArray;
use arrow_buffer::BooleanBufferBuilder;
use parquet::basic::Compression;
use parquet::file::page_index::page_metadata::{CompressedPage, RowGroupPageIndex};
use parquet::{Codec, CodecOptions, create_codec};
use std::sync::Arc;

/// Filter a record batch by a list of global indexes within the row group. Every `dispatch`
/// RecordBatch should have a row index column; the `global_indexes` are meant to match that.
fn create_boolean_mask_by_global_indices(
    start_index: u32,
    end_index: u32,
    global_indexes: &[u32],
) -> (BooleanArray, bool) {
    let len = (end_index - start_index) as usize;
    let mut builder = BooleanBufferBuilder::new(len);
    builder.append_n(len, false);

    let lo = global_indexes.partition_point(|&x| x < start_index);
    let hi = global_indexes.partition_point(|&x| x < end_index);

    let mut any_true = false;

    for &idx in &global_indexes[lo..hi] {
        any_true = true;
        builder.set_bit((idx - start_index) as usize, true);
    }

    (BooleanArray::new(builder.finish(), None), any_true)
}

pub struct IndexerFactory {
    projection: Option<Projection>,
}

impl IndexerFactory {
    pub fn new(projection: Option<Projection>) -> Self {
        Self { projection }
    }
}

impl UnaryFactory<RowGroupBuffer, RowGroupCompressedPage> for IndexerFactory {
    type Unary = Indexer;

    fn build_unary(self) -> Self::Unary {
        Indexer::new(self.projection)
    }
}

pub struct Indexer {
    projection: Option<Projection>,
    codec: Option<Box<dyn Codec>>,
}

impl Indexer {
    pub fn new(projection: Option<Projection>) -> Self {
        Self {
            projection,
            codec: create_codec(Compression::SNAPPY, &CodecOptions::default()).unwrap(),
        }
    }
}

pub struct RowGroupCompressedPage {
    pub worker_id: usize,
    pub page: CompressedPage,
    pub index: Arc<RowGroupPageIndex>,
    pub row_group_metadata_handle: RowGroupMetadataHandle,
    pub boolean_mask: Option<BooleanArray>,
}

impl Unary<RowGroupBuffer, RowGroupCompressedPage> for Indexer {
    fn consume<OP: Sender<RowGroupCompressedPage>>(
        &mut self,
        buffer: RowGroupBuffer,
        output: &mut OP,
    ) -> crate::operations::unary::Result<()> {
        let row_group = buffer.metadata.get();
        let index = Arc::new(
            RowGroupPageIndex::load(
                &buffer,
                row_group.arrow_metadata.metadata().clone(),
                row_group.row_group,
                self.projection.clone().map(|p| p.column_indices),
            )
            .unwrap(),
        ); // TODO no unwrap
        let mut total = 0;
        let mut extracted: Vec<_> = index.extract_compressed_pages(&buffer.bytes).collect();
        extracted.reverse();
        for page in extracted {
            total += 1;
            let mask = if page.is_dictionary() {
                None
            } else {
                match buffer.filtered_indices.as_ref() {
                    None => None,
                    Some(f) => {
                        let (mask, _any_true) = create_boolean_mask_by_global_indices(
                            page.info.values_start_offset as u32,
                            page.info.values_end_offset as u32,
                            f,
                        );
                        // todo: We can filter out this entire page early if we see none of the values in the column will be relevant
                        // this broke the page based reader so we're not doing it for now
                        // if !any_true {
                        //     continue;
                        // }
                        Some(mask)
                    }
                }
            };
            output.send(RowGroupCompressedPage {
                worker_id: WORKER_IDX.get(),
                index: index.clone(),
                row_group_metadata_handle: buffer.metadata.clone(),
                boolean_mask: mask,
                page,
            })?;
        }
        Ok(())
    }
}
