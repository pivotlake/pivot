use crate::identified::Identifier;
use crate::operations::channels::{Sender, WorkerIdOutput};
use crate::operations::unary;
use crate::operations::unary::parquet::indexer::RowGroupCompressedPage;
use crate::operations::unary::{Unary, UnaryFactory};
use crate::table::RowGroupMetadataHandle;
use arrow_array::BooleanArray;
use parquet::basic::Compression;
use parquet::column::page::Page;
use parquet::file::page_index::page_metadata::{PageInfo, RowGroupPageIndex};
use parquet::{Codec, CodecOptions, create_codec};
use std::sync::Arc;

pub struct DecompressorFactory;

impl UnaryFactory<RowGroupCompressedPage, PageWithInfo> for DecompressorFactory {
    type Unary = Decompressor;

    fn build_unary(self) -> Self::Unary {
        Decompressor::new()
    }
}

pub struct Decompressor {
    codec: Option<Box<dyn Codec>>,
}

impl Decompressor {
    pub fn new() -> Self {
        Self {
            codec: create_codec(Compression::SNAPPY, &CodecOptions::default()).unwrap(),
        }
    }
}

pub struct PageWithInfo {
    pub worker_id: usize,
    pub page: Page,
    pub info: PageInfo,
    pub mask: Option<BooleanArray>,
    pub index: Arc<RowGroupPageIndex>,
    pub row_group_metadata_handle: RowGroupMetadataHandle,
}

impl WorkerIdOutput for PageWithInfo {
    fn worker_id(&self) -> Identifier {
        self.worker_id
    }
}

impl Unary<RowGroupCompressedPage, PageWithInfo> for Decompressor {
    fn consume<OP: Sender<PageWithInfo>>(
        &mut self,
        page: RowGroupCompressedPage,
        output: &mut OP,
    ) -> unary::Result<()> {
        Ok(output.send(PageWithInfo {
            worker_id: page.worker_id,
            page: page.page.decompress(&mut self.codec).unwrap(),
            info: page.page.info,
            mask: page.boolean_mask,
            index: page.index,
            row_group_metadata_handle: page.row_group_metadata_handle,
        })?)
    }
}
