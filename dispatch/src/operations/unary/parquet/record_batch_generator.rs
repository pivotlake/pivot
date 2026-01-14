use crate::data_flow::WorkStatus;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::parquet::decompressor::PageWithInfo;
use crate::operations::unary::{Unary, UnaryFactory};
use crate::record_batch_metadata::with_row_group_metadata;
use crate::table::Projection;
use arrow_array::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::page_based_reader::RowGroupPageBasedReader;
use std::collections::HashMap;
use tracing::debug;

pub struct RecordBatchGeneratorFactory(Option<Projection>);

impl UnaryFactory<PageWithInfo, RecordBatch> for RecordBatchGeneratorFactory {
    type Unary = RecordBatchGenerator;

    fn build_unary(self) -> Self::Unary {
        RecordBatchGenerator::new(self.0)
    }
}

pub struct RecordBatchGenerator {
    projection: Option<Projection>,
    readers: HashMap<usize, RowGroupPageBasedReader>,
}

impl RecordBatchGenerator {
    pub fn new(projection: Option<Projection>) -> Self {
        Self {
            projection,
            readers: HashMap::new(),
        }
    }

    fn try_read_record_batch_and_output<OP: Sender<RecordBatch>>(
        &mut self,
        output: &mut OP,
    ) -> unary::Result<WorkStatus> {
        let mut did_run = false;
        let extracted: Vec<_> = self
            .readers
            .extract_if(|idx, r| {
                let offset = r.rows_read();
                let batch = r.read_record_batch();
                match batch {
                    Ok(Some(b)) => {
                        if b.num_rows() > 0 {
                            output
                                .send(with_row_group_metadata(b, *idx, offset))
                                .unwrap();
                        }
                        did_run = true;
                        false
                    }
                    Ok(None) => true,
                    Err(_e) => false,
                }
            })
            .collect();
        if extracted.len() > 0 {
            debug!(
                "Extracted {:?} ",
                extracted.into_iter().map(|(a, _b)| a).collect::<Vec<_>>()
            )
        }
        if did_run {
            return Ok(WorkStatus::Ran);
        }
        Ok(WorkStatus::Pending)
    }
}

impl Unary<PageWithInfo, RecordBatch> for RecordBatchGenerator {
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        page: PageWithInfo,
        output: &mut OP,
    ) -> unary::Result<()> {
        let reader = self
            .readers
            .entry(page.row_group_metadata_handle.index())
            .or_insert_with(|| {
                let projection = self
                    .projection
                    .clone()
                    .map(|p| {
                        ProjectionMask::leaves(
                            page.row_group_metadata_handle
                                .get()
                                .arrow_metadata
                                .parquet_schema(),
                            p.column_indices,
                        )
                    })
                    .unwrap_or_else(ProjectionMask::all);
                RowGroupPageBasedReader::try_new_with_projection(&page.index, 8192, projection)
                    .unwrap()
            });

        reader
            .push_page(
                page.info.column_idx,
                page.info.slot_idx,
                page.page,
                page.mask.as_ref(),
            )
            .unwrap();
        self.try_read_record_batch_and_output(output)?;
        Ok(())
    }

    fn run<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<WorkStatus> {
        self.try_read_record_batch_and_output(output)
    }

    fn finish<OP: Sender<RecordBatch>>(&mut self, _output: &mut OP) -> unary::Result<bool> {
        Ok(self.readers.values().all(|r| !r.has_more_rows()))
    }
}
