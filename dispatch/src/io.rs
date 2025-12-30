use crate::identified::Identifier;
use crate::table::RowGroupMetadataHandle;
use parquetd::{ParquetReader, Projection};
use std::any::Any;

/// The top level context object set as the `ParquetReader`'s `user_data`. This will be returned
/// whenever an IO op is completed and has information on which pipeline made the IO
/// request, as well which operation/input and any specific context.
pub struct PipelineIOContext {
    pub pipeline_identifier: Identifier,
    pub context: PipelineIO,
}

/// An IO request for a parquet within a pipeline, either from an operation somewhere within the
/// pipeline, or from an input
pub enum PipelineIO {
    /// An IO request from an operation.
    ///
    /// `Identifier` is the id of the operation
    /// `RowGroupMetadataHandle` is the row group to pull
    /// `Box<dyn Any>` is context the operation can provide
    /// that will be sent back together with the result of the request.
    Operation(Identifier, RowGroupMetadataHandle, Box<dyn Any>),
    /// In IO request from an input.
    ///
    /// `Identifier` is the id of the input
    /// `RowGroupMetadataHandle` is the row group to pull
    Input(Identifier, RowGroupMetadataHandle),
}

/// An IOSubmitter allows an operation to submit parquet IO operations to the ParquetReader, with the context
/// (pipeline, operation_id) so that it will be returned to the requesting operation once
/// completed (submitted via the `consume` function with the given context)
pub struct OperationIOSubmitter<'a> {
    reader: &'a mut ParquetReader<PipelineIOContext>,
    pipeline_io_counter: &'a mut usize,
    operation_id: Identifier,
    pipeline_id: Identifier,
}

impl<'a> OperationIOSubmitter<'a> {
    pub fn new(
        reader: &'a mut ParquetReader<PipelineIOContext>,
        counter: &'a mut usize,
        operation_id: Identifier,
        pipeline_id: Identifier,
    ) -> Self {
        Self {
            reader,
            pipeline_io_counter: counter,
            operation_id,
            pipeline_id,
        }
    }

    /// Submit IO for parquet to the given worker's uring. The result, once ready, will be fed
    /// back to the calling operation via the `consume` function.
    ///
    /// The `ctx` is an argument allowing the caller to provide a generic type that will be
    /// returned to it to provide context when handling the result of the IO.
    pub fn submit_operation_io(
        &mut self,
        row_group_metadata_handle: RowGroupMetadataHandle,
        projection: Option<&Projection>,
        ctx: Box<dyn Any>,
    ) -> parquetd::Result<()> {
        *self.pipeline_io_counter += 1;
        let metadata = row_group_metadata_handle.get().clone();
        self.reader.submit_projected_row_group_read(
            PipelineIOContext {
                pipeline_identifier: self.pipeline_id,
                context: PipelineIO::Operation(self.operation_id, row_group_metadata_handle, ctx),
            },
            metadata,
            projection,
        )?;
        self.reader.submit()?;
        Ok(())
    }
}
