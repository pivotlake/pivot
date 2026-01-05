use crate::identified::Identifier;
use crate::table::RowGroupMetadataHandle;
use parquetd::Projection;
use std::any::Any;

/// The top level context object set as the `ParquetReader`'s `user_data`. This will be returned
/// whenever an IO op is completed and has information on which pipeline made the IO
/// request, as well which operation/input and any specific context.
pub struct PipelineIORequest {
    pub pipeline_identifier: Identifier,
    pub request: IORequest,
}

pub struct RowGroupFetch {
    pub row_group_metadata_handle: RowGroupMetadataHandle,
    pub projection: Option<Projection>,
}

/// An IO request for a parquet within a pipeline, either from an operation somewhere within the
/// pipeline, or from an input
pub enum IORequest {
    /// An IO request from an operation.
    ///
    /// `Identifier` is the id of the operation
    /// `RowGroupFetch` is the description of the row group to fetch and how
    /// `Box<dyn Any + Send>` is context the operation can provide
    /// that will be sent back together with the result of the request.
    Operation(Identifier, RowGroupFetch, Box<dyn Any + Send>),
    /// In IO request from an input.
    ///
    /// `Identifier` is the id of the input
    /// `RowGroupFetch` is the description of the row group to fetch and how
    Input(Identifier, RowGroupFetch),
}

impl IORequest {
    pub fn row_group_fetch(&self) -> &RowGroupFetch {
        match self {
            IORequest::Operation(_, r, _) => r,
            IORequest::Input(_, r) => r,
        }
    }
}

/// An IOSubmitter allows an operation to submit parquet IO operations to the ParquetReader, with the context
/// (pipeline, operation_id) so that it will be returned to the requesting operation once
/// completed (submitted via the `consume` function with the given context)
pub struct OperationIOSubmitter<'a> {
    pipeline_id: Identifier,
    operation_id: Identifier,
    queue: &'a mut crossbeam_deque::Worker<PipelineIORequest>,
}

impl<'a> OperationIOSubmitter<'a> {
    pub fn new(
        pipeline_id: Identifier,
        operation_id: Identifier,
        queue: &'a mut crossbeam_deque::Worker<PipelineIORequest>,
    ) -> Self {
        Self {
            pipeline_id,
            operation_id,
            queue,
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
        ctx: Box<dyn Any + Send>,
    ) {
        self.queue.push(PipelineIORequest {
            pipeline_identifier: self.pipeline_id,
            request: IORequest::Operation(
                self.operation_id,
                RowGroupFetch {
                    row_group_metadata_handle,
                    projection: projection.cloned(),
                },
                ctx,
            ),
        });
    }
}
