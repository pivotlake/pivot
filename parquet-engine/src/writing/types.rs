//! Data passed between the Parquet write stages.
//!
//! Merge jobs carry file-ordering work between the node-local and global
//! stages. [`ColumnChunkJob`] then describes one top-level column of one row
//! group. The encoder turns it into [`EncodedColumnChunk`], and the assembler
//! combines those results into a Parquet file.

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;
use dispatch::memory::{FileBytes, Slab};
use dispatch::{
    Identifier, KWayMergeTask, LocatedBatch, MergeRun, MergedOutput, NodeIdOutput, OrderBy,
    WorkerIdOutput,
};
use thriftparquet::footer::Statistics;
use thriftparquet::general::Encoding;

/// Unique identity of a row group within one write pipeline.
pub(crate) type RowGroupId = u64;

/// Unique identity of an output file within one write pipeline.
pub(crate) type FileId = u64;

/// File-level information shared by all row groups during assembly.
pub(crate) struct FileAssemblyInfo {
    pub(crate) file_id: FileId,
    pub(crate) row_group_count: usize,
    pub(crate) partition: Option<crate::PartitionValues>,
    /// Compressed row-group body target for compaction output; Parquet headers
    /// and footers are excluded. `None` for INSERT and untargeted rewrites. An
    /// individually oversized row group remains one file.
    pub(crate) max_file_size: Option<usize>,
}

/// A complete Parquet file and the metadata needed to add it to the table.
///
/// `bytes` references worker-owned ring slabs and must be released by the
/// worker that assembled the file.
pub struct AssembledFile {
    pub bytes: FileBytes,
    /// Retained to avoid parsing the footer immediately after writing it.
    pub metadata: thriftparquet::footer::FileMetaData,
    pub partition: Option<crate::PartitionValues>,
}

/// Shared routing and schema information for every column chunk in one row
/// group.
pub(crate) struct RowGroupContext {
    pub(crate) row_group_id: RowGroupId,
    pub(crate) assembly_worker: usize,
    pub(crate) schema: SchemaRef,
    pub(crate) file_info: Arc<FileAssemblyInfo>,
}

/// Identities, routing, and row-group sizing assigned when a file candidate is
/// completed.
#[derive(Clone)]
pub(crate) struct FilePlan {
    pub(crate) file_id: FileId,
    pub(crate) base_row_group_id: RowGroupId,
    pub(crate) assembly_worker: usize,
    pub(crate) partition: Option<crate::PartitionValues>,
    pub(crate) target_rows_per_group: usize,
    pub(crate) max_file_size: Option<usize>,
}

/// A file whose row order is final and ready for row-group planning.
pub(crate) struct ReadyFile {
    pub(crate) plan: FilePlan,
    pub(crate) batches: Vec<LocatedBatch>,
    pub(crate) row_count: usize,
    pub(crate) target_node: usize,
}

impl NodeIdOutput for ReadyFile {
    fn node_id(&self) -> usize {
        self.target_node
    }
}

/// Shared state linking the node-local merge results for one file.
pub(crate) struct FileMergeContext {
    pub(crate) plan: FilePlan,
    pub(crate) order_by: Arc<[OrderBy]>,
    pub(crate) row_count: usize,
    pub(crate) local_outputs: Box<[OnceLock<MergedOutput>]>,
    pub(crate) nodes_remaining: AtomicUsize,
}

pub(crate) struct NodeMergeRequest {
    pub(crate) context: Arc<FileMergeContext>,
    pub(crate) node_id: usize,
    pub(crate) runs: Vec<MergeRun>,
}

/// Either an unordered file that bypasses merging or one node's sorted runs.
pub(crate) enum FileOrderInput {
    Ready(ReadyFile),
    Merge(NodeMergeRequest),
}

impl NodeIdOutput for FileOrderInput {
    fn node_id(&self) -> usize {
        match self {
            Self::Ready(file) => file.node_id(),
            Self::Merge(request) => request.node_id,
        }
    }
}

/// Work emitted by a node-local merge planner.
pub(crate) enum LocalMergeJob {
    Ready(ReadyFile),
    Identity {
        context: Arc<FileMergeContext>,
        node_id: usize,
        output: MergedOutput,
    },
    Task {
        context: Arc<FileMergeContext>,
        node_id: usize,
        task: KWayMergeTask,
    },
}

impl NodeIdOutput for LocalMergeJob {
    fn node_id(&self) -> usize {
        match self {
            Self::Ready(file) => file.node_id(),
            Self::Identity { node_id, .. } | Self::Task { node_id, .. } => *node_id,
        }
    }
}

/// One completed node-local result, or an unordered file passing through.
pub(crate) enum LocalMergeResult {
    Ready(ReadyFile),
    Merged {
        context: Arc<FileMergeContext>,
        node_id: usize,
        output: MergedOutput,
    },
}

/// Work emitted after every participating node has completed its local merge.
pub(crate) enum GlobalMergeJob {
    Ready(ReadyFile),
    Identity {
        context: Arc<FileMergeContext>,
        output: MergedOutput,
        node_id: usize,
    },
    Task {
        context: Arc<FileMergeContext>,
        task: KWayMergeTask,
    },
}

impl NodeIdOutput for GlobalMergeJob {
    fn node_id(&self) -> usize {
        match self {
            Self::Ready(file) => file.node_id(),
            Self::Identity { node_id, .. } => *node_id,
            Self::Task { task, .. } => task.node_id(),
        }
    }
}

/// One top-level column of one row group, ready for materialization and Parquet
/// encoding.
pub(crate) struct ColumnChunkJob {
    pub(crate) context: Arc<RowGroupContext>,
    pub(crate) column_index: usize,
    /// Array slices in row order. The encoder concatenates them before
    /// encoding the column chunk.
    pub(crate) batches: Arc<[ArrayRef]>,
    pub(crate) shredding: Option<Arc<arrow_schema::DataType>>,
    pub(crate) target_node: usize,
}

impl NodeIdOutput for ColumnChunkJob {
    fn node_id(&self) -> usize {
        self.target_node
    }
}

/// One encoded Parquet page, including its Thrift header and compressed body.
pub(crate) struct EncodedPage {
    pub(crate) num_rows: i64,
    pub(crate) uncompressed_size: usize,
    pub(crate) header_len: usize,
    /// The page on the wire, header followed by the compressed body, in one or
    /// more slabs (a page holding a huge single value can exceed one 2MB
    /// buffer). Every stage after this moves the slabs rather than their
    /// bytes, so this is the only place a page's bytes are written.
    pub(crate) bytes: Vec<Slab>,
}

/// Encoded pages and footer data for one primitive Parquet leaf.
pub(crate) struct EncodedLeaf {
    pub(crate) path: Vec<String>,
    pub(crate) physical_type: i32,
    pub(crate) statistics: Statistics,
    pub(crate) dictionary_page: Option<EncodedPage>,
    pub(crate) data_page_encoding: Encoding,
    pub(crate) data_pages: Vec<EncodedPage>,
}

/// One fully encoded top-level column, containing its primitive leaves in
/// schema order.
pub(crate) struct EncodedColumnChunk {
    pub(crate) context: Arc<RowGroupContext>,
    pub(crate) column_index: usize,
    pub(crate) leaves: Vec<EncodedLeaf>,
}

impl WorkerIdOutput for EncodedColumnChunk {
    fn worker_id(&self) -> Identifier {
        self.context.assembly_worker
    }
}
