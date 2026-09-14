//! Data passed between the Parquet write stages.
//!
//! Merge jobs carry file-ordering work between the node-local and global
//! stages. [`ColumnChunkJob`] then describes one top-level column of one row
//! group. The shredder splits it into one [`LeafChunkJob`] per primitive leaf,
//! the encoder turns each into an [`EncodedLeafChunk`], and the assembler
//! combines those results into a Parquet file.

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};

use crate::thrift::footer::Statistics;
use crate::thrift::general::Encoding;
use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;
use dispatch::memory::{FileBytes, Slab};
use dispatch::{
    Identifier, KWayMergeTask, LocatedBatch, MergeRun, MergedMapping, NodeIdOutput, OrderBy,
    WorkerIdOutput,
};

use crate::thrift::general::CompressionCodec;

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
    /// and footers are excluded. `None` for INSERT. An individually oversized
    /// row group remains one file.
    pub(crate) max_file_size: Option<usize>,
}

/// A complete Parquet file and the metadata needed to add it to the table.
///
/// `bytes` references worker-owned ring slabs and must be released by the
/// worker that assembled the file.
pub struct AssembledFile {
    pub bytes: FileBytes,
    /// Retained to avoid parsing the footer immediately after writing it.
    pub metadata: crate::thrift::footer::FileMetaData,
    pub partition: Option<crate::PartitionValues>,
}

/// Shared routing and schema information for every column chunk in one row
/// group.
pub(crate) struct RowGroupContext {
    pub(crate) row_group_id: RowGroupId,
    pub(crate) assembly_worker: usize,
    pub(crate) schema: SchemaRef,
    /// Primitive leaves across every column of `schema`: the number of encoded
    /// chunks that complete the row group.
    pub(crate) leaf_count: usize,
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

/// A file whose row order is final and ready for row-group planning: its
/// rows are `batches` read in `rows` order. The rows are taken a column and
/// a row group at a time, as each is encoded, never whole.
pub(crate) struct ReadyFile {
    pub(crate) plan: FilePlan,
    pub(crate) batches: Vec<LocatedBatch>,
    pub(crate) rows: FileRows,
    pub(crate) row_count: usize,
    pub(crate) target_node: usize,
}

/// The order a file's rows are read from its batches in.
pub(crate) enum FileRows {
    /// The batches as they are, one after the other.
    InOrder,
    /// The i-th row is row `.1` of batch `.0` of the i-th entry (see
    /// [`MergedMapping`](dispatch::MergedMapping)).
    Mapped(Arc<Vec<(u32, u32)>>),
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
    /// The decoded size of the file's rows, which decides whether a merge
    /// gathers them whole or orders them by mapping (see
    /// [`file_merge`](super::file_merge)).
    pub(crate) in_memory_bytes: usize,
    pub(crate) local_outputs: Box<[OnceLock<MergedMapping>]>,
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
        output: MergedMapping,
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
        output: MergedMapping,
    },
}

/// Work emitted after every participating node has completed its local merge.
pub(crate) enum GlobalMergeJob {
    Ready(ReadyFile),
    Identity {
        context: Arc<FileMergeContext>,
        output: MergedMapping,
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

/// Which rows of a file's batches make up one row group, in row order.
pub(crate) enum RowGroupRows {
    /// Runs of consecutive rows of single batches, as `(batch, rows)`: the
    /// row group is those slices of the batches, copied nowhere.
    Slices(Vec<(usize, std::ops::Range<usize>)>),
    /// Rows scattered over the batches, `rows` of the file's row order, to be
    /// gathered into a copy.
    Gather {
        mapping: Arc<Vec<(u32, u32)>>,
        rows: std::ops::Range<usize>,
    },
}

/// One top-level column of one row group, ready to be split into its primitive
/// leaves.
pub(crate) struct ColumnChunkJob {
    pub(crate) context: Arc<RowGroupContext>,
    pub(crate) column_index: usize,
    /// Where this column's leaves start in the row group's depth-first leaf
    /// numbering.
    pub(crate) first_leaf_index: usize,
    /// The column across every batch of the file, and which of their rows are
    /// the row group's. The shredder takes them on receipt and walks them a
    /// bounded number of rows per step.
    pub(crate) chunks: Arc<[ArrayRef]>,
    pub(crate) rows: Arc<RowGroupRows>,
    pub(crate) shredding: Option<Arc<arrow_schema::DataType>>,
    pub(crate) target_node: usize,
}

impl NodeIdOutput for ColumnChunkJob {
    fn node_id(&self) -> usize {
        self.target_node
    }
}

/// One primitive leaf of one row group, split out by the shredder and ready for
/// materialization and Parquet encoding.
pub(crate) struct LeafChunkJob {
    pub(crate) context: Arc<RowGroupContext>,
    /// The leaf's position in the row group's depth-first leaf numbering, which
    /// is the order the assembler lays column chunks out in.
    pub(crate) leaf_index: usize,
    /// The leaf's `path_in_schema`, from the top-level column down.
    pub(crate) path: Vec<String>,
    /// The present rows' values in row order, one chunk per shredder step. The
    /// encoder concatenates them before encoding the leaf.
    pub(crate) value_chunks: Vec<ArrayRef>,
    /// One level per row of the row group; `None` when nothing on the path is
    /// nullable.
    pub(crate) def_levels: Option<Vec<i16>>,
    pub(crate) max_def_level: i16,
    pub(crate) target_node: usize,
}

impl NodeIdOutput for LeafChunkJob {
    fn node_id(&self) -> usize {
        self.target_node
    }
}

/// One encoded Parquet page, including its Thrift header and compressed body.
pub(crate) struct EncodedPage {
    pub(crate) num_rows: i64,
    pub(crate) uncompressed_size: usize,
    pub(crate) header_len: usize,
    pub(crate) bytes: Slab,
}

/// Encoded pages and footer data for one primitive Parquet leaf.
pub(crate) struct EncodedLeaf {
    pub(crate) path: Vec<String>,
    pub(crate) physical_type: i32,
    pub(crate) statistics: Statistics,
    pub(crate) dictionary_page: Option<EncodedPage>,
    pub(crate) data_page_encoding: Encoding,
    pub(crate) data_pages: Vec<EncodedPage>,
    /// The codec this leaf's pages were compressed with, which the footer
    /// records per column chunk.
    pub(crate) codec: CompressionCodec,
}

/// One fully encoded primitive leaf of one row group, on its way to the file's
/// assembly worker.
pub(crate) struct EncodedLeafChunk {
    pub(crate) context: Arc<RowGroupContext>,
    /// See [`LeafChunkJob::leaf_index`].
    pub(crate) leaf_index: usize,
    pub(crate) leaf: EncodedLeaf,
}

impl WorkerIdOutput for EncodedLeafChunk {
    fn worker_id(&self) -> Identifier {
        self.context.assembly_worker
    }
}
