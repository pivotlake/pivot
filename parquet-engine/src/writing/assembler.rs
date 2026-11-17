//! Assembles encoded leaf chunks into complete Parquet files.
//!
//! Every encoded chunk carries an assembly-worker id. Routing all chunks for a
//! file to that worker lets [`FileAssembler`] gather them without shared state.
//! It first assembles each row group, then orders the completed row groups and
//! writes the file footer. A file is emitted from `consume` as soon as its last
//! expected row group arrives.
//!
//! An [`EncodedLeafChunk`] is one primitive leaf of one row group, numbered in
//! the schema's depth-first leaf order. Leaves are written in that order, with
//! dictionary pages before data pages, and their offsets and statistics are
//! recorded in the footer.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;

use crate::thrift::footer::{
    ColumnChunk, ColumnIndex, ColumnMetaData, ColumnOrder, FileMetaData, LogicalType, OffsetIndex,
    PageLocation, RowGroup, SchemaElement,
};
use crate::thrift::general::Encoding;
use crate::thrift::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};
use arrow_schema::{DataType, FieldRef, SchemaRef};
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult};

use dispatch::memory::{FileBytes, Slab, SlabAllocator};

use super::compaction::MemoryReservation;
use super::error::{WriteError, WriteResult};
use super::types::{
    AssembledFile, EncodedLeaf, EncodedLeafChunk, FileId, RowGroupContext, RowGroupId,
    WrittenFileMetadata,
};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Parquet repetition type for a required (non-null) field.
const REPETITION_REQUIRED: i32 = 0;
/// Parquet repetition type for an optional (nullable) field — a field whose rows
/// carry definition levels.
const REPETITION_OPTIONAL: i32 = 1;
/// Parquet format version written into the footer.
const PARQUET_VERSION: i32 = 1;

pub(super) type FileAssemblerFactory = DefaultUnaryFactory<FileAssembler>;

pub(super) fn factories(worker_count: usize) -> Vec<FileAssemblerFactory> {
    DefaultUnaryFactory::create_for_workers(worker_count)
}

/// Items accumulated until an expected count is reached. The same helper is
/// used for leaves within a row group and row groups within a file.
struct Gathering<V> {
    remaining: usize,
    context: Arc<RowGroupContext>,
    items: Vec<V>,
}

impl<V> Gathering<V> {
    fn new(remaining: usize, context: Arc<RowGroupContext>) -> Self {
        Self {
            remaining,
            context,
            items: Vec::new(),
        }
    }

    /// Returns `true` when this item completes the collection.
    fn push(&mut self, item: V) -> bool {
        self.items.push(item);
        self.remaining -= 1;
        self.remaining == 0
    }
}

/// Per-worker state for row groups and files that are still being assembled.
#[derive(Default)]
pub(super) struct FileAssembler {
    chunks_by_row_group: HashMap<RowGroupId, Gathering<EncodedLeafChunk>>,
    row_groups_by_file: HashMap<FileId, Gathering<AssembledRowGroup>>,
    /// Holds the magic and footer of every file this worker assembles, so those
    /// few bytes share ring buffers instead of pinning one per piece until the
    /// file is released. Initialized on first use so an inactive assembler holds
    /// no ring buffer.
    allocator: Option<SlabAllocator>,
}

impl Unary<EncodedLeafChunk, AssembledFile> for FileAssembler {
    fn consume(
        &mut self,
        chunk: EncodedLeafChunk,
        sender: &mut dyn Sender<AssembledFile>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        // A row group is ready once one encoded chunk has arrived for every
        // primitive leaf of the schema.
        let row_group_id = chunk.context.row_group_id;
        let leaf_count = chunk.context.leaf_count;
        let context = chunk.context.clone();
        if !self
            .chunks_by_row_group
            .entry(row_group_id)
            .or_insert_with(|| Gathering::new(leaf_count, context))
            .push(chunk)
        {
            return Ok(());
        }

        let Gathering {
            context,
            items: chunks,
            ..
        } = self.chunks_by_row_group.remove(&row_group_id).unwrap();
        let group = AssembledRowGroup::new(row_group_id, chunks)?;

        let file_id = context.file_info.file_id;
        let row_group_count = context.file_info.row_group_count;
        if !self
            .row_groups_by_file
            .entry(file_id)
            .or_insert_with(|| Gathering::new(row_group_count, context))
            .push(group)
        {
            return Ok(());
        }

        let Gathering {
            context,
            items: mut groups,
            ..
        } = self.row_groups_by_file.remove(&file_id).unwrap();
        // Parallel encoding may complete row groups out of order.
        groups.sort_by_key(|group| group.row_group_id);
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let files = build_files(
            &context.schema,
            groups,
            context.file_info.max_file_size,
            allocator,
        )?;
        for (bytes, metadata) in files {
            sender.send(AssembledFile {
                bytes,
                metadata,
                partition: context.file_info.partition.clone(),
                sorted_by: context.file_info.sorted_by.clone(),
            })?;
        }
        Ok(())
    }
}

/// Assembles each row group of a merge and hands the row groups on one at a
/// time, in file order, for a writer that appends them to a file as they
/// come. Parallel encoding may complete row groups out of order, so one that
/// is ahead of its turn waits here.
#[derive(Default)]
pub(super) struct RowGroupAssembler {
    chunks_by_row_group: HashMap<RowGroupId, Gathering<EncodedLeafChunk>>,
    assembled: BTreeMap<RowGroupId, EncodedRowGroup>,
    next_row_group_id: RowGroupId,
}

pub(super) fn row_group_assembler_factories(
    worker_count: usize,
) -> Vec<DefaultUnaryFactory<RowGroupAssembler>> {
    DefaultUnaryFactory::create_for_workers(worker_count)
}

impl Unary<EncodedLeafChunk, EncodedRowGroup> for RowGroupAssembler {
    fn consume(
        &mut self,
        chunk: EncodedLeafChunk,
        sender: &mut dyn Sender<EncodedRowGroup>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let row_group_id = chunk.context.row_group_id;
        let leaf_count = chunk.context.leaf_count;
        let context = chunk.context.clone();
        if !self
            .chunks_by_row_group
            .entry(row_group_id)
            .or_insert_with(|| Gathering::new(leaf_count, context))
            .push(chunk)
        {
            return Ok(());
        }
        let Gathering {
            context,
            items: chunks,
            ..
        } = self.chunks_by_row_group.remove(&row_group_id).unwrap();
        let group = AssembledRowGroup::new(row_group_id, chunks)?;
        // The row group's rows are encoded and no longer needed decoded, so
        // the memory reserved for them is given back as `context` goes, and
        // the row group holds only what its encoded bytes take.
        let memory_reservations = context
            .memory_reservations
            .first()
            .map(|rows_memory| rows_memory.reserve_encoded_row_group(group.bytes.len()))
            .into_iter()
            .collect();
        self.assembled.insert(
            row_group_id,
            EncodedRowGroup {
                group,
                memory_reservations,
            },
        );
        while let Some(group) = self.assembled.remove(&self.next_row_group_id) {
            self.next_row_group_id += 1;
            sender.send(group)?;
        }
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<EncodedRowGroup>) -> UnaryResult<bool> {
        assert!(
            self.chunks_by_row_group.is_empty() && self.assembled.is_empty(),
            "every row group of the merge was assembled and handed on"
        );
        Ok(true)
    }
}

/// One encoded row group, complete and ready to be appended to a file through
/// a [`ParquetFileBuilder`].
pub struct EncodedRowGroup {
    group: AssembledRowGroup,
    memory_reservations: Vec<Arc<MemoryReservation>>,
}

impl EncodedRowGroup {
    /// How many bytes the row group takes in a file.
    pub fn byte_len(&self) -> usize {
        self.group.bytes.len()
    }

    /// Take the memory reserved for the row group's rows. Its writer drops
    /// it once the row group is stored.
    pub fn take_memory_reservations(&mut self) -> Vec<Arc<MemoryReservation>> {
        std::mem::take(&mut self.memory_reservations)
    }

    /// How many rows the row group holds.
    pub fn row_count(&self) -> usize {
        self.group
            .columns
            .first()
            .and_then(|chunk| chunk.meta_data.as_ref())
            .map_or(0, |meta| meta.num_values as usize)
    }
}

/// Lays a Parquet file out a row group at a time, for a writer that appends
/// each piece where the last one ended: every appended row group returns the
/// bytes that come next in the file, and finishing returns the page index and
/// the footer.
pub struct ParquetFileBuilder {
    schema: SchemaRef,
    row_groups: Vec<RowGroup>,
    /// The page index of every chunk of `row_groups`, kept until the file
    /// ends: the index is written after the last row group.
    page_indexes: Vec<Vec<ChunkPageIndex>>,
    row_count: i64,
    /// How many bytes of the file have been handed out.
    len: usize,
    row_group_bytes: usize,
}

impl ParquetFileBuilder {
    /// A file of `schema` with nothing in it yet.
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            row_groups: Vec::new(),
            page_indexes: Vec::new(),
            row_count: 0,
            len: 0,
            row_group_bytes: 0,
        }
    }

    /// How many bytes the appended row groups take, without the file's
    /// leading magic or its footer.
    pub fn row_group_bytes(&self) -> usize {
        self.row_group_bytes
    }

    /// The leading magic, when nothing of the file has been handed out.
    fn start_file(&mut self, allocator: &mut SlabAllocator) -> FileBytes {
        let mut out = FileBytes::new();
        if self.len == 0 {
            out.push_slab(run_of(PARQUET_MAGIC, allocator));
            self.len = out.len();
        }
        out
    }

    /// Add `group` as the file's next row group and return the bytes that go
    /// at the file's current end.
    pub fn append_row_group(
        &mut self,
        group: EncodedRowGroup,
        allocator: &mut SlabAllocator,
    ) -> FileBytes {
        let mut out = self.start_file(allocator);
        let group = group.group;
        let base = self.len as i64;
        self.len += group.bytes.len();
        self.row_group_bytes += group.bytes.len();
        out.append(group.bytes);

        // Rebase each column's page offsets to the file, and tally the row
        // group's size and (from any column — all cover the same rows) row count.
        let mut total_byte_size = 0i64;
        let mut group_rows = 0i64;
        let columns: Vec<ColumnChunk> = group
            .columns
            .into_iter()
            .map(|mut chunk| {
                chunk.file_offset += base;
                if let Some(meta) = chunk.meta_data.as_mut() {
                    meta.data_page_offset += base;
                    if let Some(dict_offset) = meta.dictionary_page_offset.as_mut() {
                        *dict_offset += base;
                    }
                    total_byte_size += meta.total_uncompressed_size;
                    group_rows = meta.num_values;
                }
                chunk
            })
            .collect();
        self.row_count += group_rows;
        self.page_indexes.push(
            group
                .page_indexes
                .into_iter()
                .map(|mut index| {
                    for location in &mut index.offset_index.page_locations {
                        location.offset += base;
                    }
                    index
                })
                .collect(),
        );
        self.row_groups.push(RowGroup {
            columns,
            total_byte_size,
            num_rows: group_rows,
        });
        out
    }

    /// End the file: returns the bytes that go at its current end, which
    /// close it with the page index and the footer, and the metadata just
    /// written into them, so a consumer that needs the file's row-group
    /// metadata can take it directly instead of reading it back.
    pub fn finish(
        self,
        allocator: &mut SlabAllocator,
    ) -> UnaryResult<(FileBytes, WrittenFileMetadata)> {
        Ok(self.finish_file(allocator)?)
    }

    fn finish_file(
        mut self,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<(FileBytes, WrittenFileMetadata)> {
        let mut out = self.start_file(allocator);
        let offset_indexes = self.write_page_index(&mut out, allocator)?;
        // Readers following the spec only honour a chunk's `min_value`/`max_value`
        // when the footer names an order for every leaf column.
        let schema_elements = build_schema_elements(&self.schema)?;
        let leaf_count = schema_elements
            .iter()
            .filter(|element| element.num_children.is_none())
            .count();
        let file_meta = FileMetaData {
            version: PARQUET_VERSION,
            schema: schema_elements,
            num_rows: self.row_count,
            row_groups: self.row_groups,
            created_by: Some("pivotdb".to_string()),
            column_orders: Some(vec![ColumnOrder::TYPE_ORDER; leaf_count]),
        };
        write_footer(&mut out, &file_meta, allocator)?;
        Ok((
            out,
            WrittenFileMetadata {
                footer: file_meta,
                offset_indexes,
            },
        ))
    }

    /// Append the page index of every chunk at the file's current end and
    /// point each chunk's footer entry at its part of it. The column indexes
    /// come first and the offset indexes last, so the offset indexes, which
    /// are what a reader fetching some rows of a row group needs, sit next
    /// to the footer and are read together with it. Returns the offset
    /// indexes, per row group and chunk.
    fn write_page_index(
        &mut self,
        out: &mut FileBytes,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<Vec<Vec<OffsetIndex>>> {
        let index_start = self.len;
        let mut index = Vec::new();
        let page_indexes = std::mem::take(&mut self.page_indexes);
        for (row_group, chunk_indexes) in self.row_groups.iter_mut().zip(&page_indexes) {
            for (chunk, chunk_index) in row_group.columns.iter_mut().zip(chunk_indexes) {
                let Some(column_index) = &chunk_index.column_index else {
                    continue;
                };
                let start = index.len();
                column_index.write_thrift(&mut ThriftCompactOutputProtocol::new(&mut index))?;
                chunk.column_index_offset = Some((index_start + start) as i64);
                chunk.column_index_length = Some((index.len() - start) as i32);
            }
        }
        for (row_group, chunk_indexes) in self.row_groups.iter_mut().zip(&page_indexes) {
            for (chunk, chunk_index) in row_group.columns.iter_mut().zip(chunk_indexes) {
                let start = index.len();
                chunk_index
                    .offset_index
                    .write_thrift(&mut ThriftCompactOutputProtocol::new(&mut index))?;
                chunk.offset_index_offset = Some((index_start + start) as i64);
                chunk.offset_index_length = Some((index.len() - start) as i32);
            }
        }
        self.len += index.len();
        push_runs_of(out, &index, allocator);
        Ok(page_indexes
            .into_iter()
            .map(|chunk_indexes| {
                chunk_indexes
                    .into_iter()
                    .map(|chunk_index| chunk_index.offset_index)
                    .collect()
            })
            .collect())
    }
}

/// The page index of one column chunk: where its data pages are, and their
/// value ranges when the encoder recorded them.
struct ChunkPageIndex {
    offset_index: OffsetIndex,
    column_index: Option<ColumnIndex>,
}

/// Encoded bytes and footer column metadata for one row group. Column and
/// page offsets are relative to the row group until a [`ParquetFileBuilder`]
/// rebases them.
struct AssembledRowGroup {
    row_group_id: RowGroupId,
    bytes: FileBytes,
    columns: Vec<ColumnChunk>,
    /// Each chunk's page index, in the order of `columns`.
    page_indexes: Vec<ChunkPageIndex>,
}

impl AssembledRowGroup {
    /// Writes a row group's encoded leaves in schema order.
    fn new(row_group_id: RowGroupId, mut chunks: Vec<EncodedLeafChunk>) -> WriteResult<Self> {
        // Workers may return leaves in any order.
        chunks.sort_by_key(|chunk| chunk.leaf_index);

        let mut bytes = FileBytes::new();
        let mut columns = Vec::with_capacity(chunks.len());
        let mut page_indexes = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let (column, page_index) = write_leaf_chunk(&mut bytes, chunk.leaf)?;
            columns.push(column);
            page_indexes.push(page_index);
        }
        Ok(Self {
            row_group_id,
            bytes,
            columns,
            page_indexes,
        })
    }
}

/// Build one untargeted file, or split a compaction result into the fewest
/// possible byte-balanced files at row-group boundaries. The target applies to
/// compressed row-group bodies, not Parquet headers and footers. Row groups are
/// already in global sort order, so consecutive ranges also give consecutive
/// file key ranges. An individually oversized group stays alone.
fn build_files(
    schema: &SchemaRef,
    groups: Vec<AssembledRowGroup>,
    target_size: Option<usize>,
    allocator: &mut SlabAllocator,
) -> WriteResult<Vec<(FileBytes, WrittenFileMetadata)>> {
    let ranges = match target_size {
        Some(target_size) if !groups.is_empty() => {
            let weights = groups
                .iter()
                .map(|group| group.bytes.len())
                .collect::<Vec<_>>();
            balanced_row_group_ranges(&weights, target_size)
        }
        _ => std::iter::once(0..groups.len()).collect(),
    };
    let mut groups = groups.into_iter();
    ranges
        .into_iter()
        .map(|range| {
            let file_groups = groups.by_ref().take(range.len()).collect();
            build_file(schema, file_groups, allocator)
        })
        .collect()
}

/// Balance compressed row-group bodies across the minimum number of files.
/// Capping planning weights at the target makes every oversized group fill a
/// partition by itself without letting its excess distort the other ranges.
fn balanced_row_group_ranges(weights: &[usize], target_size: usize) -> Vec<Range<usize>> {
    debug_assert!(!weights.is_empty());
    let weights = weights
        .iter()
        .map(|&weight| weight.min(target_size))
        .collect::<Vec<_>>();
    let file_count = minimum_file_count_for_target(&weights, target_size);
    linear_partitions(&weights, file_count)
}

/// Greedily count the fewest target-sized files that can hold `weights` in
/// order. Every individual weight must fit.
fn minimum_file_count_for_target(weights: &[usize], target_size: usize) -> usize {
    debug_assert!(!weights.is_empty());
    debug_assert!(weights.iter().all(|&weight| weight <= target_size));
    let mut file_count = 1;
    let mut current_size = 0usize;
    for &weight in weights {
        if current_size > 0 && current_size.saturating_add(weight) > target_size {
            file_count += 1;
            current_size = 0;
        }
        current_size += weight;
    }
    file_count
}

/// Split positive `weights` into exactly `parts` contiguous ranges while
/// minimizing the greatest range sum (the linear-partition problem).
fn linear_partitions(weights: &[usize], parts: usize) -> Vec<Range<usize>> {
    debug_assert!(!weights.is_empty());
    debug_assert!((1..=weights.len()).contains(&parts));

    // Test whether `weights` can fit into at most `parts` ranges under a
    // candidate limit. Increasing the limit cannot require more ranges, so
    // binary search finds the smallest feasible limit. If fewer than `parts`
    // ranges are needed, splitting them reaches the exact count without making
    // any range larger. For example, with two parts, `[8, 1, 1, 8]` needs three
    // ranges under a limit of 8, while a limit of 9 permits the balanced ranges
    // `[8, 1]` and `[1, 8]`.
    let mut low = *weights.iter().max().unwrap();
    let mut high = weights.iter().sum();
    while low < high {
        let limit = low + (high - low) / 2;
        let mut needed = 1usize;
        let mut current = 0usize;
        for &weight in weights {
            if current > 0 && current.saturating_add(weight) > limit {
                needed += 1;
                current = 0;
            }
            current += weight;
        }
        if needed <= parts {
            high = limit;
        } else {
            low = limit + 1;
        }
    }

    // Reconstruct from the end, filling each range to the optimal limit while
    // reserving at least one row group for every range still to its left.
    let mut reversed = Vec::with_capacity(parts);
    let mut end = weights.len();
    for remaining_parts in (2..=parts).rev() {
        let mut start = end - 1;
        let mut sum = weights[start];
        while start > remaining_parts - 1 && sum.saturating_add(weights[start - 1]) <= low {
            start -= 1;
            sum += weights[start];
        }
        reversed.push(start..end);
        end = start;
    }
    reversed.push(0..end);
    reversed.reverse();
    reversed
}

/// Stitch several assembled row groups into one Parquet file. Returns the bytes
/// and the metadata just written into them. The magic, page index and footer
/// are carved from `allocator`.
fn build_file(
    schema: &SchemaRef,
    groups: Vec<AssembledRowGroup>,
    allocator: &mut SlabAllocator,
) -> WriteResult<(FileBytes, WrittenFileMetadata)> {
    let mut builder = ParquetFileBuilder::new(schema.clone());
    let mut out = FileBytes::new();
    for group in groups {
        out.append(builder.append_row_group(
            EncodedRowGroup {
                group,
                memory_reservations: Vec::new(),
            },
            allocator,
        ));
    }
    let (footer, file_meta) = builder.finish_file(allocator)?;
    out.append(footer);
    Ok((out, file_meta))
}

/// Append one leaf's column chunk to `out`: its dictionary page (if any) followed
/// by its data pages, contiguously. Returns the chunk metadata and its page
/// index, with offsets relative to `out` (rebased to the file by
/// [`ParquetFileBuilder::append_row_group`]).
fn write_leaf_chunk(
    out: &mut FileBytes,
    leaf: EncodedLeaf,
) -> WriteResult<(ColumnChunk, ChunkPageIndex)> {
    if leaf.data_pages.is_empty() {
        return Err(WriteError::MissingPages {
            path: leaf.path.join("."),
        });
    }
    let chunk_start = out.len() as i64;
    let mut uncompressed = 0i64;

    // The dictionary page, if any, precedes the data pages.
    let dictionary_page_offset = leaf.dictionary_page.map(|dict| {
        let offset = out.len() as i64;
        uncompressed += (dict.header_len + dict.uncompressed_size) as i64;
        out.push_slab(dict.bytes);
        offset
    });

    let data_page_offset = out.len() as i64;
    let mut num_values = 0i64;
    let mut page_locations = Vec::with_capacity(leaf.data_pages.len());
    for page in leaf.data_pages {
        page_locations.push(PageLocation {
            offset: out.len() as i64,
            compressed_page_size: page.bytes.size() as i32,
            first_row_index: num_values,
        });
        num_values += page.num_rows;
        uncompressed += (page.header_len + page.uncompressed_size) as i64;
        out.push_slab(page.bytes);
    }
    let compressed = out.len() as i64 - chunk_start;

    // A dictionary chunk encodes its data pages as RLE_DICTIONARY indices and
    // its dictionary page as PLAIN; any other chunk reports the one encoding
    // its data pages carry.
    let encodings = if dictionary_page_offset.is_some() {
        vec![Encoding::PLAIN as i32, Encoding::RLE_DICTIONARY as i32]
    } else {
        vec![leaf.data_page_encoding as i32]
    };
    let chunk = ColumnChunk {
        file_offset: chunk_start,
        meta_data: Some(ColumnMetaData {
            physical_type: leaf.physical_type,
            encodings,
            path_in_schema: leaf.path,
            codec: leaf.codec,
            num_values,
            total_uncompressed_size: uncompressed,
            total_compressed_size: compressed,
            data_page_offset,
            dictionary_page_offset,
            statistics: Some(leaf.statistics),
            encoding_stats: None,
        }),
        // The index is written when the file ends, which is when its place
        // in the file is known.
        offset_index_offset: None,
        offset_index_length: None,
        column_index_offset: None,
        column_index_length: None,
    };
    let page_index = ChunkPageIndex {
        offset_index: OffsetIndex { page_locations },
        column_index: leaf.column_index,
    };
    Ok((chunk, page_index))
}

/// Build the footer schema: a root group element followed by every field's,
/// depth-first — the flat, pre-order element list Parquet stores a schema tree
/// as, and the order the reader rebuilds it from.
fn build_schema_elements(schema: &SchemaRef) -> WriteResult<Vec<SchemaElement>> {
    let mut elements = vec![SchemaElement {
        physical_type: None,
        type_length: None,
        repetition_type: None,
        name: "schema".to_string(),
        num_children: Some(schema.fields().len() as i32),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
        logical_type: None,
    }];
    for field in schema.fields() {
        push_schema_element(field, &mut elements)?;
    }
    Ok(elements)
}

/// The field id `field` declares in its metadata under
/// [`PARQUET_FIELD_ID_METADATA_KEY`](crate::PARQUET_FIELD_ID_METADATA_KEY), or
/// `None` when it declares none. A value that is not an id is an error: a
/// table format that resolves columns by id would silently stop finding the
/// column.
fn declared_field_id(field: &FieldRef) -> WriteResult<Option<i32>> {
    field
        .metadata()
        .get(crate::PARQUET_FIELD_ID_METADATA_KEY)
        .map(|value| {
            value.parse::<i32>().map_err(|_| {
                WriteError::Schema(format!(
                    "field `{}` declares the Parquet field id `{value}`, which is not an integer",
                    field.name()
                ))
            })
        })
        .transpose()
}

/// Append `field`'s schema element, and for a struct its children's after it. A
/// nullable field is OPTIONAL, which is what tells the reader its leaves carry
/// definition levels. A field id the arrow field declares in its metadata is
/// written with the element, so a reader resolving columns by id finds it.
fn push_schema_element(field: &FieldRef, elements: &mut Vec<SchemaElement>) -> WriteResult<()> {
    let repetition_type = Some(if field.is_nullable() {
        REPETITION_OPTIONAL
    } else {
        REPETITION_REQUIRED
    });
    let field_id = declared_field_id(field)?;
    match field.data_type() {
        DataType::Struct(children) => {
            elements.push(SchemaElement {
                physical_type: None,
                type_length: None,
                repetition_type,
                name: field.name().clone(),
                num_children: Some(children.len() as i32),
                converted_type: None,
                scale: None,
                precision: None,
                field_id,
                // The VARIANT annotation is the whole difference between a
                // variant column and a plain struct of binary leaves: it is what
                // a reader keys off to treat the group as semi-structured.
                logical_type: crate::is_variant_field(field).then_some(LogicalType::Variant),
            });
            for child in children {
                push_schema_element(child, elements)?;
            }
        }
        data_type => {
            // A leaf is described by its physical type plus whatever annotation
            // its type needs, which the two write-path halves of `arrow_map`
            // give together.
            let annotation = crate::arrow_to_annotation(data_type);
            elements.push(SchemaElement {
                physical_type: Some(crate::arrow_to_parquet_physical(data_type)?),
                type_length: annotation.type_length,
                repetition_type,
                name: field.name().clone(),
                num_children: None,
                converted_type: annotation.converted_type,
                scale: annotation.scale,
                precision: annotation.precision,
                field_id,
                logical_type: annotation.logical_type,
            })
        }
    }
    Ok(())
}

/// Write the trailing footer: `[FileMetaData][u32 LE footer length][PAR1]`. It
/// is one more run of the file, small enough to build in one piece.
fn write_footer(
    out: &mut FileBytes,
    file_meta: &FileMetaData,
    allocator: &mut SlabAllocator,
) -> WriteResult<()> {
    let mut footer = Vec::new();
    file_meta.write_thrift(&mut ThriftCompactOutputProtocol::new(&mut footer))?;
    footer.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    footer.extend_from_slice(PARQUET_MAGIC);
    push_runs_of(out, &footer, allocator);
    Ok(())
}

/// Append a copy of `bytes` to `out`, in as many runs as it takes: a run is
/// at most one ring buffer long.
fn push_runs_of(out: &mut FileBytes, bytes: &[u8], allocator: &mut SlabAllocator) {
    for run in bytes.chunks(dispatch::memory::BUFFER_SIZE) {
        out.push_slab(run_of(run, allocator));
    }
}

/// A run holding a copy of `bytes`, for the small pieces of a file that are not
/// pages: the leading magic, the page index and the footer.
fn run_of(bytes: &[u8], allocator: &mut SlabAllocator) -> Slab {
    let mut run = allocator.get_slab_of_size(bytes.len(), false);
    run.as_mut_slice().copy_from_slice(bytes);
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParquetTable;
    use crate::types::arrow_map::CONVERTED_DECIMAL;
    use crate::writing::encoder::encode_leaf;
    use crate::writing::leaves;
    use crate::writing::types::{EncodedLeafChunk, FileAssemblyInfo};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal64Type, Int64Type};
    use arrow_array::{Array, ArrayRef, Decimal64Array, Int64Array, RecordBatch, StructArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{Field, Fields, Schema};
    use dispatch::Dispatch;
    use dispatch::test_utils::feed_unary;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::sync::Arc;

    fn context(schema: &SchemaRef) -> Arc<RowGroupContext> {
        file_context(schema, 0)
    }

    /// The context of the only row group of file `file_id`.
    fn file_context(schema: &SchemaRef, file_id: FileId) -> Arc<RowGroupContext> {
        Arc::new(RowGroupContext {
            row_group_id: file_id,
            assembly_worker: 0,
            schema: schema.clone(),
            leaf_count: schema
                .fields()
                .iter()
                .map(|field| leaves::count_leaves(field))
                .sum(),
            file_info: Arc::new(FileAssemblyInfo {
                file_id,
                row_group_count: 1,
                partition: None,
                sorted_by: Arc::default(),
                max_file_size: None,
            }),
            memory_reservations: Vec::new(),
        })
    }

    #[test]
    fn target_ranges_balance_the_minimum_file_count() {
        let weights = [8, 1, 1, 8];

        assert_eq!(minimum_file_count_for_target(&weights, 10), 2);
        assert_eq!(balanced_row_group_ranges(&weights, 10), [0..2, 2..4]);
    }

    #[test]
    fn oversized_groups_are_singleton_files() {
        assert_eq!(
            balanced_row_group_ranges(&[4, 11, 4, 4], 10),
            [0..1, 1..2, 2..4]
        );
    }

    #[test]
    fn linear_partitioning_is_contiguous_balanced_and_exact_count() {
        let weights = [8, 1, 1, 8];
        let ranges = linear_partitions(&weights, 2);

        assert_eq!(ranges, [0..2, 2..4]);
        assert_eq!(
            ranges
                .iter()
                .map(|range| weights[range.clone()].iter().sum::<usize>())
                .collect::<Vec<_>>(),
            [9, 9]
        );

        let ranges = linear_partitions(&[1, 1, 1, 1], 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, 4);
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(ranges.iter().all(|range| !range.is_empty()));
    }

    /// A finished file stays in ring memory until it is uploaded, so the magic
    /// and footer of many small files have to share ring buffers: sixteen
    /// one-row files are all alive at once here in a ring of four buffers.
    #[test]
    fn many_small_files_stay_alive_in_a_few_ring_buffers() {
        dispatch::memory::init_test_free_pool(4);
        let field = Field::new("n", DataType::Int64, false);
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let mut page_allocator = SlabAllocator::new(false);
        let chunks: Vec<EncodedLeafChunk> = (0..16)
            .map(|file_id| {
                let values: ArrayRef = Arc::new(Int64Array::from(vec![file_id as i64]));
                let leaf = leaves::to_parquet_leaves(&field, &values)
                    .unwrap()
                    .remove(0);
                EncodedLeafChunk {
                    context: file_context(&schema, file_id),
                    leaf_index: 0,
                    leaf: encode_leaf(leaf, &mut page_allocator).unwrap(),
                }
            })
            .collect();
        let mut assembler = FileAssembler::default();

        let files: Vec<AssembledFile> = feed_unary(&mut assembler, chunks);

        assert_eq!(files.len(), 16);
    }

    /// Encode a batch the way the pipeline does: convert each column to Parquet
    /// leaves and encode them (dictionary or PLAIN, the encoder's choice), then
    /// `AssembledRowGroup::new` and `build_file`. Returns the bytes of a
    /// single-row-group file.
    fn encode(batch: &RecordBatch) -> Vec<u8> {
        // Pages are written into ring memory, so this needs a memory context of
        // its own: a thread can hold only one, and a test may also spin up a
        // Dispatch. The file is copied out before the thread ends, which is
        // where its slabs are released.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    dispatch::memory::init_test_free_pool(64);
                    let mut allocator = SlabAllocator::new(false);
                    let schema = batch.schema();
                    let context = context(&schema);
                    let mut chunks = Vec::new();
                    for (column, field) in schema.fields().iter().enumerate() {
                        for leaf in leaves::to_parquet_leaves(field, batch.column(column)).unwrap()
                        {
                            chunks.push(EncodedLeafChunk {
                                context: context.clone(),
                                leaf_index: chunks.len(),
                                leaf: encode_leaf(leaf, &mut allocator).unwrap(),
                            });
                        }
                    }
                    let group = AssembledRowGroup::new(0, chunks).unwrap();
                    build_file(&schema, vec![group], &mut allocator)
                        .unwrap()
                        .0
                        .runs()
                        .flatten()
                        .copied()
                        .collect()
                })
                .join()
                .unwrap()
        })
    }

    /// Write `batch` and read it back through arrow-rs's strict reader (our
    /// DuckDB-readability proxy).
    fn round_trip(batch: &RecordBatch) -> RecordBatch {
        let bytes = encode(batch);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");
        std::fs::write(&path, &bytes).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let read: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(read.len(), 1);
        read.into_iter().next().unwrap()
    }

    #[test]
    fn writer_records_arrow_field_ids_in_the_footer() {
        let with_id = |field: Field, id: &str| {
            field.with_metadata(std::collections::HashMap::from([(
                crate::PARQUET_FIELD_ID_METADATA_KEY.to_string(),
                id.to_string(),
            )]))
        };
        let inner = Fields::from(vec![with_id(Field::new("a", DataType::Int64, true), "7")]);
        let schema = Arc::new(Schema::new(vec![
            with_id(Field::new("n", DataType::Int64, true), "3"),
            with_id(Field::new("s", DataType::Struct(inner.clone()), true), "5"),
            Field::new("unnumbered", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(StructArray::new(
                    inner,
                    vec![Arc::new(Int64Array::from(vec![2])) as ArrayRef],
                    None,
                )),
                Arc::new(Int64Array::from(vec![3])),
            ],
        )
        .unwrap();

        let bytes = encode(&batch);

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes)).unwrap();
        let fields = reader.parquet_schema().root_schema().get_fields().to_vec();
        let ids: Vec<Option<i32>> = fields
            .iter()
            .map(|field| {
                field
                    .get_basic_info()
                    .has_id()
                    .then(|| field.get_basic_info().id())
            })
            .collect();
        assert_eq!(ids, [Some(3), Some(5), None]);
        let child = &fields[1].get_fields()[0];
        assert_eq!(child.get_basic_info().id(), 7);
    }

    #[test]
    fn a_field_id_that_is_not_an_integer_is_refused() {
        let field = Field::new("n", DataType::Int64, true).with_metadata(
            std::collections::HashMap::from([(
                crate::PARQUET_FIELD_ID_METADATA_KEY.to_string(),
                "seven".to_string(),
            )]),
        );
        let schema = Arc::new(Schema::new(vec![field]));

        let error = build_schema_elements(&schema).unwrap_err();

        assert!(matches!(error, WriteError::Schema(_)), "{error}");
    }

    /// A nullable column and a nullable struct round-trip through arrow-rs: the
    /// footer carries a real group with OPTIONAL fields, and the definition
    /// levels place the nulls back where they were. This is the shape a shredded
    /// variant is built out of.
    #[test]
    fn a_nested_nullable_column_round_trips() {
        let inner = Fields::from(vec![Field::new("a", DataType::Int64, true)]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Struct(inner.clone()), true),
        ]));
        // Row 1's struct is absent entirely; row 2 has a struct whose field is
        // null — two different depths of absence, one level apart.
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(StructArray::new(
                    inner,
                    vec![Arc::new(Int64Array::from(vec![Some(10), None, None]))],
                    Some(NullBuffer::from(vec![true, false, true])),
                )),
            ],
        )
        .unwrap();

        let got = round_trip(&batch);

        assert_eq!(
            got.column(0).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(1), None, Some(3)])
        );
        let structs = got.column(1).as_struct();
        assert!(structs.is_null(1), "row 1's struct is absent");
        assert_eq!(
            structs.column(0).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(10), None, None])
        );
    }

    /// A decimal leaf's schema element carries the full decimal description:
    /// the precision-chosen storage (INT64 here, with no type_length),
    /// precision/scale, the Decimal logical type, and the legacy DECIMAL
    /// converted type. A wide decimal stores as a fixed 16-byte array instead.
    #[test]
    fn a_decimal_column_annotates_its_schema_element() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Decimal64(10, 2), false),
            Field::new("total", DataType::Decimal128(20, 2), false),
        ]));

        let elements = build_schema_elements(&schema).unwrap();

        let narrow = &elements[1];
        assert_eq!(
            narrow.physical_type,
            Some(crate::thrift::general::Type::INT64 as i32)
        );
        assert_eq!(narrow.type_length, None);
        assert_eq!(narrow.converted_type, Some(CONVERTED_DECIMAL));
        assert_eq!(narrow.precision, Some(10));
        assert_eq!(narrow.scale, Some(2));
        assert_eq!(
            narrow.logical_type,
            Some(LogicalType::Decimal {
                scale: 2,
                precision: 10
            })
        );
        let wide = &elements[2];
        assert_eq!(
            wide.physical_type,
            Some(crate::thrift::general::Type::FIXED_LEN_BYTE_ARRAY as i32)
        );
        assert_eq!(wide.type_length, Some(16));
    }

    /// A decimal column's row-group statistics decode into bounds at the
    /// column's own type, negative values included, which is what lets a decimal
    /// predicate skip a row group.
    #[test]
    fn a_decimal_columns_statistics_decode_into_bounds() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal64(10, 2),
            false,
        )]));
        let values = Decimal64Array::from(vec![12345_i64, -67890, 100])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");
        std::fs::write(&path, encode(&batch)).unwrap();
        let dispatch = Dispatch::spin_up(2, 64, None);

        let table =
            Arc::new(ParquetTable::from_files(dispatch.dispatcher(), &[path], &[]).unwrap());

        let stats = table.row_groups()[0].column_statistics(0).unwrap();
        let min = stats.min().unwrap().into_inner();
        let max = stats.max().unwrap().into_inner();
        assert_eq!(min.as_primitive::<Decimal64Type>().value(0), -67890);
        assert_eq!(max.as_primitive::<Decimal64Type>().value(0), 12345);
        dispatch.exit();
    }
}
