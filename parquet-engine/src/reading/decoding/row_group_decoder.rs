//! Per-row-group decoder that decodes the ranges of a row group into Arrow
//! [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) keeps one [`RowGroupDecoder`] per
//! row group it has decoded rows of. A [`DecodeRange`] that starts where the
//! decoder stopped goes on from the page it has open; any other range
//! repositions it, which costs a skip into the range's first page of every
//! column.
//!
//! A pushed-down comparison of a plain column gates the decode of the rest:
//! each batch decodes the compared leaves first, and where the comparisons
//! reject most of its rows the other leaves decode only the rows that pass.

use crate::reading::decoding::ScanComparison;
use crate::reading::decoding::column_decoder::{ColumnDecoder, Error, Result, create_leaf_decoder};
use crate::reading::decoding::leaf_decoders::{LeafDecoder, take_rows};
use crate::reading::range_cutter::row_group_pages::{
    DecodeRange, LeafCursors, PageContent, StoredPage,
};
use crate::reading::record_batch_metadata::with_row_index;
use crate::types::filter_mask::FilterMask;
use crate::types::leaves::{leaf_fields, plan_leaves, resolve_output_reads};
use crate::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata, RowSelection};
use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};
use crate::types::projection::Projection;
use arrow_array::{Array, ArrayRef, Datum, RecordBatch, Scalar};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_ord::cmp;
use arrow_schema::{DataType, FieldRef, Fields, Schema, SchemaRef};
use dispatch::memory::SlabAllocator;
use planner::expression::CompareType;
use std::cmp::min;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// An output column whose pushed-down equality constant can prune the row
/// group, and the decoded leaf that constant was installed on.
#[derive(Clone, Copy)]
pub(crate) struct PrunableColumn {
    /// The column's position in the emitted batch.
    pub output_idx: usize,
    /// The leaf whose dictionary decides the pruning.
    pub leaf: usize,
}

/// A pushed-down comparison evaluated on a decoded leaf, so that the row
/// group's other leaves decode only the rows it passes.
pub(crate) struct Gate {
    /// The decoded leaf the comparison reads.
    leaf: usize,
    operator: CompareType,
    /// The constant compared against, of the leaf's own type.
    value: Scalar<ArrayRef>,
}

impl Gate {
    /// A bit per row of `leaf_rows`, set where the comparison holds. A null
    /// passes no comparison.
    fn find_passing_rows(&self, leaf_rows: &ArrayRef) -> BooleanBuffer {
        let compare = match self.operator {
            CompareType::Equal => cmp::eq,
            CompareType::NotEqual => cmp::neq,
            CompareType::Less => cmp::lt,
            CompareType::Greater => cmp::gt,
            CompareType::LessEqual => cmp::lt_eq,
            CompareType::GreaterEqual => cmp::gt_eq,
        };
        let passes =
            compare(leaf_rows, &self.value).expect("a gate's constant has its leaf's type");
        match passes.nulls() {
            Some(nulls) => passes.values() & nulls.inner(),
            None => passes.values().clone(),
        }
    }
}

/// Whether a comparison of `leaf_type` values means to the decoder what it
/// means to the query's filter, which decides the rows for real: integers,
/// dates, timestamps and strings order one way everywhere.
fn compares_like_the_query(leaf_type: &DataType) -> bool {
    leaf_type.is_integer()
        || matches!(
            leaf_type,
            DataType::Boolean
                | DataType::Date32
                | DataType::Timestamp(_, _)
                | DataType::Utf8View
                | DataType::BinaryView
        )
}

/// The most of a batch's rows, in percent, its gates may pass for the other
/// leaves to decode by selection. A batch that keeps more decodes whole:
/// picking out most of the rows costs more than decoding all of them.
const MOST_PASSING_PERCENT_TO_SELECT: usize = 35;

/// How a row group's projected columns decode: which leaves are read, one
/// view per output column, and the leaves whose dictionaries can prune the
/// row group. Built once per row group and shared by every decoder of its
/// rows; each decoder creates its own leaf decoders from it.
pub(crate) struct DecodePlan {
    /// The fields of the leaves read, in the order the fetcher requested
    /// them, so a page's `column_idx` indexes straight into this.
    leaf_fields: Vec<FieldRef>,
    /// The file leaves the decoders read, in the same order.
    pub file_leaves: Vec<usize>,
    /// One view per output column, in projection order. Several can fold the
    /// same decoded leaf.
    column_decoders: Vec<ColumnDecoder>,
    /// Output schema (projected). A pushed-down extract column carries the
    /// type the extract emits, not the variant it was read from.
    schema: SchemaRef,
    pub prunable: Vec<PrunableColumn>,
    /// The pushed-down equality constant each prunable leaf's decoder gets.
    eq_constants: Vec<(usize, Scalar<ArrayRef>)>,
    /// The pushed-down comparisons the decoders evaluate themselves.
    gates: Vec<Gate>,
}

impl DecodePlan {
    pub fn new(
        row_group_metadata: &QueryRowGroupMetadata,
        projection: &Projection,
        comparisons: &[ScanComparison],
    ) -> Result<Self> {
        // Expand projected columns into this file's leaves. Variant layouts can
        // differ between files, so each row group resolves them independently,
        // and the fetcher resolved them the same way.
        let fields = row_group_metadata.get_metadata().schema.fields();
        let leaves = leaf_fields(fields);
        let reads = resolve_output_reads(fields, row_group_metadata, projection);
        let plan = plan_leaves(&reads);

        let mut column_decoders = Vec::with_capacity(projection.column_indices.len());
        for ((output_idx, &column), positions) in projection
            .column_indices
            .iter()
            .enumerate()
            .zip(plan.output_positions)
        {
            column_decoders.push(ColumnDecoder::new(
                column,
                projection.extract_at(output_idx),
                &reads[output_idx],
                positions,
                &leaves,
                fields,
            )?);
        }

        let (prunable, eq_constants) = prunable_columns(
            &column_decoders,
            &plan.file_leaves,
            row_group_metadata.columns(),
            projection,
            comparisons,
        );
        let leaf_fields: Vec<FieldRef> = plan
            .file_leaves
            .iter()
            .map(|&leaf| leaves[leaf].clone())
            .collect();
        let gates = plan_gates(&column_decoders, &leaf_fields, projection, comparisons);

        let output_fields: Fields = column_decoders
            .iter()
            .map(|decoder| decoder.output_field().clone())
            .collect();
        Ok(Self {
            leaf_fields,
            file_leaves: plan.file_leaves,
            column_decoders,
            schema: Arc::new(Schema::new(output_fields)),
            prunable,
            eq_constants,
            gates,
        })
    }

    /// How many leaves are decoded.
    pub fn leaf_count(&self) -> usize {
        self.file_leaves.len()
    }

    /// Fresh leaf decoders, one per decoded leaf, with the pushed-down
    /// constants installed on the prunable ones.
    pub fn create_leaf_decoders(
        &self,
        row_group_metadata: &QueryRowGroupMetadata,
    ) -> Result<Vec<Box<dyn LeafDecoder>>> {
        let column_chunks = row_group_metadata.columns();
        let mut leaf_decoders = self
            .leaf_fields
            .iter()
            .zip(&self.file_leaves)
            .map(|(field, &leaf)| create_leaf_decoder(field.data_type(), &column_chunks[leaf]))
            .collect::<Result<Vec<_>>>()?;
        for (leaf, constant) in &self.eq_constants {
            leaf_decoders[*leaf].set_eq_constant(constant);
        }
        Ok(leaf_decoders)
    }
}

/// A batch of a row group's rows, and, when the table format deleted any of
/// them, a bit per row of the batch set for the rows it left live.
pub struct DecodedBatch {
    pub batch: RecordBatch,
    pub live_rows: Option<BooleanBuffer>,
}

/// Decodes the ranges of a single row group into [`RecordBatch`]es.
pub struct RowGroupDecoder {
    metadata: QueryRowGroupMetadata,
    plan: Arc<DecodePlan>,
    leaf_decoders: Vec<Box<dyn LeafDecoder>>,
    /// Max rows per batch.
    batch_size: usize,
    /// The next row to emit and the row the attached range ends at, in
    /// decoded-row terms (see [`DecodeRange::rows`]).
    next_row: usize,
    end_row: usize,
    /// The range being decoded.
    range: Option<DecodeRange>,
    /// One bit per row of the attached range, set for the rows the table
    /// format left live, when it deleted any of them.
    live_in_range: Option<BooleanBuffer>,
    /// The row group's ranges still to decode, on any worker.
    ranges_left: Arc<AtomicUsize>,
    /// Per leaf, the index of the next page to hand the leaf decoder.
    next_page: Vec<usize>,
    /// Whether to append row-group-id / row-index metadata columns.
    add_row_group_metadata: bool,
    /// Whether emitted batches may drop rows that fail a pushed-down
    /// equality constant. The materializer path must not: it addresses rows
    /// by their position inside the row group, so every row has to stay in
    /// place.
    filter_batches: bool,
    /// Whether the attached range's batches still try their gates. A batch
    /// the gates pass most of ends the trying for the rest of its range:
    /// neighbouring rows tend to pass alike, and each try costs a comparison
    /// over the batch.
    gating_range: bool,
}

/// The most page bytes a batch spans, across the leaves it decodes, before
/// it is cut to fewer rows. The operators after the scan read a batch's
/// values through its views into the pages, and a batch of wide values that
/// spans more than a worker's cache holds has those bytes fetched from memory
/// a second time.
const BATCH_PAGE_BYTES: usize = 512 * 1024;

/// The fewest rows a batch is cut to on account of its width: below this the
/// per-batch cost of every operator outweighs what a narrower span saves.
const MIN_BATCH_ROWS: usize = 2048;

/// How many rows a batch of this row group holds: `max_rows`, halved while
/// a batch would span more than [`BATCH_PAGE_BYTES`] of the decoded leaves'
/// pages, down to [`MIN_BATCH_ROWS`].
fn choose_batch_rows(
    plan: &DecodePlan,
    metadata: &QueryRowGroupMetadata,
    max_rows: usize,
) -> usize {
    let row_group_rows = metadata.num_rows().max(1) as usize;
    let column_chunks = metadata.columns();
    let page_bytes: usize = plan
        .file_leaves
        .iter()
        .map(|&leaf| column_chunks[leaf].total_uncompressed_size.max(0) as usize)
        .sum();
    let bytes_per_row = page_bytes / row_group_rows;
    let mut rows = max_rows;
    while rows / 2 >= MIN_BATCH_ROWS && rows * bytes_per_row > BATCH_PAGE_BYTES {
        rows /= 2;
    }
    rows
}

impl RowGroupDecoder {
    /// A decoder for the row group of `range`, ready to take it on, reading
    /// through the dictionaries the range carries. Every range of a row
    /// group carries the same dictionaries, built once by the cutter.
    pub fn new(
        range: &DecodeRange,
        batch_size: usize,
        add_row_group_metadata: bool,
    ) -> Result<Self> {
        let metadata = range.metadata().clone();
        let plan = range.plan().clone();
        let batch_size = choose_batch_rows(&plan, &metadata, batch_size);
        let filter_batches =
            !add_row_group_metadata && !matches!(metadata.selection(), RowSelection::Indices(_));
        let mut leaf_decoders = plan.create_leaf_decoders(&metadata)?;
        for (leaf, decoder) in leaf_decoders.iter_mut().enumerate() {
            if let Some(dictionary) = &range.column(leaf).dictionary {
                decoder.adopt_dictionary(dictionary.clone());
            }
        }
        Ok(Self {
            next_page: vec![0; plan.leaf_count()],
            leaf_decoders,
            plan,
            batch_size,
            next_row: 0,
            end_row: 0,
            range: None,
            live_in_range: None,
            ranges_left: range.ranges_left().clone(),
            add_row_group_metadata,
            filter_batches,
            gating_range: false,
            metadata,
        })
    }

    pub fn row_group_index(&self) -> usize {
        self.metadata.row_group_index
    }

    /// Whether every range of the row group is decoded, on any worker, so
    /// no further range can arrive.
    pub fn row_group_done(&self) -> bool {
        self.ranges_left.load(Ordering::Acquire) == 0
    }

    /// Takes on `range`. A range starting at the row this decoder stopped
    /// at continues from the page it has open; any other range repositions
    /// every leaf at the range's first page.
    ///
    /// A dictionary encoded page in a range that carries no dictionary
    /// could never be decoded, so it is an error rather than a wait.
    pub fn attach(&mut self, range: DecodeRange) -> Result<()> {
        assert!(self.exhausted(), "a decoder takes on one range at a time");
        for leaf in 0..self.leaf_decoders.len() {
            let column = range.column(leaf);
            if column.dictionary.is_none()
                && column
                    .pages
                    .iter()
                    .any(|page| page.content.is_dictionary_encoded())
            {
                return Err(Error::MissingDictionary { leaf });
            }
        }
        let rows = range.rows();
        let continues = rows.start as usize == self.end_row;
        self.live_in_range = self.live_rows_of(&rows);
        self.next_row = rows.start as usize;
        self.end_row = rows.end as usize;
        self.gating_range = !self.plan.gates.is_empty();
        if !continues {
            self.reposition_at(&range);
        }
        for leaf in 0..self.leaf_decoders.len() {
            for page in &range.column(leaf).pages {
                if page.idx < self.next_page[leaf] {
                    continue;
                }
                self.next_page[leaf] = page.idx + 1;
                let decompressed = self.page_to_decode(leaf, page, &rows);
                self.leaf_decoders[leaf].insert_page(decompressed);
            }
        }
        self.range = Some(range);
        Ok(())
    }

    /// Readies every leaf for the pages of `range`, whose rows do not follow
    /// the rows this decoder emitted last.
    ///
    /// A leaf restarts at the range's first page, and when the range starts
    /// inside that page the rows before it are skipped as the page is read.
    /// The skip is shorter wherever a decoder of the row group recorded its
    /// cursor at a range start inside the page: the leaf resumes from the
    /// latest such cursor and skips only the ranges between it and `range`,
    /// recording its own cursors at each range start it passes for the
    /// decoders that come after.
    fn reposition_at(&mut self, range: &DecodeRange) {
        let start_cursors = range.start_cursors();
        let cursors_apply = matches!(self.metadata.selection(), RowSelection::All);
        let resumed_from: Vec<Option<usize>> = (0..self.leaf_decoders.len())
            .map(|leaf| {
                let first = range
                    .column(leaf)
                    .pages
                    .first()
                    .expect("a ready range has pages in every column");
                let cursor = (cursors_apply && first.first_row < range.rows().start)
                    .then(|| start_cursors.find_latest(leaf, range.index(), first.first_row))
                    .flatten();
                match cursor {
                    Some((cursor_range, cursor)) => {
                        let page = self.page_at_cursor(leaf, first);
                        self.leaf_decoders[leaf].resume_page(page, cursor);
                        self.next_page[leaf] = first.idx + 1;
                        Some(cursor_range)
                    }
                    None => {
                        self.leaf_decoders[leaf].restart_at_page(first.idx);
                        self.next_page[leaf] = first.idx;
                        None
                    }
                }
            })
            .collect();
        let Some(earliest) = resumed_from.iter().flatten().copied().min() else {
            return;
        };
        for passed in earliest..range.index() {
            let passed_rows =
                (start_cursors.first_row(passed + 1) - start_cursors.first_row(passed)) as usize;
            let skips = |leaf: usize| resumed_from[leaf].is_some_and(|from| from <= passed);
            for leaf in (0..self.leaf_decoders.len()).filter(|&leaf| skips(leaf)) {
                self.leaf_decoders[leaf].skip_rows(passed_rows);
            }
            if !start_cursors.is_recorded_at(passed + 1) {
                let cursors = (0..self.leaf_decoders.len())
                    .map(|leaf| {
                        skips(leaf)
                            .then(|| self.leaf_decoders[leaf].open_page_cursor())
                            .flatten()
                    })
                    .collect();
                start_cursors.record(passed + 1, cursors);
            }
        }
    }

    /// Gives back the range once every row of it is emitted, recording
    /// where each leaf stands for the decoder of the range that follows.
    pub fn finish_range(&mut self) -> DecodeRange {
        assert!(
            self.exhausted(),
            "a range is finished once its rows are emitted"
        );
        let range = self
            .range
            .take()
            .expect("a range is finished once, after being attached");
        let next = range.index() + 1;
        if matches!(self.metadata.selection(), RowSelection::All)
            && !range.start_cursors().is_recorded_at(next)
        {
            let cursors: LeafCursors = self
                .leaf_decoders
                .iter()
                .map(|leaf| leaf.open_page_cursor())
                .collect();
            if cursors.iter().any(Option::is_some) {
                range.start_cursors().record(next, cursors);
            }
        }
        range
    }

    /// The stored page for a leaf decoder to resume inside, at a cursor.
    fn page_at_cursor(&self, leaf: usize, page: &StoredPage) -> DecompressedPage {
        let PageContent::Data(data) = &page.content else {
            panic!("a cursor lies inside a page with rows to decode")
        };
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: self.metadata.clone(),
            column_idx: leaf,
            idx: page.idx,
            first_row: page.first_row,
            data: DecompressedPageType::Data(DataPage {
                header: data.header.clone(),
                data: data.data.clone(),
                filter_mask: None,
            }),
        }
    }

    /// The stored page as the leaf decoder takes it. The first page of a
    /// range that starts inside it gets a mask skipping the rows before the
    /// range; an index selection's pages carry their own masks.
    fn page_to_decode(
        &self,
        leaf: usize,
        page: &StoredPage,
        rows: &Range<u32>,
    ) -> DecompressedPage {
        let data = match &page.content {
            PageContent::Data(data) => {
                let filter_mask = match self.metadata.selection() {
                    RowSelection::All if page.first_row < rows.start => {
                        let span = page.row_span();
                        Some(FilterMask::for_row_range(
                            span.start,
                            span.end,
                            &(rows.start..span.end),
                        ))
                    }
                    RowSelection::All => None,
                    RowSelection::Indices(_) => data.filter_mask.clone(),
                };
                DecompressedPageType::Data(DataPage {
                    header: data.header.clone(),
                    data: data.data.clone(),
                    filter_mask,
                })
            }
            PageContent::Skipped(header) => DecompressedPageType::SkippedData {
                header: header.clone(),
            },
        };
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: self.metadata.clone(),
            column_idx: leaf,
            idx: page.idx,
            first_row: page.first_row,
            data,
        }
    }

    /// The rows of `rows`, a range of the row group's rows, the table format
    /// left live, as a bit per row, when it deleted any of them. The
    /// materializer's index list was drawn from rows that already passed
    /// here, so an index selection has none deleted.
    fn live_rows_of(&self, rows: &Range<u32>) -> Option<BooleanBuffer> {
        let metadata = self.metadata.get_metadata();
        if metadata.deleted_rows.is_none()
            || !matches!(self.metadata.selection(), RowSelection::All)
            || metadata.count_deleted_in(rows.clone()) == 0
        {
            return None;
        }
        let mut live = BooleanBufferBuilder::new(rows.len());
        live.append_n(rows.len(), true);
        for position in metadata.deleted_rows_in(rows.clone()) {
            live.set_bit((position - rows.start) as usize, false);
        }
        Some(live.finish())
    }

    /// Whether every row of the attached range has been emitted.
    pub fn exhausted(&self) -> bool {
        self.next_row == self.end_row
    }

    /// Attempts to produce the next [`DecodedBatch`].
    ///
    /// Returns `Ok(Some(batch))` if every column decoder has enough buffered
    /// rows, `Ok(None)` if more pages are needed, or an error if decoding
    /// fails.
    pub fn try_read(&mut self, allocator: &mut SlabAllocator) -> Result<Option<DecodedBatch>> {
        loop {
            let size = min(self.batch_size, self.end_row - self.next_row);
            let available = min(
                self.leaf_decoders
                    .iter()
                    .map(|leaf| leaf.available())
                    .min()
                    .unwrap(),
                size,
            );
            if size == 0 || available == 0 {
                return Ok(None);
            }
            let first_row = self.next_row;
            // The rows the table format deleted are named by the batch's
            // bits rather than dropped here, for the parent to copy the rest
            // into full batches.
            let live_rows = self.live_in_range.as_ref().and_then(|live| {
                let live = live.slice(live.len() - (self.end_row - first_row), available);
                (live.count_set_bits() < available).then_some(live)
            });
            // Every distinct leaf is decoded once; the output columns then fold
            // their own views of the result, so two extracts over one variant
            // column share the decode instead of repeating it. The gates drop
            // rows only where a batch may drop rows at all.
            let decoded = if self.gating_range && self.filter_batches && live_rows.is_none() {
                self.read_rows_passing_gates(allocator, available)?
            } else {
                Some(self.read_every_row(allocator, available)?)
            };
            self.next_row += available;
            // Rows that all fail the gates make no batch; the rows after
            // them are read next.
            let Some(decoded) = decoded else {
                continue;
            };
            let columns = self
                .plan
                .column_decoders
                .iter()
                .map(|decoder| decoder.read(&decoded))
                .collect::<Result<Vec<_>>>()?;
            let record_batch = RecordBatch::try_new(self.plan.schema.clone(), columns)?;
            // Give each column carrying a pushed-down equality constant a
            // chance to drop rows that provably fail it, before the batch
            // travels any further. Dictionary encoded string columns filter it
            // with a cheap view comparison; the rest leave it untouched. Never
            // done on the materializer path, which needs every row to stay in
            // place, nor to a batch whose live rows are named by its bits.
            let record_batch = if self.filter_batches && live_rows.is_none() {
                self.plan.prunable.iter().fold(record_batch, |batch, p| {
                    self.leaf_decoders[p.leaf].fast_filter_record_batch(batch, p.output_idx)
                })
            } else {
                record_batch
            };
            let batch = if self.add_row_group_metadata {
                with_row_index(record_batch, first_row)
            } else {
                record_batch
            };
            return Ok(Some(DecodedBatch { batch, live_rows }));
        }
    }

    /// Decodes the next `rows` rows of every leaf.
    fn read_every_row(
        &mut self,
        allocator: &mut SlabAllocator,
        rows: usize,
    ) -> Result<Vec<ArrayRef>> {
        self.leaf_decoders
            .iter_mut()
            .map(|leaf| leaf.read(allocator, rows).map_err(Into::into))
            .collect()
    }

    /// Reads the next `rows` rows through the gates: decodes the gates'
    /// leaves, and where the gates pass few enough of the rows decodes the
    /// other leaves for those rows only, returning every leaf cut down to
    /// them, or `None` when no row passes. Rows the gates pass most of are
    /// decoded whole, and so is the rest of the range.
    fn read_rows_passing_gates(
        &mut self,
        allocator: &mut SlabAllocator,
        rows: usize,
    ) -> Result<Option<Vec<ArrayRef>>> {
        let mut gate_leaf_rows: Vec<Option<ArrayRef>> = vec![None; self.leaf_decoders.len()];
        let mut passing: Option<BooleanBuffer> = None;
        for gate in &self.plan.gates {
            if gate_leaf_rows[gate.leaf].is_none() {
                gate_leaf_rows[gate.leaf] =
                    Some(self.leaf_decoders[gate.leaf].read(allocator, rows)?);
            }
            let passing_gate = gate.find_passing_rows(gate_leaf_rows[gate.leaf].as_ref().unwrap());
            passing = Some(match passing {
                Some(passing_earlier_gates) => &passing_earlier_gates & &passing_gate,
                None => passing_gate,
            });
        }
        let passing = passing.expect("a gated range has a gate");
        if passing.count_set_bits() * 100 > rows * MOST_PASSING_PERCENT_TO_SELECT {
            self.gating_range = false;
            let every_row = gate_leaf_rows
                .into_iter()
                .zip(&mut self.leaf_decoders)
                .map(|(gate_rows, leaf)| match gate_rows {
                    Some(gate_rows) => Ok(gate_rows),
                    None => leaf.read(allocator, rows).map_err(Into::into),
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Some(every_row));
        }
        let kept: Vec<u32> = passing.set_indices().map(|row| row as u32).collect();
        let kept_rows = gate_leaf_rows
            .into_iter()
            .zip(&mut self.leaf_decoders)
            .map(|(gate_rows, leaf)| match gate_rows {
                Some(gate_rows) => Ok(take_rows(&gate_rows, &kept)),
                None => leaf
                    .read_selected(allocator, rows, &kept)
                    .map_err(Into::into),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((!kept.is_empty()).then_some(kept_rows))
    }
}

/// The pushed-down comparisons the row group's decoders can evaluate
/// themselves, each on the decoded leaf it reads.
///
/// A comparison gates only when its column is emitted as exactly one decoded
/// leaf unchanged, so the leaf's rows are the values the query compares; when
/// the leaf's type compares the same way here as in the query's filter; and
/// when its constant is of that type.
fn plan_gates(
    column_decoders: &[ColumnDecoder],
    leaf_fields: &[FieldRef],
    projection: &Projection,
    comparisons: &[ScanComparison],
) -> Vec<Gate> {
    comparisons
        .iter()
        .filter(|comparison| comparison.path.is_empty())
        .filter_map(|comparison| {
            let output_idx = projection.column_indices.iter().enumerate().position(
                |(output_idx, &column)| {
                    column == comparison.column_idx && projection.extract_at(output_idx).is_none()
                },
            )?;
            let leaf = column_decoders[output_idx].untransformed_leaf()?;
            let leaf_type = leaf_fields[leaf].data_type();
            let comparable = compares_like_the_query(leaf_type)
                && comparison.value.get().0.data_type() == leaf_type;
            comparable.then(|| Gate {
                leaf,
                operator: comparison.operator,
                value: comparison.value.clone(),
            })
        })
        .collect()
}

/// Decides which output columns the row group can be pruned by, and the
/// constant each one's leaf decoder gets.
///
/// A constant only goes in when its output column emits exactly one decoded
/// leaf unchanged, because that is the only shape both uses of the constant can
/// read: the row group is pruned from the leaf's own dictionary, and the batch
/// filter compares the emitted array against the dictionary's view of the
/// constant. A column that casts its leaf or reconstructs a variant emits
/// something else entirely.
///
/// It also goes in only when every data page of that chunk is dictionary
/// encoded. The decoder uses the constant to skip building a dictionary that
/// excludes it, which is sound only when an excluded dictionary prunes the whole
/// row group. A PLAIN fallback page could hold the constant even if the
/// dictionary does not, so such a chunk is still scanned and its dictionary must
/// be built to decode it.
///
/// A constant whose type does not match the leaf is dropped by the leaf decoder,
/// which forgoes the pushdown; the query's `Filter` still applies the
/// comparison.
fn prunable_columns(
    column_decoders: &[ColumnDecoder],
    file_leaves: &[usize],
    column_chunks: &[ColumnChunkMeta],
    projection: &Projection,
    comparisons: &[ScanComparison],
) -> (Vec<PrunableColumn>, Vec<(usize, Scalar<ArrayRef>)>) {
    let mut prunable = Vec::new();
    let mut eq_constants = Vec::new();
    for (output_idx, &column) in projection.column_indices.iter().enumerate() {
        let Some(leaf) = column_decoders[output_idx].untransformed_leaf() else {
            continue;
        };
        // A whole-column read answers a comparison on the column itself; a
        // pushed extract answers one that names the very path it reads.
        let path = projection
            .extract_at(output_idx)
            .map(|extract| extract.path.as_slice())
            .unwrap_or_default();
        let Some(predicate) = comparisons.iter().find(|p| {
            matches!(p.operator, CompareType::Equal) && p.column_idx == column && p.path == path
        }) else {
            continue;
        };
        if !column_chunks[file_leaves[leaf]].data_pages_all_dictionary {
            continue;
        }
        eq_constants.push((leaf, predicate.value.clone()));
        prunable.push(PrunableColumn { output_idx, leaf });
    }
    (prunable, eq_constants)
}
