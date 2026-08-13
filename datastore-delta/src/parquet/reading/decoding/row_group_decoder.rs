//! Per-row-group decoder that accumulates decompressed pages and produces
//! Arrow [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) owns one [`RowGroupDecoder`] per
//! in-flight row group. Pages are inserted as they arrive from the
//! decompressor. When every projected column has enough buffered rows,
//! [`try_read`](RowGroupDecoder::try_read) decodes the next batch.

use crate::parquet::reading::decoding::column_decoder::{
    ColumnDecoder, Result, create_leaf_decoder,
};
use crate::parquet::reading::decoding::leaf_decoders::LeafDecoder;
use crate::parquet::reading::decoding::{AppliedPredicate, ScanEqualityPredicate};
use crate::parquet::reading::record_batch_metadata::with_row_group_metadata;
use crate::parquet::types::leaves::{leaf_fields, plan_leaves, resolve_output_reads};
use crate::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::parquet::types::page::DecompressedPage;
use crate::parquet::types::projection::Projection;
use arrow_arith::boolean::and_kleene;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, new_empty_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;
use arrow_schema::{Fields, Schema, SchemaRef};
use arrow_select::filter::filter_record_batch;
use dispatch::memory::SlabAllocator;
use planner::expression::CompareType;
use std::cmp::min;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// An output column whose pushed-down equality constant can prune the row
/// group, and the decoded leaf that constant was installed on.
#[derive(Clone, Copy)]
struct PrunableColumn {
    /// The column's position in the emitted batch.
    output_idx: usize,
    /// The leaf whose dictionary decides the pruning.
    leaf: usize,
}

/// Decodes pages for a single row group into [`RecordBatch`]es.
///
/// Pages are inserted out-of-order via [`insert_page`](Self::insert_page).
/// Each call to [`try_read`](Self::try_read) checks whether all column
/// decoders have at least `batch_size` rows available and, if so, produces one
/// batch. The decoder tracks how many rows have been emitted (`row_offset`)
/// and is [`exhausted`](Self::exhausted) once all rows have been read.
pub struct RowGroupDecoder {
    /// Global row-group index (used for routing and metadata tagging).
    row_group_idx: usize,
    /// One decoder per distinct leaf column chunk, in the order the fetcher
    /// requested them, so a page's `column_idx` indexes straight into this.
    leaf_decoders: Vec<Box<dyn LeafDecoder>>,
    /// One view per output column, in projection order. Several can fold the
    /// same decoded leaf.
    column_decoders: Vec<ColumnDecoder>,
    /// Output schema (projected). A pushed-down extract column carries the
    /// type the extract emits, not the variant it was read from.
    schema: SchemaRef,
    /// Max rows per batch.
    batch_size: usize,
    /// Total rows to emit (filtered count, or full row-group count).
    total: usize,
    /// Rows emitted so far.
    row_offset: usize,
    /// Whether to append row-group-id / row-index metadata columns.
    add_row_group_metadata: bool,
    /// The output columns carrying a pushed-down equality constant whose chunk
    /// is sound to prune by (all data pages dictionary encoded), each with the
    /// leaf holding that constant. When any such leaf's dictionary excludes it,
    /// the whole row group is pruned.
    prunable: Vec<PrunableColumn>,
    /// The row group's shared pruned flag is the same `Arc` every page of this
    /// row group carries. It is set here when a prunable column's dictionary
    /// excludes its constant: the row group cannot contain a matching row, so
    /// it emits nothing and is treated as exhausted. The decompressor reads it
    /// so it can skip the row group's remaining, not-yet-decompressed pages
    /// instead of decompressing them only for this decoder to discard.
    pruned: Arc<AtomicBool>,
    /// Whether emitted batches may drop rows that fail a pushed-down
    /// equality constant. The materializer path must not: it addresses rows
    /// by their position inside the row group, so every row has to stay in
    /// place.
    filter_batches: bool,
    /// Comparisons this scan alone applies; the plan carries no `Filter` for
    /// them, so every emitted row must satisfy each of them.
    applied_predicates: Arc<Vec<AppliedPredicate>>,
    /// How many of the decoded columns reach the batch. Any beyond this were
    /// read only so a claimed comparison could be evaluated.
    output_columns: usize,
    /// Leaves read for a claimed comparison's answer alone: their runs become
    /// mask bits and their values are never materialized.
    mask_only: Vec<MaskOnlyLeaf>,
}

/// Upper bound on the output-column index a mask can answer, so the per-batch
/// bookkeeping is a fixed-size array rather than an allocation. A projection
/// wider than this simply keeps the ordinary decode-and-compare path.
const MAX_TRACKED_MASK_COLUMNS: usize = 256;

/// A leaf whose only job is to answer a claimed comparison.
#[derive(Clone, Copy)]
struct MaskOnlyLeaf {
    /// The leaf to read the mask from.
    leaf: usize,
    /// The column position it would otherwise have occupied, so the ordinary
    /// path can still evaluate it if the mask turns out unavailable.
    output_idx: usize,
}

impl RowGroupDecoder {
    pub fn new(
        row_group_metadata: QueryRowGroupMetadata,
        projection: &Projection,
        batch_size: usize,
        add_row_group_metadata: bool,
        eq_predicates: &[ScanEqualityPredicate],
        applied_predicates: Arc<Vec<AppliedPredicate>>,
        output_columns: usize,
    ) -> Result<Self> {
        let pruned = row_group_metadata.pruned_flag();
        // Expand projected columns into this file's leaves. Variant layouts can
        // differ between files, so each row group resolves them independently,
        // and the fetcher resolved them the same way.
        let fields = row_group_metadata.get_metadata().schema.fields();
        let leaves = leaf_fields(fields);
        let reads = resolve_output_reads(fields, &row_group_metadata, projection);
        let plan = plan_leaves(&reads);
        let filter_batches =
            !add_row_group_metadata && row_group_metadata.filtered_indices().is_none();

        let column_chunks = row_group_metadata.columns();
        let mut leaf_decoders = plan
            .file_leaves
            .iter()
            .map(|&leaf| create_leaf_decoder(leaves[leaf].data_type(), &column_chunks[leaf]))
            .collect::<Result<Vec<_>>>()?;

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

        // A claimed comparison whose column the plan never selects can be
        // answered from the chunk's dictionary keys, so that leaf is read for a
        // mask and its values are never built. Requires the column to resolve
        // to one untransformed leaf whose data pages are all dictionary
        // encoded; anything else keeps the ordinary decode-and-compare path.
        let mut mask_only: Vec<MaskOnlyLeaf> = Vec::new();
        for predicate in applied_predicates.iter() {
            if predicate.output_idx < output_columns {
                continue;
            }
            let Some(leaf) = column_decoders[predicate.output_idx].untransformed_leaf() else {
                continue;
            };
            if !column_chunks[plan.file_leaves[leaf]].data_pages_all_dictionary {
                continue;
            }
            let negated = matches!(predicate.compare_type, CompareType::NotEqual);
            leaf_decoders[leaf].set_mask_predicate(&predicate.value, negated);
            mask_only.push(MaskOnlyLeaf {
                leaf,
                output_idx: predicate.output_idx,
            });
        }

        let prunable = install_eq_constants(
            &column_decoders,
            &mut leaf_decoders,
            &plan.file_leaves,
            column_chunks,
            projection,
            eq_predicates,
        );

        // Columns past `output_columns` are read only so a claimed comparison
        // can be evaluated; they never reach the batch, so the schema stops
        // where the plan's projection does.
        let output_fields: Fields = column_decoders
            .iter()
            .take(output_columns)
            .map(|decoder| decoder.output_field().clone())
            .collect();
        Ok(Self {
            row_group_idx: row_group_metadata.index(),
            leaf_decoders,
            column_decoders,
            schema: Arc::new(Schema::new(output_fields)),
            batch_size,
            total: row_group_metadata
                .filtered_indices()
                .as_ref()
                .map(|f| f.len())
                .unwrap_or(row_group_metadata.num_rows() as usize),
            row_offset: 0,
            add_row_group_metadata,
            prunable,
            pruned,
            filter_batches,
            applied_predicates,
            output_columns,
            mask_only,
        })
    }

    /// The conjunction of every comparison this scan applies itself, or `None`
    /// when it owns none.
    ///
    /// Reads straight from `columns` (the full decoded set, wider than the
    /// emitted batch when a column is read only for a comparison). A null
    /// compares as null, which the filter drops, matching SQL.
    fn evaluate_applied_predicates(
        &self,
        columns: &[ArrayRef],
        answered_by_mask: &[bool],
    ) -> Result<Option<BooleanArray>> {
        let mut combined: Option<BooleanArray> = None;
        for predicate in self.applied_predicates.iter() {
            if answered_by_mask.get(predicate.output_idx) == Some(&true) {
                continue;
            }
            let mask = predicate.evaluate(&columns[predicate.output_idx])?;
            combined = Some(match combined {
                Some(previous) => and_kleene(&previous, &mask)?,
                None => mask,
            });
        }
        Ok(combined)
    }

    /// Returns the global row-group index this decoder is responsible for.
    pub fn row_group_idx(&self) -> usize {
        self.row_group_idx
    }

    /// Returns `true` when the row group has been pruned (no row can match a
    /// pushed-down equality predicate) or all its rows have been emitted.
    pub fn exhausted(&self) -> bool {
        self.pruned() || self.total - self.row_offset == 0
    }

    /// Returns `true` if this row group was pruned by dictionary pushdown and
    /// should be dropped without emitting any rows.
    pub fn pruned(&self) -> bool {
        self.pruned.load(Ordering::Relaxed)
    }

    /// Routes a decompressed page to the appropriate column decoder, then
    /// re-evaluates dictionary pruning (a just-loaded dictionary page may
    /// exclude a pushed-down constant, allowing the whole row group to be
    /// dropped before its data pages are decoded).
    pub fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        self.leaf_decoders[page.column_idx].insert_page(page, allocator);
        if !self.pruned()
            && self
                .prunable
                .iter()
                .any(|p| self.leaf_decoders[p.leaf].dict_excludes_eq_constant())
        {
            // Publish to the shared flag (seen by every page of this row group):
            // the decoder discards the rest, and the decompressor can skip the
            // row group's remaining, not-yet-decompressed pages.
            self.pruned.store(true, Ordering::Relaxed);
        }
    }

    /// Attempts to produce the next [`RecordBatch`].
    ///
    /// Returns `Ok(Some(batch))` if every column decoder has enough buffered
    /// rows, `Ok(None)` if more pages are needed, or an error if decoding
    /// fails.
    pub fn try_read(&mut self, allocator: &mut SlabAllocator) -> Result<Option<RecordBatch>> {
        if self.pruned() {
            return Ok(None);
        }
        let size = min(self.batch_size, self.total - self.row_offset);
        let available = min(
            self.leaf_decoders
                .iter()
                .map(|leaf| leaf.available())
                .min()
                .unwrap(),
            size,
        );

        if size > 0 && available > 0 {
            // Every distinct leaf is decoded once; the output columns then fold
            // their own views of the result, so two extracts over one variant
            // column share the decode instead of repeating it.
            // Leaves answering a claimed comparison and nothing else are read
            // as a mask: their runs settle the comparison directly, so the
            // 4-bytes-per-row expansion of a heavily run-encoded column never
            // happens. Everything else decodes normally.
            let mut mask_from_keys: Option<BooleanBuffer> = None;
            let mut answered_by_mask = [false; MAX_TRACKED_MASK_COLUMNS];
            let mut skip_leaf = vec![false; self.leaf_decoders.len()];
            for entry in &self.mask_only {
                if !self.leaf_decoders[entry.leaf].can_read_predicate_mask() {
                    continue;
                }
                let bits = self.leaf_decoders[entry.leaf].read_predicate_mask(available)?;
                skip_leaf[entry.leaf] = true;
                if entry.output_idx < MAX_TRACKED_MASK_COLUMNS {
                    answered_by_mask[entry.output_idx] = true;
                }
                mask_from_keys = Some(match mask_from_keys {
                    Some(previous) => &previous & &bits,
                    None => bits,
                });
            }

            let decoded = self
                .leaf_decoders
                .iter_mut()
                .enumerate()
                .map(|(idx, leaf)| match skip_leaf[idx] {
                    // Never read: the columns that would have consumed this
                    // leaf are exactly the ones the mask replaced.
                    true => Ok(new_empty_array(&DataType::Null)),
                    false => leaf.read(allocator, available).map_err(Into::into),
                })
                .collect::<Result<Vec<_>>>()?;
            let columns = self
                .column_decoders
                .iter()
                .enumerate()
                .map(|(idx, decoder)| match answered_by_mask.get(idx) {
                    Some(true) => Ok(new_empty_array(&DataType::Null)),
                    _ => decoder.read(&decoded),
                })
                .collect::<Result<Vec<_>>>()?;

            // Comparisons the plan handed over: no `Filter` re-checks these, so
            // they are evaluated here against the full decoded set of columns -
            // including any read solely to answer them - and every row that
            // fails is dropped before the batch is built.
            let claimed = self.evaluate_applied_predicates(&columns, &answered_by_mask)?;

            // An explicit row count, because a claimed comparison can leave the
            // batch with no columns at all (`COUNT(*)` over a column read only
            // to answer the predicate), and a column-less batch has nothing to
            // infer its length from.
            let record_batch = RecordBatch::try_new_with_options(
                self.schema.clone(),
                columns.into_iter().take(self.output_columns).collect(),
                &RecordBatchOptions::new().with_row_count(Some(available)),
            )?;
            // Give each column carrying a pushed-down equality constant a
            // chance to drop rows that provably fail it, before the batch
            // travels any further. Dictionary encoded string columns filter it
            // with a cheap view comparison; the rest leave it untouched. Never
            // done on the materializer path, which needs every row to stay in
            // place.
            let record_batch = if self.filter_batches {
                self.prunable.iter().fold(record_batch, |batch, p| {
                    self.leaf_decoders[p.leaf].fast_filter_record_batch(batch, p.output_idx)
                })
            } else {
                record_batch
            };
            let batch = if self.add_row_group_metadata {
                with_row_group_metadata(record_batch, self.row_group_idx, self.row_offset)
            } else {
                record_batch
            };
            // Applied last, and after the row-group metadata is stamped: the
            // row index is a dense `offset..offset + rows` range over the rows
            // as decoded, so a row dropped before stamping would shift every
            // later row's id and send late materialization to the wrong rows.
            let claimed = match (claimed, mask_from_keys) {
                (Some(kernel), Some(keys)) => {
                    Some(and_kleene(&kernel, &BooleanArray::new(keys, None))?)
                }
                (Some(kernel), None) => Some(kernel),
                (None, Some(keys)) => Some(BooleanArray::new(keys, None)),
                (None, None) => None,
            };
            let batch = match claimed {
                Some(mask) => filter_record_batch(&batch, &mask)?,
                None => batch,
            };
            self.row_offset += available;
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
}

/// Installs each pushed-down equality constant on the leaf that answers it, and
/// returns the columns the row group can then be pruned by.
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
fn install_eq_constants(
    column_decoders: &[ColumnDecoder],
    leaf_decoders: &mut [Box<dyn LeafDecoder>],
    file_leaves: &[usize],
    column_chunks: &[ColumnChunkMeta],
    projection: &Projection,
    eq_predicates: &[ScanEqualityPredicate],
) -> Vec<PrunableColumn> {
    let mut prunable = Vec::new();
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
        let Some(predicate) = eq_predicates
            .iter()
            .find(|p| p.column_idx == column && p.path == path)
        else {
            continue;
        };
        if !column_chunks[file_leaves[leaf]].data_pages_all_dictionary {
            continue;
        }
        leaf_decoders[leaf].set_eq_constant(&predicate.value);
        prunable.push(PrunableColumn { output_idx, leaf });
    }
    prunable
}
