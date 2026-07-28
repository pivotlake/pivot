//! [`Filter`] — filters rows by one or more boolean conditions.

use crate::compile::{Error, ExprEvalFn};
use crate::expression::Expression;
use arrow_array::{Array, ArrayRef, BooleanArray, NullArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use dispatch::arrays::take::take;
use dispatch::memory::SlabAllocator;
use dispatch::{RecordBatchOperatorSpec, RowDelivery, RowSelection};
use std::fmt;
use std::sync::Arc;

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(Debug)]
pub struct Filter {
    pub conditions: Vec<Expression>,
    /// When survivors reach the operator below, decided by what that operator
    /// is: the plan's `annotate_filter_delivery` pass sets this once the tree
    /// is built. Coalesced until that pass says otherwise.
    pub delivery: RowDelivery,
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conds = self
            .conditions
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" AND ");
        write!(f, "Filter({conds})")
    }
}

impl Filter {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let filters = Arc::new(
            self.conditions
                .iter()
                .map(|e| e.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        assert!(!filters.is_empty());

        // Conditions apply progressively: each one's input_row_map are what the
        // next condition evaluates, so a selective early condition spares the
        // later ones most of the rows (see the shrink decision below).
        let later_kernel_counts: Vec<usize> = {
            let per_condition: Vec<usize> = self
                .conditions
                .iter()
                .map(Expression::count_kernels)
                .collect();
            (0..per_condition.len())
                .map(|i| per_condition[i + 1..].iter().sum())
                .collect()
        };
        // For each condition, the union of columns the LATER conditions read
        // (sorted, deduplicated), so a shrink at that point keeps only those.
        let later_columns_by_condition: Vec<Vec<usize>> = (0..self.conditions.len())
            .map(|i| {
                let mut refs = Vec::new();
                for condition in &self.conditions[i + 1..] {
                    condition.collect_column_refs(&mut refs);
                }
                refs.sort_unstable();
                refs.dedup();
                refs
            })
            .collect();
        let later_columns_by_condition = Arc::new(later_columns_by_condition);
        // Cycles one gathered element-copy costs, measured in kernel-pass
        // units: a copy is an L1 load + store plus gather bookkeeping (~4
        // cycles), while a vectorized kernel touches an element in ~1.
        const GATHER_COST_IN_KERNEL_PASSES: f64 = 4.0;

        let delivery = self.delivery;
        Ok(input.filter_with_delivery(
            move || {
                let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
                let later_kernel_counts = later_kernel_counts.clone();
                let later_columns_by_condition = later_columns_by_condition.clone();
                let mut condition_mask = ConditionMask::new();
                let mut selection_scratch: Vec<u32> = Vec::new();
                let mut remap_scratch: Vec<u32> = Vec::new();
                let mut mini_schemas: Vec<Option<SchemaRef>> =
                    vec![None; later_kernel_counts.len()];
                // Each row of `current` mapped to its position in the input
                // batch; identity until the first shrink compacts `current`.
                let mut input_row_map: Vec<u32> = Vec::new();
                move |batch: &RecordBatch,
                      allocator: &mut SlabAllocator,
                      selected_indices: &mut Vec<u32>| {
                    let mut current = batch.clone();
                    let mut has_input_row_map = false;
                    condition_mask.reset();
                    for (i, (eval, &later_kernels)) in
                        eval_fns.iter_mut().zip(&later_kernel_counts).enumerate()
                    {
                        let result = eval(&current);
                        let (arr, _) = result.as_datum().get();
                        let mask = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                        let selected_count = condition_mask.intersect(mask, current.num_rows());
                        if selected_count == 0 {
                            selected_indices.clear();
                            return RowSelection::Indices;
                        }
                        // Shrink or carry the mask? Arrow kernels cannot skip
                        // rows, so rows that already failed a condition ("dead",
                        // the `1.0 - selected_fraction` fraction) are still computed over by
                        // every later_kernels condition unless the batch is
                        // physically compacted first. Compacting is not free
                        // either: a gather copies every selected row once per
                        // column. Compare the two options in expected per-row
                        // cost (the batch's row count multiplies both sides, so
                        // it cancels):
                        //
                        //   wasted ops/row if we keep the dead rows around:
                        //       remaining kernel passes x fraction dead
                        //   copy ops/row if we compact now:
                        //       GATHER_COST_IN_KERNEL_PASSES x columns x fraction selected_fraction
                        //
                        // Copy when the copy is cheaper than the waste it
                        // removes.
                        //
                        // When the columns the later conditions read are known,
                        // the shrink keeps only those; cheap zeroed placeholders
                        // (which nothing reads) stand in for the rest, since the
                        // final output gathers full-width rows straight from the
                        // input batch through the input row map.
                        let needed_columns = &later_columns_by_condition[i];
                        let prune = needed_columns.len() < current.num_columns();
                        let compacted_column_count = if prune {
                            needed_columns.len()
                        } else {
                            current.num_columns()
                        };
                        let selected_fraction = selected_count as f64 / current.num_rows() as f64;
                        if later_kernels as f64 * (1.0 - selected_fraction)
                            > GATHER_COST_IN_KERNEL_PASSES
                                * compacted_column_count as f64
                                * selected_fraction
                        {
                            condition_mask.collect_indices(&mut selection_scratch);
                            current = if prune {
                                compact_columns_for_later_conditions(
                                    allocator,
                                    &current,
                                    needed_columns,
                                    &selection_scratch,
                                    &mut mini_schemas[i],
                                )
                            } else {
                                compact_record_batch_rows(allocator, &current, &selection_scratch)
                            };
                            remap_selection_to_input(
                                &mut input_row_map,
                                has_input_row_map,
                                &selection_scratch,
                                &mut remap_scratch,
                            );
                            has_input_row_map = true;
                            condition_mask.reset();
                        }
                    }
                    // The mask is always active here: a shrink only pays off
                    // while conditions remain (its cost model multiplies the
                    // kernels left to run), and every remaining condition
                    // reactivates the mask.
                    debug_assert!(condition_mask.is_active());
                    // Report the selected rows as positions in the input batch.
                    if has_input_row_map {
                        condition_mask.collect_indices(&mut selection_scratch);
                        selected_indices.clear();
                        selected_indices.extend(
                            selection_scratch
                                .iter()
                                .map(|&pos| input_row_map[pos as usize]),
                        );
                    } else {
                        condition_mask.collect_indices(selected_indices);
                    }
                    RowSelection::Indices
                }
            },
            delivery,
        ))
    }
}

/// The conjunction of the condition results evaluated since the last row
/// compaction, one bit per row of the current batch. Kept as raw words in a
/// reused buffer so intersecting the next condition's mask, counting the
/// selected rows, and collecting their positions each cost one pass and no
/// allocation. A null condition result always excludes its row.
struct ConditionMask {
    selected_row_bits: Vec<u64>,
    active: bool,
}

impl ConditionMask {
    fn new() -> Self {
        Self {
            selected_row_bits: Vec::new(),
            active: false,
        }
    }

    fn reset(&mut self) {
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    /// Intersect a condition's mask into the selection and return how many
    /// rows remain selected.
    fn intersect(&mut self, mask: &BooleanArray, rows: usize) -> usize {
        if !self.active {
            let word_count = rows.div_ceil(64);
            self.selected_row_bits.clear();
            self.selected_row_bits.resize(word_count, u64::MAX);
            if !rows.is_multiple_of(64) {
                self.selected_row_bits[word_count - 1] = (1u64 << (rows % 64)) - 1;
            }
            self.active = true;
        }
        let mut selected_count = 0;
        intersect_and_count_words(
            &mut self.selected_row_bits,
            mask.values(),
            &mut selected_count,
        );
        if let Some(nulls) = mask.nulls() {
            selected_count = 0;
            intersect_and_count_words(
                &mut self.selected_row_bits,
                nulls.inner(),
                &mut selected_count,
            );
        }
        selected_count
    }

    /// The ascending selected row positions, written into `selected_indices`.
    fn collect_indices(&self, selected_indices: &mut Vec<u32>) {
        selected_indices.clear();
        let selected_count: usize = self
            .selected_row_bits
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();
        selected_indices.reserve(selected_count);
        // SAFETY: `reserve(selected_count)` guarantees capacity for every set bit.
        unsafe {
            let ptr = selected_indices.as_mut_ptr();
            let mut len = 0;
            for (word_idx, &word) in self.selected_row_bits.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    ptr.add(len)
                        .write((word_idx * 64 + bits.trailing_zeros() as usize) as u32);
                    len += 1;
                    bits &= bits - 1;
                }
            }
            selected_indices.set_len(len);
        }
    }
}

/// AND a boolean buffer into the selection words, accumulating the selected
/// popcount. Bit-offset buffers (never produced by the compare kernels) take
/// a per-bit path.
fn intersect_and_count_words(
    words: &mut [u64],
    bits: &arrow::buffer::BooleanBuffer,
    selected_count: &mut usize,
) {
    if bits.offset() == 0 {
        let bytes = bits.values();
        for (word_idx, word) in words.iter_mut().enumerate() {
            let mut chunk = [0u8; 8];
            let start = word_idx * 8;
            let end = bytes.len().min(start + 8);
            chunk[..end - start].copy_from_slice(&bytes[start..end]);
            *word &= u64::from_le_bytes(chunk);
            *selected_count += word.count_ones() as usize;
        }
    } else {
        for (word_idx, word) in words.iter_mut().enumerate() {
            let mut mask_word = 0u64;
            for bit in 0..64 {
                let row = word_idx * 64 + bit;
                if row < bits.len() && bits.value(row) {
                    mask_word |= 1 << bit;
                }
            }
            *word &= mask_word;
            *selected_count += word.count_ones() as usize;
        }
    }
}

/// Update the mapping from positions in the about-to-be-compacted batch to
/// positions in the input batch: narrow the existing map to the selected
/// positions, or initialize it from them when rows were never compacted
/// before. The selected positions must be ascending and in bounds for the
/// batch being compacted.
fn remap_selection_to_input(
    input_row_map: &mut Vec<u32>,
    has_input_row_map: bool,
    selected: &[u32],
    scratch: &mut Vec<u32>,
) {
    if !has_input_row_map {
        input_row_map.clear();
        input_row_map.extend_from_slice(selected);
        return;
    }
    scratch.clear();
    scratch.extend(selected.iter().map(|&pos| input_row_map[pos as usize]));
    std::mem::swap(input_row_map, scratch);
}

/// Compact the columns later conditions read to the rows at `indices`; every
/// other column becomes a same-length [`NullArray`], which is valid for any
/// length at zero cost. The remaining conditions provably never read those
/// columns, and the final output never gathers them from this batch (it
/// reads the input batch through the row map instead), so the mini batch's
/// schema marks them as nullable `Null` fields — cached in `schema_cache`
/// since the kept set is fixed per condition.
fn compact_columns_for_later_conditions(
    allocator: &mut SlabAllocator,
    batch: &RecordBatch,
    needed_columns: &[usize],
    indices: &[u32],
    schema_cache: &mut Option<SchemaRef>,
) -> RecordBatch {
    let schema = schema_cache
        .get_or_insert_with(|| {
            let fields: Vec<Field> = batch
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(idx, field)| {
                    if needed_columns.binary_search(&idx).is_ok() {
                        field.as_ref().clone()
                    } else {
                        Field::new(field.name(), DataType::Null, true)
                    }
                })
                .collect();
            Arc::new(Schema::new(fields))
        })
        .clone();
    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(idx, column)| {
            if needed_columns.binary_search(&idx).is_ok() {
                take(allocator, column, indices).expect("indices are in bounds")
            } else {
                Arc::new(NullArray::new(indices.len())) as ArrayRef
            }
        })
        .collect();
    RecordBatch::try_new(schema, columns).expect("columns keep the cached schema")
}

/// Compact every column of `batch` to the rows at `indices`.
fn compact_record_batch_rows(
    allocator: &mut SlabAllocator,
    batch: &RecordBatch,
    indices: &[u32],
) -> RecordBatch {
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(allocator, column, indices).expect("indices are in bounds"))
        .collect();
    RecordBatch::try_new(batch.schema(), columns).expect("columns keep the batch schema")
}
