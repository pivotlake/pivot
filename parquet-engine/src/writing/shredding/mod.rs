//! Shredding: storing a variant column's common paths as typed Parquet leaves
//! beside the binary blob, so a reader can fetch `attrs.user.id` without
//! decoding every document.
//!
//! A variant column is a struct of `{metadata, value}` — the document's
//! dictionary and its binary encoding. Shredding widens that to
//! `{metadata, value, typed_value{..}}`, where `typed_value` mirrors the chosen
//! paths as real columns. Nothing is lost: a row whose value at a shredded path
//! is missing or of another type keeps it in `value`, so the pair is always a
//! complete document. [`infer`] picks the paths.
//!
//! The decision is made **per file**, from that file's own rows. Files are free
//! to disagree — the read path resolves a path against each file's schema and
//! falls back to `value` where a file didn't shred it — and that is what keeps
//! the choice cheap: no cross-file agreement to reach, no table-wide schema to
//! migrate when the data drifts.
//!
//! The entry points:
//!
//! - [`widen_batch`] brings a compaction input's variant columns to the union
//!   of the input files' layouts ([`plan_shredding_union`]) at the head of the
//!   write pipeline, so batches of files that each shredded differently share
//!   one schema and can be regrouped into fresh files. Every typed value stays
//!   where it is; only a path some file typed as a kind the union did not
//!   keep is folded back into its leftover.
//! - [`unshred_batch`] folds a variant column back to `{metadata, value}`, for
//!   a pipeline that wants whole documents. Ingest's batches are already that
//!   shape and pass straight through.
//! - [`plan_file_shredding`] runs in the
//!   [`row_group_planner`](super::row_group_planner) once a file's rows are
//!   known. It picks the file's layout without rewriting the rows: from the
//!   documents, or from the typed leaves and leftovers of rows that came in
//!   shredded.
//! - [`shred_column`] applies the plan in the [`shredder`](super::shredder)
//!   stage, one slice of a row group's variant column at a time. Documents are
//!   shredded; rows in the union layout are brought to the plan by moving only
//!   the values whose leaf changed ([`reshred`]). The rewrite is the heavy
//!   half: it parallelizes across workers there, and slicing it keeps any one
//!   worker turn short.
//!
//! So compaction re-shreds without rebuilding a document: each output file
//! decides afresh from the rows it actually got, and pays only for the paths
//! its inputs disagreed on.

mod infer;
mod reshred;
mod shred;
mod union;

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};
use dispatch::arrays::take::concat_chunks;
use dispatch::memory::SlabAllocator;
use parquet_variant_compute::{VariantArray, unshred_variant};

use crate::types::leaves::leaf_range;
use crate::types::metadata::RowGroupMetadata;
use union::InputLayout;

use super::error::WriteResult;

/// Fold every variant column of `batch` back to the plain `{metadata, value}`
/// pair, dropping any typed leaves it arrived with. A batch whose variants are
/// already unshredded (or which has none) is returned untouched.
pub(crate) fn unshred_batch(
    batch: RecordBatch,
    allocator: &mut SlabAllocator,
) -> WriteResult<RecordBatch> {
    map_variant_columns(batch, |_, array| {
        // Already the plain pair: nothing to fold, and rebuilding it would copy
        // every document for nothing. This is the ingest path.
        if array.typed_value_field().is_none() {
            return Ok(None);
        }
        let unshredded = unshred_variant(array)?;
        Ok(Some(copy_value_field_to_slabs(unshredded, allocator)?))
    })
}

/// Rebuilds an unshredded column's `value` field over slab memory.
///
/// [`unshred_variant`] reassembles each row's document with `parquet_variant`'s
/// own builders, which write into a `Vec<u8>` on the global allocator. That
/// buffer becomes the emitted array's data buffer and lives as long as the
/// batch does, and a write holds every batch it has gathered so far, so a whole
/// compaction's worth accumulates beside the buffer pool instead of inside it.
/// Copying the finished column into slabs puts it on the same accounted,
/// pre-faulted memory the rest of the dataflow runs on, and frees the `Vec` as
/// soon as this returns, leaving only the batch in flight on the heap.
///
/// Only `value` is rebuilt. `metadata` is carried over from the input array,
/// which the decoder already built in slab memory, so copying it would spend a
/// second slab on bytes that are already accounted for.
fn copy_value_field_to_slabs(
    array: VariantArray,
    allocator: &mut SlabAllocator,
) -> WriteResult<VariantArray> {
    let (fields, mut columns, nulls) = array.into_inner().into_parts();
    let value_index = fields
        .iter()
        .position(|field| field.name() == "value")
        .expect("an unshredded variant column has a value field");

    let value = columns[value_index].clone();
    columns[value_index] = concat_chunks(allocator, std::slice::from_ref(&value))?;

    let rebuilt = StructArray::try_new(fields, columns, nulls)?;
    Ok(VariantArray::try_new(&rebuilt)?)
}

/// The union of the layouts `row_groups`, a compaction's inputs, give each
/// variant column, by column: the layout [`widen_batch`] brings every input
/// batch to. `None` for a plain column, and for a variant column no input
/// shredded. Where inputs typed one path as different kinds, the union keeps
/// the kind holding the most rows by the footers' null counts.
pub(super) fn plan_shredding_union(
    row_groups: &[Arc<RowGroupMetadata>],
) -> Arc<[Option<Arc<DataType>>]> {
    let Some(first) = row_groups.first() else {
        return Arc::from(Vec::new());
    };
    (0..first.schema.fields().len())
        .map(|column| {
            if !crate::is_variant_field(first.schema.field(column)) {
                return None;
            }
            let inputs: Vec<(DataType, Vec<u64>)> = row_groups
                .iter()
                .map(|row_group| {
                    let fields = row_group.schema.fields();
                    let rows_per_leaf = leaf_range(fields, column)
                        .map(|leaf| {
                            let nulls = row_group
                                .leaf_statistics(leaf)
                                .and_then(|statistics| statistics.null_count)
                                .unwrap_or(0);
                            (row_group.num_rows - nulls).max(0) as u64
                        })
                        .collect();
                    (fields[column].data_type().clone(), rows_per_leaf)
                })
                .collect();
            let layouts: Vec<InputLayout<'_>> = inputs
                .iter()
                .map(|(column, rows_per_leaf)| InputLayout {
                    column,
                    rows_per_leaf,
                })
                .collect();
            union::union_layout(&layouts).map(Arc::new)
        })
        .collect()
}

/// The union of the layouts `batches` give each variant column, as
/// [`plan_shredding_union`] would find from their files' footers: each
/// batch votes with its leaves' non-null counts.
#[cfg(test)]
pub(crate) fn plan_shredding_union_of_batches(
    batches: &[RecordBatch],
) -> Arc<[Option<Arc<DataType>>]> {
    let Some(first) = batches.first() else {
        return Arc::from(Vec::new());
    };
    (0..first.schema().fields().len())
        .map(|column| {
            if !crate::is_variant_field(first.schema().field(column)) {
                return None;
            }
            let inputs: Vec<(DataType, Vec<u64>)> = batches
                .iter()
                .map(|batch| {
                    let mut rows_per_leaf = Vec::new();
                    push_leaf_rows(batch.column(column), &mut rows_per_leaf);
                    (
                        batch.schema().field(column).data_type().clone(),
                        rows_per_leaf,
                    )
                })
                .collect();
            let layouts: Vec<InputLayout<'_>> = inputs
                .iter()
                .map(|(column, rows_per_leaf)| InputLayout {
                    column,
                    rows_per_leaf,
                })
                .collect();
            union::union_layout(&layouts).map(Arc::new)
        })
        .collect()
}

#[cfg(test)]
fn push_leaf_rows(array: &ArrayRef, rows_per_leaf: &mut Vec<u64>) {
    use arrow_array::Array;
    match array.as_any().downcast_ref::<StructArray>() {
        Some(object) => object
            .columns()
            .iter()
            .for_each(|child| push_leaf_rows(child, rows_per_leaf)),
        None => rows_per_leaf.push((array.len() - array.null_count()) as u64),
    }
}

/// Bring every variant column of `batch`, a compaction input's, to its
/// layout in `union` (see [`plan_shredding_union`]). A column already in that
/// layout, or one whose inputs typed nothing, passes through untouched.
pub(crate) fn widen_batch(
    batch: RecordBatch,
    union: &[Option<Arc<DataType>>],
    allocator: &mut SlabAllocator,
) -> WriteResult<RecordBatch> {
    map_variant_columns(batch, |column, array| {
        let Some(target) = &union[column] else {
            return Ok(None);
        };
        let widened =
            DataType::Struct(shred::column_fields(Some(&shred::typed_value_type(target))));
        if array.data_type() == &widened {
            return Ok(None);
        }
        Ok(Some(reshred::widen(array, target, allocator)?))
    })
}

/// One file's shredding decision: the widened schema its footer will
/// describe, and each variant column's inferred layout for the encode workers
/// to apply. Plain columns, and variants with nothing worth shredding, carry
/// `None`.
pub(super) struct FileShredding {
    pub(super) schema: arrow_schema::SchemaRef,
    pub(super) column_shredding: Vec<Option<Arc<arrow_schema::DataType>>>,
}

/// Decide one file's shredding from all of its rows without rewriting any of
/// them: infer each variant column's layout, and derive the widened field a
/// column will have by shredding zero rows of it — the schema comes out of the
/// same code that later shreds the values.
///
/// `chunks` must hold a whole output file's rows, since the layout is chosen
/// from all of them together; the rewrite itself happens slice by slice in the
/// shredder stage ([`shred_column`]).
pub(super) fn plan_file_shredding(chunks: &[RecordBatch]) -> WriteResult<FileShredding> {
    let schema = chunks[0].schema();
    let mut fields: Vec<FieldRef> = schema.fields().to_vec();
    let mut column_shredding = vec![None; fields.len()];
    for column in 0..fields.len() {
        if !crate::is_variant_field(schema.field(column)) {
            continue;
        }
        let arrays = chunks
            .iter()
            .map(|chunk| VariantArray::try_new(chunk.column(column).as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let arrived_shredded = arrays
            .iter()
            .any(|array| array.typed_value_field().is_some());
        let shredding_type = match infer::infer_shredding_type(&arrays) {
            Some(shredding_type) => shredding_type,
            // Rows that came in shredded still have to be brought to the plain
            // pair: an empty layout is that plan.
            None if arrived_shredded => DataType::Struct(Fields::empty()),
            None => continue,
        };
        fields[column] = Arc::new(shredded_column_field(
            fields[column].name(),
            &shredding_type,
        )?);
        column_shredding[column] = Some(Arc::new(shredding_type));
    }
    Ok(FileShredding {
        schema: Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        column_shredding,
    })
}

/// The field a variant column named `name` has once shredded into
/// `shredding`: the layout's struct under the variant extension tag, which is
/// what keeps the column recognized as a variant by the footer's VARIANT
/// annotation the assembler writes. An empty layout is the plain pair.
fn shredded_column_field(name: &str, shredding: &DataType) -> WriteResult<Field> {
    let typed_value = match shredding {
        DataType::Struct(fields) if fields.is_empty() => None,
        shredding => Some(shred::typed_value_type(shredding)),
    };
    let column = DataType::Struct(shred::column_fields(typed_value.as_ref()));
    let no_rows = arrow_array::new_empty_array(&column);
    Ok(VariantArray::try_new(no_rows.as_ref())?.field(name))
}

/// Apply one column's planned `shredding` to a run of its rows: the heavy half
/// of shredding, run on a slice of a row group at a time by the shredder stage.
/// Documents are shredded; rows in the union layout are brought to the plan
/// by moving only the values whose leaf changed. The columns are built on
/// slabs from `allocator`.
pub(super) fn shred_column(
    values: &ArrayRef,
    shredding: &DataType,
    allocator: &mut SlabAllocator,
) -> WriteResult<ArrayRef> {
    let variant = VariantArray::try_new(values.as_ref())?;
    let shredded = if variant.typed_value_field().is_some() {
        reshred::reshred(&variant, shredding, allocator)?
    } else {
        shred::shred_into_slabs(&variant, shredding, allocator)?
    };
    Ok(Arc::new(shredded.into_inner()))
}

/// Rebuild `batch` with `f` applied to each of its variant columns, widening or
/// narrowing the schema to whatever shape `f` returns. `f` returning `None`
/// leaves that column alone, and a batch with no variant column at all is
/// returned as-is — the common case for a table of plain columns, which must not
/// pay for this.
fn map_variant_columns(
    batch: RecordBatch,
    mut f: impl FnMut(usize, &VariantArray) -> WriteResult<Option<VariantArray>>,
) -> WriteResult<RecordBatch> {
    let schema = batch.schema();
    let variant_columns: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| crate::is_variant_field(schema.field(i)))
        .collect();
    if variant_columns.is_empty() {
        return Ok(batch);
    }

    let mut fields: Vec<FieldRef> = schema.fields().to_vec();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    let mut changed = false;
    for column in variant_columns {
        let array = VariantArray::try_new(columns[column].as_ref())?;
        let Some(mapped) = f(column, &array)? else {
            continue;
        };
        // `VariantArray::field` re-applies the variant extension tag, so the
        // rebuilt column is still recognized as a variant — by the next stage
        // here, and by the footer's VARIANT annotation the assembler writes.
        fields[column] = Arc::new(mapped.field(fields[column].name()));
        columns[column] = Arc::new(mapped.into_inner());
        changed = true;
    }
    if !changed {
        return Ok(batch);
    }
    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}
