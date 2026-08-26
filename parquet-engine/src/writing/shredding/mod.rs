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
//! - [`unshred_batch`] folds a variant column back to `{metadata, value}` at
//!   the head of the write pipeline. Ingest's batches are already that shape
//!   and pass straight through; compaction's come from files that each
//!   shredded differently, and this is what makes them one schema again so
//!   they can be regrouped into fresh files.
//! - [`plan_file_shredding`] runs in the
//!   [`row_group_planner`](super::row_group_planner) once a file's rows are
//!   known. It picks the file's layout without rewriting the rows.
//! - [`shred_gathered_column`] applies the plan, on the encode workers, to
//!   each materialized row group's variant column — the rewrite is the heavy
//!   half, and it parallelizes there.
//!
//! Between them, compaction re-shreds: every input file's layout is folded away
//! and each output file decides afresh from the rows it actually got.

mod infer;
mod shred;

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::{FieldRef, Schema};
use dispatch::arrays::concat_chunks;
use dispatch::memory::SlabAllocator;
use parquet_variant_compute::{VariantArray, shred_variant, unshred_variant};

use super::error::WriteResult;

/// Fold every variant column of `batch` back to the plain `{metadata, value}`
/// pair, dropping any typed leaves it arrived with. A batch whose variants are
/// already unshredded (or which has none) is returned untouched.
pub(crate) fn unshred_batch(
    batch: RecordBatch,
    allocator: &mut SlabAllocator,
) -> WriteResult<RecordBatch> {
    map_variant_columns(batch, |array| {
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
/// from all of them together; the rewrite itself happens per row group on the
/// encode workers ([`shred_gathered_column`]).
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
        let Some(shredding_type) = infer::infer_shredding_type(&arrays) else {
            continue;
        };
        let no_rows = VariantArray::try_new(chunks[0].column(column).slice(0, 0).as_ref())?;
        let widened = shred_variant(&no_rows, &shredding_type)?;
        // `VariantArray::field` re-applies the variant extension tag, so the
        // widened column is still recognized as a variant by the footer's
        // VARIANT annotation the assembler writes.
        fields[column] = Arc::new(widened.field(fields[column].name()));
        column_shredding[column] = Some(Arc::new(shredding_type));
    }
    Ok(FileShredding {
        schema: Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        column_shredding,
    })
}

/// Apply one column's planned `shredding` to a materialized row group's
/// values: the heavy half of shredding, run per column chunk on the encode
/// workers. The shredded columns are built on slabs from `allocator`.
pub(super) fn shred_gathered_column(
    values: &ArrayRef,
    shredding: &arrow_schema::DataType,
    allocator: &mut SlabAllocator,
) -> WriteResult<ArrayRef> {
    let variant = VariantArray::try_new(values.as_ref())?;
    Ok(Arc::new(
        shred::shred_into_slabs(&variant, shredding, allocator)?.into_inner(),
    ))
}

/// Rebuild `batch` with `f` applied to each of its variant columns, widening or
/// narrowing the schema to whatever shape `f` returns. `f` returning `None`
/// leaves that column alone, and a batch with no variant column at all is
/// returned as-is — the common case for a table of plain columns, which must not
/// pay for this.
fn map_variant_columns(
    batch: RecordBatch,
    mut f: impl FnMut(&VariantArray) -> WriteResult<Option<VariantArray>>,
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
        let Some(mapped) = f(&array)? else {
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
