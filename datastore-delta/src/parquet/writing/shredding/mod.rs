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
//! Two entry points, both driven from the [`partition`](super::partition) stage:
//!
//! - [`unshred_batch`] folds a variant column back to `{metadata, value}` as
//!   batches arrive. Ingest's are already that shape and pass straight through;
//!   compaction's come from files that each shredded differently, and this is
//!   what makes them one schema again so they can be accumulated together and
//!   re-cut into files.
//! - [`plan_shredding`] runs once a file's rows are known, and is where a file
//!   picks its own layout, from all of its row groups together. Only the
//!   decision is made there: rewriting the rows into that layout is by far the
//!   heaviest step of a variant write, so it runs as [`shred_column`] in the
//!   encoder stage, one row group at a time on the parallel side.
//!
//! Between them, compaction re-shreds: every input file's layout is folded away
//! and each output file decides afresh from the rows it actually got.

mod infer;

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, FieldRef, Schema, SchemaRef};
use parquet_variant_compute::{VariantArray, shred_variant, unshred_variant};

use super::error::WriteResult;

/// Fold every variant column of `batch` back to the plain `{metadata, value}`
/// pair, dropping any typed leaves it arrived with. A batch whose variants are
/// already unshredded (or which has none) is returned untouched.
pub(crate) fn unshred_batch(batch: RecordBatch) -> WriteResult<RecordBatch> {
    map_variant_columns(batch, |array| {
        // Already the plain pair: nothing to fold, and rebuilding it would copy
        // every document for nothing. This is the ingest path.
        if array.typed_value_field().is_none() {
            return Ok(None);
        }
        Ok(Some(unshred_variant(array)?))
    })
}

/// One file's shredding decision: the widened schema its footer will carry, and
/// the type each variant column shreds into. `shredding_types` is index-aligned
/// with the schema's columns; a column that is not a variant, or that no path is
/// worth shredding for, holds `None` and its values reach the encoder untouched.
pub(super) struct ShreddingPlan {
    pub(super) schema: SchemaRef,
    pub(super) shredding_types: Arc<[Option<DataType>]>,
}

/// Decide the shredding for one file from its row `groups`: infer each variant
/// column's type from a sample of the file's own rows, and widen the schema to
/// the shape [`shred_column`] will produce. The rows themselves are not
/// rewritten here; the encoder applies the plan one row group at a time, in
/// parallel.
///
/// `groups` must hold a whole output file's rows: the layout is inferred from
/// all of them together and applied to each, so every row group of the file is
/// written against the one schema, which is what the footer describes.
pub(super) fn plan_shredding(groups: &[RecordBatch]) -> WriteResult<ShreddingPlan> {
    let schema = groups[0].schema();
    let mut shredding_types: Vec<Option<DataType>> = vec![None; schema.fields().len()];
    let variant_columns: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| crate::parquet::is_variant_field(schema.field(i)))
        .collect();
    if variant_columns.is_empty() {
        return Ok(ShreddingPlan {
            schema,
            shredding_types: shredding_types.into(),
        });
    }

    let mut fields: Vec<FieldRef> = schema.fields().to_vec();
    for column in variant_columns {
        let arrays = groups
            .iter()
            .map(|group| VariantArray::try_new(group.column(column).as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(shredding_type) = infer::infer_shredding_type(&arrays) else {
            continue;
        };
        // The widened field is exactly what `shred_variant` will produce for
        // every row group: its shape is a function of the shredding type alone,
        // so shredding a zero-row slice yields it without touching any values.
        // `VariantArray::field` re-applies the variant extension tag, so the
        // widened column is still recognized as a variant by the encoder and by
        // the footer's VARIANT annotation the assembler writes.
        let shredded = shred_variant(&arrays[0].slice(0, 0), &shredding_type)?;
        fields[column] = Arc::new(shredded.field(fields[column].name()));
        shredding_types[column] = Some(shredding_type);
    }

    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    Ok(ShreddingPlan {
        schema,
        shredding_types: shredding_types.into(),
    })
}

/// Apply a [`ShreddingPlan`] to one row group's variant column: rewrite `values`
/// into `shredding_type`, the heavy step the encoder runs chunk by chunk on the
/// parallel side of the pipeline.
pub(super) fn shred_column(values: &ArrayRef, shredding_type: &DataType) -> WriteResult<ArrayRef> {
    let array = VariantArray::try_new(values.as_ref())?;
    Ok(Arc::new(
        shred_variant(&array, shredding_type)?.into_inner(),
    ))
}

/// Rebuild `batch` with `f` applied to each of its variant columns, widening or
/// narrowing the schema to whatever shape `f` returns. `f` returning `None`
/// leaves that column alone, and a batch with no variant column at all is
/// returned as-is — the common case for a table of plain columns, which must not
/// pay for this.
fn map_variant_columns(
    batch: RecordBatch,
    f: impl Fn(&VariantArray) -> WriteResult<Option<VariantArray>>,
) -> WriteResult<RecordBatch> {
    let schema = batch.schema();
    let variant_columns: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| crate::parquet::is_variant_field(schema.field(i)))
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
