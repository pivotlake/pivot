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
//! - [`shred_groups`] runs once a file's rows are known, and is where a file
//!   picks its own layout, from all of its row groups together.
//!
//! Between them, compaction re-shreds: every input file's layout is folded away
//! and each output file decides afresh from the rows it actually got.

mod infer;

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{FieldRef, Schema};
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

/// Shred every variant column of one file's row `groups` into the paths the
/// file's own rows favour. A column no path is worth shredding for is left as
/// the plain pair, and a file with no variant column at all is returned as it
/// came.
///
/// `groups` must hold a whole output file's rows: the layout is inferred from
/// all of them together and then applied to each, so every row group of the file
/// is written against the one schema, which is what the footer describes.
pub(super) fn shred_groups(groups: Vec<RecordBatch>) -> WriteResult<Vec<RecordBatch>> {
    let Some(schema) = groups.first().map(RecordBatch::schema) else {
        return Ok(groups);
    };
    let variant_columns: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| crate::parquet::is_variant_field(schema.field(i)))
        .collect();
    if variant_columns.is_empty() {
        return Ok(groups);
    }

    let mut fields: Vec<FieldRef> = schema.fields().to_vec();
    let mut columns: Vec<Vec<ArrayRef>> = groups
        .iter()
        .map(|group| group.columns().to_vec())
        .collect();
    let rows: Vec<usize> = groups.iter().map(RecordBatch::num_rows).collect();
    drop(groups);
    for column in variant_columns {
        let arrays = columns
            .iter()
            .map(|group| VariantArray::try_new(group[column].as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(shredding_type) = infer::infer_shredding_type(&arrays) else {
            continue;
        };
        // Consuming `arrays` releases each group's unshredded column as soon as
        // its shredded form replaces it, so the file is never held twice over.
        for (group, array) in columns.iter_mut().zip(arrays) {
            let shredded = shred_variant(&array, &shredding_type)?;
            // `VariantArray::field` re-applies the variant extension tag, so the
            // rebuilt column is still recognized as a variant by the next stage
            // here, and by the footer's VARIANT annotation the assembler writes.
            fields[column] = Arc::new(shredded.field(fields[column].name()));
            group[column] = Arc::new(shredded.into_inner());
        }
    }

    // The widened schema is the file's, so every group is rebuilt against it.
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    columns
        .into_iter()
        .zip(rows)
        .map(|(columns, rows)| {
            let options = RecordBatchOptions::new().with_row_count(Some(rows));
            Ok(RecordBatch::try_new_with_options(
                schema.clone(),
                columns,
                &options,
            )?)
        })
        .collect()
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
