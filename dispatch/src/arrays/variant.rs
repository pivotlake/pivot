//! Cross-batch schema unification for variant columns.
//!
//! A variant column's physical layout is chosen per file (each file shreds
//! independently), so batches flowing through one dataflow routinely disagree
//! on the column's struct type. An operator that merges rows across batches
//! (e.g. a Top-N gathering its candidates) needs one common layout first:
//! every layout folds back to the canonical `{metadata, value}` pair without
//! losing a document, so that is what mismatched columns are rebuilt as.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use parquet_variant_compute::{VariantArray, unshred_variant};

/// Rebuilds `batches` onto one schema when a variant column's physical layout
/// differs between them, folding that column back to the canonical
/// `{metadata, value}` pair in every batch. Batches already sharing a schema
/// pass through untouched, which is the common case for a table with no
/// variant column. Mismatches on non-variant columns are left for the caller's
/// merge to reject.
pub fn unify_variant_layouts(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>, ArrowError> {
    let fold = mismatched_variant_columns(&batches);
    if fold.is_empty() {
        return Ok(batches);
    }
    batches
        .into_iter()
        .map(|batch| fold_batch_columns_to_canonical(&batch, &fold))
        .collect()
}

/// The column positions whose physical variant layout differs across
/// `batches`: a variant struct anywhere in the column, with a data type that
/// is not identical in every batch. Empty means the batches already agree, the
/// common case. Schema-only work, so a caller can make the fold decision
/// centrally and spread [`fold_batch_columns_to_canonical`] over workers.
pub fn mismatched_variant_columns(batches: &[RecordBatch]) -> Vec<usize> {
    let Some(first) = batches.first() else {
        return Vec::new();
    };
    let first_schema = first.schema();
    if batches
        .iter()
        .all(|batch| batch.schema() == first_schema || batch.schema_ref() == &first_schema)
    {
        return Vec::new();
    }
    (0..first_schema.fields().len())
        .filter(|&column| {
            let differs = batches
                .iter()
                .any(|b| b.column(column).data_type() != first.column(column).data_type());
            differs
                && batches
                    .iter()
                    .any(|b| is_variant_struct(b.column(column).data_type()))
        })
        .collect()
}

/// Rebuilds `batch` with each of `columns` folded to the canonical
/// `{metadata, value}` pair. A column of the batch that is already canonical
/// folds to itself cheaply (the restamp copies no data).
pub fn fold_batch_columns_to_canonical(
    batch: &RecordBatch,
    fold: &[usize],
) -> Result<RecordBatch, ArrowError> {
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for &column in fold {
        let folded = fold_to_canonical(&columns[column])?;
        // Keep the field's name, nullability, and metadata (the variant
        // extension tag); only the physical layout changes.
        fields[column] = Arc::new(
            (*fields[column])
                .clone()
                .with_data_type(folded.data_type().clone()),
        );
        columns[column] = folded;
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// Folds every variant column of `batch` to the canonical `{metadata, value}`
/// pair with canonical nullability, so an operator that splices rows from many
/// batches into one array (a join storing its build side, or accumulating
/// matched probe rows) always sees one physical type for the column. A batch
/// whose variant columns are already canonical, or which has none, passes
/// through untouched.
pub fn canonicalize_batch_variants(batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
    let fold: Vec<usize> = (0..batch.num_columns())
        .filter(|&column| {
            let data_type = batch.column(column).data_type();
            is_variant_struct(data_type) && !is_canonical_variant_struct(data_type)
        })
        .collect();
    if fold.is_empty() {
        return Ok(batch);
    }
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for &column in &fold {
        let folded = fold_to_canonical(&columns[column])?;
        // Keep the field's name, nullability, and metadata (the variant
        // extension tag); only the physical layout changes.
        fields[column] = Arc::new(
            (*fields[column])
                .clone()
                .with_data_type(folded.data_type().clone()),
        );
        columns[column] = folded;
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// Whether `data_type` is already the canonical pair with canonical
/// nullability, so [`canonicalize_batch_variants`] has nothing to rebuild.
fn is_canonical_variant_struct(data_type: &DataType) -> bool {
    let DataType::Struct(fields) = data_type else {
        return false;
    };
    fields.len() == 2
        && fields[0].name() == "metadata"
        && !fields[0].is_nullable()
        && fields[1].name() == "value"
        && fields[1].is_nullable()
}

/// Whether `data_type` is a variant column's physical struct: the canonical
/// `{metadata, value}` pair or a shredded widening of it.
fn is_variant_struct(data_type: &DataType) -> bool {
    let DataType::Struct(fields) = data_type else {
        return false;
    };
    fields.iter().any(|f| f.name() == "metadata")
        && fields
            .iter()
            .all(|f| matches!(f.name().as_str(), "metadata" | "value" | "typed_value"))
}

/// Folds one variant column back to the canonical `{metadata, value}` pair,
/// restamping the pair's nullability so every batch lands on the identical
/// struct type regardless of what its file's footer declared.
fn fold_to_canonical(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    let variant = VariantArray::try_new(array.as_ref())?;
    // Folds the typed leaves back into the binary value column; a no-op for
    // an already-canonical input.
    let unshredded = unshred_variant(&variant)?;
    let inner = unshredded.into_inner();
    let fields = Fields::from(
        inner
            .fields()
            .iter()
            .map(|field| {
                let nullable = field.name() != "metadata";
                Arc::new(Field::new(
                    field.name(),
                    field.data_type().clone(),
                    nullable,
                ))
            })
            .collect::<Vec<_>>(),
    );
    Ok(Arc::new(StructArray::try_new(
        fields,
        inner.columns().to_vec(),
        inner.nulls().cloned(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::DataType;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    fn docs_batch(rows: Vec<&str>, shred_age: bool) -> RecordBatch {
        let json: ArrayRef = Arc::new(StringArray::from(rows));
        let variant = json_to_variant(&json).unwrap();
        let variant = if shred_age {
            let shred = ShreddedSchemaBuilder::new()
                .with_path("age", &DataType::Int64)
                .unwrap()
                .build();
            shred_variant(&variant, &shred).unwrap()
        } else {
            variant
        };
        let field = variant.field("d");
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![field])),
            vec![Arc::new(variant.into_inner()) as _],
        )
        .unwrap()
    }

    /// Batches with per-file shredding fold onto one canonical layout that
    /// concatenates, and the documents survive the fold.
    #[test]
    fn folds_mismatched_layouts_to_one_schema() {
        let shredded = docs_batch(vec![r#"{"age":30}"#], true);
        let plain = docs_batch(vec![r#"{"age":31}"#], false);

        let unified = unify_variant_layouts(vec![shredded, plain]).unwrap();

        let schema = unified[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &unified).unwrap();
        let variant = VariantArray::try_new(merged.column(0).as_ref()).unwrap();
        let ages: Vec<i64> = (0..variant.len())
            .map(|row| {
                variant
                    .value(row)
                    .get_object_field("age")
                    .unwrap()
                    .as_int64()
                    .unwrap()
            })
            .collect();
        assert_eq!(ages, vec![30, 31]);
    }

    /// Batches that already agree pass through untouched.
    #[test]
    fn passes_matching_schemas_through() {
        let a = docs_batch(vec![r#"{"age":30}"#], true);
        let b = docs_batch(vec![r#"{"age":31}"#], true);
        let a_column = a.column(0).clone();

        let unified = unify_variant_layouts(vec![a, b]).unwrap();

        assert!(Arc::ptr_eq(unified[0].column(0), &a_column));
    }

    /// A mismatch on a non-variant column is not this helper's to resolve.
    #[test]
    fn leaves_non_variant_mismatches_alone() {
        let ints =
            RecordBatch::try_from_iter([("v", Arc::new(Int64Array::from(vec![1])) as ArrayRef)])
                .unwrap();
        let texts =
            RecordBatch::try_from_iter([("v", Arc::new(StringArray::from(vec!["x"])) as ArrayRef)])
                .unwrap();

        let unified = unify_variant_layouts(vec![ints, texts]).unwrap();

        assert_eq!(unified[0].column(0).data_type(), &DataType::Int64);
        assert_eq!(unified[1].column(0).data_type(), &DataType::Utf8);
    }
}
