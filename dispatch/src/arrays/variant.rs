//! Normalization of physical layouts for variant columns.
//!
//! A variant column's physical layout is chosen per file (each file shreds
//! independently), so batches flowing through one dataflow routinely disagree
//! on the column's struct type. Operators that combine rows from several
//! batches need one physical type for the column. These helpers unshred variant
//! columns to the `{metadata, value}` layout and normalize its child-field
//! nullability without copying unaffected columns.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use parquet_variant_compute::{VariantArray, unshred_variant};

/// Rebuilds `batches` when a variant column's physical layout differs between
/// them, unshredding that column to the normalized `{metadata, value}` layout
/// in every batch. If no variant layout differs, the batches pass through
/// untouched. Mismatches on non-variant columns are left for the caller's merge
/// to reject.
pub fn unify_variant_layouts(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>, ArrowError> {
    let variant_columns = mismatched_variant_columns(&batches);
    if variant_columns.is_empty() {
        return Ok(batches);
    }
    batches
        .into_iter()
        .map(|batch| unshred_variant_columns(&batch, &variant_columns))
        .collect()
}

/// The positions of columns whose physical variant layout differs across
/// `batches`: the data type is not identical in every batch and at least one
/// batch has the field-name shape of a variant struct at that position. Empty
/// means no variant column needs unshredding. This examines schemas only, so a
/// caller can make the decision centrally and spread [`unshred_variant_columns`]
/// over workers.
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
                    .any(|b| looks_like_variant_struct(b.column(column).data_type()))
        })
        .collect()
}

/// Rebuilds `batch` with each listed variant column converted to the normalized
/// unshredded `{metadata, value}` layout. An already-unshredded column reuses
/// its data buffers; only its array and field metadata are rebuilt.
pub fn unshred_variant_columns(
    batch: &RecordBatch,
    variant_columns: &[usize],
) -> Result<RecordBatch, ArrowError> {
    let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    let mut columns = batch.columns().to_vec();
    for &column in variant_columns {
        let unshredded = unshred_and_normalize(&columns[column])?;
        // Keep the field's name, nullability, and metadata (the variant
        // extension tag); only the physical layout changes.
        fields[column] = Arc::new(
            (*fields[column])
                .clone()
                .with_data_type(unshredded.data_type().clone()),
        );
        columns[column] = unshredded;
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// Converts every variant column of `batch` to the normalized unshredded
/// `{metadata, value}` layout. Operators that combine rows from independently
/// shredded batches can call this on each input to ensure one physical type per
/// column. A batch with no variant columns requiring normalization passes
/// through untouched.
pub fn unshred_batch_variants(batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
    let variant_columns: Vec<usize> = (0..batch.num_columns())
        .filter(|&column| {
            let data_type = batch.column(column).data_type();
            looks_like_variant_struct(data_type) && !is_normalized_unshredded_layout(data_type)
        })
        .collect();
    if variant_columns.is_empty() {
        return Ok(batch);
    }
    unshred_variant_columns(&batch, &variant_columns)
}

/// Whether `data_type` has the normalized unshredded field names, order, and
/// nullability, so [`unshred_batch_variants`] has nothing to rebuild.
fn is_normalized_unshredded_layout(data_type: &DataType) -> bool {
    let DataType::Struct(fields) = data_type else {
        return false;
    };
    fields.len() == 2
        && fields[0].name() == "metadata"
        && !fields[0].is_nullable()
        && fields[1].name() == "value"
        && fields[1].is_nullable()
}

/// Whether `data_type` has the top-level field-name shape of a physical variant
/// layout. This is a structural heuristic: the extension tag belongs to the
/// containing [`Field`], not the struct's [`DataType`].
fn looks_like_variant_struct(data_type: &DataType) -> bool {
    let DataType::Struct(fields) = data_type else {
        return false;
    };
    fields.iter().any(|f| f.name() == "metadata")
        && fields
            .iter()
            .all(|f| matches!(f.name().as_str(), "metadata" | "value" | "typed_value"))
}

/// Unshreds one variant column to `{metadata, value}` and normalizes the child
/// fields' nullability so batches land on the same struct type regardless of
/// what their files' footers declared.
fn unshred_and_normalize(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    let variant = VariantArray::try_new(array.as_ref())?;
    // Merges typed leaves into the encoded `value` child; a no-op for an
    // already-unshredded input.
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
            let shredding_schema = ShreddedSchemaBuilder::new()
                .with_path("age", &DataType::Int64)
                .unwrap()
                .build();
            shred_variant(&variant, &shredding_schema).unwrap()
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

    #[test]
    fn unshreds_mismatched_layouts_to_one_schema() {
        let shredded = docs_batch(vec![r#"{"age":30}"#], true);
        let unshredded = docs_batch(vec![r#"{"age":31}"#], false);

        let unified = unify_variant_layouts(vec![shredded, unshredded]).unwrap();

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

    #[test]
    fn passes_matching_schemas_through() {
        let a = docs_batch(vec![r#"{"age":30}"#], true);
        let b = docs_batch(vec![r#"{"age":31}"#], true);
        let a_column = a.column(0).clone();

        let unified = unify_variant_layouts(vec![a, b]).unwrap();

        assert!(Arc::ptr_eq(unified[0].column(0), &a_column));
    }

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
