//! Flatten nested Arrow structs into Parquet's depth-first leaf columns.
//!
//! The inverse lives in `catalog::parquet::types::leaves`: a Parquet row group
//! owns one column chunk per primitive leaf, while Arrow presents the result as
//! top-level arrays containing nested `StructArray`s.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, make_array};
use arrow_buffer::NullBuffer;
use arrow_schema::{ArrowError, DataType, FieldRef};

use super::types::DefinitionLevels;

pub(super) struct LeafColumn {
    pub(super) leaf: usize,
    pub(super) column: usize,
    pub(super) field: FieldRef,
    pub(super) path: Arc<[String]>,
    pub(super) values: ArrayRef,
    pub(super) levels: DefinitionLevels,
}

/// Flatten all top-level columns in schema order. Struct children are visited
/// depth-first, exactly matching Parquet footer and column-chunk order.
pub(super) fn flatten(batch: &RecordBatch) -> Result<Vec<LeafColumn>, ArrowError> {
    let mut leaves = Vec::new();
    let mut path = Vec::new();
    let mut nullable = Vec::new();
    for (column, (field, array)) in batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .enumerate()
    {
        push_field(column, field, array, &mut path, &mut nullable, &mut leaves)?;
    }
    Ok(leaves)
}

fn push_field(
    column: usize,
    field: &FieldRef,
    array: &ArrayRef,
    path: &mut Vec<String>,
    nullable: &mut Vec<Option<NullBuffer>>,
    leaves: &mut Vec<LeafColumn>,
) -> Result<(), ArrowError> {
    path.push(field.name().clone());
    if field.is_nullable() {
        nullable.push(array.nulls().cloned());
    }

    match field.data_type() {
        DataType::Struct(children) => {
            let values = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    ArrowError::CastError(format!(
                        "field '{}' declares a struct but its array is {}",
                        field.name(),
                        array.data_type()
                    ))
                })?;
            for (child, values) in children.iter().zip(values.columns()) {
                push_field(column, child, values, path, nullable, leaves)?;
            }
        }
        _ => {
            // A non-nullable child's physical array can still carry nulls under
            // a null ancestor. Union every path bitmap into the leaf so casts,
            // dictionary construction, and PLAIN encoding all see the effective
            // (not merely local) validity without copying the values.
            let mut inherited = None;
            let ancestor_levels = nullable.len() - usize::from(field.is_nullable());
            for nulls in nullable[..ancestor_levels].iter().flatten() {
                inherited = NullBuffer::union(inherited.as_ref(), Some(nulls));
            }
            let values = if inherited.is_none() {
                array.clone()
            } else {
                let effective = NullBuffer::union(array.nulls(), inherited.as_ref());
                make_array(array.to_data().into_builder().nulls(effective).build()?)
            };
            leaves.push(LeafColumn {
                leaf: leaves.len(),
                column,
                field: field.clone(),
                path: path.clone().into(),
                values,
                levels: if nullable.is_empty() {
                    DefinitionLevels::required()
                } else {
                    DefinitionLevels {
                        nulls: nullable.clone().into(),
                        offset: 0,
                    }
                },
            });
        }
    }

    if field.is_nullable() {
        nullable.pop();
    }
    path.pop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StructArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{Field, Fields, Schema};

    #[test]
    fn computes_nested_definition_levels() {
        let child = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let children = Fields::from(vec![Field::new("value", DataType::Int64, true)]);
        let object = StructArray::new(
            children.clone(),
            vec![child],
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "object",
                DataType::Struct(children),
                true,
            )])),
            vec![Arc::new(object)],
        )
        .unwrap();

        let leaves = flatten(&batch).unwrap();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].levels.max(), 2);
        assert_eq!(
            (0..3)
                .map(|row| leaves[0].levels.value(row))
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
        assert_eq!(leaves[0].values.null_count(), 2);
    }
}
