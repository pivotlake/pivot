//! File-level Parquet VARIANT shredding.
//!
//! Parquet fixes a column's physical schema for the whole file, so shredding is
//! chosen after a file-worth of batches has been gathered and before it is cut
//! into row groups. For every VARIANT location we count the supported physical
//! types and select the most frequent one. Values of another type remain in the
//! spec's binary `value` fallback, preserving the document exactly.
//!
//! Object fields are inferred recursively. Hard limits bound both work and the
//! resulting footer for documents used as high-cardinality maps (for example,
//! user-provided telemetry attributes). Reaching a limit only forgoes an
//! optimization: omitted fields remain readable from `value`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet_variant::Variant;
use parquet_variant_compute::{VariantArray, VariantType, shred_variant, unshred_variant};

use super::error::WriteResult;

/// Enough for wide telemetry documents while preventing dynamic keys from
/// creating unbounded column counts. This is per VARIANT column, per file.
const MAX_INFERRED_FIELDS: usize = 256;
/// A malformed or machine-generated document should not turn schema inference
/// into unbounded recursion. Parquet's two wrappers per shredded object make
/// even this limit far deeper than practical query paths.
const MAX_INFERRED_DEPTH: usize = 16;

const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";

/// Convert every VARIANT batch to the canonical unshredded shape. Compaction
/// can receive batches from files with different shredding layouts; normalizing
/// them first gives `concat_batches` one stable schema. Already-unshredded
/// arrays are returned without rebuilding their values.
pub(super) fn unshred_batches(batches: Vec<RecordBatch>) -> WriteResult<Vec<RecordBatch>> {
    batches.into_iter().map(unshred_batch).collect()
}

fn unshred_batch(batch: RecordBatch) -> WriteResult<RecordBatch> {
    if !batch
        .schema()
        .fields()
        .iter()
        .any(|field| is_variant(field))
    {
        return Ok(batch);
    }

    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        if !is_variant(field) {
            fields.push(field.as_ref().clone());
            columns.push(column.clone());
            continue;
        }

        let variant = VariantArray::try_new(column.as_ref())?;
        let variant = if variant.typed_value_field().is_some() {
            unshred_variant(&variant)?
        } else {
            variant
        };
        // Files may disagree on whether the top-level column was declared
        // optional. Widening to nullable gives concatenation one canonical,
        // lossless schema; `shred_file` narrows the output again when the
        // complete file contains no SQL-null rows.
        fields.push(variant.field(field.name()).with_nullable(true));
        columns.push(Arc::new(variant.into_inner()) as ArrayRef);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Infer and apply one shredding layout to each VARIANT column in a complete
/// output file. Non-VARIANT columns are shared unchanged.
pub(super) fn shred_file(batch: RecordBatch) -> WriteResult<RecordBatch> {
    if !batch
        .schema()
        .fields()
        .iter()
        .any(|field| is_variant(field))
    {
        return Ok(batch);
    }

    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        if !is_variant(field) {
            fields.push(field.as_ref().clone());
            columns.push(column.clone());
            continue;
        }

        let variant = VariantArray::try_new(column.as_ref())?;
        let mut inference = Inference::new();
        for row in 0..variant.len() {
            if !variant.is_null(row) {
                inference.observe(variant.try_value(row)?, 0);
            }
        }

        let Some(shredding_type) = inference.root.inferred_type() else {
            // No supported value was found (for example, an all-null or
            // list/boolean-only file). The canonical binary representation is
            // already valid and avoids adding a useless typed leaf.
            fields.push(field.as_ref().clone());
            columns.push(column.clone());
            continue;
        };
        let shredded = shred_variant(&variant, &shredding_type)?;
        fields.push(shredded.field(field.name()));
        columns.push(Arc::new(shredded.into_inner()) as ArrayRef);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

#[inline]
pub(super) fn is_variant(field: &Field) -> bool {
    field.extension_type_name() == Some(VARIANT_EXTENSION_NAME)
        && field.try_extension_type::<VariantType>().is_ok()
}

/// Types the current Pivot leaf encoder and decoder both support. Integer
/// widths are deliberately grouped into Int64: JSON encodes integers at their
/// smallest physical width, and widening lets one typed leaf capture all of
/// them without overflow.
#[derive(Clone, Copy)]
enum Kind {
    Object = 0,
    Integer = 1,
    Float = 2,
    String = 3,
    Binary = 4,
}

const KINDS: [Kind; 5] = [
    Kind::Object,
    Kind::Integer,
    Kind::Float,
    Kind::String,
    Kind::Binary,
];

#[derive(Default)]
struct Node {
    counts: [usize; KINDS.len()],
    /// `shred_variant` accepts zero-scale decimals for an Int64 target. That
    /// conversion changes the Variant's semantic type, so such a value makes
    /// integer shredding unsafe at this location.
    blocks_integer: bool,
    fields: HashMap<String, Node>,
}

struct Inference {
    root: Node,
    remaining_fields: usize,
}

impl Inference {
    fn new() -> Self {
        Self {
            root: Node::default(),
            remaining_fields: MAX_INFERRED_FIELDS,
        }
    }

    fn observe(&mut self, value: Variant<'_, '_>, depth: usize) {
        observe_node(&mut self.root, value, depth, &mut self.remaining_fields);
    }
}

fn observe_node(
    node: &mut Node,
    value: Variant<'_, '_>,
    depth: usize,
    remaining_fields: &mut usize,
) {
    let kind = match value {
        Variant::Int8(_) | Variant::Int16(_) | Variant::Int32(_) | Variant::Int64(_) => {
            Kind::Integer
        }
        Variant::Float(_) | Variant::Double(_) => Kind::Float,
        Variant::String(_) | Variant::ShortString(_) => Kind::String,
        Variant::Binary(_) => Kind::Binary,
        Variant::Object(object) => {
            node.counts[Kind::Object as usize] += 1;
            if depth >= MAX_INFERRED_DEPTH {
                return;
            }
            for (name, child) in object.iter() {
                if let Some(stats) = node.fields.get_mut(name) {
                    observe_node(stats, child, depth + 1, remaining_fields);
                } else if *remaining_fields > 0 {
                    *remaining_fields -= 1;
                    let mut stats = Node::default();
                    observe_node(&mut stats, child, depth + 1, remaining_fields);
                    node.fields.insert(name.to_owned(), stats);
                }
            }
            return;
        }
        // The reader does not yet decode BOOLEAN leaves, and lists require
        // repetition levels. Temporal/decimal/UUID values are likewise outside
        // today's writable leaf set. They safely remain in `value`.
        Variant::Null
        | Variant::BooleanTrue
        | Variant::BooleanFalse
        | Variant::List(_)
        | Variant::Date(_)
        | Variant::TimestampMicros(_)
        | Variant::TimestampNtzMicros(_)
        | Variant::TimestampNanos(_)
        | Variant::TimestampNtzNanos(_)
        | Variant::Time(_)
        | Variant::Uuid(_) => return,
        Variant::Decimal4(decimal) => {
            node.blocks_integer |= decimal.scale() == 0;
            return;
        }
        Variant::Decimal8(decimal) => {
            node.blocks_integer |= decimal.scale() == 0;
            return;
        }
        Variant::Decimal16(decimal) => {
            node.blocks_integer |= decimal.scale() == 0;
            return;
        }
    };
    node.counts[kind as usize] += 1;
}

impl Node {
    /// Try candidates in descending frequency. An object with no inferable
    /// children is not useful, so the next-most-common primitive may win.
    fn inferred_type(&self) -> Option<DataType> {
        let mut candidates = KINDS;
        candidates.sort_by_key(|kind| std::cmp::Reverse(self.counts[*kind as usize]));
        let integers = self.counts[Kind::Integer as usize];
        let floats = self.counts[Kind::Float as usize];
        for kind in candidates {
            if self.counts[kind as usize] == 0 {
                break;
            }
            let data_type = match kind {
                // The Arrow shredding kernel deliberately permits numeric
                // coercions. Preserve Variant semantics instead: Int64 may
                // widen integer widths, but must not absorb decimals; Float64
                // may widen Float, but must not turn integers into doubles.
                // If floats outnumber integers, skipping both avoids choosing a
                // tiny integer leaf merely because the useful float leaf is
                // unsafe.
                Kind::Integer if self.blocks_integer || floats > integers => None,
                Kind::Integer => Some(DataType::Int64),
                Kind::Float if integers > 0 => None,
                Kind::Float => Some(DataType::Float64),
                Kind::String => Some(DataType::Utf8View),
                Kind::Binary => Some(DataType::BinaryView),
                Kind::Object => self.object_type(),
            };
            if data_type.is_some() {
                return data_type;
            }
        }
        None
    }

    fn object_type(&self) -> Option<DataType> {
        let mut fields: Vec<Field> = self
            .fields
            .iter()
            .filter_map(|(name, node)| {
                node.inferred_type()
                    .map(|data_type| Field::new(name, data_type, true))
            })
            .collect();
        // Variant object field order is not stable across input metadata
        // dictionaries. A sorted schema keeps output deterministic.
        fields.sort_unstable_by(|a, b| a.name().cmp(b.name()));
        (!fields.is_empty()).then(|| DataType::Struct(Fields::from(fields)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_select::concat::concat_batches;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant};

    fn variant_batch(rows: Vec<Option<&str>>) -> RecordBatch {
        let json: ArrayRef = Arc::new(StringArray::from(rows));
        let variant = json_to_variant(&json).unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![variant.field("doc")])),
            vec![Arc::new(variant.into_inner())],
        )
        .unwrap()
    }

    #[test]
    fn chooses_modal_types_recursively() {
        let batch = variant_batch(vec![
            Some(r#"{"age":10,"name":"a"}"#),
            Some(r#"{"age":20,"name":"b"}"#),
            Some(r#"{"age":"unknown","name":3}"#),
        ]);

        let shredded = shred_file(batch).unwrap();
        let doc = VariantArray::try_new(shredded.column(0).as_ref()).unwrap();
        let DataType::Struct(root) = doc.typed_value_field().unwrap().data_type() else {
            panic!("object should shred to a struct");
        };
        let age = root.iter().find(|field| field.name() == "age").unwrap();
        let DataType::Struct(age_fields) = age.data_type() else {
            panic!("each object field is a shredded variant wrapper");
        };
        assert_eq!(
            age_fields
                .iter()
                .find(|field| field.name() == "typed_value")
                .unwrap()
                .data_type(),
            &DataType::Int64
        );
    }

    #[test]
    fn unsupported_only_values_stay_unshredded() {
        let batch = variant_batch(vec![Some("true"), Some("false"), Some("null")]);

        let batch = shred_file(batch).unwrap();
        let doc = VariantArray::try_new(batch.column(0).as_ref()).unwrap();
        assert!(doc.typed_value_field().is_none());
    }

    #[test]
    fn differently_shredded_input_files_normalize_before_concat() {
        fn shredded_batch(json: &str, data_type: &DataType) -> RecordBatch {
            let json: ArrayRef = Arc::new(StringArray::from(vec![json]));
            let variant = json_to_variant(&json).unwrap();
            let schema = ShreddedSchemaBuilder::new()
                .with_path("value", data_type)
                .unwrap()
                .build();
            let variant = shred_variant(&variant, &schema).unwrap();
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![variant.field("doc")])),
                vec![Arc::new(variant.into_inner())],
            )
            .unwrap()
        }

        let batches = unshred_batches(vec![
            shredded_batch(r#"{"value":1}"#, &DataType::Int64),
            shredded_batch(r#"{"value":"two"}"#, &DataType::Utf8View),
        ])
        .unwrap();
        let combined = concat_batches(&batches[0].schema(), &batches).unwrap();
        let variant = VariantArray::try_new(combined.column(0).as_ref()).unwrap();

        assert_eq!(combined.num_rows(), 2);
        assert!(variant.typed_value_field().is_none());
    }
}
