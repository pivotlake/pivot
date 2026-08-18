mod common;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int64Type};
use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray, StringViewArray,
    StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use common::*;
use datastore_delta::parquet::{ParquetTable, table_input};
use dispatch::{AggregationKind, AggregationSlot, Projection};

#[test]
fn scan_all_columns() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(
        &dispatch,
        &[strings_and_ints(
            &["a", "b", "c", "d", "e"],
            &[1, 2, 3, 4, 5],
        )],
        true,
    );

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    assert_eq!(results[0].num_columns(), 2);
}

#[test]
fn limit_can_abandon_parquet_reads() {
    let dispatch = dispatch(4);
    let dir = TempDir::new().unwrap();
    let batch = strings_and_ints(
        &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    );
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    for part in 0..8 {
        let file = std::fs::File::create(dir.path().join(format!("part{part}.parquet"))).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props.clone())).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .limit(10, 0)
        .collect()
        .unwrap();

    assert_eq!(
        results.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        10
    );
}

#[test]
fn scan_column_subset() {
    let dispatch = dispatch(1);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);

    let results = table_input(&dispatch, &table, Projection::columns([1]), false)
        .collect()
        .unwrap();

    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![10, 20, 30]);
}

// Regression: a projection with no data columns must still emit one row per
// table row. The column-driven page pipeline produces nothing for zero columns
// (no pages → no decoder), so a bare empty projection silently returned zero
// rows; it now routes to the dedicated empty-projection scan, and the row count
// survives CopyOut (which feeds `collect`).
#[test]
fn scan_empty_projection_emits_row_count() {
    let dispatch = dispatch(1);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]))],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);

    let results = table_input(&dispatch, &table, Projection::all(0), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
    assert!(results.iter().all(|b| b.num_columns() == 0));
}

#[test]
fn scan_nested_struct_column() {
    let dispatch = dispatch(1);
    // A struct column `user{id, name}` alongside a scalar `ts`, written by
    // arrow-rs (a trusted nested writer) and read back through pivot.
    let inner = Fields::from(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8View, false),
    ]);
    let user = StructArray::new(
        inner.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as _,
            Arc::new(StringViewArray::from(vec!["a", "bb", "ccc"])) as _,
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("user", DataType::Struct(inner), false),
        Field::new("ts", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(user) as _,
            Arc::new(Int64Array::from(vec![10, 20, 30])) as _,
        ],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    // Two top-level columns (`user{id, name}` and `ts`) over three leaf chunks.
    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let got = &results[0];
    assert_eq!(got.num_columns(), 2);
    let user = got.column(0).as_struct();
    assert_eq!(
        user.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![1, 2, 3])
    );
    assert_eq!(
        user.column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap(),
        &StringViewArray::from(vec!["a", "bb", "ccc"])
    );
    assert_eq!(
        got.column(1).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![10, 20, 30])
    );
}

#[test]
fn scan_nullable_column() {
    let dispatch = dispatch(1);
    let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let values = Int64Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as _]).unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    assert_eq!(
        results[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![Some(1), None, Some(3), None, Some(5)])
    );
}

#[test]
fn scan_nullable_float_with_boundary_nulls() {
    let dispatch = dispatch(1);
    let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, true)]));
    let values = Float64Array::from(vec![None, Some(1.5), Some(-2.0), None]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as _]).unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    assert_eq!(
        results[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap(),
        &Float64Array::from(vec![None, Some(1.5), Some(-2.0), None])
    );
}

/// Regression: an all-null dictionary-encoded column writes a dictionary page
/// with zero uncompressed bytes; the decompressor must not hand the snappy
/// decoder zero output buffers (it panics). Not variant-specific.
#[test]
fn scan_all_null_dictionary_column() {
    let dispatch = dispatch(1);
    let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
    let values = Int64Array::from(vec![None, None, None]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as _]).unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true); // dictionary enabled

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    assert_eq!(
        results[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![None, None, None])
    );
}

#[test]
fn scan_variant_column() {
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{GetOptions, json_to_variant, variant_get};

    let dispatch = dispatch(1);
    // A VARIANT-tagged column written by arrow-rs; pivot must read its binary
    // metadata/value leaves back as binary (not string) so `variant_get` works.
    let json: ArrayRef = Arc::new(StringArray::from(vec![r#"{"age":30}"#, r#"{"age":25}"#]));
    let variant = json_to_variant(&json).unwrap();
    let field = variant.field("doc");
    let doc = variant.into_inner();
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(doc) as _]).unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    // One top-level column `doc` (its metadata/value leaves reassemble).
    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    let age_field = Arc::new(Field::new("age", DataType::Int64, true));
    let ages = variant_get(
        results[0].column(0),
        GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
            .with_as_type(Some(age_field)),
    )
    .unwrap();
    let ages = ages.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(
        (0..ages.len()).map(|i| ages.value(i)).collect::<Vec<_>>(),
        vec![30, 25]
    );
}

#[test]
fn scan_shredded_variant() {
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    // A VARIANT-tagged column with `age` shredded into a typed `typed_value`
    // leaf (null where a row lacks it), written by arrow-rs; pivot reconstructs
    // the shredded variant and `variant_get` reads the typed leaf.
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"age":30}"#,
        r#"{"name":"bob"}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let field = shredded.field("doc");
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    // One top-level column `doc`, spanning this file's four shredded leaves.
    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    let as_int = Some(Arc::new(Field::new("age", DataType::Int64, true)));
    let ages = variant_get(
        results[0].column(0),
        GetOptions::new_with_path(VariantPath::try_from("age").unwrap()).with_as_type(as_int),
    )
    .unwrap();
    assert_eq!(
        ages.as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![Some(30), None])
    );
}

#[test]
fn scan_shredded_variant_with_a_leading_null() {
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    // `age` is absent in the first row (the shredded leaf's first def-level run
    // is "null") and present in the second, exercising the multi-level scatter
    // when it leads with an absent run.
    let json: ArrayRef = Arc::new(StringArray::from(vec![r#"{"name":"x"}"#, r#"{"age":42}"#]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    let ages = variant_get(
        results[0].column(0),
        GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
            .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
    )
    .unwrap();
    assert_eq!(
        ages.as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![None, Some(42)])
    );
}

#[test]
fn scan_variant_with_a_null_document_row() {
    use arrow_array::Array;
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{GetOptions, json_to_variant, variant_get};

    let dispatch = dispatch(1);
    // The second document is SQL NULL: its required `metadata` leaf decodes
    // with a null there, which only a struct-level null mask may carry.
    let json: ArrayRef = Arc::new(StringArray::from(vec![Some(r#"{"age":30}"#), None]));
    let variant = json_to_variant(&json).unwrap();
    let field = variant.field("doc");
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(variant.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    let doc = results[0].column(0);
    assert!(doc.is_null(1));
    let ages = variant_get(
        doc,
        GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
            .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
    )
    .unwrap();
    assert_eq!(
        ages.as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![Some(30), None])
    );
}

#[test]
fn scan_shredded_variant_with_a_null_document_row() {
    use arrow_array::Array;
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    // A NULL document in a SHREDDED file: the null must survive through the
    // deeper struct nesting (typed_value groups) as a struct-level mask.
    let json: ArrayRef = Arc::new(StringArray::from(vec![Some(r#"{"age":30}"#), None]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    let doc = results[0].column(0);
    assert!(doc.is_null(1));
    let ages = variant_get(
        doc,
        GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
            .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
    )
    .unwrap();
    assert_eq!(
        ages.as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![Some(30), None])
    );
}

#[test]
fn scan_variant_same_path_shredded_as_different_types() {
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    // Both files shred the SAME path, one as Int64 and one as Utf8, so their
    // typed leaves carry different arrow types.
    for (i, (json_rows, ty)) in [
        (vec![r#"{"age":30}"#], DataType::Int64),
        (vec![r#"{"age":"forty"}"#], DataType::Utf8),
    ]
    .into_iter()
    .enumerate()
    {
        let json: ArrayRef = Arc::new(StringArray::from(json_rows));
        let shred = ShreddedSchemaBuilder::new()
            .with_path("age", &ty)
            .unwrap()
            .build();
        let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![shredded.field("doc")])),
            vec![Arc::new(shredded.into_inner()) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.path().join(format!("part{i}.parquet"))).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props.clone())).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    // Reading as BIGINT hits file 0's typed leaf and nulls file 1's string.
    let mut ages: Vec<Option<i64>> = results
        .iter()
        .flat_map(|b| {
            let a = variant_get(
                b.column(0),
                GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
                    .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
            )
            .unwrap();
            a.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    ages.sort();
    assert_eq!(ages, vec![None, Some(30)]);
}

#[test]
fn scan_nullable_strings_produce_valid_views() {
    use arrow_array::Array;

    // Null rows still occupy view slots, and kernels may touch masked slots
    // before applying validity, so every slot must hold a valid view.
    for dictionary in [false, true] {
        let dispatch = dispatch(1);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)])),
            vec![Arc::new(StringViewArray::from(vec![
                Some("a string too long to inline in a view"),
                None,
                Some("b"),
                None,
            ]))],
        )
        .unwrap();
        let (_dir, table) = parquet_table(&dispatch, &[batch], dictionary);

        let results = table_input(&dispatch, &table, Projection::all(1), false)
            .collect()
            .unwrap();

        let strings = results[0].column(0);
        strings.to_data().validate_full().unwrap();
        assert_eq!(strings.null_count(), 2);
    }
}

#[test]
fn scan_dictionary_encoded_variant() {
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"age":30}"#,
        r#"{"age":30}"#,
        r#"{"age":25}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true); // dictionary ENABLED

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    let ages = variant_get(
        results[0].column(0),
        GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
            .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
    )
    .unwrap();
    assert_eq!(
        ages.as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![30, 30, 25])
    );
}

/// A batch laid out `a, s{x, y}, b`: four leaves where a struct sits between
/// scalars, so leaf (column-chunk) indices and top-level field indices diverge.
fn struct_between_scalars_batch() -> RecordBatch {
    let inner = Fields::from(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Utf8View, false),
    ]);
    let s = StructArray::new(
        inner.clone(),
        vec![
            Arc::new(Int64Array::from(vec![10, 20])) as _,
            Arc::new(StringViewArray::from(vec!["p", "q"])) as _,
        ],
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("s", DataType::Struct(inner), false),
        Field::new("b", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as _,
            Arc::new(s) as _,
            Arc::new(Int64Array::from(vec![100, 200])) as _,
        ],
    )
    .unwrap()
}

#[test]
fn scan_struct_between_scalars_keeps_leaf_order() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[struct_between_scalars_batch()], false);

    let results = table_input(&dispatch, &table, Projection::all(3), false)
        .collect()
        .unwrap();

    let got = &results[0];
    assert_eq!(got.num_columns(), 3);
    let s = got.column(1).as_struct();
    assert_eq!(
        got.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![1, 2])
    );
    assert_eq!(
        s.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![10, 20])
    );
    assert_eq!(
        s.column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap(),
        &StringViewArray::from(vec!["p", "q"])
    );
    assert_eq!(
        got.column(2).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![100, 200])
    );
}

#[test]
fn scan_scalar_after_struct() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[struct_between_scalars_batch()], false);

    // Column `b` sits after the struct, so its leaf chunk is at index 3, not 2;
    // projecting the column must fetch and decode that chunk.
    let results = table_input(&dispatch, &table, Projection::columns([2]), false)
        .collect()
        .unwrap();

    let got = &results[0];
    assert_eq!(got.num_columns(), 1);
    assert_eq!(got.schema().field(0).name(), "b");
    assert_eq!(collect_i64s(&results, 0), vec![100, 200]);
}

#[test]
fn scan_struct_only() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[struct_between_scalars_batch()], false);

    // Projecting the struct column reads both its leaves and reassembles them.
    let results = table_input(&dispatch, &table, Projection::columns([1]), false)
        .collect()
        .unwrap();

    let got = &results[0];
    assert_eq!(got.num_columns(), 1);
    let s = got.column(0).as_struct();
    assert_eq!(
        s.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![10, 20])
    );
    assert_eq!(
        s.column(1)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap(),
        &StringViewArray::from(vec!["p", "q"])
    );
}

#[test]
fn scan_variant_shredded_differently_per_file() {
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    // Two files, same top-level `doc` column, DIFFERENT physical layouts:
    // file 0 shreds `age` (Int64), file 1 shreds `name` (Utf8). The projection
    // must resolve to each file's own leaves; a single fixed leaf list would
    // silently mis-decode one of the files.
    let files = [
        (
            vec![r#"{"age":30}"#, r#"{"age":25}"#],
            "age",
            DataType::Int64,
        ),
        (
            vec![r#"{"name":"x"}"#, r#"{"name":"y"}"#],
            "name",
            DataType::Utf8,
        ),
    ];
    for (i, (rows, path, ty)) in files.iter().enumerate() {
        let json: ArrayRef = Arc::new(StringArray::from(rows.clone()));
        let shred = ShreddedSchemaBuilder::new()
            .with_path(*path, ty)
            .unwrap()
            .build();
        let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![shredded.field("doc")])),
            vec![Arc::new(shredded.into_inner()) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.path().join(format!("part{i}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props.clone())).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(1), false)
        .collect()
        .unwrap();

    // `age` is present (typed) in file 0 and absent (null) in file 1.
    let mut ages: Vec<Option<i64>> = results
        .iter()
        .flat_map(|b| {
            let a = variant_get(
                b.column(0),
                GetOptions::new_with_path(VariantPath::try_from("age").unwrap())
                    .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
            )
            .unwrap();
            a.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    ages.sort();
    assert_eq!(ages, vec![None, None, Some(25), Some(30)]);
}

#[test]
fn scan_multiple_parquet_files() {
    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8View, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    for (i, batch) in [
        strings_and_ints(&["a", "b"], &[1, 2]),
        strings_and_ints(&["c", "d", "e"], &[3, 4, 5]),
    ]
    .iter()
    .enumerate()
    {
        let file = std::fs::File::create(dir.path().join(format!("part{i}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), Some(props.clone())).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
    }
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 5);
}

#[test]
fn list_columns_are_rejected_at_load() {
    use arrow_array::Array;
    use arrow_array::builder::{Int64Builder, ListBuilder};

    // The decoder has no repetition-level support, so a LIST column must fail
    // the load cleanly rather than misdecode its pages.
    let dispatch = dispatch(1);
    let mut tags = ListBuilder::new(Int64Builder::new());
    tags.append_value([Some(1), Some(2)]);
    tags.append_value([Some(3)]);
    let tags = tags.finish();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "tags",
            tags.data_type().clone(),
            true,
        )])),
        vec![Arc::new(tags) as _],
    )
    .unwrap();
    let dir = TempDir::new().unwrap();
    let file = std::fs::File::create(dir.path().join("lists.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let result = ParquetTable::from_files(&dispatch, &parquet_files_in(dir.path()), &[]);

    let err = result.expect_err("LIST columns must not load");
    assert!(err.to_string().contains("not supported"), "{err}");
}

/// Writes a one-column timestamp file at `unit`, the way another engine would,
/// and returns the directory holding it.
fn timestamp_file(unit: arrow_schema::TimeUnit, counts: &[i64]) -> TempDir {
    use arrow_array::{TimestampMicrosecondArray, TimestampMillisecondArray};
    use arrow_schema::TimeUnit;

    let counts = counts.to_vec();
    let values: ArrayRef = match unit {
        TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from(counts)),
        TimeUnit::Millisecond => Arc::new(TimestampMillisecondArray::from(counts)),
        other => panic!("the reader only takes microseconds; {other:?} is for the reject case"),
    };
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(unit, None),
            false,
        )])),
        vec![values],
    )
    .unwrap();

    let dir = TempDir::new().unwrap();
    let file = std::fs::File::create(dir.path().join("timestamps.parquet")).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    dir
}

/// A microsecond timestamp column reads back unscaled, sub-second part and all,
/// whether or not the table declares the column: the file's unit is the one a
/// pivot timestamp counts in.
#[test]
fn scan_reads_a_microsecond_timestamp_column() {
    use arrow_array::types::TimestampMicrosecondType;
    use arrow_schema::TimeUnit;

    let dispatch = dispatch(1);
    let counts = [1_700_000_000_123_456, 0, -1_500_000];
    let dir = timestamp_file(TimeUnit::Microsecond, &counts);
    let declared = [planner::catalog::Column {
        name: "ts".to_string(),
        col_type: planner::types::Type::Timestamp,
    }];

    for columns in [&declared[..], &[][..]] {
        let table = Arc::new(
            ParquetTable::from_files(&dispatch, &parquet_files_in(dir.path()), columns).unwrap(),
        );
        let results = table_input(&dispatch, &table, Projection::all(1), false)
            .collect()
            .unwrap();

        assert_eq!(
            *results[0].column(0).data_type(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let read: Vec<i64> = results
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<TimestampMicrosecondType>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(read, counts);
    }
}

/// A file at any other resolution fails the load: its counts mean something
/// else, and reading them as microseconds would misplace every value by the
/// ratio between the two units.
/// A file whose timestamp leaf is stamped UTC-adjusted, the way most engines
/// write instants, reads back as a zone-carrying timestamp when nothing is
/// declared, and as whichever timestamp type the table does declare.
#[test]
fn scan_honors_a_files_utc_adjusted_flag() {
    use arrow_array::TimestampMicrosecondArray;
    use arrow_array::types::TimestampMicrosecondType;
    use arrow_schema::TimeUnit;

    let dispatch = dispatch(1);
    let counts = vec![1_700_000_000_123_456i64, 0];
    let values: ArrayRef =
        Arc::new(TimestampMicrosecondArray::from(counts.clone()).with_timezone("UTC"));
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        )])),
        vec![values],
    )
    .unwrap();
    let dir = TempDir::new().unwrap();
    let file = std::fs::File::create(dir.path().join("instants.parquet")).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let declared_tz = [planner::catalog::Column {
        name: "ts".to_string(),
        col_type: planner::types::Type::TimestampTz,
    }];
    let declared_naive = [planner::catalog::Column {
        name: "ts".to_string(),
        col_type: planner::types::Type::Timestamp,
    }];
    let cases: [(&[planner::catalog::Column], DataType); 3] = [
        (
            &[],
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        ),
        (
            &declared_tz,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        ),
        (
            &declared_naive,
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ),
    ];
    for (columns, expected_type) in cases {
        let table = Arc::new(
            ParquetTable::from_files(&dispatch, &parquet_files_in(dir.path()), columns).unwrap(),
        );
        let results = table_input(&dispatch, &table, Projection::all(1), false)
            .collect()
            .unwrap();

        assert_eq!(*results[0].column(0).data_type(), expected_type);
        let read: Vec<i64> = results
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<TimestampMicrosecondType>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(read, counts);
    }
}

#[test]
fn a_timestamp_file_in_another_unit_is_rejected() {
    use arrow_schema::TimeUnit;

    let dispatch = dispatch(1);
    let dir = timestamp_file(TimeUnit::Millisecond, &[1_700_000_000_000]);

    let err = ParquetTable::from_files(&dispatch, &parquet_files_in(dir.path()), &[])
        .expect_err("a millisecond timestamp file must not load");

    let message = err.to_string();
    assert!(
        message.contains("MILLIS") && message.contains("microsecond"),
        "unhelpful rejection: {message}"
    );
}

#[test]
fn materialize_rejects_corrupt_footer_without_panicking() {
    // A file whose trailing `[footer_len][PAR1]` claims a footer larger than the
    // whole file. The footer reader must surface a clean error, not underflow the
    // offset math (`size - 8 - footer_len`) into a wild cache read.
    let dispatch = dispatch(1);
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("corrupt.parquet");
    // 16 bytes: 8 filler + footer_len = u32::MAX + "PAR1".
    let mut bytes = vec![0u8; 8];
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(b"PAR1");
    std::fs::write(&path, &bytes).unwrap();

    let result = ParquetTable::from_files(&dispatch, &[&path], &[]);
    assert!(
        result.is_err(),
        "a footer longer than the file must error, not panic"
    );
}

#[test]
fn scan_empty_table() {
    let dispatch = dispatch(1);
    let (_dir, table) = parquet_table(&dispatch, &[strings_and_ints(&[], &[])], true);

    let results = table_input(&dispatch, &table, Projection::all(0), false)
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 0);
}

/// A column written `DELTA_BINARY_PACKED` by arrow's writer reads back with the
/// same values as the plain-encoded column beside it, which is the check the
/// hand-built decoder unit tests cannot make: it decodes what another
/// implementation actually wrote.
#[test]
fn scan_delta_binary_packed_column() {
    let dispatch = dispatch(2);
    let rows: Vec<i64> = (0..50_000).map(|i| 1_000_000 + i * 37 % 999_983).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("delta", DataType::Int64, false),
        Field::new("plain", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(rows.clone())) as ArrayRef,
            Arc::new(Int64Array::from(rows.clone())) as ArrayRef,
        ],
    )
    .unwrap();
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(false)
        .set_column_encoding(
            "delta".into(),
            parquet::basic::Encoding::DELTA_BINARY_PACKED,
        )
        .set_column_encoding("plain".into(), parquet::basic::Encoding::PLAIN)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(dir.path().join("data.parquet")).unwrap(),
        schema,
        Some(props),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    let mut delta: Vec<i64> = Vec::new();
    let mut plain: Vec<i64> = Vec::new();
    for batch in &results {
        delta.extend(
            batch
                .column(0)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values(),
        );
        plain.extend(
            batch
                .column(1)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values(),
        );
    }
    delta.sort_unstable();
    plain.sort_unstable();
    let mut expected = rows;
    expected.sort_unstable();
    assert_eq!(delta, expected);
    assert_eq!(plain, expected);
}

/// A string column written `DELTA_LENGTH_BYTE_ARRAY` by arrow's writer reads
/// back with the same values as the plain-encoded copy beside it. The column is
/// sized past one 2 MiB buffer so values land across a buffer boundary too, and
/// mixes lengths either side of the twelve bytes a view inlines.
#[test]
fn scan_delta_length_byte_array_column() {
    let dispatch = dispatch(2);
    let rows: Vec<String> = (0..60_000)
        .map(|i: usize| {
            if i.is_multiple_of(5) {
                format!("s{i}")
            } else {
                format!("a much longer value that will not inline, number {i:012}")
            }
        })
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("delta", DataType::Utf8View, false),
        Field::new("plain", DataType::Utf8View, false),
    ]));
    let column: ArrayRef = Arc::new(StringViewArray::from(
        rows.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![column.clone(), column.clone()]).unwrap();
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(false)
        .set_column_encoding(
            "delta".into(),
            parquet::basic::Encoding::DELTA_LENGTH_BYTE_ARRAY,
        )
        .set_column_encoding("plain".into(), parquet::basic::Encoding::PLAIN)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(dir.path().join("data.parquet")).unwrap(),
        schema,
        Some(props),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let table = parquet_table_from_dir(&dispatch, dir.path());

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    let mut delta = collect_strings(&results, 0);
    let mut plain = collect_strings(&results, 1);
    delta.sort();
    plain.sort();
    let mut expected = rows;
    expected.sort();
    assert_eq!(delta, expected);
    assert_eq!(plain, expected);
}

/// A column our own writer delta-encodes reads back through our own scan, so
/// the two halves of the encoding agree with each other and not only with
/// arrow's reader.
#[test]
fn scan_a_column_our_writer_delta_encoded() {
    // The rows are accumulated into ring-backed row groups before they are
    // encoded, so the ring has to hold this file's 200k rows (~10 MB of keys and
    // string bytes) as well as the reads that follow.
    let dispatch = dispatch_with_buffers(2, 32);
    let keys: Vec<i64> = (0..200_000).map(|i| 5_000_000 + i * 7919).collect();
    let names: Vec<String> = (0..200_000)
        .map(|i| format!("value {i} with enough tail to not inline"))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("name", DataType::Utf8View, false),
        ])),
        vec![
            Arc::new(Int64Array::from(keys.clone())) as ArrayRef,
            Arc::new(StringViewArray::from(
                names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let dir = TempDir::new().unwrap();
    let written = write_parquet_files(&dispatch, dir.path(), vec![batch]);
    let table = parquet_table_from_dir(&dispatch, &written);

    let results = table_input(&dispatch, &table, Projection::all(2), false)
        .collect()
        .unwrap();

    let mut read_keys: Vec<i64> = Vec::new();
    let mut read_names: Vec<String> = Vec::new();
    for batch in &results {
        read_keys.extend(
            batch
                .column(0)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values(),
        );
        read_names.extend(
            batch
                .column(1)
                .as_string_view()
                .iter()
                .map(|v| v.unwrap().to_string()),
        );
    }
    read_keys.sort_unstable();
    read_names.sort();
    let mut expected_keys = keys;
    let mut expected_names = names;
    expected_keys.sort_unstable();
    expected_names.sort();
    assert_eq!(read_keys, expected_keys);
    assert_eq!(read_names, expected_names);
}

fn shredded_variant_batch(rows: &[&str], path: &str, data_type: &DataType) -> RecordBatch {
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
    let shredding = ShreddedSchemaBuilder::new()
        .with_path(path, data_type)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shredding).unwrap();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as ArrayRef],
    )
    .unwrap()
}

fn corrupt_parquet_leaves(dir: &std::path::Path, table: &ParquetTable, leaves: &[usize]) {
    use std::io::{Seek, SeekFrom, Write};

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.join("data.parquet"))
        .unwrap();
    for &leaf in leaves {
        let chunk = &table.row_groups()[0].columns[leaf];
        let offset = chunk
            .dictionary_page_offset
            .unwrap_or(chunk.data_page_offset);
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        file.write_all(&vec![0xff; chunk.total_compressed_size as usize])
            .unwrap();
    }
}

/// A pushed extract on a path shredded into a typed leaf, with every row's
/// value in that leaf, reads the leaf directly and emits a plain scalar column
/// instead of the whole variant.
#[test]
fn scan_pushed_extract_reads_a_shredded_leaf_directly() {
    use dispatch::VariantExtract;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![r#"{"age":30}"#, r#"{"age":25}"#]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["age".to_string()],
            as_type: Some(DataType::Int64),
        })],
    );
    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    assert_eq!(results[0].column(0).data_type(), &DataType::Int64);
    assert_eq!(
        results[0]
            .column(0)
            .as_primitive::<arrow_array::types::Int64Type>(),
        &Int64Array::from(vec![30, 25])
    );
}

#[test]
fn scan_pushed_numeric_extract_reads_only_its_typed_leaf() {
    use dispatch::VariantExtract;

    // Setup
    let dispatch = dispatch(1);
    let batch = shredded_variant_batch(
        &[r#"{"age":30}"#, r#"{"age":null}"#, r#"{}"#],
        "age",
        &DataType::Int64,
    );
    let (dir, table) = parquet_table(&dispatch, &[batch], true);
    corrupt_parquet_leaves(dir.path(), &table, &[0, 1, 2]);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["age".to_string()],
            as_type: Some(DataType::Int64),
        })],
    );

    // Execute
    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    // Assert
    assert_eq!(
        results[0].column(0).as_primitive::<Int64Type>(),
        &Int64Array::from(vec![Some(30), None, None])
    );
}

/// A typed extract of a path that no row has: the file shreds every row
/// perfectly, so the untyped fallback is statistically all NULL and the path
/// resolves to SQL NULL everywhere.
#[test]
fn scan_pushed_extract_of_an_absent_path_yields_nulls() {
    use dispatch::VariantExtract;

    let dispatch = dispatch(1);
    let batch =
        shredded_variant_batch(&[r#"{"age":30}"#, r#"{"age":25}"#], "age", &DataType::Int64);
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["latency".to_string()],
            as_type: Some(DataType::Float32),
        })],
    );

    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    assert_eq!(
        results[0].column(0).as_primitive::<Float32Type>(),
        &Float32Array::from(vec![None, None])
    );
}

/// Like the flat case, but the path diverges below the top level: the file
/// shreds `user.name`, the extract wants `user.age`, and every untyped
/// fallback is all NULL.
#[test]
fn scan_pushed_extract_of_a_nested_absent_path_yields_nulls() {
    use dispatch::VariantExtract;

    let dispatch = dispatch(1);
    let batch = shredded_variant_batch(
        &[r#"{"user":{"name":"bob"}}"#, r#"{"user":{"name":"amy"}}"#],
        "user.name",
        &DataType::Utf8View,
    );
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["user".to_string(), "age".to_string()],
            as_type: Some(DataType::Float32),
        })],
    );

    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    assert_eq!(
        results[0].column(0).as_primitive::<Float32Type>(),
        &Float32Array::from(vec![None, None])
    );
}

/// A bare extract of a provably absent path still emits a variant column with
/// every row SQL NULL, so it merges with files whose layouts do hold the path.
#[test]
fn scan_pushed_bare_extract_of_an_absent_path_yields_a_null_variant() {
    use dispatch::VariantExtract;

    let dispatch = dispatch(1);
    let batch =
        shredded_variant_batch(&[r#"{"age":30}"#, r#"{"age":25}"#], "age", &DataType::Int64);
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["latency".to_string()],
            as_type: None,
        })],
    );

    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    let column = results[0].column(0);
    let DataType::Struct(fields) = column.data_type() else {
        panic!("expected a variant struct, got {:?}", column.data_type());
    };
    assert_eq!(fields[0].name(), "metadata");
    assert_eq!(fields[1].name(), "value");
    assert_eq!(column.null_count(), 2);
}

#[test]
fn scan_pushed_text_extract_distinguishes_json_null_from_missing() {
    use dispatch::VariantExtract;

    // Setup
    let dispatch = dispatch(1);
    let batch = shredded_variant_batch(
        &[
            r#"{"name":"bob"}"#,
            r#"{"name":null}"#,
            r#"{"other":"present"}"#,
        ],
        "name",
        &DataType::Utf8View,
    );
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["name".to_string()],
            as_type: Some(DataType::Utf8View),
        })],
    );

    // Execute
    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    // Assert
    assert_eq!(
        results[0].column(0).as_string_view(),
        &StringViewArray::from(vec![Some("bob"), Some("null"), None])
    );
}

#[test]
fn scan_pushed_text_extract_skips_an_ancestor_json_null() {
    use dispatch::VariantExtract;

    // Setup
    let dispatch = dispatch(1);
    let batch = shredded_variant_batch(
        &[r#"{"user":{"name":"bob"}}"#, r#"{"user":null}"#, r#"{}"#],
        "user.name",
        &DataType::Utf8View,
    );
    let (dir, table) = parquet_table(&dispatch, &[batch], true);
    corrupt_parquet_leaves(dir.path(), &table, &[0, 1, 2, 3]);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["user".to_string(), "name".to_string()],
            as_type: Some(DataType::Utf8View),
        })],
    );

    // Execute
    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    // Assert
    assert_eq!(
        results[0].column(0).as_string_view(),
        &StringViewArray::from(vec![Some("bob"), None, None])
    );
}

/// A pushed extract on a path that is not shredded here reads its binary
/// fallback and applies the SQL cast per row.
#[test]
fn scan_pushed_extract_falls_back_for_an_unshredded_path() {
    use dispatch::VariantExtract;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"age":30}"#,
        r#"{"name":"bob"}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["name".to_string()],
            as_type: Some(DataType::Utf8View),
        })],
    );
    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    assert_eq!(results[0].column(0).data_type(), &DataType::Utf8View);
    assert_eq!(
        results[0].column(0).as_string_view(),
        &StringViewArray::from(vec![None, Some("bob")])
    );
}

/// A bare extract (no cast) yields the sub-variant at the path, read from only
/// its subtree; casting the emitted sub-variant recovers the value.
#[test]
fn scan_pushed_bare_extract_yields_a_subvariant() {
    use dispatch::VariantExtract;
    use parquet_variant::VariantPath;
    use parquet_variant_compute::{
        GetOptions, ShreddedSchemaBuilder, json_to_variant, shred_variant, variant_get,
    };

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![r#"{"age":30}"#, r#"{"age":25}"#]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], false);

    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["age".to_string()],
            as_type: None,
        })],
    );
    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    // The emitted column is the `age` sub-variant; cast it back to read the value.
    let ages =
        variant_get(
            results[0].column(0),
            GetOptions::new_with_path(VariantPath::default())
                .with_as_type(Some(Arc::new(Field::new("age", DataType::Int64, true)))),
        )
        .unwrap();
    assert_eq!(
        ages.as_any().downcast_ref::<Int64Array>().unwrap(),
        &Int64Array::from(vec![Some(30), Some(25)])
    );
}

/// A shredded variant path that a pushed extract reads as its own typed leaf,
/// unchanged, can carry a pushed-down equality constant: the scan drops rows
/// whose dictionary entry differs before the batch leaves the decoder.
#[test]
fn scan_pushed_extract_filters_by_an_equality_constant() {
    use arrow_array::Scalar;
    use datastore_delta::parquet::{
        ScanEqualityPredicate, table_input_with_filter_and_eq_predicates,
    };
    use dispatch::VariantExtract;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"name":"alice"}"#,
        r#"{"name":"bob"}"#,
        r#"{"name":"carol"}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("name", &DataType::Utf8View)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["name".to_string()],
            as_type: Some(DataType::Utf8View),
        })],
    );
    let predicate = ScanEqualityPredicate {
        column_idx: 0,
        path: vec!["name".to_string()],
        value: Scalar::new(Arc::new(StringViewArray::from(vec!["bob"])) as ArrayRef),
    };

    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        projection,
        false,
        None,
        None,
        Arc::new(vec![predicate]),
    )
    .collect()
    .unwrap();

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert_eq!(
        results[0].column(0).as_string_view(),
        &StringViewArray::from(vec![Some("bob")])
    );
}

/// A path this file does not shred makes the extract rebuild the whole variant,
/// so the emitted column is no longer the leaf the constant's dictionary
/// describes. The constant must not be installed there, and every row survives
/// for the query's own `Filter` to judge.
#[test]
fn scan_pushed_extract_ignores_an_equality_constant_it_cannot_apply() {
    use arrow_array::Scalar;
    use datastore_delta::parquet::{
        ScanEqualityPredicate, table_input_with_filter_and_eq_predicates,
    };
    use dispatch::VariantExtract;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"age":30}"#,
        r#"{"name":"bob"}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["name".to_string()],
            as_type: Some(DataType::Utf8View),
        })],
    );
    let predicate = ScanEqualityPredicate {
        column_idx: 0,
        path: vec!["name".to_string()],
        value: Scalar::new(Arc::new(StringViewArray::from(vec!["bob"])) as ArrayRef),
    };

    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        projection,
        false,
        None,
        None,
        Arc::new(vec![predicate]),
    )
    .collect()
    .unwrap();

    assert_eq!(
        results[0].column(0).as_string_view(),
        &StringViewArray::from(vec![None, Some("bob")])
    );
}

/// A shredded typed leaf the extract has to cast reaches the batch as the cast
/// array, not as the leaf. The constant describes the leaf's dictionary views,
/// so it must not be installed: applying it to the cast column would compare
/// against views that no longer exist.
#[test]
fn scan_pushed_extract_ignores_an_equality_constant_across_a_cast() {
    use arrow_array::Scalar;
    use datastore_delta::parquet::{
        ScanEqualityPredicate, table_input_with_filter_and_eq_predicates,
    };
    use dispatch::VariantExtract;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"name":"alice"}"#,
        r#"{"name":"bob"}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("name", &DataType::Utf8View)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    // The leaf is Utf8View, so emitting Utf8 casts its representation after
    // decoding while preserving the extracted text.
    let projection = Projection::columns_with_extracts(
        vec![0],
        vec![Some(VariantExtract {
            path: vec!["name".to_string()],
            as_type: Some(DataType::Utf8),
        })],
    );
    let predicate = ScanEqualityPredicate {
        column_idx: 0,
        path: vec!["name".to_string()],
        value: Scalar::new(Arc::new(StringViewArray::from(vec!["bob"])) as ArrayRef),
    };

    let results = table_input_with_filter_and_eq_predicates(
        &dispatch,
        &table,
        projection,
        false,
        None,
        None,
        Arc::new(vec![predicate]),
    )
    .collect()
    .unwrap();

    assert_eq!(
        results[0].column(0).as_string::<i32>(),
        &arrow_array::StringArray::from(vec!["alice", "bob"])
    );
}

/// Two pushed extracts on one variant column resolve to overlapping leaves: the
/// shredded `age` leaf is also part of the whole-column read that unshredded
/// `name` falls back to. The column is decoded once, so each output has to fold
/// its own view of that one decode and still come back with its own value.
#[test]
fn scan_two_pushed_extracts_on_one_column_share_the_read() {
    use dispatch::VariantExtract;
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = dispatch(1);
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"age":30,"name":"alice"}"#,
        r#"{"age":25,"name":"bob"}"#,
    ]));
    let shred = ShreddedSchemaBuilder::new()
        .with_path("age", &DataType::Int64)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner()) as _],
    )
    .unwrap();
    let (_dir, table) = parquet_table(&dispatch, &[batch], true);
    let extract = |path: &str, as_type| {
        Some(VariantExtract {
            path: vec![path.to_string()],
            as_type: Some(as_type),
        })
    };
    let projection = Projection::columns_with_extracts(
        vec![0, 0],
        vec![
            extract("age", DataType::Int64),
            extract("name", DataType::Utf8View),
        ],
    );

    let results = table_input(&dispatch, &table, projection, false)
        .collect()
        .unwrap();

    assert_eq!(
        results[0].column(0).as_primitive::<Int64Type>(),
        &Int64Array::from(vec![30, 25])
    );
    assert_eq!(
        results[0].column(1).as_string_view(),
        &StringViewArray::from(vec!["alice", "bob"])
    );
}

// A late-materialized fetch whose one row group's rows arrive split across
// many metadata batches (spread over the workers) must still issue exactly one
// request for the group: the decoder tracks row groups by index and drops
// pages of a group it already finished, so a second request for the same group
// silently loses its rows.
#[test]
fn materialize_reads_a_row_group_split_across_batches_exactly_once() {
    use arrow_array::Int64Array;
    use datastore_delta::parquet::materialize;

    let dispatch = dispatch(4);
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batches: Vec<RecordBatch> = (0..8)
        .map(|f| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from_iter_values(
                    f * 100_000..(f + 1) * 100_000,
                ))],
            )
            .unwrap()
        })
        .collect();
    let dir = tempfile::TempDir::new().unwrap();
    for (i, batch) in batches.iter().enumerate() {
        let file = std::fs::File::create(dir.path().join(format!("f{i}.parquet"))).unwrap();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }
    let table = parquet_table_from_dir(&dispatch, dir.path());
    // The LIMIT gathers the metadata batches onto one worker and re-emits them
    // in a burst, which is what spreads one row group's batches across the
    // other workers downstream.
    let narrow = table_input(&dispatch, &table, Projection::columns([]), true).limit(400_000, 0);

    let results = materialize(narrow, table.clone(), Projection::columns([0]))
        .collect()
        .unwrap();

    let total: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 400_000);
}
