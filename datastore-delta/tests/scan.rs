mod common;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, StringViewArray, StructArray,
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

    let result = ParquetTable::from_directory(&dispatch, dir.path(), &[]);

    let err = result.expect_err("LIST columns must not load");
    assert!(err.to_string().contains("not supported"), "{err}");
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
    let dispatch = dispatch(2);
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
    let spec = dispatch::values_input(&dispatch, vec![batch]).record_batches();
    let files: Vec<datastore_delta::parquet::writing::EncodedFile> =
        datastore_delta::parquet::writing::encode_record_batches(
            spec,
            Arc::from([]),
            Arc::from([]),
            400_000,
            1,
        )
        .collect()
        .unwrap();
    std::fs::write(dir.path().join("data.parquet"), &files[0].bytes).unwrap();
    let table = parquet_table_from_dir(&dispatch, dir.path());

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
