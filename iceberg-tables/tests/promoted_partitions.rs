//! A partition of a table whose partition column was widened by another
//! tool: a value written before the promotion and the same value written
//! after it are one partition, whichever width a manifest holds it in.
//!
//! Every data file here stores the wide type, so nothing depends on reading
//! a file that stores the narrow one.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal64Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    Scalar,
};
use arrow_schema::{Field, Schema as ArrowSchema};
use iceberg::spec::{NestedField, PrimitiveType, Schema, Type as IcebergType};
use iceberg_tables::Table;
use iceberg_tables::files::data_files::AddedDataFile;
use object_storage::{FileRef, ObjectPath};
use parquet::arrow::ArrowWriter;
use parquet_engine::{FileStats, PartitionValues};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use support::{
    Fixture, append, column, commit_replace, create_table_of, describe, find_partition_holding,
    list_partitions, paths, publish_changed, selected_sorted,
};

/// A widening of the column `v` a table is partitioned by: the Pivot type
/// it has before and after, the Iceberg type it is promoted to, and a value
/// of each width for a number a test names.
struct Promotion {
    narrow_type: Type,
    wide_type: Type,
    promoted_to: PrimitiveType,
    build_narrow: fn(i32) -> ArrayRef,
    build_wide: fn(i32) -> ArrayRef,
}

fn build_decimal(number: i32, precision: u8) -> ArrayRef {
    let unscaled = Decimal64Array::from(vec![i64::from(number)]);
    Arc::new(unscaled.with_precision_and_scale(precision, 2).unwrap())
}

/// An `int` widened to a `long`, a `float` to a `double`, and a decimal to
/// one of more digits.
fn list_promotions() -> [Promotion; 3] {
    [
        Promotion {
            narrow_type: Type::Int32,
            wide_type: Type::Int64,
            promoted_to: PrimitiveType::Long,
            build_narrow: |number| Arc::new(Int32Array::from(vec![number])),
            build_wide: |number| Arc::new(Int64Array::from(vec![i64::from(number)])),
        },
        Promotion {
            narrow_type: Type::Float32,
            wide_type: Type::Float64,
            promoted_to: PrimitiveType::Double,
            build_narrow: |number| Arc::new(Float32Array::from(vec![number as f32 + 0.5])),
            build_wide: |number| Arc::new(Float64Array::from(vec![f64::from(number) + 0.5])),
        },
        Promotion {
            narrow_type: Type::Decimal {
                precision: 9,
                scale: 2,
            },
            wide_type: Type::Decimal {
                precision: 18,
                scale: 2,
            },
            promoted_to: PrimitiveType::Decimal {
                precision: 18,
                scale: 2,
            },
            build_narrow: |number| build_decimal(number, 9),
            build_wide: |number| build_decimal(number, 18),
        },
    ]
}

/// An entry for `name`, a file of one row in the partition `v = value`,
/// written into the fixture's directory holding `stored`, the value in the
/// width the column is widened to.
fn write_entry(
    fixture: &Fixture,
    name: &str,
    value: &ArrayRef,
    stored: &ArrayRef,
) -> AddedDataFile {
    let field = Field::new("v", stored.data_type().clone(), false);
    let schema = Arc::new(ArrowSchema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(stored)]).unwrap();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let path = ObjectPath::new(format!("{name}.parquet"));
    let key = fixture.location.resolve_table_path(&path);
    fixture.location.store().put(&key, &bytes).unwrap();
    let bounds = || HashMap::from([("v".to_string(), Arc::clone(value))]);
    AddedDataFile {
        file: FileRef {
            path,
            size: bytes.len() as u64,
        },
        partition: Some(PartitionValues::from([(
            "v".to_string(),
            Scalar::new(Arc::clone(value)),
        )])),
        stats: Some(Arc::new(FileStats {
            num_records: Some(1),
            min_values: bounds(),
            max_values: bounds(),
            null_counts: HashMap::from([("v".to_string(), 0)]),
        })),
        sort_order_id: None,
    }
}

/// Make `v` of the fixture's table a `promoted_to` from now on, the way
/// another tool promotes a column, and load the table at that version.
fn promote_column(fixture: &Fixture, table: &Table, promoted_to: &PrimitiveType) -> Table {
    let fields = table.metadata().current_schema().as_struct().fields();
    let promoted_fields = fields.iter().map(|field| {
        let promoted_type = IcebergType::Primitive(promoted_to.clone());
        Arc::new(NestedField::optional(field.id, &field.name, promoted_type))
    });
    let promoted = Schema::builder()
        .with_fields(promoted_fields)
        .build()
        .unwrap();
    publish_changed(fixture, |builder| {
        let builder = builder.add_schema(promoted).unwrap();
        builder.set_current_schema(-1).unwrap()
    })
}

/// A table partitioned by `v` holding `before.parquet` in the partition of
/// the number `before`, written when `v` was narrow, and `after.parquet`
/// in that of the number `after`, written once `v` was widened.
fn write_across_promotion(promotion: &Promotion, before: i32, after: i32) -> (Fixture, Table) {
    let columns = [column("v", promotion.narrow_type.clone())];
    let (fixture, created) = create_table_of(&columns, &["v".to_string()], &[]);
    let narrow = (promotion.build_narrow)(before);
    let stored = (promotion.build_wide)(before);
    let written = append(
        &created,
        vec![write_entry(&fixture, "before", &narrow, &stored)],
    );
    let promoted = promote_column(&fixture, &written, &promotion.promoted_to);
    let wide = (promotion.build_wide)(after);
    let table = append(
        &promoted,
        vec![write_entry(&fixture, "after", &wide, &wide)],
    );
    (fixture, table)
}

/// The filter `v = value`, a value of the type `v` is widened to.
fn v_equals(promotion: &Promotion, value: ArrayRef) -> Vec<Expression> {
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: promotion.wide_type.clone(),
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(value))),
        compare_type: CompareType::Equal,
        return_type: Type::Boolean,
    })]
}

#[test]
fn a_value_written_before_and_after_a_promotion_is_one_partition_a_scan_selects_whole() {
    for promotion in list_promotions() {
        let (_fixture, table) = write_across_promotion(&promotion, 7, 7);

        let partitions = list_partitions(&table);
        let scanned = selected_sorted(&table, &[]);
        let filtered = selected_sorted(&table, &v_equals(&promotion, (promotion.build_wide)(7)));

        assert_eq!(partitions.len(), 1);
        assert_eq!(scanned, ["after.parquet", "before.parquet"]);
        assert_eq!(filtered, ["after.parquet", "before.parquet"]);
    }
}

#[test]
fn different_values_written_before_and_after_a_promotion_are_two_partitions() {
    for promotion in list_promotions() {
        let (_fixture, table) = write_across_promotion(&promotion, 7, 9);

        let partitions = list_partitions(&table);
        let filtered = selected_sorted(&table, &v_equals(&promotion, (promotion.build_wide)(9)));

        assert_eq!(partitions.len(), 2);
        assert_eq!(filtered, ["after.parquet"]);
    }
}

#[test]
fn a_float_and_the_double_of_the_same_decimal_digits_are_two_partitions() {
    let columns = [column("v", Type::Float32)];
    let (fixture, created) = create_table_of(&columns, &["v".to_string()], &[]);
    let narrow: ArrayRef = Arc::new(Float32Array::from(vec![0.1f32]));
    let stored: ArrayRef = Arc::new(Float64Array::from(vec![f64::from(0.1f32)]));
    let written = append(
        &created,
        vec![write_entry(&fixture, "before", &narrow, &stored)],
    );
    let promoted = promote_column(&fixture, &written, &PrimitiveType::Double);
    let wide: ArrayRef = Arc::new(Float64Array::from(vec![0.1f64]));

    let table = append(
        &promoted,
        vec![write_entry(&fixture, "after", &wide, &wide)],
    );

    assert_eq!(list_partitions(&table).len(), 2);
}

#[test]
fn a_file_written_before_a_promotion_is_found_with_its_partition_value_in_the_wide_type() {
    for promotion in list_promotions() {
        let (_fixture, table) = write_across_promotion(&promotion, 7, 9);
        let partition = find_partition_holding(&table, "before.parquet");

        let found = table
            .find_datastore_entries(&partition, &[ObjectPath::new("before.parquet")])
            .unwrap();

        let entry = found[0].as_ref().unwrap();
        let value = entry.partition.as_ref().unwrap()["v"].clone().into_inner();
        assert_eq!(&value, &(promotion.build_wide)(7));
    }
}

#[test]
fn a_commit_replaces_the_file_written_before_a_promotion_in_its_partition() {
    // The spec crate's rewrite writes the manifest of a file it removes
    // again under the table's current schema, which holds a decimal of
    // fewer digits and neither an `int` under a `long` nor a `float` under
    // a `double`: the widening of a decimal is the one a commit replaces a
    // file across.
    let [_, _, promotion] = list_promotions();
    let (fixture, table) = write_across_promotion(&promotion, 7, 7);
    let wide = (promotion.build_wide)(7);
    let merged = write_entry(&fixture, "merged", &wide, &wide);

    let replaced = commit_replace(&table, &["before.parquet"], vec![merged])
        .unwrap()
        .unwrap();

    assert_eq!(paths(&replaced), ["after.parquet", "merged.parquet"]);
    assert_eq!(list_partitions(&replaced).len(), 1);
    assert_eq!(
        selected_sorted(&replaced, &[]),
        ["after.parquet", "merged.parquet"]
    );
}

#[test]
fn the_partition_of_a_manifest_written_before_a_promotion_is_listed_with_no_manifest_read() {
    // A manifest-list entry names no partition of a float or a double,
    // whose NaNs its bounds leave out.
    let [integer, _, decimal] = list_promotions();
    for promotion in [integer, decimal] {
        let (fixture, table) = write_across_promotion(&promotion, 7, 7);
        fixture.delete_data_manifests(&table);

        let partitions = list_partitions(&table);

        assert_eq!(partitions.len(), 1);
    }
}

#[test]
fn a_description_names_the_partition_of_files_written_before_and_after_a_promotion_alike() {
    let [promotion, _, _] = list_promotions();
    let (_fixture, table) = write_across_promotion(&promotion, 7, 7);

    let description = describe(&table);

    let partitions: Vec<&str> = description
        .files
        .iter()
        .map(|file| file.partition.as_str())
        .collect();
    assert_eq!(partitions, ["v=7", "v=7"]);
}
