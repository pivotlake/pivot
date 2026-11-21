//! What a scan reads of a table's data manifests, and what it finds held of
//! them: a manifest is parsed once for every part of it, by partition and by
//! column, and a scan after it, of whichever version lists the manifest,
//! selects its files through those parts.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use iceberg::spec::Transform;
use iceberg_tables::files::data_files::DataFileEntry;
use object_storage::{FileRef, ObjectPath};
use parquet_engine::FileStats;
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use support::{
    append, column, create_partitioned_table, create_table_of, int_scalar, load, repartition,
    selected, v_equals,
};

/// The filter `w = value`, on the second column of a table of `v` and `w`.
fn w_equals(value: i64) -> Vec<Expression> {
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 1,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(int_scalar(value))),
        compare_type: CompareType::Equal,
        return_type: Type::Boolean,
    })]
}

/// An entry for `name`, a file of ten rows holding `v` in every row of the
/// column `v` and `w` in every row of the column `w`.
fn entry_of_v_and_w(name: &str, v: i64, w: i64) -> DataFileEntry {
    let bound = |value: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![value])) };
    let bounds = HashMap::from([("v".to_string(), bound(v)), ("w".to_string(), bound(w))]);
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new(format!("{name}.parquet")),
            size: 10,
        },
        partition: None,
        stats: Some(Arc::new(FileStats {
            num_records: Some(10),
            min_values: bounds.clone(),
            max_values: bounds,
            null_counts: HashMap::from([("v".to_string(), 0), ("w".to_string(), 0)]),
        })),
        sort_order_id: None,
    }
}

#[test]
fn a_second_scan_of_every_partition_reads_no_manifest_that_holds_several() {
    let (fixture, created) = create_partitioned_table();
    append(
        &created,
        vec![
            fixture.write_file("seven", 7),
            fixture.write_file("nine", 9),
        ],
    );
    let opened = load(&fixture);
    selected(&opened, &[]);
    fixture.delete_data_manifests(&opened);

    let files = selected(&opened, &[]);

    assert_eq!(files, ["seven.parquet", "nine.parquet"]);
}

#[test]
fn a_scan_of_one_partition_leaves_the_files_of_the_others_in_its_manifest_held() {
    let (fixture, created) = create_partitioned_table();
    append(
        &created,
        vec![
            fixture.write_file("seven", 7),
            fixture.write_file("nine", 9),
        ],
    );
    let opened = load(&fixture);
    selected(&opened, &v_equals(7));
    fixture.delete_data_manifests(&opened);

    let files = selected(&opened, &v_equals(9));

    assert_eq!(files, ["nine.parquet"]);
}

#[test]
fn a_scan_filtered_on_a_column_no_scan_compared_before_reads_no_manifest() {
    let columns = [column("v", Type::Int64), column("w", Type::Int64)];
    let (fixture, created) = create_table_of(&columns, &[], &[]);
    append(
        &created,
        vec![
            entry_of_v_and_w("low", 1, 10),
            entry_of_v_and_w("high", 2, 20),
        ],
    );
    let opened = load(&fixture);
    opened.select_scan(&v_equals(5)).unwrap();
    fixture.delete_data_manifests(&opened);

    let selection = opened.select_scan(&w_equals(99)).unwrap();

    assert!(selection.row_groups.row_groups().is_empty());
}

#[test]
fn a_version_under_another_partition_spec_scans_through_the_parts_an_earlier_one_read() {
    let (fixture, created) = create_partitioned_table();
    let written = append(&created, vec![fixture.write_file("seven", 7)]);
    written.select_scan(&[]).unwrap();
    let repartitioned = repartition(&fixture, &[("v", "v_bucket", Transform::Bucket(4))]);
    fixture.delete_data_manifests(&repartitioned);

    let files = selected(&repartitioned, &[]);

    assert_eq!(files, ["seven.parquet"]);
}
