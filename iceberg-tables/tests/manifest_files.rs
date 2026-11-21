//! A version's live files as maintenance takes them, a data manifest at a
//! time: each file with its bounds on the table's sort key, read through
//! the parts of the manifests the cache holds.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, StringViewArray,
    TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use iceberg_tables::files::data_files::AddedDataFile;
use iceberg_tables::{Error, Table};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::FileStats;
use parquet_engine::{BoundSide, SortValue};
use planner::types::Type;
use support::{
    append, column, create_partitioned_table, create_table, create_table_laid_out_by,
    create_table_of, entry, load, partition_entry, replace,
};

/// The names of the files each data manifest of `table` lists, manifest by
/// manifest in manifest-list order, each manifest's sorted.
fn list_files_per_manifest(table: &Table) -> Vec<Vec<String>> {
    let manifest_paths = table.list_data_manifest_paths();
    let manifests = table.load_manifest_files(&manifest_paths).unwrap();
    manifests
        .iter()
        .map(|partitions| {
            let files = partitions.iter().flat_map(|partition| &partition.files);
            let mut names: Vec<String> = files.map(|file| file.file().path.to_string()).collect();
            names.sort_unstable();
            names
        })
        .collect()
}

#[test]
fn every_manifest_lists_the_files_its_commit_added() {
    let (_fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("early", 10, 1, 2)]);
    let second = append(
        &first,
        vec![entry("late", 10, 3, 4), entry("later", 10, 5, 6)],
    );

    let files = list_files_per_manifest(&second);

    // A commit lists its own manifest before the ones it carries over.
    assert_eq!(
        files,
        [vec!["late.parquet", "later.parquet"], vec!["early.parquet"]]
    );
}

#[test]
fn a_manifests_files_come_by_partition_with_their_bounds_on_the_sort_key() {
    let by_v = ["v".to_string()];
    let (_fixture, created) = create_table_laid_out_by(&by_v, &by_v);
    let written = append(
        &created,
        vec![
            partition_entry("nine", 9, 0),
            partition_entry("seven", 7, 0),
        ],
    );

    let manifests = written
        .load_manifest_files(&written.list_data_manifest_paths())
        .unwrap();

    let read_range = |file: &iceberg_tables::CompactionInput| {
        let read = |side| match file.read_sort_bound(0, side) {
            SortValue::Integer(value) => value,
            other => panic!("the sort column is bounded by an integer, not {other:?}"),
        };
        (read(BoundSide::Lower), read(BoundSide::Upper))
    };
    let ranges_per_partition: Vec<Vec<(i128, i128)>> = manifests[0]
        .iter()
        .map(|partition| partition.files.iter().map(read_range).collect())
        .collect();
    assert_eq!(ranges_per_partition, [vec![(7, 7)], vec![(9, 9)]]);
}

#[test]
fn a_files_bound_on_a_sort_column_of_any_type_reads_as_a_value_of_that_type() {
    let narrow_decimal = Decimal64Array::from(vec![-1234])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let wide_decimal = Decimal128Array::from(vec![10_i128.pow(30)])
        .with_precision_and_scale(38, 2)
        .unwrap();
    let zoned = TimestampMicrosecondArray::from(vec![-9]).with_timezone("UTC");
    // Each sort column's type, the one value a file holds of it, and what
    // the file's bounds on it read as.
    let held: Vec<(Type, ArrayRef, SortValue<'_>)> = vec![
        (
            Type::Boolean,
            Arc::new(BooleanArray::from(vec![true])),
            SortValue::Boolean(true),
        ),
        (
            Type::Int8,
            Arc::new(Int8Array::from(vec![-8])),
            SortValue::Integer(-8),
        ),
        (
            Type::Int16,
            Arc::new(Int16Array::from(vec![-16])),
            SortValue::Integer(-16),
        ),
        (
            Type::Int32,
            Arc::new(Int32Array::from(vec![-32])),
            SortValue::Integer(-32),
        ),
        (
            Type::Int64,
            Arc::new(Int64Array::from(vec![i64::MIN])),
            SortValue::Integer(i128::from(i64::MIN)),
        ),
        (
            Type::UInt8,
            Arc::new(UInt8Array::from(vec![8])),
            SortValue::Integer(8),
        ),
        (
            Type::UInt16,
            Arc::new(UInt16Array::from(vec![16])),
            SortValue::Integer(16),
        ),
        (
            Type::UInt32,
            Arc::new(UInt32Array::from(vec![u32::MAX])),
            SortValue::Integer(i128::from(u32::MAX)),
        ),
        (
            Type::UInt64,
            Arc::new(UInt64Array::from(vec![64])),
            SortValue::Integer(64),
        ),
        (
            Type::Float32,
            Arc::new(Float32Array::from(vec![-0.0])),
            SortValue::Float(-0.0),
        ),
        (
            Type::Float64,
            Arc::new(Float64Array::from(vec![f64::INFINITY])),
            SortValue::Float(f64::INFINITY),
        ),
        (
            Type::Date,
            Arc::new(Date32Array::from(vec![-3])),
            SortValue::Integer(-3),
        ),
        (
            Type::Timestamp,
            Arc::new(TimestampMicrosecondArray::from(vec![9])),
            SortValue::Integer(9),
        ),
        (Type::TimestampTz, Arc::new(zoned), SortValue::Integer(-9)),
        (
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["a value longer than a view"])),
            SortValue::String("a value longer than a view"),
        ),
        (
            Type::Decimal {
                precision: 10,
                scale: 2,
            },
            Arc::new(narrow_decimal),
            SortValue::Integer(-1234),
        ),
        (
            Type::Decimal {
                precision: 38,
                scale: 2,
            },
            Arc::new(wide_decimal),
            SortValue::Integer(10_i128.pow(30)),
        ),
    ];
    let names: Vec<String> = (0..held.len()).map(|place| format!("c{place}")).collect();
    let columns: Vec<_> = names
        .iter()
        .zip(&held)
        .map(|(name, (column_type, _, _))| column(name, column_type.clone()))
        .collect();
    let values: HashMap<String, ArrayRef> = names
        .iter()
        .zip(&held)
        .map(|(name, (_, value, _))| (name.clone(), Arc::clone(value)))
        .collect();
    let (_fixture, created) = create_table_of(&columns, &[], &names);
    let bounded = AddedDataFile {
        file: FileRef {
            path: ObjectPath::new("bounded.parquet"),
            size: 10,
        },
        partition: None,
        stats: Some(Arc::new(FileStats {
            num_records: Some(10),
            min_values: values.clone(),
            max_values: values.clone(),
            null_counts: HashMap::new(),
        })),
        sort_order_id: None,
    };
    let written = append(&created, vec![bounded]);

    let manifests = written
        .load_manifest_files(&written.list_data_manifest_paths())
        .unwrap();
    let file = &manifests[0][0].files[0];
    let read = |side| -> Vec<SortValue<'_>> {
        let places = 0..file.count_sort_columns();
        places
            .map(|place| file.read_sort_bound(place, side))
            .collect()
    };
    let (lower, upper) = (read(BoundSide::Lower), read(BoundSide::Upper));
    let stats = file.build_sort_key_stats();

    let expected: Vec<SortValue<'_>> = held.iter().map(|(_, _, bound)| *bound).collect();
    assert_eq!(lower, expected);
    assert_eq!(upper, expected);
    assert_eq!(stats.min_values, values);
    assert_eq!(stats.max_values, values);
}

#[test]
fn a_rewritten_manifest_lists_the_files_it_carried_over_under_another_path() {
    let (_fixture, created) = create_table(&[]);
    let appended = append(
        &created,
        vec![entry("kept", 10, 1, 1), entry("replaced", 10, 2, 2)],
    );
    let rewritten = replace(&appended, &["replaced.parquet"], "merged");

    let paths_before = appended.list_data_manifest_paths();
    let paths_after = rewritten.list_data_manifest_paths();
    let files = list_files_per_manifest(&rewritten);

    assert!(paths_after.iter().all(|path| !paths_before.contains(path)));
    assert_eq!(files, [vec!["merged.parquet"], vec!["kept.parquet"]]);
}

#[test]
fn the_files_of_a_manifest_a_scan_read_are_listed_with_no_manifest_read() {
    let (fixture, created) = create_partitioned_table();
    append(
        &created,
        vec![
            fixture.write_file("seven", 7),
            fixture.write_file("nine", 9),
        ],
    );
    let opened = load(&fixture);
    opened.select_scan(&[]).unwrap();
    fixture.delete_data_manifests(&opened);

    let files = list_files_per_manifest(&opened);

    assert_eq!(files, [vec!["nine.parquet", "seven.parquet"]]);
}

#[test]
fn the_files_of_a_manifest_a_version_does_not_list_are_refused() {
    let (_fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("early", 10, 1, 2)]);
    let rewritten = replace(&appended, &["early.parquet"], "merged");

    let files = rewritten.load_manifest_files(&appended.list_data_manifest_paths());

    assert!(matches!(files, Err(Error::UnlistedManifest { .. })));
}
