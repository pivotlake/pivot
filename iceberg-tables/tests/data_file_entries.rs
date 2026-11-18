//! A data file as a manifest records it and as a table reads it back: its
//! statistics, its partition tuple and its numbers survive a commit, the
//! partition it sits in is named by a key that orders as the tuple does,
//! and what Iceberg cannot record is refused or left out.

mod support;

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Datum as _, Float64Array, Int32Array, Int64Array,
    Scalar, StringViewArray, UInt64Array,
};
use arrow_schema::DataType;
use iceberg::spec::{
    Datum, FormatVersion, Literal, PartitionSpec, PrimitiveLiteral, Struct, TableMetadata,
    TableMetadataBuilder, Transform,
};
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::types::schema::{build_partition_spec, build_schema};
use iceberg_tables::{Error, Table, TableLocation};
use object_storage::{FileRef, LocalStore, ObjectPath, ObjectStore};
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::Column;
use planner::types::Type;
use support::{
    append, column, create_table, create_table_of, describe, entry, live_entry, load,
    publish_changed, repartition,
};
use table_file_index::IndexedRecord;

/// The data sequence number `table` reads its live file `path` at.
fn read_commit_sequence(table: &Table, path: &str) -> u64 {
    let path = ObjectPath::new(path);
    let mut sequence = None;
    table
        .visit_partitions(&mut |_, index| {
            if let Some(file) = index.find(&path) {
                sequence = Some(file.facts().commit_sequence());
            }
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    sequence.unwrap_or_else(|| panic!("{path} is live"))
}

/// An entry for `name` of `rows` rows in the partition `partition`, with no
/// column bounds.
fn entry_in(name: &str, rows: i64, partition: PartitionValues) -> DataFileEntry {
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new(format!("{name}.parquet")),
            size: 1,
        },
        partition: Some(partition),
        partition_spec_id: None,
        stats: Some(stats(rows, &[], &[])),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    }
}

#[test]
fn an_entry_round_trips_through_a_manifest_record() {
    let columns = [
        column("id", Type::UInt64),
        column("name", Type::Utf8),
        column("day", Type::Date),
    ];
    let (fixture, created) = create_table_of(&columns, &["day".to_string()], &[]);
    let on_day = |day: i32| {
        PartitionValues::from([("day".to_string(), scalar(Date32Array::from(vec![day])))])
    };
    let entry = DataFileEntry {
        file: FileRef {
            path: ObjectPath::new("pivot-1.parquet"),
            size: 123,
        },
        partition: Some(on_day(20_000)),
        partition_spec_id: None,
        stats: Some(stats(
            7,
            &[
                (
                    "id",
                    Arc::new(UInt64Array::from(vec![1])),
                    Arc::new(UInt64Array::from(vec![9])),
                ),
                (
                    "name",
                    Arc::new(StringViewArray::from(vec!["a"])),
                    Arc::new(StringViewArray::from(vec!["z"])),
                ),
            ],
            &[("name", 2)],
        )),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    };
    let hundred_rows_before = append(&created, vec![entry_in("pivot-0", 100, on_day(19_999))]);

    let data_file = entry
        .create_data_file(created.metadata(), created.location())
        .unwrap();
    append(&hundred_rows_before, vec![entry.clone()]);
    let loaded = load(&fixture);
    let read = live_entry(&loaded, "pivot-1.parquet");

    assert_eq!(data_file.record_count(), 7);
    assert_eq!(data_file.lower_bounds()[&1], Datum::long(1));
    assert_eq!(data_file.upper_bounds()[&2], Datum::string("z"));
    assert_eq!(data_file.null_value_counts()[&2], 2);
    assert_eq!(
        data_file.partition().fields(),
        &[Some(Literal::Primitive(PrimitiveLiteral::Int(20_000)))]
    );
    assert_eq!(read.file, entry.file);
    assert!(parquet_engine::scalar_values_equal(
        read.partition.as_ref().unwrap(),
        entry.partition.as_ref().unwrap()
    ));
    let read_stats = read.stats.unwrap();
    assert_eq!(read_stats.num_records, Some(7));
    assert_eq!(
        read_stats.min_values["id"].as_ref().data_type(),
        &DataType::UInt64
    );
    assert_eq!(read_stats.null_counts["name"], 2);
    assert_eq!(read.first_row_id, Some(100));
    assert_eq!(read_commit_sequence(&loaded, "pivot-1.parquet"), 2);
}

#[test]
fn an_existing_entry_keeps_its_own_numbers() {
    let (fixture, created) = create_table(&[]);
    let table = append(
        &created,
        vec![entry("a", 10, 1, 10), entry("b", 10, 11, 20)],
    );
    let table = append(&table, vec![entry("c", 10, 21, 30)]);
    let removed = live_entry(&table, "a.parquet");

    table
        .commit(&[removed], vec![entry("d", 10, 1, 10)])
        .unwrap()
        .expect("no other writer");
    let loaded = load(&fixture);

    // The replace rewrote the manifest of the first commit, which lists `b`
    // as existing from then on, under the third commit's number.
    assert_eq!(read_commit_sequence(&loaded, "b.parquet"), 1);
    assert_eq!(read_commit_sequence(&loaded, "d.parquet"), 3);
    assert!(live_entry(&loaded, "b.parquet").partition.is_none());
}

#[test]
fn a_null_partition_value_round_trips_as_a_null_scalar() {
    let columns = [column("name", Type::Utf8)];
    let (fixture, created) = create_table_of(&columns, &["name".to_string()], &[]);
    let no_name = Scalar::new(arrow_array::new_null_array(
        &planner::types::physical_arrow_type(&Type::Utf8),
        1,
    ));
    let entry = entry_in(
        "pivot-1",
        1,
        PartitionValues::from([("name".to_string(), no_name)]),
    );

    let data_file = entry
        .create_data_file(created.metadata(), created.location())
        .unwrap();
    append(&created, vec![entry]);
    let read = live_entry(&load(&fixture), "pivot-1.parquet")
        .partition
        .unwrap();

    assert_eq!(data_file.partition().fields(), &[None]);
    assert!(read["name"].get().0.is_null(0));
}

/// A derived partition value (a day of a timestamp) is recorded in the
/// type its transform produces, keyed by the field's name, and reads back
/// as a scalar of that type.
#[test]
fn a_derived_partition_value_is_recorded_in_its_transforms_type() {
    let (fixture, _created) = create_table_of(&[column("at", Type::Timestamp)], &[], &[]);
    let table = repartition(&fixture, &[("at", "at_day", Transform::Day)]);
    let entry = entry_in(
        "pivot-1",
        1,
        PartitionValues::from([(
            "at_day".to_string(),
            scalar(Date32Array::from(vec![19_723])),
        )]),
    );

    let data_file = entry
        .create_data_file(table.metadata(), table.location())
        .unwrap();
    append(&table, vec![entry]);
    let read = live_entry(&load(&fixture), "pivot-1.parquet")
        .partition
        .unwrap();

    assert_eq!(
        data_file.partition().fields(),
        &[Some(Literal::date(19_723))]
    );
    let (day, _) = read["at_day"].get();
    assert_eq!(day.as_primitive::<Date32Type>().value(0), 19_723);
}

#[test]
fn a_file_of_an_earlier_partition_spec_reads_its_tuple_by_that_specs_fields() {
    let columns = [column("v", Type::Int64), column("at", Type::Timestamp)];
    let (fixture, created) = create_table_of(&columns, &["v".to_string()], &[]);
    let seven = PartitionValues::from([("v".to_string(), scalar(Int64Array::from(vec![7])))]);
    append(&created, vec![entry_in("by-value", 1, seven)]);
    let repartitioned = repartition(&fixture, &[("at", "at_day", Transform::Day)]);
    let new_years_day = PartitionValues::from([(
        "at_day".to_string(),
        scalar(Date32Array::from(vec![19_723])),
    )]);
    let written = append(&repartitioned, vec![entry_in("by-day", 1, new_years_day)]);

    let by_value = live_entry(&written, "by-value.parquet");
    let by_day = live_entry(&written, "by-day.parquet");

    let value_tuple = by_value.partition.unwrap();
    assert_eq!(value_tuple.keys().collect::<Vec<_>>(), ["v"]);
    let (value, _) = value_tuple["v"].get();
    assert_eq!(value.as_primitive::<Int64Type>().value(0), 7);
    let day_tuple = by_day.partition.unwrap();
    assert_eq!(day_tuple.keys().collect::<Vec<_>>(), ["at_day"]);
    let (day, _) = day_tuple["at_day"].get();
    assert_eq!(day.as_primitive::<Date32Type>().value(0), 19_723);
    assert_eq!(
        (by_value.partition_spec_id, by_day.partition_spec_id),
        (
            Some(created.metadata().default_partition_spec_id()),
            Some(written.metadata().default_partition_spec_id())
        )
    );
}

#[test]
fn a_file_of_a_partition_spec_the_table_no_longer_lists_is_refused_by_its_spec() {
    let (fixture, created) = create_table(&["v".to_string()]);
    let seven = PartitionValues::from([("v".to_string(), scalar(Int64Array::from(vec![7])))]);
    append(&created, vec![entry_in("by-value", 1, seven)]);
    repartition(&fixture, &[]);
    let by_value_spec_id = created.metadata().default_partition_spec_id();
    let unlisted = publish_changed(&fixture, |builder| {
        builder.remove_partition_specs(&[by_value_spec_id]).unwrap()
    });

    let entries = unlisted.entries();

    assert!(matches!(
        entries,
        Err(Error::UnlistedPartitionSpec { file, spec_id })
            if file == "by-value.parquet" && spec_id == by_value_spec_id
    ));
}

#[test]
fn a_day_partition_is_shown_by_its_fields_name_and_its_date() {
    let (fixture, _created) = create_table_of(&[column("at", Type::Timestamp)], &[], &[]);
    let table = repartition(&fixture, &[("at", "at_day", Transform::Day)]);
    let write_on_day = |name: &str, day: Option<i32>| {
        let on_day = scalar(Date32Array::from(vec![day]));
        let partition = PartitionValues::from([("at_day".to_string(), on_day)]);
        fixture.write_data_file(entry_in(name, 10, partition), 0)
    };
    let written = append(
        &table,
        vec![
            write_on_day("new-years-day", Some(19_723)),
            write_on_day("unknown-day", None),
        ],
    );

    let description = describe(&written);

    let mut shown: Vec<(&str, &str)> = description
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.partition.as_str()))
        .collect();
    shown.sort_unstable();
    assert_eq!(
        shown,
        [
            ("new-years-day.parquet", "at_day=2024-01-01"),
            ("unknown-day.parquet", "at_day="),
        ]
    );
}

#[test]
fn a_partitions_key_reads_back_as_its_spec_and_tuple() {
    let columns = [
        column("flag", Type::Boolean),
        column("count", Type::Int64),
        column("name", Type::Utf8),
        column("ratio", Type::Float64),
    ];
    let names: Vec<String> = columns.iter().map(|column| column.name.clone()).collect();
    let (_fixture, created) = create_table_of(&columns, &names, &[]);
    let tuple = PartitionValues::from([
        ("flag".to_string(), scalar(BooleanArray::from(vec![true]))),
        ("count".to_string(), scalar(Int64Array::from(vec![-7]))),
        (
            "name".to_string(),
            scalar(StringViewArray::from(vec!["a\0b"])),
        ),
        (
            "ratio".to_string(),
            scalar(Float64Array::from(vec![None::<f64>])),
        ),
    ]);
    let written = append(&created, vec![entry_in("a", 1, tuple)]);
    let partition = written.partitions().unwrap().remove(0);

    let read_back = written.find_partition(&partition.encode_key()).unwrap();
    let cut_short = written.find_partition(&partition.encode_key()[..5]);

    assert_eq!(
        partition.tuple(),
        &Struct::from_iter([
            Some(Literal::bool(true)),
            Some(Literal::long(-7)),
            Some(Literal::string("a\0b")),
            None,
        ])
    );
    assert_eq!(read_back.spec_id(), partition.spec_id());
    assert_eq!(read_back.tuple(), partition.tuple());
    assert_eq!(read_back.encode_key(), partition.encode_key());
    assert!(matches!(cut_short, Err(Error::UnknownPartition { .. })));
}

#[test]
fn partition_tuples_order_as_their_values_do() {
    let columns = [column("number", Type::Int32), column("text", Type::Utf8)];
    let names: Vec<String> = columns.iter().map(|column| column.name.clone()).collect();
    let (_fixture, created) = create_table_of(&columns, &names, &[]);
    let int = |value: i32| Some(Literal::int(value));
    let text = |value: &str| Some(Literal::string(value));
    let entry_of = |position: usize, number: Option<i32>, word: &str| {
        let tuple = PartitionValues::from([
            ("number".to_string(), scalar(Int32Array::from(vec![number]))),
            (
                "text".to_string(),
                scalar(StringViewArray::from(vec![word])),
            ),
        ]);
        entry_in(&format!("file-{position}"), 1, tuple)
    };
    let entries = [
        (Some(7), "b"),
        (Some(-3), "z"),
        (Some(7), "ab"),
        (None, "a"),
        (Some(7), "a"),
    ]
    .into_iter()
    .enumerate()
    .map(|(position, (number, word))| entry_of(position, number, word))
    .collect();

    let written = append(&created, entries);

    let tuples: Vec<Vec<Option<Literal>>> = written
        .partitions()
        .unwrap()
        .iter()
        .map(|partition| partition.tuple().fields().to_vec())
        .collect();
    assert_eq!(
        tuples,
        [
            vec![None, text("a")],
            vec![int(-3), text("z")],
            vec![int(7), text("a")],
            vec![int(7), text("ab")],
            vec![int(7), text("b")],
        ]
    );
}

/// The metadata of a table over `columns`, partitioned by `partition_by`,
/// that no version was published of, and where it would live: a temporary
/// local store.
struct Unpublished {
    _dir: tempfile::TempDir,
    location: TableLocation,
    metadata: TableMetadata,
}

impl Unpublished {
    fn new(columns: Vec<Column>, partition_by: &[&str]) -> Self {
        let schema = build_schema(&columns).unwrap();
        let partition_by: Vec<String> = partition_by.iter().map(|c| c.to_string()).collect();
        let spec = build_partition_spec(&schema, &partition_by).unwrap();
        Self::with_spec(columns, spec)
    }

    /// A table partitioned by `transform` of the column `source`, the
    /// partition field named `field`.
    fn partitioned_by_transform(
        columns: Vec<Column>,
        source: &str,
        field: &str,
        transform: Transform,
    ) -> Self {
        let schema = build_schema(&columns).unwrap();
        let spec = PartitionSpec::builder(Arc::new(schema))
            .add_partition_field(source, field, transform)
            .unwrap()
            .build()
            .unwrap()
            .into_unbound();
        Self::with_spec(columns, spec)
    }

    fn with_spec(columns: Vec<Column>, spec: iceberg::spec::UnboundPartitionSpec) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let location =
            TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t")).unwrap();
        let schema = build_schema(&columns).unwrap();
        let metadata = TableMetadataBuilder::new(
            schema,
            spec,
            iceberg::spec::SortOrder::unsorted_order(),
            location.table_uri(),
            FormatVersion::V3,
            HashMap::new(),
        )
        .unwrap()
        .build()
        .unwrap()
        .metadata;
        Self {
            _dir: dir,
            location,
            metadata,
        }
    }
}

fn scalar(array: impl Array + 'static) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(array))
}

fn stats(
    rows: i64,
    bounds: &[(&str, ArrayRef, ArrayRef)],
    nulls: &[(&str, i64)],
) -> Arc<FileStats> {
    Arc::new(FileStats {
        num_records: Some(rows),
        min_values: bounds
            .iter()
            .map(|(name, min, _)| (name.to_string(), min.clone()))
            .collect(),
        max_values: bounds
            .iter()
            .map(|(name, _, max)| (name.to_string(), max.clone()))
            .collect(),
        null_counts: nulls
            .iter()
            .map(|(name, count)| (name.to_string(), *count))
            .collect(),
    })
}

#[test]
fn bounds_iceberg_cannot_hold_are_left_out() {
    let fixture = Unpublished::new(
        vec![column("big", Type::UInt64), column("f", Type::Float64)],
        &[],
    );
    let entry = DataFileEntry {
        file: FileRef {
            path: ObjectPath::new("pivot-1.parquet"),
            size: 1,
        },
        partition: None,
        partition_spec_id: None,
        stats: Some(stats(
            1,
            &[
                (
                    "big",
                    Arc::new(UInt64Array::from(vec![0])),
                    Arc::new(UInt64Array::from(vec![u64::MAX])),
                ),
                (
                    "f",
                    Arc::new(Float64Array::from(vec![f64::NAN])),
                    Arc::new(Float64Array::from(vec![1.0])),
                ),
            ],
            &[],
        )),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    };

    let data_file = entry
        .create_data_file(&fixture.metadata, &fixture.location)
        .unwrap();

    assert!(data_file.lower_bounds().is_empty());
    assert!(data_file.upper_bounds().is_empty());
}

#[test]
fn a_partitioned_table_refuses_an_entry_without_a_tuple() {
    let fixture = Unpublished::new(vec![column("day", Type::Date)], &["day"]);
    let entry = DataFileEntry {
        file: FileRef {
            path: ObjectPath::new("pivot-1.parquet"),
            size: 1,
        },
        partition: None,
        partition_spec_id: None,
        stats: Some(stats(1, &[], &[])),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    };

    let error = entry
        .create_data_file(&fixture.metadata, &fixture.location)
        .unwrap_err();

    assert!(
        matches!(error, Error::PartitionNotConstant { .. }),
        "{error}"
    );
}

/// A derived partition field the tuple lacks is an error, not a NULL: a
/// NULL day would hide the file from every reader pruning by day.
#[test]
fn a_missing_derived_partition_value_is_refused() {
    let fixture = Unpublished::partitioned_by_transform(
        vec![column("at", Type::Timestamp)],
        "at",
        "at_day",
        Transform::Day,
    );
    let entry = DataFileEntry {
        file: FileRef {
            path: ObjectPath::new("pivot-1.parquet"),
            size: 1,
        },
        partition: Some(PartitionValues::new()),
        partition_spec_id: None,
        stats: Some(stats(1, &[], &[])),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    };

    let error = entry
        .create_data_file(&fixture.metadata, &fixture.location)
        .unwrap_err();

    assert!(error.to_string().contains("at_day"), "{error}");
}

#[test]
fn an_entry_without_stats_cannot_be_recorded() {
    let fixture = Unpublished::new(vec![column("id", Type::Int64)], &[]);
    let entry = DataFileEntry {
        file: FileRef {
            path: ObjectPath::new("pivot-1.parquet"),
            size: 1,
        },
        partition: None,
        partition_spec_id: None,
        stats: None,
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    };

    let error = entry
        .create_data_file(&fixture.metadata, &fixture.location)
        .unwrap_err();

    assert!(matches!(error, Error::MissingFileStats { .. }), "{error}");
}

/// Two strings that differ only after their first 26 code points.
const FIRST_PAGE: &str = "https://example.com/pages/a";
const LAST_PAGE: &str = "https://example.com/pages/z";

/// An entry for one file whose strings in each of `columns` run from
/// [`FIRST_PAGE`] to [`LAST_PAGE`].
fn build_page_entry(columns: &[&str]) -> DataFileEntry {
    let bound = |page: &str| -> ArrayRef { Arc::new(StringViewArray::from(vec![page])) };
    let bounds: Vec<(&str, ArrayRef, ArrayRef)> = columns
        .iter()
        .map(|column| (*column, bound(FIRST_PAGE), bound(LAST_PAGE)))
        .collect();
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new("pivot-1.parquet"),
            size: 1,
        },
        partition: None,
        partition_spec_id: None,
        stats: Some(stats(2, &bounds, &[])),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    }
}

#[test]
fn a_sort_columns_long_string_bounds_are_recorded_whole() {
    let columns = [column("url", Type::Utf8)];
    let (_fixture, table) = create_table_of(&columns, &[], &["url".to_string()]);
    let entry = build_page_entry(&["url"]);

    let data_file = entry
        .create_data_file(table.metadata(), table.location())
        .unwrap();

    assert_eq!(data_file.lower_bounds()[&1], Datum::string(FIRST_PAGE));
    assert_eq!(data_file.upper_bounds()[&1], Datum::string(LAST_PAGE));
}

#[test]
fn a_long_string_bound_of_a_column_outside_the_sort_order_is_cut() {
    let columns = [column("url", Type::Utf8), column("referrer", Type::Utf8)];
    let (_fixture, table) = create_table_of(&columns, &[], &["url".to_string()]);
    let entry = build_page_entry(&["referrer"]);

    let data_file = entry
        .create_data_file(table.metadata(), table.location())
        .unwrap();

    // The first 16 code points below, and above them with the last one
    // incremented.
    assert_eq!(
        data_file.lower_bounds()[&2],
        Datum::string("https://example.")
    );
    assert_eq!(
        data_file.upper_bounds()[&2],
        Datum::string("https://example/")
    );
}

#[test]
fn every_sort_column_reads_its_long_string_bounds_back_whole_from_a_manifest() {
    let columns = [column("site", Type::Utf8), column("url", Type::Utf8)];
    let sort_by = ["site".to_string(), "url".to_string()];
    let (fixture, created) = create_table_of(&columns, &[], &sort_by);

    append(&created, vec![build_page_entry(&["site", "url"])]);
    let read = live_entry(&load(&fixture), "pivot-1.parquet")
        .stats
        .unwrap();

    for column in ["site", "url"] {
        let (lowest, highest) = (&read.min_values[column], &read.max_values[column]);
        assert_eq!(lowest.as_string_view().value(0), FIRST_PAGE);
        assert_eq!(highest.as_string_view().value(0), LAST_PAGE);
    }
}
