//! Which partitions a table indexes and which manifests it reads: a scan
//! indexes only the partitions its filters keep, computed from the filters
//! alone where they leave every partition field only a few values, and a
//! table lists its partitions and those a commit added to from as few
//! manifests as say it.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, Scalar, TimestampMicrosecondArray};
use iceberg::spec::{Datum, Literal, PrimitiveLiteral, Transform};
use iceberg::transform::create_transform_function;
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::{Partition, Table};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::{FileStats, PartitionValues};
use planner::expression::{Compare, CompareType, Expression, InList, Ref};
use planner::types::Type;
use support::{
    Fixture, SpecField, append, column, create_partitioned_table, create_table_of, int_scalar,
    load, partition_entry, repartition, selected, v_equals,
};

/// The path of the data manifest `later` lists that `earlier`, the version
/// before it, does not.
fn find_joined_manifest(earlier: &Table, later: &Table) -> String {
    let listed_earlier: Vec<&str> = earlier
        .manifest_list()
        .data_manifests()
        .map(|manifest| manifest.manifest_path.as_str())
        .collect();
    later
        .manifest_list()
        .data_manifests()
        .map(|manifest| manifest.manifest_path.as_str())
        .find(|path| !listed_earlier.contains(path))
        .expect("an append lists a manifest of its own")
        .to_string()
}

#[test]
fn a_table_opens_without_reading_a_data_manifest() {
    let (fixture, created) = create_partitioned_table();
    let written = append(
        &created,
        vec![
            partition_entry("seven", 7, 0),
            partition_entry("nine", 9, 0),
        ],
    );
    fixture.delete_data_manifests(&written);

    let opened = load(&fixture);

    assert_eq!(opened.total_rows().unwrap(), 20);
}

#[test]
fn a_scan_filtered_on_the_partition_indexes_only_the_partitions_it_keeps() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("seven", 7)]);
    let second = append(&first, vec![partition_entry("nine", 9, 0)]);
    let opened = load(&fixture);
    fixture.delete_manifest(&find_joined_manifest(&first, &second));

    let files = selected(&opened, &v_equals(7));

    assert_eq!(files, ["seven.parquet"]);
}

#[test]
fn a_scan_that_keeps_only_partitions_without_nulls_reports_the_column_free_of_them() {
    let (fixture, created) = create_partitioned_table();
    let dense = append(&created, vec![fixture.write_file("seven", 7)]);
    let holed = append(&dense, vec![partition_entry("nine", 9, 3)]);
    let opened = load(&fixture);
    fixture.delete_manifest(&find_joined_manifest(&dense, &holed));

    let nullability = opened.select_scan(&v_equals(7)).unwrap().nullability;

    assert_eq!(nullability, [false]);
}

#[test]
fn a_scan_of_every_partition_reports_a_column_with_a_null_in_one_of_them_nullable() {
    let (fixture, created) = create_partitioned_table();
    let dense = append(&created, vec![fixture.write_file("seven", 7)]);
    let holed = fixture.write_data_file(partition_entry("nine", 9, 3), 9);
    append(&dense, vec![holed]);
    let opened = load(&fixture);

    let nullability = opened.select_scan(&[]).unwrap().nullability;

    assert_eq!(nullability, [true]);
}

#[test]
fn a_scan_that_keeps_no_file_reports_every_column_nullable() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![partition_entry("seven", 7, 0)]);
    let opened = load(&fixture);

    let selection = opened.select_scan(&v_equals(8)).unwrap();

    assert!(selection.row_groups.row_groups().is_empty());
    assert_eq!(selection.nullability, [true]);
}

/// The value of `v` each of `partitions` holds the files of.
fn list_values(partitions: &[Arc<Partition>]) -> Vec<i64> {
    partitions
        .iter()
        .map(|partition| match &partition.tuple().fields()[0] {
            Some(Literal::Primitive(PrimitiveLiteral::Long(value))) => *value,
            other => panic!("a partition of `v` holds a long, not {other:?}"),
        })
        .collect()
}

/// The sequence number of the commit that wrote `table`'s version.
fn sequence_number(table: &Table) -> i64 {
    table.metadata().last_sequence_number()
}

#[test]
fn a_filtered_scan_reads_only_the_manifests_whose_bounds_admit_its_partition() {
    let (fixture, created) = create_partitioned_table();
    let early = append(
        &created,
        vec![partition_entry("one", 1, 0), fixture.write_file("two", 2)],
    );
    let late = append(
        &early,
        vec![
            partition_entry("eight", 8, 0),
            partition_entry("nine", 9, 0),
        ],
    );
    let opened = load(&fixture);
    fixture.delete_manifest(&find_joined_manifest(&early, &late));

    let files = selected(&opened, &v_equals(2));

    assert_eq!(files, ["two.parquet"]);
}

#[test]
fn the_partitions_of_manifests_holding_one_each_are_listed_with_no_manifest_read() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![partition_entry("seven", 7, 0)]);
    append(&first, vec![partition_entry("nine", 9, 0)]);
    let opened = load(&fixture);
    fixture.delete_data_manifests(&opened);

    let partitions = opened.partitions().unwrap();

    assert_eq!(list_values(&partitions), [7, 9]);
}

#[test]
fn nothing_is_kept_of_a_manifest_read_for_the_partitions_it_holds() {
    let (fixture, created) = create_partitioned_table();
    append(
        &created,
        vec![
            partition_entry("seven", 7, 0),
            partition_entry("nine", 9, 0),
        ],
    );
    let opened = load(&fixture);

    let partitions = opened.partitions().unwrap();
    fixture.delete_data_manifests(&opened);
    let listed_again = opened.partitions();

    assert_eq!(list_values(&partitions), [7, 9]);
    assert!(listed_again.is_err());
}

#[test]
fn the_partitions_added_to_after_a_commit_are_those_of_the_files_committed_since() {
    let (_fixture, created) = create_partitioned_table();
    let first = append(
        &created,
        vec![
            partition_entry("seven", 7, 0),
            partition_entry("nine", 9, 0),
        ],
    );
    let second = append(&first, vec![partition_entry("nine-again", 9, 0)]);
    let third = append(&second, vec![partition_entry("three", 3, 0)]);

    let since_first = third
        .list_partitions_added_after(sequence_number(&first))
        .unwrap();
    let since_third = third
        .list_partitions_added_after(sequence_number(&third))
        .unwrap();

    assert_eq!(list_values(&since_first), [3, 9]);
    assert!(since_third.is_empty());
}

#[test]
fn a_commit_that_merges_manifests_adds_only_to_the_partition_of_its_own_file() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![partition_entry("seven", 7, 0)]);
    let second = append(&first, vec![partition_entry("nine", 9, 0)]);
    fixture.metadata_store.publish_changed(|builder| {
        let merge_from_two_manifests = HashMap::from([(
            "commit.manifest.min-count-to-merge".to_string(),
            "2".to_string(),
        )]);
        builder.set_properties(merge_from_two_manifests).unwrap()
    });
    let altered = load(&fixture);

    let merged = append(&altered, vec![partition_entry("three", 3, 0)]);
    let since_second = merged
        .list_partitions_added_after(sequence_number(&second))
        .unwrap();

    assert_eq!(merged.manifest_list().data_manifests().count(), 1);
    assert_eq!(list_values(&since_second), [3]);
}

#[test]
fn a_rewrite_adds_to_the_partition_of_its_output_and_to_no_other() {
    let (_fixture, created) = create_partitioned_table();
    let appended = append(
        &created,
        vec![
            partition_entry("seven", 7, 0),
            partition_entry("nine", 9, 0),
        ],
    );
    let rewritten_file = appended
        .entries()
        .unwrap()
        .into_iter()
        .find(|entry| entry.file.path.as_str() == "seven.parquet")
        .unwrap();

    let rewritten = appended
        .commit(
            &[rewritten_file],
            vec![partition_entry("seven-rewritten", 7, 0)],
        )
        .unwrap()
        .expect("no other writer");
    let since_append = rewritten
        .list_partitions_added_after(sequence_number(&appended))
        .unwrap();

    assert_eq!(list_values(&since_append), [7]);
}

#[test]
fn a_partition_indexed_before_a_commit_is_read_after_it_from_the_manifest_it_wrote_alone() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("early", 7)]);
    selected(&first, &v_equals(7));

    let second = append(&first, vec![fixture.write_file("late", 7)]);
    fixture.delete_data_manifests(&first);
    let files = selected(&second, &v_equals(7));
    fixture.delete_data_manifests(&second);
    let files_again = selected(&second, &v_equals(7));

    assert_eq!(files, ["early.parquet", "late.parquet"]);
    assert_eq!(files_again, files);
}

#[test]
fn the_first_scan_of_a_partition_after_a_commit_reads_the_manifest_the_commit_wrote() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("early", 7)]);
    selected(&first, &v_equals(7));

    let second = append(&first, vec![fixture.write_file("late", 7)]);
    fixture.delete_manifest(&find_joined_manifest(&first, &second));
    let scan = second.select_scan(&v_equals(7));

    assert!(scan.is_err());
}

#[test]
fn a_partitions_key_names_the_partition_and_its_files_again() {
    let (_fixture, created) = create_partitioned_table();
    let written = append(&created, vec![partition_entry("seven", 7, 0)]);
    let listed = written.partitions().unwrap();

    let found = written.find_partition(&listed[0].encode_key()).unwrap();
    let files = written.partition_files(&found).unwrap();

    assert_eq!(list_values(&[found]), [7]);
    assert_eq!(files.files().count(), 1);
}

const DAY_MICROS: i64 = 86_400_000_000;

/// A table of the `Int64` column `v` and the `Timestamp` column `at`,
/// partitioned by `fields`.
fn create_events_table(fields: &[SpecField]) -> (Fixture, Table) {
    let columns = [column("v", Type::Int64), column("at", Type::Timestamp)];
    let (fixture, _created) = create_table_of(&columns, &[], &[]);
    let table = repartition(&fixture, fields);
    (fixture, table)
}

/// An entry for `name`, a file of ten rows of `v` at the start of the day
/// `day` after the epoch, in the partition those values have under
/// `fields`.
fn event(fields: &[SpecField], name: &str, v: i64, day: i64) -> DataFileEntry {
    event_at(fields, name, v, day * DAY_MICROS)
}

/// An entry for `name`, a file of ten rows of `v` at `at_micros`
/// microseconds after the epoch, in the partition those values have under
/// `fields`.
fn event_at(fields: &[SpecField], name: &str, v: i64, at_micros: i64) -> DataFileEntry {
    let v: ArrayRef = Arc::new(Int64Array::from(vec![v]));
    let at: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![at_micros]));
    let partition = fields.iter().map(|(source, name, transform)| {
        let source = if *source == "v" { &v } else { &at };
        let value = create_transform_function(transform)
            .unwrap()
            .transform(Arc::clone(source))
            .unwrap();
        (name.to_string(), Scalar::new(value))
    });
    let bounds = || HashMap::from([("v".to_string(), v.clone()), ("at".to_string(), at.clone())]);
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new(format!("{name}.parquet")),
            size: 10,
        },
        partition: Some(partition.collect()),
        partition_spec_id: None,
        stats: Some(Arc::new(FileStats {
            num_records: Some(10),
            min_values: bounds(),
            max_values: bounds(),
            null_counts: HashMap::new(),
        })),
        sort_order_id: None,
        first_row_id: None,
    }
}

/// [`event`], with the file written into the table's directory.
fn write_event(
    fixture: &Fixture,
    fields: &[SpecField],
    name: &str,
    v: i64,
    day: i64,
) -> DataFileEntry {
    fixture.write_data_file(event(fields, name, v, day), v)
}

/// The comparison of the column at `column_idx` with `constant`.
fn compare(column_idx: usize, compare_type: CompareType, constant: Scalar<ArrayRef>) -> Expression {
    let return_type = [Type::Int64, Type::Timestamp][column_idx].clone();
    Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx,
            return_type,
            name: None,
        })),
        right: Box::new(Expression::Constant(constant)),
        compare_type,
        return_type: Type::Boolean,
    })
}

/// The filter `v IN (values)`.
fn v_in(values: &[i64]) -> Expression {
    Expression::InList(InList {
        input: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int64,
            name: None,
        })),
        values: values
            .iter()
            .map(|value| Expression::Constant(int_scalar(*value)))
            .collect(),
    })
}

/// The filters that keep `at` from the start of the day `first` to before
/// the start of the day `end`.
fn at_within_days(first: i64, end: i64) -> Vec<Expression> {
    at_within(first * DAY_MICROS, end * DAY_MICROS)
}

/// The filters that keep `at` from `first` to before `end`, each in
/// microseconds after the epoch.
fn at_within(first: i64, end: i64) -> Vec<Expression> {
    let instant = |micros: i64| -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(TimestampMicrosecondArray::from(vec![micros])) as ArrayRef)
    };
    vec![
        compare(1, CompareType::GreaterEqual, instant(first)),
        compare(1, CompareType::Less, instant(end)),
    ]
}

/// Microseconds from the epoch to `instant`, a timestamp written
/// `1969-12-31T22:30:00` that may stop after any of its fields: `1969` is
/// the start of that year and `1969-12-31T22` the start of that hour.
fn parse_instant(instant: &str) -> i64 {
    let unwritten_fields = &"0000-01-01T00:00:00"[instant.len()..];
    let parsed = Datum::timestamp_from_str(format!("{instant}{unwritten_fields}")).unwrap();
    let PrimitiveLiteral::Long(micros) = parsed.literal() else {
        panic!("a timestamp is held as a long, not {parsed:?}");
    };
    *micros
}

/// A table partitioned by `transform` of `at`, holding one file for each of
/// `units`, a year, month, day or hour as [`parse_instant`] reads it, named
/// by the unit and in the partition the unit's start has. No file records
/// bounds of `at`, so a scan filtered on `at` selects every file of the
/// partitions it keeps.
fn create_table_of_units(transform: Transform, units: &[&str]) -> (Fixture, Table) {
    let fields = [("at", "at_unit", transform)];
    let (fixture, created) = create_events_table(&fields);
    let files = units.iter().map(|unit| {
        let mut entry = event_at(&fields, unit, 0, parse_instant(unit));
        let stats = Arc::get_mut(entry.stats.as_mut().unwrap()).unwrap();
        stats.min_values.remove("at");
        stats.max_values.remove("at");
        fixture.write_data_file(entry, 0)
    });
    append(&created, files.collect());
    let written = load(&fixture);
    (fixture, written)
}

/// The units whose files a scan of `table`, one [`create_table_of_units`]
/// made, selects over `filters`, in order of time.
fn list_selected_units(table: &Table, filters: &[Expression]) -> Vec<String> {
    let mut units: Vec<String> = selected(table, filters)
        .iter()
        .map(|name| name.trim_end_matches(".parquet").to_string())
        .collect();
    units.sort();
    units
}

/// The units a scan keeping `at` from `first` to before `end`, each as
/// [`parse_instant`] reads it, selects of a table of `units` partitioned by
/// `transform` ([`create_table_of_units`]): as planned with the table's data
/// manifests in place, and as planned again once they are deleted, when
/// only the filters and the partitions the first plan indexed name a file.
fn plan_range_twice(
    transform: Transform,
    units: &[&str],
    first: &str,
    end: &str,
) -> [Vec<String>; 2] {
    let (fixture, written) = create_table_of_units(transform, units);
    let range = at_within(parse_instant(first), parse_instant(end));
    let with_manifests = list_selected_units(&written, &range);
    fixture.delete_data_manifests(&written);
    [with_manifests, list_selected_units(&written, &range)]
}

#[test]
fn a_second_plan_whose_partition_an_equality_determines_reads_no_manifest() {
    let fields = [("v", "v", Transform::Identity)];
    let (fixture, created) = create_events_table(&fields);
    append(
        &created,
        vec![
            write_event(&fixture, &fields, "seven", 7, 0),
            event(&fields, "nine", 9, 0),
        ],
    );
    let opened = load(&fixture);
    let first = selected(&opened, &v_equals(7));
    fixture.delete_data_manifests(&opened);

    let second = selected(&opened, &v_equals(7));

    assert_eq!(first, ["seven.parquet"]);
    assert_eq!(second, first);
}

#[test]
fn an_in_list_on_a_bucketed_column_selects_the_files_holding_its_values() {
    let fields = [("v", "v_bucket", Transform::Bucket(4))];
    let (fixture, created) = create_events_table(&fields);
    let files = [-5, 1, 2, 3, 4, 6]
        .map(|v| write_event(&fixture, &fields, &format!("v{v}"), v, 0))
        .to_vec();
    let written = append(&created, files);

    let mut files = selected(&written, &[v_in(&[-5, 2, 4, 40])]);
    files.sort();

    assert_eq!(files, ["v-5.parquet", "v2.parquet", "v4.parquet"]);
}

#[test]
fn a_bounded_range_of_days_selects_the_files_of_the_days_it_spans() {
    let fields = [("at", "at_day", Transform::Day)];
    let (fixture, created) = create_events_table(&fields);
    let files = [-2, -1, 0, 1]
        .map(|day| write_event(&fixture, &fields, &format!("day{day}"), 0, day))
        .to_vec();
    append(&created, files);
    let written = load(&fixture);
    let first = selected(&written, &at_within_days(-1, 1));
    fixture.delete_data_manifests(&written);

    let second = selected(&written, &at_within_days(-1, 1));

    assert_eq!(first, ["day-1.parquet", "day0.parquet"]);
    assert_eq!(second, first);
}

#[test]
fn a_range_from_the_start_of_one_unit_to_the_start_of_another_selects_the_first_and_not_the_last() {
    let cases = [
        (
            Transform::Year,
            ["1999", "2000", "2001", "2002", "2003"],
            ("2000", "2002"),
            ["2000", "2001"],
        ),
        (
            Transform::Month,
            ["1999-11", "1999-12", "2000-01", "2000-02", "2000-03"],
            ("1999-12", "2000-02"),
            ["1999-12", "2000-01"],
        ),
        (
            Transform::Hour,
            [
                "1999-12-31T22",
                "1999-12-31T23",
                "2000-01-01T00",
                "2000-01-01T01",
                "2000-01-01T02",
            ],
            ("1999-12-31T23", "2000-01-01T01"),
            ["1999-12-31T23", "2000-01-01T00"],
        ),
    ];

    for (transform, units, (first, end), expected) in cases {
        let [with_manifests, without_manifests] = plan_range_twice(transform, &units, first, end);

        assert_eq!(with_manifests, expected, "{transform:?}");
        assert_eq!(without_manifests, expected, "{transform:?}");
    }
}

#[test]
fn a_range_that_starts_and_ends_inside_units_selects_both_of_them() {
    let cases = [
        (
            Transform::Year,
            ["1999", "2000", "2001", "2002", "2003"],
            ("2000-07-01", "2002-03-01"),
            ["2000", "2001", "2002"],
        ),
        (
            Transform::Month,
            ["1999-11", "1999-12", "2000-01", "2000-02", "2000-03"],
            ("1999-12-15", "2000-02-15"),
            ["1999-12", "2000-01", "2000-02"],
        ),
        (
            Transform::Hour,
            [
                "1999-12-31T22",
                "1999-12-31T23",
                "2000-01-01T00",
                "2000-01-01T01",
                "2000-01-01T02",
            ],
            ("1999-12-31T23:30", "2000-01-01T01:30"),
            ["1999-12-31T23", "2000-01-01T00", "2000-01-01T01"],
        ),
    ];

    for (transform, units, (first, end), expected) in cases {
        let [with_manifests, without_manifests] = plan_range_twice(transform, &units, first, end);

        assert_eq!(with_manifests, expected, "{transform:?}");
        assert_eq!(without_manifests, expected, "{transform:?}");
    }
}

#[test]
fn a_range_across_the_epoch_selects_the_units_it_spans_on_both_sides() {
    let cases = [
        (
            Transform::Year,
            ["1967", "1968", "1969", "1970", "1971", "1972"],
            ("1968-07-01", "1971-07-01"),
            ["1968", "1969", "1970", "1971"],
        ),
        (
            Transform::Month,
            [
                "1969-10", "1969-11", "1969-12", "1970-01", "1970-02", "1970-03",
            ],
            ("1969-11-15", "1970-02-15"),
            ["1969-11", "1969-12", "1970-01", "1970-02"],
        ),
        (
            Transform::Day,
            [
                "1969-12-29",
                "1969-12-30",
                "1969-12-31",
                "1970-01-01",
                "1970-01-02",
                "1970-01-03",
            ],
            ("1969-12-30T12", "1970-01-02T12"),
            ["1969-12-30", "1969-12-31", "1970-01-01", "1970-01-02"],
        ),
        (
            Transform::Hour,
            [
                "1969-12-31T21",
                "1969-12-31T22",
                "1969-12-31T23",
                "1970-01-01T00",
                "1970-01-01T01",
                "1970-01-01T02",
            ],
            ("1969-12-31T22:30", "1970-01-01T01:30"),
            [
                "1969-12-31T22",
                "1969-12-31T23",
                "1970-01-01T00",
                "1970-01-01T01",
            ],
        ),
    ];

    for (transform, units, (first, end), expected) in cases {
        let [with_manifests, without_manifests] = plan_range_twice(transform, &units, first, end);

        assert_eq!(with_manifests, expected, "{transform:?}");
        assert_eq!(without_manifests, expected, "{transform:?}");
    }
}

/// The spec crate projects an upper bound before the epoch onto its own
/// year, month or day and the next, where some writers put such a row, so
/// one partition past the last that holds a row of the range is selected.
#[test]
fn a_range_of_years_months_or_days_before_the_epoch_also_selects_the_unit_after_its_last() {
    let cases = [
        (
            Transform::Year,
            ["1966", "1967", "1968", "1969", "1970"],
            ("1967", "1969"),
            ["1967", "1968", "1969"],
        ),
        (
            Transform::Year,
            ["1967", "1968", "1969", "1970", "1971"],
            ("1968", "1970"),
            ["1968", "1969", "1970"],
        ),
        (
            Transform::Month,
            ["1969-08", "1969-09", "1969-10", "1969-11", "1969-12"],
            ("1969-09", "1969-11"),
            ["1969-09", "1969-10", "1969-11"],
        ),
        (
            Transform::Month,
            ["1969-10", "1969-11", "1969-12", "1970-01", "1970-02"],
            ("1969-11", "1970-01"),
            ["1969-11", "1969-12", "1970-01"],
        ),
        (
            Transform::Day,
            [
                "1969-12-27",
                "1969-12-28",
                "1969-12-29",
                "1969-12-30",
                "1969-12-31",
            ],
            ("1969-12-28", "1969-12-30"),
            ["1969-12-28", "1969-12-29", "1969-12-30"],
        ),
        (
            Transform::Day,
            [
                "1969-12-29",
                "1969-12-30",
                "1969-12-31",
                "1970-01-01",
                "1970-01-02",
            ],
            ("1969-12-30", "1970-01-01"),
            ["1969-12-30", "1969-12-31", "1970-01-01"],
        ),
    ];

    for (transform, units, (first, end), expected) in cases {
        let [with_manifests, without_manifests] = plan_range_twice(transform, &units, first, end);

        assert_eq!(with_manifests, expected, "{transform:?} to {end}");
        assert_eq!(without_manifests, expected, "{transform:?} to {end}");
    }
}

/// The spec crate projects an upper bound before the epoch onto its own
/// hour alone.
#[test]
fn a_range_of_hours_before_the_epoch_selects_only_the_hours_it_spans() {
    let cases = [
        (
            [
                "1969-12-31T19",
                "1969-12-31T20",
                "1969-12-31T21",
                "1969-12-31T22",
                "1969-12-31T23",
            ],
            ("1969-12-31T20", "1969-12-31T22"),
            ["1969-12-31T20", "1969-12-31T21"],
        ),
        (
            [
                "1969-12-31T21",
                "1969-12-31T22",
                "1969-12-31T23",
                "1970-01-01T00",
                "1970-01-01T01",
            ],
            ("1969-12-31T22", "1970-01-01T00"),
            ["1969-12-31T22", "1969-12-31T23"],
        ),
    ];

    for (units, (first, end), expected) in cases {
        let [with_manifests, without_manifests] =
            plan_range_twice(Transform::Hour, &units, first, end);

        assert_eq!(with_manifests, expected, "to {end}");
        assert_eq!(without_manifests, expected, "to {end}");
    }
}

#[test]
fn filters_determining_two_partition_fields_select_the_files_of_their_combinations() {
    let fields = [
        ("v", "v", Transform::Identity),
        ("at", "at_day", Transform::Day),
    ];
    let (fixture, created) = create_events_table(&fields);
    let files = [("a", 1, 0), ("b", 2, 1), ("c", 3, 0), ("d", 1, 5)]
        .map(|(name, v, day)| write_event(&fixture, &fields, name, v, day))
        .to_vec();
    append(&created, files);
    let written = load(&fixture);
    let mut filters = at_within_days(0, 2);
    filters.push(v_in(&[1, 2]));
    let first = selected(&written, &filters);
    fixture.delete_data_manifests(&written);

    let second = selected(&written, &filters);

    assert_eq!(first, ["a.parquet", "b.parquet"]);
    assert_eq!(second, first);
}

#[test]
fn a_partition_computed_from_the_filter_without_a_file_adds_nothing_to_the_scan() {
    let (fixture, created) = create_partitioned_table();
    let written = append(
        &created,
        vec![
            partition_entry("seven", 7, 0),
            fixture.write_file("nine", 9),
        ],
    );

    let between_the_two = written.select_scan(&v_equals(8)).unwrap();
    let files = selected(&written, &[v_in(&[8, 9])]);

    assert!(between_the_two.row_groups.row_groups().is_empty());
    assert_eq!(files, ["nine.parquet"]);
}

#[test]
fn an_equality_selects_no_file_of_the_partition_without_a_value() {
    let (fixture, created) = create_partitioned_table();
    let mut unset = partition_entry("unset", 7, 10);
    let no_value: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
    unset.partition = Some(PartitionValues::from([(
        "v".to_string(),
        Scalar::new(no_value),
    )]));
    let written = append(&created, vec![fixture.write_file("seven", 7), unset]);

    let files = selected(&written, &v_equals(7));

    assert_eq!(files, ["seven.parquet"]);
}

#[test]
fn a_range_spanning_more_partitions_than_are_computed_from_a_filter_lists_those_the_manifests_hold()
{
    let fields = [("v", "v", Transform::Identity)];
    let (fixture, created) = create_events_table(&fields);
    append(
        &created,
        vec![
            write_event(&fixture, &fields, "five", 5, 0),
            write_event(&fixture, &fields, "late", 900, 0),
        ],
    );
    let written = load(&fixture);
    let wide_range = [
        compare(0, CompareType::GreaterEqual, int_scalar(0)),
        compare(0, CompareType::Less, int_scalar(1000)),
    ];
    let first = selected(&written, &wide_range);
    fixture.delete_data_manifests(&written);

    let second = written.select_scan(&wide_range);

    assert_eq!(first, ["five.parquet", "late.parquet"]);
    assert!(second.is_err());
}

#[test]
fn filters_determining_every_spec_in_use_select_the_files_written_under_each() {
    let by_value = [("v", "v", Transform::Identity)];
    let by_day = [("at", "at_day", Transform::Day)];
    let (fixture, created) = create_events_table(&by_value);
    append(
        &created,
        vec![
            write_event(&fixture, &by_value, "old", 7, 3),
            write_event(&fixture, &by_value, "other", 9, 3),
        ],
    );
    let repartitioned = repartition(&fixture, &by_day);
    append(
        &repartitioned,
        vec![
            write_event(&fixture, &by_day, "new", 7, 3),
            write_event(&fixture, &by_day, "later", 7, 8),
        ],
    );
    let written = load(&fixture);
    let mut filters = at_within_days(3, 4);
    filters.extend(v_equals(7));
    let mut first = selected(&written, &filters);
    fixture.delete_data_manifests(&written);

    let mut second = selected(&written, &filters);

    first.sort();
    second.sort();
    assert_eq!(first, ["new.parquet", "old.parquet"]);
    assert_eq!(second, first);
}

#[test]
fn filters_determining_one_of_two_specs_in_use_list_the_partitions_the_manifests_hold() {
    let by_value = [("v", "v", Transform::Identity)];
    let by_day = [("at", "at_day", Transform::Day)];
    let (fixture, created) = create_events_table(&by_value);
    append(
        &created,
        vec![write_event(&fixture, &by_value, "old", 7, 3)],
    );
    let repartitioned = repartition(&fixture, &by_day);
    let written = append(
        &repartitioned,
        vec![
            write_event(&fixture, &by_day, "new", 7, 3),
            write_event(&fixture, &by_day, "later", 7, 8),
        ],
    );

    let mut files = selected(&written, &v_equals(7));
    files.sort();

    assert_eq!(files, ["later.parquet", "new.parquet", "old.parquet"]);
}
