//! A managed Iceberg table partitioned by a transform of a column, through
//! SQL: `CREATE TABLE` records the terms as a partition spec, an INSERT cuts
//! its files by the transformed values and records each file's partition
//! tuple, a filter on the source column reads the partitions it keeps, a
//! DELETE takes out the partitions its conditions cover and no other row,
//! compaction leaves every row in the partition its file records, and a term
//! the table cannot be partitioned by is refused by name.

mod iceberg_support;

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{ArrayRef, Datum as _, Int64Array, Scalar, TimestampMicrosecondArray};
use arrow_schema::DataType;
use catalog::datastore::DatastoreTransaction as _;
use datastore_pivotlake::test_support::metadata_versions;
use datastore_pivotlake::{CatalogTable, FileEntry, TableFormat};
use iceberg_support::{Database, collect_i64s, single_i64};
use iceberg_tables::iceberg::spec::{Datum, PrimitiveLiteral, TableMetadata};
use planner::catalog::{SchemaQualifiedTableName, split_partition_terms};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;

const HOUR_MICROS: i64 = 3_600_000_000;
const DAY_MICROS: i64 = 24 * HOUR_MICROS;
/// 2024-02-29, in days after the epoch.
const LEAP_DAY: i64 = 19_782;

/// One row of a table partitioned by time: its instant, in microseconds
/// after the epoch, with the year and the month it falls in, counted from
/// 1970 and from January 1970.
#[derive(Clone, Copy)]
struct Instant {
    micros: i64,
    year: i64,
    month: i64,
}

impl Instant {
    fn count_days(&self) -> i64 {
        self.micros.div_euclid(DAY_MICROS)
    }

    fn count_hours(&self) -> i64 {
        self.micros.div_euclid(HOUR_MICROS)
    }
}

/// A year and a month, counted from 1970 and from January 1970.
type YearAndMonth = (i64, i64);

/// The start of a year, a month, a day and an hour on each side of the
/// epoch, each with the year and the month it starts and those of the
/// microsecond before it.
const UNIT_STARTS: [(&str, YearAndMonth, YearAndMonth); 9] = [
    ("1969-01-01T00:00:00", (-1, -12), (-2, -13)),
    ("1969-12-01T00:00:00", (-1, -1), (-1, -2)),
    ("1969-12-31T00:00:00", (-1, -1), (-1, -1)),
    ("1969-12-31T23:00:00", (-1, -1), (-1, -1)),
    ("1970-01-01T00:00:00", (0, 0), (-1, -1)),
    ("2001-01-01T00:00:00", (31, 372), (30, 371)),
    ("2001-03-01T00:00:00", (31, 374), (31, 373)),
    ("2001-03-15T00:00:00", (31, 374), (31, 374)),
    ("2001-03-15T07:00:00", (31, 374), (31, 374)),
];

/// Every unit start with the microsecond before it and the one after, in
/// order: where a partition value rounded the wrong way puts a row in the
/// wrong file.
fn list_instants() -> Vec<Instant> {
    UNIT_STARTS
        .iter()
        .flat_map(|(start, (year, month), (year_before, month_before))| {
            let start = Datum::timestamp_from_str(start).unwrap();
            let PrimitiveLiteral::Long(micros) = *start.literal() else {
                panic!("a timestamp is held as a long, not {start:?}");
            };
            [
                Instant {
                    micros: micros - 1,
                    year: *year_before,
                    month: *month_before,
                },
                Instant {
                    micros,
                    year: *year,
                    month: *month,
                },
                Instant {
                    micros: micros + 1,
                    year: *year,
                    month: *month,
                },
            ]
        })
        .collect()
}

/// The instant `micros` after the epoch, as a SQL constant.
fn write_timestamp(micros: i64) -> String {
    format!("TIMESTAMP '{}'", Datum::timestamp_micros(micros))
}

/// What a query asks of `ts`: as SQL, and as the same comparison of a row's
/// instant.
type TimeCondition = (String, Box<dyn Fn(i64) -> bool>);

/// What a query asks of `ts` against `bounds`, instants in order: an
/// equality with each, an open range to either side of it, and the two
/// bounded ranges between each and the next.
fn list_time_conditions(bounds: &[i64]) -> Vec<TimeCondition> {
    let mut conditions: Vec<TimeCondition> = Vec::new();
    for (position, &low) in bounds.iter().enumerate() {
        let low_sql = write_timestamp(low);
        conditions.push((format!("ts = {low_sql}"), Box::new(move |ts| ts == low)));
        conditions.push((format!("ts < {low_sql}"), Box::new(move |ts| ts < low)));
        conditions.push((format!("ts >= {low_sql}"), Box::new(move |ts| ts >= low)));
        let Some(&high) = bounds.get(position + 1) else {
            continue;
        };
        let high_sql = write_timestamp(high);
        conditions.push((
            format!("ts >= {low_sql} AND ts < {high_sql}"),
            Box::new(move |ts| low <= ts && ts < high),
        ));
        conditions.push((
            format!("ts > {low_sql} AND ts <= {high_sql}"),
            Box::new(move |ts| low < ts && ts <= high),
        ));
    }
    conditions
}

/// Which of a year, a month, a day and an hour begin at each instant of
/// [`UNIT_STARTS`], in order.
const UNITS_BEGUN: [&[&str]; 9] = [
    &["year", "month", "day", "hour"],
    &["month", "day", "hour"],
    &["day", "hour"],
    &["hour"],
    &["year", "month", "day", "hour"],
    &["year", "month", "day", "hour"],
    &["month", "day", "hour"],
    &["day", "hour"],
    &["hour"],
];

/// What a DELETE asks of `ts`: as SQL, as the same comparison of a row's
/// instant, and whether a table partitioned by a unit of time has to take
/// it (`Some(true)`), has to refuse it (`Some(false)`), or may do either.
struct DeleteCondition {
    sql: String,
    holds_for: Box<dyn Fn(i64) -> bool>,
    is_taken: Option<bool>,
}

/// What a DELETE asks of `ts` against each instant of [`UNIT_STARTS`]: an
/// equality with it, an open range to either side of it, and the two
/// bounded ranges between it and the next. A table partitioned by `unit`
/// has a row on both sides of each of those instants, so it takes exactly
/// the ranges that end where a unit begins and include that instant on the
/// upper side. A bound at or before 1970 may be refused as well: a
/// partition there counts as covered only as far as a writer that filed
/// such a time one partition late allows.
fn list_delete_conditions(unit: &str) -> Vec<DeleteCondition> {
    let bounds: Vec<(i64, bool)> = UNIT_STARTS
        .iter()
        .zip(UNITS_BEGUN)
        .map(|((start, _, _), units_begun)| {
            let start = Datum::timestamp_from_str(start).unwrap();
            let PrimitiveLiteral::Long(micros) = *start.literal() else {
                panic!("a timestamp is held as a long, not {start:?}");
            };
            (micros, units_begun.contains(&unit))
        })
        .collect();
    let judge =
        |micros: i64, ends_on_unit_starts: bool| (micros > 0).then_some(ends_on_unit_starts);
    let mut conditions = Vec::new();
    for (position, &(low, low_begins_unit)) in bounds.iter().enumerate() {
        let low_sql = write_timestamp(low);
        conditions.push(DeleteCondition {
            sql: format!("ts = {low_sql}"),
            holds_for: Box::new(move |ts| ts == low),
            is_taken: Some(false),
        });
        conditions.push(DeleteCondition {
            sql: format!("ts < {low_sql}"),
            holds_for: Box::new(move |ts| ts < low),
            is_taken: judge(low, low_begins_unit),
        });
        conditions.push(DeleteCondition {
            sql: format!("ts >= {low_sql}"),
            holds_for: Box::new(move |ts| ts >= low),
            is_taken: judge(low, low_begins_unit),
        });
        let Some(&(high, high_begins_unit)) = bounds.get(position + 1) else {
            continue;
        };
        let high_sql = write_timestamp(high);
        conditions.push(DeleteCondition {
            sql: format!("ts >= {low_sql} AND ts < {high_sql}"),
            holds_for: Box::new(move |ts| low <= ts && ts < high),
            is_taken: judge(low, low_begins_unit && high_begins_unit),
        });
        conditions.push(DeleteCondition {
            sql: format!("ts > {low_sql} AND ts <= {high_sql}"),
            holds_for: Box::new(move |ts| low < ts && ts <= high),
            is_taken: Some(false),
        });
    }
    conditions
}

/// The conditions a table partitioned by a unit of time answered a DELETE
/// by wrongly.
struct MisdeletedConditions {
    /// Those whose DELETE left other rows than it should have: the rows the
    /// condition is not true of when it was taken, every row when it was
    /// refused.
    left_wrong_rows: Vec<String>,
    /// Those whose DELETE counted other rows than the condition is true of.
    miscounted: Vec<String>,
    /// Those taken that had to be refused, or refused that had to be taken.
    misjudged: Vec<String>,
}

/// Create a table partitioned by `unit(ts)` holding a row per instant of
/// [`list_instants`], its id the instant's position, and delete from it by
/// every condition of [`list_delete_conditions`], each from the full table.
fn delete_by_every_time_condition(unit: &str) -> MisdeletedConditions {
    let instants = list_instants();
    let mut db = Database::new();
    db.execute(&format!(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP) PARTITIONED BY ({unit}(ts))"
    ));
    let rows: Vec<String> = (0..)
        .zip(&instants)
        .map(|(id, instant): (i64, _)| format!("({id}, {})", write_timestamp(instant.micros)))
        .collect();
    let insert_every_row = format!("INSERT INTO events VALUES {}", rows.join(", "));
    db.execute(&insert_every_row);
    let every_id: Vec<i64> = (0..instants.len() as i64).collect();
    let mut misdeleted = MisdeletedConditions {
        left_wrong_rows: Vec::new(),
        miscounted: Vec::new(),
        misjudged: Vec::new(),
    };
    for condition in list_delete_conditions(unit) {
        let kept_ids: Vec<i64> = (0..)
            .zip(&instants)
            .filter(|(_, instant)| !(condition.holds_for)(instant.micros))
            .map(|(id, _)| id)
            .collect();

        let deleted = db.try_execute(&format!("DELETE FROM events WHERE {}", condition.sql));

        let left_ids = collect_i64s(&db.execute("SELECT id FROM events ORDER BY id"), 0);
        let expected_ids = if deleted.is_ok() {
            &kept_ids
        } else {
            &every_id
        };
        if &left_ids != expected_ids {
            misdeleted.left_wrong_rows.push(condition.sql.clone());
        }
        if let Ok(deleted) = &deleted
            && single_i64(deleted) != (every_id.len() - kept_ids.len()) as i64
        {
            misdeleted.miscounted.push(condition.sql.clone());
        }
        if condition
            .is_taken
            .is_some_and(|is_taken| is_taken != deleted.is_ok())
        {
            misdeleted.misjudged.push(condition.sql.clone());
        }
        if deleted.is_ok() {
            db.execute("DELETE FROM events");
            db.execute(&insert_every_row);
        }
    }
    misdeleted
}

/// The copy of the table `name` the datastore of `db` holds now.
fn hold_table(db: &Database, name: &str) -> CatalogTable {
    db.datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
        .expect("the table exists")
}

/// One field of a partition spec: its transform, its source column and its
/// name.
type PartitionField = (String, String, String);

fn field(transform: &str, source: &str, name: &str) -> PartitionField {
    (transform.to_string(), source.to_string(), name.to_string())
}

/// The fields of the default partition spec the current metadata file of
/// the table `name` of `db` records.
fn read_partition_fields(db: &Database, name: &str) -> Vec<PartitionField> {
    let directory = db.table_directory(name);
    let versions = metadata_versions(&db.root_uri(), &directory).unwrap();
    let current_file = db
        .table_path(name)
        .join("metadata")
        .join(format!("v{:020}.metadata.json", versions.last().unwrap()));
    let metadata: TableMetadata =
        serde_json::from_slice(&std::fs::read(current_file).unwrap()).unwrap();
    let schema = metadata.current_schema();
    metadata
        .default_partition_spec()
        .fields()
        .iter()
        .map(|field| {
            (
                field.transform.to_string(),
                schema
                    .name_by_field_id(field.source_id)
                    .unwrap()
                    .to_string(),
                field.name.clone(),
            )
        })
        .collect()
}

/// The value the tuple of the file `entry` holds of the time partition
/// field `field`: a count of years, months or hours, or a day.
fn read_time_value(entry: &FileEntry, field: &str) -> i64 {
    let tuple = entry.partition.as_ref().expect("the file has a tuple");
    let (value, _) = tuple[field].get();
    match value.data_type() {
        DataType::Date32 => i64::from(value.as_primitive::<Date32Type>().value(0)),
        DataType::Int32 => i64::from(value.as_primitive::<Int32Type>().value(0)),
        other => panic!("a time partition value is an int or a date, not {other}"),
    }
}

/// The value the tuple of the file `entry` holds of the string partition
/// field `field`.
fn read_string_value(entry: &FileEntry, field: &str) -> String {
    let tuple = entry.partition.as_ref().expect("the file has a tuple");
    let (value, _) = tuple[field].get();
    value.as_string_view().value(0).to_string()
}

fn count_rows(entry: &FileEntry) -> i64 {
    entry.stats.as_ref().unwrap().num_records.unwrap()
}

/// Per live file of the table `name` of `db`, the value its tuple holds of
/// the time partition field `field` and its row count, sorted.
fn list_time_partition_files(db: &Database, name: &str, field: &str) -> Vec<(i64, i64)> {
    let mut files: Vec<(i64, i64)> = hold_table(db, name)
        .live_entries()
        .unwrap()
        .iter()
        .map(|entry| (read_time_value(entry, field), count_rows(entry)))
        .collect();
    files.sort();
    files
}

/// Per live file of the table `name` of `db`, partitioned by `day(ts)` and
/// `region`, the day and the region its tuple holds and its row count,
/// sorted.
fn list_day_and_region_files(db: &Database, name: &str) -> Vec<(i64, String, i64)> {
    let mut files: Vec<(i64, String, i64)> = hold_table(db, name)
        .live_entries()
        .unwrap()
        .iter()
        .map(|entry| {
            (
                read_time_value(entry, "ts_day"),
                read_string_value(entry, "region"),
                count_rows(entry),
            )
        })
        .collect();
    files.sort();
    files
}

/// How many of `instants` fall in each unit `count_units` counts, by the
/// unit's count, sorted: what a table partitioned by that unit holds a file
/// of when one INSERT wrote them all.
fn count_rows_per_unit(
    instants: &[Instant],
    count_units: impl Fn(&Instant) -> i64,
) -> Vec<(i64, i64)> {
    let mut rows_per_unit: BTreeMap<i64, i64> = BTreeMap::new();
    for instant in instants {
        *rows_per_unit.entry(count_units(instant)).or_default() += 1;
    }
    rows_per_unit.into_iter().collect()
}

/// What a table partitioned by one time transform of its `ts` column shows
/// once rows are inserted into it.
struct TimePartitionedTable {
    fields: Vec<PartitionField>,
    /// Per live file, its partition value and its row count, sorted.
    files: Vec<(i64, i64)>,
    /// The conditions on `ts` that a query answered with other rows than
    /// the ones they hold for.
    misread_conditions: Vec<String>,
}

/// Create a table partitioned by `unit(ts)`, insert a row per instant of
/// `instants`, its id the instant's position, and query it by every
/// condition on the unit starts among them.
fn fill_table_partitioned_by(unit: &str, instants: &[Instant]) -> TimePartitionedTable {
    let mut db = Database::new();
    db.execute(&format!(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP) WITH (partition_by = '{unit}(ts)')"
    ));
    let rows: Vec<String> = (0..)
        .zip(instants)
        .map(|(id, instant): (i64, _)| format!("({id}, {})", write_timestamp(instant.micros)))
        .collect();
    db.execute(&format!("INSERT INTO events VALUES {}", rows.join(", ")));
    let unit_starts: Vec<i64> = instants
        .iter()
        .skip(1)
        .step_by(3)
        .map(|instant| instant.micros)
        .collect();
    let mut misread_conditions = Vec::new();
    for (condition, holds_for) in list_time_conditions(&unit_starts) {
        let returned = db.execute(&format!(
            "SELECT id FROM events WHERE {condition} ORDER BY id"
        ));
        let expected: Vec<i64> = (0..)
            .zip(instants)
            .filter(|(_, instant)| holds_for(instant.micros))
            .map(|(id, _)| id)
            .collect();
        if collect_i64s(&returned, 0) != expected {
            misread_conditions.push(condition);
        }
    }
    TimePartitionedTable {
        fields: read_partition_fields(&db, "events"),
        files: list_time_partition_files(&db, "events", &format!("ts_{unit}")),
        misread_conditions,
    }
}

/// `column <compare_type> constant`, over the column at `column_idx`, of
/// `column_type`.
fn compare_column(
    column_idx: usize,
    column_type: Type,
    compare_type: CompareType,
    constant: ArrayRef,
) -> Expression {
    Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx,
            return_type: column_type,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(constant))),
        compare_type,
        return_type: Type::Boolean,
    })
}

/// `ts <compare_type> micros`, over the `ts` column the tables here declare
/// second.
fn compare_ts(compare_type: CompareType, micros: i64) -> Expression {
    compare_column(
        1,
        Type::Timestamp,
        compare_type,
        Arc::new(TimestampMicrosecondArray::from(vec![micros])),
    )
}

#[test]
fn a_table_partitioned_by_year_holds_a_file_per_year_and_reads_its_rows_by_their_instants() {
    let instants = list_instants();

    let table = fill_table_partitioned_by("year", &instants);

    assert_eq!(table.fields, [field("year", "ts", "ts_year")]);
    assert_eq!(
        table.files,
        count_rows_per_unit(&instants, |instant| instant.year)
    );
    assert_eq!(table.misread_conditions, Vec::<String>::new());
}

#[test]
fn a_table_partitioned_by_month_holds_a_file_per_month_and_reads_its_rows_by_their_instants() {
    let instants = list_instants();

    let table = fill_table_partitioned_by("month", &instants);

    assert_eq!(table.fields, [field("month", "ts", "ts_month")]);
    assert_eq!(
        table.files,
        count_rows_per_unit(&instants, |instant| instant.month)
    );
    assert_eq!(table.misread_conditions, Vec::<String>::new());
}

#[test]
fn a_table_partitioned_by_day_holds_a_file_per_day_and_reads_its_rows_by_their_instants() {
    let instants = list_instants();

    let table = fill_table_partitioned_by("day", &instants);

    assert_eq!(table.fields, [field("day", "ts", "ts_day")]);
    assert_eq!(
        table.files,
        count_rows_per_unit(&instants, Instant::count_days)
    );
    assert_eq!(table.misread_conditions, Vec::<String>::new());
}

#[test]
fn a_table_partitioned_by_hour_holds_a_file_per_hour_and_reads_its_rows_by_their_instants() {
    let instants = list_instants();

    let table = fill_table_partitioned_by("hour", &instants);

    assert_eq!(table.fields, [field("hour", "ts", "ts_hour")]);
    assert_eq!(
        table.files,
        count_rows_per_unit(&instants, Instant::count_hours)
    );
    assert_eq!(table.misread_conditions, Vec::<String>::new());
}

#[test]
fn a_delete_from_a_table_partitioned_by_year_takes_out_the_rows_of_its_condition_or_none() {
    let misdeleted = delete_by_every_time_condition("year");

    assert_eq!(misdeleted.left_wrong_rows, Vec::<String>::new());
    assert_eq!(misdeleted.miscounted, Vec::<String>::new());
    assert_eq!(misdeleted.misjudged, Vec::<String>::new());
}

#[test]
fn a_delete_from_a_table_partitioned_by_month_takes_out_the_rows_of_its_condition_or_none() {
    let misdeleted = delete_by_every_time_condition("month");

    assert_eq!(misdeleted.left_wrong_rows, Vec::<String>::new());
    assert_eq!(misdeleted.miscounted, Vec::<String>::new());
    assert_eq!(misdeleted.misjudged, Vec::<String>::new());
}

#[test]
fn a_delete_from_a_table_partitioned_by_day_takes_out_the_rows_of_its_condition_or_none() {
    let misdeleted = delete_by_every_time_condition("day");

    assert_eq!(misdeleted.left_wrong_rows, Vec::<String>::new());
    assert_eq!(misdeleted.miscounted, Vec::<String>::new());
    assert_eq!(misdeleted.misjudged, Vec::<String>::new());
}

#[test]
fn a_delete_from_a_table_partitioned_by_hour_takes_out_the_rows_of_its_condition_or_none() {
    let misdeleted = delete_by_every_time_condition("hour");

    assert_eq!(misdeleted.left_wrong_rows, Vec::<String>::new());
    assert_eq!(misdeleted.miscounted, Vec::<String>::new());
    assert_eq!(misdeleted.misjudged, Vec::<String>::new());
}

#[test]
fn a_date_and_a_zoned_timestamp_take_the_time_transforms() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (d DATE, zoned TIMESTAMPTZ) \
         WITH (partition_by = 'month(d), hour(zoned)')",
    );

    db.execute(
        "INSERT INTO events VALUES \
         (DATE '1969-12-31', TIMESTAMPTZ '1969-12-31 23:59:59.999999+00'), \
         (DATE '1970-01-01', TIMESTAMPTZ '1970-01-01 00:00:00+00')",
    );

    let mut tuples: Vec<(i64, i64)> = hold_table(&db, "events")
        .live_entries()
        .unwrap()
        .iter()
        .map(|entry| {
            (
                read_time_value(entry, "d_month"),
                read_time_value(entry, "zoned_hour"),
            )
        })
        .collect();
    tuples.sort();
    assert_eq!(tuples, [(-1, -1), (0, 0)]);
    let before_the_epoch = db
        .execute("SELECT count(*) FROM events WHERE zoned < TIMESTAMPTZ '1970-01-01 00:00:00+00'");
    assert_eq!(single_i64(&before_the_epoch), 1);
}

#[test]
fn a_transform_and_a_column_cut_files_by_both_in_the_order_given() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP, region VARCHAR) \
         WITH (partition_by = 'day(ts), region')",
    );

    db.execute(
        "INSERT INTO events VALUES \
         (1, TIMESTAMP '2024-02-29 10:00:00', 'eu'), \
         (2, TIMESTAMP '2024-02-29 11:00:00', 'us'), \
         (3, TIMESTAMP '2024-03-01 00:00:00', 'eu'), \
         (4, TIMESTAMP '2024-02-29 23:59:59.999999', 'eu')",
    );

    assert_eq!(
        read_partition_fields(&db, "events"),
        [
            field("day", "ts", "ts_day"),
            field("identity", "region", "region")
        ]
    );
    assert_eq!(
        list_day_and_region_files(&db, "events"),
        [
            (LEAP_DAY, "eu".to_string(), 2),
            (LEAP_DAY, "us".to_string(), 1),
            (LEAP_DAY + 1, "eu".to_string(), 1),
        ]
    );
    let leap_day_in_europe = db.execute(
        "SELECT id FROM events \
         WHERE ts < TIMESTAMP '2024-03-01 00:00:00' AND region = 'eu' ORDER BY id",
    );
    assert_eq!(collect_i64s(&leap_day_in_europe, 0), [1, 4]);
}

#[test]
fn bucket_and_truncate_terms_are_recorded_under_their_conventional_field_names() {
    let mut db = Database::new();

    db.execute(
        "CREATE TABLE fruit (id BIGINT, name VARCHAR) \
         WITH (partition_by = 'bucket(4, id), truncate(2, name)')",
    );
    db.execute("INSERT INTO fruit VALUES (1, 'apple'), (2, 'apricot'), (3, 'banana')");

    assert_eq!(
        read_partition_fields(&db, "fruit"),
        [
            field("bucket[4]", "id", "id_bucket"),
            field("truncate[2]", "name", "name_trunc")
        ]
    );
    let mut prefixes: Vec<String> = hold_table(&db, "fruit")
        .live_entries()
        .unwrap()
        .iter()
        .map(|entry| read_string_value(entry, "name_trunc"))
        .collect();
    prefixes.sort();
    prefixes.dedup();
    assert_eq!(prefixes, ["ap", "ba"]);
    let bananas = db.execute("SELECT id FROM fruit WHERE name = 'banana'");
    assert_eq!(collect_i64s(&bananas, 0), [3]);
}

#[test]
fn a_second_insert_adds_files_to_the_days_the_first_made_and_to_a_new_one() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (id BIGINT, ts TIMESTAMP) WITH (partition_by = 'day(ts)')");
    db.execute(
        "INSERT INTO events VALUES \
         (1, TIMESTAMP '2024-02-28 10:00:00'), (2, TIMESTAMP '2024-02-29 10:00:00')",
    );

    db.execute(
        "INSERT INTO events VALUES \
         (3, TIMESTAMP '2024-02-29 11:00:00'), (4, TIMESTAMP '2024-03-01 11:00:00')",
    );

    assert_eq!(
        list_time_partition_files(&db, "events", "ts_day"),
        [
            (LEAP_DAY - 1, 1),
            (LEAP_DAY, 1),
            (LEAP_DAY, 1),
            (LEAP_DAY + 1, 1)
        ]
    );
    let from_the_leap_day =
        db.execute("SELECT id FROM events WHERE ts >= TIMESTAMP '2024-02-29 00:00:00' ORDER BY id");
    assert_eq!(collect_i64s(&from_the_leap_day, 0), [2, 3, 4]);
}

#[test]
fn an_insert_from_a_select_cuts_its_files_by_the_targets_transform() {
    let mut db = Database::new();
    db.execute("CREATE TABLE staged (id BIGINT, ts TIMESTAMP)");
    db.execute(
        "INSERT INTO staged VALUES \
         (1, TIMESTAMP '2024-02-29 23:59:59.999999'), \
         (2, TIMESTAMP '2024-03-01 00:00:00'), \
         (3, TIMESTAMP '2024-03-01 00:00:00.000001')",
    );
    db.execute("CREATE TABLE events (id BIGINT, ts TIMESTAMP) WITH (partition_by = 'day(ts)')");

    db.execute("INSERT INTO events SELECT id, ts FROM staged");

    assert_eq!(
        list_time_partition_files(&db, "events", "ts_day"),
        [(LEAP_DAY, 1), (LEAP_DAY + 1, 2)]
    );
}

#[test]
fn a_reopened_database_shows_the_terms_and_keeps_cutting_files_by_them() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP, region VARCHAR) \
         WITH (partition_by = 'day(ts), region')",
    );
    db.execute("INSERT INTO events VALUES (1, TIMESTAMP '2024-02-29 10:00:00', 'eu')");

    let mut reopened = Database::reopen(db.close());
    reopened.execute(
        "INSERT INTO events VALUES \
         (2, TIMESTAMP '2024-02-29 11:00:00', 'eu'), (3, TIMESTAMP '2024-03-01 11:00:00', 'eu')",
    );

    assert_eq!(
        hold_table(&reopened, "events").partition_by(),
        ["day(ts)", "region"]
    );
    assert_eq!(
        list_day_and_region_files(&reopened, "events"),
        [
            (LEAP_DAY, "eu".to_string(), 1),
            (LEAP_DAY, "eu".to_string(), 1),
            (LEAP_DAY + 1, "eu".to_string(), 1),
        ]
    );
}

#[test]
fn a_filter_on_the_source_column_reads_only_the_days_it_keeps() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (id BIGINT, ts TIMESTAMP) WITH (partition_by = 'day(ts)')");
    db.execute(
        "INSERT INTO events VALUES \
         (1, TIMESTAMP '2024-02-28 12:00:00'), \
         (2, TIMESTAMP '2024-02-29 00:00:00'), \
         (3, TIMESTAMP '2024-02-29 23:59:59'), \
         (4, TIMESTAMP '2024-03-01 00:00:00')",
    );
    let table = hold_table(&db, "events");
    let leap_day_start = LEAP_DAY * DAY_MICROS;

    let whole = table.select_scan(&[]).unwrap();
    let leap_day = table
        .select_scan(&[
            compare_ts(CompareType::GreaterEqual, leap_day_start),
            compare_ts(CompareType::Less, leap_day_start + DAY_MICROS),
        ])
        .unwrap();

    assert_eq!(whole.row_groups().len(), 3);
    assert_eq!(leap_day.row_groups().len(), 1);
    assert_eq!(leap_day.row_groups()[0].num_rows, 2);
}

/// Each bucket's file holds ids from across the whole range, so the files'
/// bounds on `id` keep both: only the bucket a file records tells which one
/// may hold the constant.
#[test]
fn a_filter_on_a_bucketed_column_reads_only_the_bucket_of_its_constant() {
    let mut db = Database::new();
    db.execute("CREATE TABLE readings (id BIGINT) WITH (partition_by = 'bucket(2, id)')");
    let rows: Vec<String> = (1..=100).map(|id| format!("({id})")).collect();
    db.execute(&format!("INSERT INTO readings VALUES {}", rows.join(", ")));
    let table = hold_table(&db, "readings");
    let id_is_fifty = compare_column(
        0,
        Type::Int64,
        CompareType::Equal,
        Arc::new(Int64Array::from(vec![50])),
    );

    let whole = table.select_scan(&[]).unwrap();
    let one_bucket = table.select_scan(&[id_is_fifty]).unwrap();

    assert_eq!(whole.row_groups().len(), 2);
    assert_eq!(one_bucket.row_groups().len(), 1);
}

/// Per live file of the table `name` of `db`, partitioned by `day(ts)` and
/// sorted by `ts`: the day its tuple holds, its row count, and the days of
/// its least and its greatest `ts`, sorted.
fn list_day_files_with_their_span(db: &Database, name: &str) -> Vec<(i64, i64, i64, i64)> {
    let mut files: Vec<(i64, i64, i64, i64)> = hold_table(db, name)
        .live_entries()
        .unwrap()
        .iter()
        .map(|entry| {
            let stats = entry.stats.as_ref().unwrap();
            let count_days = |bounds: &std::collections::HashMap<String, ArrayRef>| {
                bounds["ts"]
                    .as_primitive::<TimestampMicrosecondType>()
                    .value(0)
                    .div_euclid(DAY_MICROS)
            };
            (
                read_time_value(entry, "ts_day"),
                count_rows(entry),
                count_days(&stats.min_values),
                count_days(&stats.max_values),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn compaction_merges_the_files_of_a_day_into_a_file_of_that_day() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP) \
         WITH (partition_by = 'day(ts)', sort_by = 'ts')",
    );
    for id in 0..3 {
        db.execute(&format!(
            "INSERT INTO events VALUES \
             ({id}, TIMESTAMP '1969-12-31 23:59:5{id}'), \
             ({}, TIMESTAMP '1970-01-01 00:00:0{id}')",
            id + 10
        ));
    }

    db.compact("events", true);

    assert_eq!(
        list_day_files_with_their_span(&db, "events"),
        [(-1, 3, -1, -1), (0, 3, 0, 0)]
    );
    let rows = db.execute("SELECT id FROM events ORDER BY id");
    assert_eq!(collect_i64s(&rows, 0), [0, 1, 2, 10, 11, 12]);
    let before_the_epoch =
        db.execute("SELECT id FROM events WHERE ts < TIMESTAMP '1970-01-01 00:00:00' ORDER BY id");
    assert_eq!(collect_i64s(&before_the_epoch, 0), [0, 1, 2]);
}

#[test]
fn compaction_keeps_the_regions_of_a_day_in_files_of_their_own() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP, region VARCHAR) \
         WITH (partition_by = 'day(ts), region')",
    );
    for id in 0..2 {
        db.execute(&format!(
            "INSERT INTO events VALUES \
             ({id}, TIMESTAMP '2024-02-29 10:00:0{id}', 'eu'), \
             ({}, TIMESTAMP '2024-02-29 10:00:0{id}', 'us'), \
             ({}, TIMESTAMP '2024-03-01 10:00:0{id}', 'eu')",
            id + 10,
            id + 20
        ));
    }

    db.compact("events", true);

    assert_eq!(
        list_day_and_region_files(&db, "events"),
        [
            (LEAP_DAY, "eu".to_string(), 2),
            (LEAP_DAY, "us".to_string(), 2),
            (LEAP_DAY + 1, "eu".to_string(), 2),
        ]
    );
    let rows = db.execute("SELECT id FROM events ORDER BY id");
    assert_eq!(collect_i64s(&rows, 0), [0, 1, 10, 11, 20, 21]);
    let american = db.execute("SELECT id FROM events WHERE region = 'us' ORDER BY id");
    assert_eq!(collect_i64s(&american, 0), [10, 11]);
    let march =
        db.execute("SELECT id FROM events WHERE ts >= TIMESTAMP '2024-03-01 00:00:00' ORDER BY id");
    assert_eq!(collect_i64s(&march, 0), [20, 21]);
}

#[test]
fn a_tables_description_spells_its_partition_terms_as_they_were_written() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP, region VARCHAR, note VARCHAR) \
         WITH (partition_by = 'day(ts), region')",
    );
    db.execute("INSERT INTO events VALUES (1, TIMESTAMP '2024-02-29 10:00:00', 'eu', 'x')");

    let transaction = Arc::clone(&db.datastore).begin_transaction();
    let described = transaction.tables().unwrap().remove(0);

    assert_eq!(described.partition_by, ["day(ts)", "region"]);
    let partition_keys: Vec<(&str, bool)> = described
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column.is_partition_key))
        .collect();
    assert_eq!(
        partition_keys,
        [
            ("id", false),
            ("ts", true),
            ("region", true),
            ("note", false)
        ]
    );
    assert_eq!(described.files[0].partition, "ts_day=2024-02-29,region=eu");
}

#[test]
fn partitioned_by_records_the_spec_the_partition_by_option_records() {
    let mut db = Database::new();

    db.execute(
        "CREATE TABLE by_clause (id BIGINT, ts TIMESTAMP, region VARCHAR) \
         PARTITIONED BY (day(ts), bucket(4, id), region)",
    );
    db.execute(
        "CREATE TABLE by_option (id BIGINT, ts TIMESTAMP, region VARCHAR) \
         WITH (partition_by = 'day(ts), bucket(4, id), region')",
    );

    let by_clause = read_partition_fields(&db, "by_clause");
    assert_eq!(
        by_clause,
        [
            field("day", "ts", "ts_day"),
            field("bucket[4]", "id", "id_bucket"),
            field("identity", "region", "region")
        ]
    );
    assert_eq!(by_clause, read_partition_fields(&db, "by_option"));
}

#[test]
fn partitioned_by_and_sorted_by_are_taken_together_in_either_order() {
    let mut db = Database::new();

    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP) SORTED BY (ts, id) PARTITIONED BY (day(ts))",
    );

    let table = hold_table(&db, "events");
    assert_eq!(table.partition_by(), ["day(ts)"]);
    assert_eq!(table.sort_by(), ["ts", "id"]);
}

/// A table shows its terms comma-separated, and a term may hold a comma of
/// its own: the text reads back as the terms only when split outside
/// parentheses.
#[test]
fn the_terms_a_table_shows_are_singular_and_split_back_from_their_text() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE events (id BIGINT, ts TIMESTAMP, region VARCHAR) \
         WITH (partition_by = 'days(ts), BUCKET(16,id), region')",
    );

    let transaction = Arc::clone(&db.datastore).begin_transaction();
    let shown = transaction.tables().unwrap().remove(0).partition_by;

    assert_eq!(shown, ["day(ts)", "bucket(16, id)", "region"]);
    assert_eq!(split_partition_terms(&shown.join(",")), shown);
}

#[test]
fn partitioned_by_beside_the_partition_by_option_is_refused() {
    let mut db = Database::new();

    let error = db
        .try_execute("CREATE TABLE t (id BIGINT) PARTITIONED BY (id) WITH (partition_by = 'id')")
        .unwrap_err();

    assert!(error.contains("PARTITIONED BY"), "{error}");
    assert!(error.contains("`partition_by`"), "{error}");
}

/// What a table of an id, a name, a date and a timestamp is refused with
/// when it is to be partitioned by `partition_by`.
fn refuse_partition_by(partition_by: &str) -> String {
    let mut db = Database::new();
    db.try_execute(&format!(
        "CREATE TABLE t (id BIGINT, name VARCHAR, d DATE, ts TIMESTAMP) \
         WITH (partition_by = '{partition_by}')"
    ))
    .expect_err("the table is refused")
}

#[test]
fn a_time_transform_of_a_string_is_refused_naming_the_term() {
    let error = refuse_partition_by("id, day(name)");

    assert!(error.contains("`day(name)`"), "{error}");
}

#[test]
fn an_hour_of_a_date_is_refused_naming_the_term() {
    let error = refuse_partition_by("hour(d)");

    assert!(error.contains("`hour(d)`"), "{error}");
}

#[test]
fn an_unknown_transform_is_refused_naming_the_term() {
    let error = refuse_partition_by("fortnight(ts)");

    assert!(error.contains("`fortnight(ts)`"), "{error}");
}

#[test]
fn a_transform_of_an_undeclared_column_is_refused_naming_the_term_and_the_column() {
    let error = refuse_partition_by("day(missing)");

    assert!(error.contains("`day(missing)`"), "{error}");
    assert!(error.contains("`missing`"), "{error}");
}

#[test]
fn a_bucket_count_of_zero_is_refused_naming_the_term() {
    let error = refuse_partition_by("bucket(0, id)");

    assert!(error.contains("`bucket(0, id)`"), "{error}");
}

#[test]
fn a_second_time_transform_of_one_column_is_refused_naming_the_term() {
    let error = refuse_partition_by("day(ts), hour(ts)");

    assert!(error.contains("`hour(ts)`"), "{error}");
}

#[test]
fn a_transform_whose_field_would_be_named_like_a_column_is_refused() {
    let mut db = Database::new();

    let error = db
        .try_execute("CREATE TABLE t (ts TIMESTAMP, ts_day DATE) WITH (partition_by = 'day(ts)')")
        .unwrap_err();

    assert!(error.contains("`day(ts)`"), "{error}");
    assert!(error.contains("`ts_day`"), "{error}");
}

#[test]
fn a_table_that_adopts_files_refuses_a_transform() {
    let mut db = Database::new();

    let error = db
        .try_execute(
            "CREATE TABLE t (id BIGINT, ts TIMESTAMP) \
             WITH (with_pre_existing_parquets = 'adopted', partition_by = 'day(ts)')",
        )
        .unwrap_err();

    assert!(error.contains("`day(ts)`"), "{error}");
    assert!(error.contains("with_pre_existing_parquets"), "{error}");
}

#[test]
fn a_delta_database_refuses_a_transform_and_still_partitions_by_a_column() {
    let mut db = Database::in_format(TableFormat::Delta);

    let refused = db
        .try_execute(
            "CREATE TABLE by_day (id BIGINT, ts TIMESTAMP) WITH (partition_by = 'day(ts)')",
        )
        .unwrap_err();
    db.execute("CREATE TABLE by_region (id BIGINT, region VARCHAR) WITH (partition_by = 'region')");
    db.execute("INSERT INTO by_region VALUES (1, 'eu'), (2, 'us')");

    assert!(refused.contains("`day(ts)`"), "{refused}");
    assert!(refused.contains("Delta Lake"), "{refused}");
    assert_eq!(hold_table(&db, "by_region").partition_by(), ["region"]);
    let european = db.execute("SELECT id FROM by_region WHERE region = 'eu'");
    assert_eq!(collect_i64s(&european, 0), [1]);
}
