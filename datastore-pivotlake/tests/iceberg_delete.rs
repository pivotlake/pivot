//! `DELETE` on a managed Iceberg table, through SQL: conditions that end on
//! the table's partition bounds take the partitions they cover out of the
//! table whole, with no data file written, and conditions that cut a
//! partition are refused with nothing deleted.

mod iceberg_support;

use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampMicrosecondArray};
use datastore_pivotlake::TableFormat;
use datastore_pivotlake::test_support::live_data_file_names;
use iceberg_support::{
    Database, collect_i64s, collect_strings, insert_rows, names_in, parquet_files_in, single_i64,
};
use iceberg_tables::iceberg::spec::{Operation, TableMetadata};

/// 2026-11-13 12:00:00, in microseconds after the epoch.
const NOON_OF_THE_13TH_MICROS: i64 = 1_794_571_200_000_000;

/// A table of events partitioned by the day of `ts`, holding two rows on
/// each of the 13th, 14th and 15th of November 2026 and one whose `ts` is
/// NULL.
fn create_events_by_day(db: &mut Database) {
    db.execute("CREATE TABLE events (ts TIMESTAMP, id BIGINT) PARTITIONED BY (day(ts))");
    db.execute(
        "INSERT INTO events VALUES
            (TIMESTAMP '2026-11-13 00:00:00', 1), (TIMESTAMP '2026-11-13 23:59:59', 2),
            (TIMESTAMP '2026-11-14 00:00:00', 3), (TIMESTAMP '2026-11-14 12:00:00', 4),
            (TIMESTAMP '2026-11-15 00:00:00', 5), (TIMESTAMP '2026-11-15 12:00:00', 6),
            (NULL, 7)",
    );
}

/// The `id` of every row of `events`, in order.
fn select_event_ids(db: &mut Database) -> Vec<i64> {
    collect_i64s(&db.execute("SELECT id FROM events ORDER BY id"), 0)
}

/// The metadata of the table at `table_path` at its latest version.
fn read_current_metadata(table_path: &Path) -> TableMetadata {
    let metadata_dir = table_path.join("metadata");
    let latest = names_in(&metadata_dir)
        .into_iter()
        .filter(|name| name.ends_with(".metadata.json"))
        .max()
        .unwrap();
    serde_json::from_slice(&std::fs::read(metadata_dir.join(latest)).unwrap()).unwrap()
}

/// How many data files the table `name` holds.
fn count_data_files(db: &Database, name: &str) -> usize {
    live_data_file_names(&db.root_uri(), &db.table_directory(name))
        .unwrap()
        .len()
}

#[test]
fn a_delete_ending_on_a_day_bound_drops_the_days_before_it() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted = db.execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-15 00:00:00'");

    assert_eq!(single_i64(&deleted), 4);
    assert_eq!(select_event_ids(&mut db), vec![5, 6, 7]);
    assert_eq!(count_data_files(&db, "events"), 2);
}

#[test]
fn a_delete_inside_a_day_is_refused_and_deletes_nothing() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let refusal = db
        .try_execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-15 10:00:00'")
        .unwrap_err();

    assert!(
        refusal.contains("cut partition `ts_day=2026-11-15`"),
        "{refusal}"
    );
    assert!(refusal.contains("day(ts)"), "{refusal}");
    assert_eq!(select_event_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_inside_a_day_that_holds_no_row_drops_the_days_it_covers() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted = db.execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-16 10:00:00'");

    assert_eq!(single_i64(&deleted), 6);
    assert_eq!(select_event_ids(&mut db), vec![7]);
}

#[test]
fn a_delete_between_two_day_bounds_drops_the_days_between_them() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted = db.execute(
        "DELETE FROM events
         WHERE ts >= TIMESTAMP '2026-11-14 00:00:00' AND ts < TIMESTAMP '2026-11-15 00:00:00'",
    );

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_event_ids(&mut db), vec![1, 2, 5, 6, 7]);
}

#[test]
fn a_delete_of_a_day_named_by_truncating_the_time_drops_the_day() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted = db.execute(
        "DELETE FROM events WHERE date_trunc('day', ts) = TIMESTAMP '2026-11-14 00:00:00'",
    );

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_event_ids(&mut db), vec![1, 2, 5, 6, 7]);
}

#[test]
fn a_delete_up_to_the_last_instant_of_a_day_drops_the_day() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted =
        db.execute("DELETE FROM events WHERE ts <= TIMESTAMP '2026-11-13 23:59:59.999999'");

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_event_ids(&mut db), vec![3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_without_conditions_empties_the_table() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted = db.execute("DELETE FROM events");

    assert_eq!(single_i64(&deleted), 7);
    assert_eq!(select_event_ids(&mut db), Vec::<i64>::new());
    assert_eq!(count_data_files(&db, "events"), 0);
}

#[test]
fn a_delete_of_days_that_hold_no_row_deletes_nothing() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let deleted = db.execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-01 00:00:00'");

    assert_eq!(single_i64(&deleted), 0);
    assert_eq!(select_event_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_by_a_partition_column_drops_the_values_it_names() {
    let mut db = Database::new();
    db.execute("CREATE TABLE visits (region VARCHAR, id BIGINT) PARTITIONED BY (region)");
    db.execute("INSERT INTO visits VALUES ('eu', 1), ('us', 2), ('us', 3), ('ap', 4), (NULL, 5)");

    let deleted = db.execute("DELETE FROM visits WHERE region IN ('eu', 'us')");

    assert_eq!(single_i64(&deleted), 3);
    let regions = db.execute("SELECT id FROM visits ORDER BY id");
    assert_eq!(collect_i64s(&regions, 0), vec![4, 5]);
}

/// A table of visits partitioned by `region`, holding one row for each of
/// six regions and one whose `region` is NULL.
fn create_visits_by_region(db: &mut Database) {
    db.execute("CREATE TABLE visits (region VARCHAR, id BIGINT) PARTITIONED BY (region)");
    db.execute(
        "INSERT INTO visits VALUES
            ('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), (NULL, 7)",
    );
}

#[test]
fn a_delete_by_a_long_list_of_values_drops_each_of_them() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let deleted = db.execute("DELETE FROM visits WHERE region IN ('a', 'b', 'c', 'd', 'e')");

    assert_eq!(single_i64(&deleted), 5);
    assert_eq!(select_visit_ids(&mut db), vec![6, 7]);
}

#[test]
fn a_delete_of_the_values_outside_a_list_keeps_the_list_and_the_rows_without_a_value() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let deleted = db.execute("DELETE FROM visits WHERE region NOT IN ('a', 'b', 'c', 'd', 'e')");

    assert_eq!(single_i64(&deleted), 1);
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4, 5, 7]);
}

#[test]
fn a_delete_of_the_rows_without_a_partition_value_drops_only_them() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let deleted = db.execute("DELETE FROM visits WHERE region IS NULL");

    assert_eq!(single_i64(&deleted), 1);
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn a_delete_of_the_rows_with_a_partition_value_keeps_the_rows_without_one() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let deleted = db.execute("DELETE FROM visits WHERE region IS NOT NULL");

    assert_eq!(single_i64(&deleted), 6);
    assert_eq!(select_visit_ids(&mut db), vec![7]);
}

#[test]
fn a_delete_from_a_table_without_a_row_deletes_nothing() {
    let mut db = Database::new();
    db.execute("CREATE TABLE visits (region VARCHAR, id BIGINT) PARTITIONED BY (region)");

    let deleted = db.execute("DELETE FROM visits WHERE id = 1");

    assert_eq!(single_i64(&deleted), 0);
    assert_eq!(select_visit_ids(&mut db), Vec::<i64>::new());
}

#[test]
fn a_delete_whose_conditions_no_row_can_meet_deletes_nothing() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let deleted = db.execute("DELETE FROM visits WHERE 1 = 0");

    assert_eq!(single_i64(&deleted), 0);
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_by_the_rows_of_another_query_is_refused() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let refusal = db
        .try_execute("DELETE FROM visits WHERE region IN (SELECT region FROM visits)")
        .unwrap_err();

    assert!(
        refusal.contains("USING and subqueries are not supported"),
        "{refusal}"
    );
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_returning_its_rows_is_refused() {
    let mut db = Database::new();
    create_visits_by_region(&mut db);

    let refusal = db
        .try_execute("DELETE FROM visits WHERE region = 'a' RETURNING id")
        .unwrap_err();

    assert!(refusal.contains("RETURNING is not supported"), "{refusal}");
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_by_a_column_the_table_is_not_partitioned_by_is_refused() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let refusal = db
        .try_execute("DELETE FROM events WHERE id = 3")
        .unwrap_err();

    assert!(refusal.contains("cut partition `ts_day="), "{refusal}");
    assert_eq!(select_event_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_with_conditions_on_an_unpartitioned_table_is_refused() {
    let mut db = Database::new();
    db.execute("CREATE TABLE plain (id BIGINT)");
    db.execute("INSERT INTO plain VALUES (1), (2)");

    let refusal = db
        .try_execute("DELETE FROM plain WHERE id = 1")
        .unwrap_err();

    assert!(
        refusal.contains("files that are in no partition"),
        "{refusal}"
    );
    assert_eq!(single_i64(&db.execute("SELECT count(*) FROM plain")), 2);
}

#[test]
fn a_delete_by_a_condition_no_partition_can_be_checked_against_is_refused() {
    let mut db = Database::new();
    create_events_by_day(&mut db);

    let refusal = db
        .try_execute("DELETE FROM events WHERE id * id = 4")
        .unwrap_err();

    assert!(
        refusal.contains("cannot be checked against a partition"),
        "{refusal}"
    );
    assert_eq!(select_event_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_commits_a_delete_snapshot_and_writes_no_data_file() {
    let mut db = Database::new();
    create_events_by_day(&mut db);
    let files_before = parquet_files_in(&db.table_path("events"));

    db.execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-14 00:00:00'");

    let metadata = read_current_metadata(&db.table_path("events"));
    let summary = metadata.current_snapshot().unwrap().summary();
    assert_eq!(summary.operation, Operation::Delete);
    assert_eq!(summary.additional_properties["deleted-data-files"], "1");
    assert_eq!(summary.additional_properties["deleted-records"], "2");
    assert_eq!(parquet_files_in(&db.table_path("events")), files_before);
}

/// A table of visits partitioned by the day of `ts` and by `region`,
/// holding one row for each of two regions on each of two days.
fn create_visits_by_day_and_region(db: &mut Database) {
    db.execute(
        "CREATE TABLE visits (ts TIMESTAMP, region VARCHAR, id BIGINT)
         PARTITIONED BY (day(ts), region)",
    );
    db.execute(
        "INSERT INTO visits VALUES
            (TIMESTAMP '2026-11-13 08:00:00', 'eu', 1), (TIMESTAMP '2026-11-13 09:00:00', 'us', 2),
            (TIMESTAMP '2026-11-14 08:00:00', 'eu', 3), (TIMESTAMP '2026-11-14 09:00:00', 'us', 4)",
    );
}

/// The `id` of every row of `visits`, in order.
fn select_visit_ids(db: &mut Database) -> Vec<i64> {
    collect_i64s(&db.execute("SELECT id FROM visits ORDER BY id"), 0)
}

#[test]
fn a_delete_by_one_of_two_partition_fields_drops_its_value_across_the_other() {
    let mut db = Database::new();
    create_visits_by_day_and_region(&mut db);

    let deleted = db.execute("DELETE FROM visits WHERE region = 'eu'");

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_visit_ids(&mut db), vec![2, 4]);
}

#[test]
fn a_delete_by_both_partition_fields_drops_the_partitions_both_conditions_cover() {
    let mut db = Database::new();
    create_visits_by_day_and_region(&mut db);

    let deleted = db
        .execute("DELETE FROM visits WHERE ts < TIMESTAMP '2026-11-14 00:00:00' AND region = 'us'");

    assert_eq!(single_i64(&deleted), 1);
    assert_eq!(select_visit_ids(&mut db), vec![1, 3, 4]);
}

#[test]
fn a_delete_by_either_of_two_conditions_drops_the_partitions_one_of_them_covers() {
    let mut db = Database::new();
    create_visits_by_day_and_region(&mut db);

    let deleted = db
        .execute("DELETE FROM visits WHERE ts < TIMESTAMP '2026-11-14 00:00:00' OR region = 'us'");

    assert_eq!(single_i64(&deleted), 3);
    assert_eq!(select_visit_ids(&mut db), vec![3]);
}

#[test]
fn a_delete_that_narrows_a_covered_partition_by_another_column_is_refused() {
    let mut db = Database::new();
    create_visits_by_day_and_region(&mut db);

    let refusal = db
        .try_execute("DELETE FROM visits WHERE region = 'eu' AND id = 1")
        .unwrap_err();

    assert!(
        refusal.contains("cut partition `ts_day=2026-11-13,region=eu`"),
        "{refusal}"
    );
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4]);
}

#[test]
fn a_delete_of_every_value_but_one_keeps_that_value_and_the_rows_without_one() {
    let mut db = Database::new();
    db.execute("CREATE TABLE visits (region VARCHAR, id BIGINT) PARTITIONED BY (region)");
    db.execute("INSERT INTO visits VALUES ('eu', 1), ('us', 2), ('ap', 3), (NULL, 4)");

    let deleted = db.execute("DELETE FROM visits WHERE region <> 'eu'");

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_visit_ids(&mut db), vec![1, 4]);
}

#[test]
fn a_delete_by_a_bucketed_column_is_refused() {
    let mut db = Database::new();
    db.execute("CREATE TABLE visits (id BIGINT) PARTITIONED BY (bucket(4, id))");
    db.execute("INSERT INTO visits VALUES (1), (2), (3), (4), (5), (6), (7), (8)");

    let refusal = db
        .try_execute("DELETE FROM visits WHERE id = 5")
        .unwrap_err();

    assert!(refusal.contains("cut partition `id_bucket="), "{refusal}");
    assert_eq!(select_visit_ids(&mut db), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn a_row_inserted_into_a_covered_day_before_the_delete_commits_goes_with_the_day() {
    let mut db = Database::new();
    create_events_by_day(&mut db);
    let late_row = RecordBatch::try_from_iter([
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(vec![
                NOON_OF_THE_13TH_MICROS,
            ])) as ArrayRef,
        ),
        ("id", Arc::new(Int64Array::from(vec![8])) as ArrayRef),
    ])
    .unwrap();

    let deleted = db.execute_before_commit(
        "DELETE FROM events WHERE ts < TIMESTAMP '2026-11-14 00:00:00'",
        |datastore| insert_rows(datastore, "events", vec![late_row]),
    );

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_event_ids(&mut db), vec![3, 4, 5, 6, 7]);
}

#[test]
fn a_delete_after_a_compaction_drops_the_merged_files_of_the_days_it_covers() {
    let mut db = Database::new();
    create_events_by_day(&mut db);
    db.execute("INSERT INTO events VALUES (TIMESTAMP '2026-11-13 10:00:00', 8)");
    db.compact("events", true);

    let deleted = db.execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-14 00:00:00'");

    assert_eq!(single_i64(&deleted), 3);
    assert_eq!(select_event_ids(&mut db), vec![3, 4, 5, 6, 7]);
}

#[test]
fn a_reopened_database_does_not_show_the_deleted_days() {
    let mut db = Database::new();
    create_events_by_day(&mut db);
    db.execute("DELETE FROM events WHERE ts < TIMESTAMP '2026-11-15 00:00:00'");

    let mut reopened = Database::reopen(db.close());

    assert_eq!(select_event_ids(&mut reopened), vec![5, 6, 7]);
}

#[test]
fn a_delete_of_the_days_before_today_drops_them() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (ts TIMESTAMP, id BIGINT) PARTITIONED BY (day(ts))");
    db.execute(
        "INSERT INTO events VALUES
            (TIMESTAMP '2020-01-01 05:00:00', 1), (TIMESTAMP '2020-01-02 05:00:00', 2),
            (TIMESTAMP '2999-01-01 05:00:00', 3)",
    );

    let deleted = db.execute("DELETE FROM events WHERE ts < CURRENT_DATE");

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_event_ids(&mut db), vec![3]);
}

#[test]
fn a_delete_by_a_date_the_table_is_partitioned_by_drops_the_dates_before_its_bound() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (day DATE, id BIGINT) PARTITIONED BY (day)");
    db.execute(
        "INSERT INTO events VALUES
            (DATE '2026-11-13', 1), (DATE '2026-11-14', 2), (DATE '2026-11-15', 3)",
    );

    let deleted = db.execute("DELETE FROM events WHERE day < DATE '2026-11-15'");

    assert_eq!(single_i64(&deleted), 2);
    assert_eq!(select_event_ids(&mut db), vec![3]);
}

#[test]
fn a_delete_by_a_zoned_time_drops_the_days_before_a_bound_that_starts_a_day() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (ts TIMESTAMPTZ, id BIGINT) PARTITIONED BY (day(ts))");
    db.execute(
        "INSERT INTO events VALUES
            (TIMESTAMPTZ '2026-11-13 05:00:00+00', 1), (TIMESTAMPTZ '2026-11-14 05:00:00+00', 2)",
    );

    let refusal = db
        .try_execute("DELETE FROM events WHERE ts < TIMESTAMPTZ '2026-11-14 00:00:00+02'")
        .unwrap_err();
    let deleted = db.execute("DELETE FROM events WHERE ts < TIMESTAMPTZ '2026-11-14 00:00:00+00'");

    assert!(
        refusal.contains("cut partition `ts_day=2026-11-13`"),
        "{refusal}"
    );
    assert_eq!(single_i64(&deleted), 1);
    assert_eq!(select_event_ids(&mut db), vec![2]);
}

#[test]
fn a_delete_by_a_floating_point_partition_column_is_refused() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (score DOUBLE, id BIGINT) PARTITIONED BY (score)");
    db.execute("INSERT INTO events VALUES (1.5, 1), (2.5, 2)");

    let refusal = db
        .try_execute("DELETE FROM events WHERE score = 1.5")
        .unwrap_err();

    assert!(refusal.contains("cut partition `score=1.5`"), "{refusal}");
    assert_eq!(select_event_ids(&mut db), vec![1, 2]);
}

#[test]
fn a_delete_on_a_delta_database_is_refused() {
    let mut db = Database::in_format(TableFormat::Delta);
    db.execute("CREATE TABLE visits (region VARCHAR, id BIGINT) PARTITIONED BY (region)");
    db.execute("INSERT INTO visits VALUES ('eu', 1)");

    let refusal = db
        .try_execute("DELETE FROM visits WHERE region = 'eu'")
        .unwrap_err();

    assert!(refusal.contains("`delta` table format"), "{refusal}");
    let regions = db.execute("SELECT region FROM visits");
    assert_eq!(collect_strings(&regions, 0), vec!["eu".to_string()]);
}
