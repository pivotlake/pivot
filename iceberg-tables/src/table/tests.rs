use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
use arrow_schema::DataType;
use object_storage::{FileRef, LocalStore, ObjectPath, ObjectStore};
use parquet::arrow::ArrowWriter;
use parquet_engine::{FileStats, PartitionValues};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use uuid::Uuid;

use super::*;
use crate::commit::TableCreation;
use crate::files::file_metadata_cache::measure_footer_entry;
use crate::metadata::pivotlake_metadata_store::PivotlakeMetadataStore;
use crate::test_support::test_pool;

/// A table of one `Int64` column `v`, partitioned by it, in a directory of
/// its own.
struct Fixture {
    _directory: tempfile::TempDir,
    location: TableLocation,
}

fn table_name() -> SchemaQualifiedTableName {
    SchemaQualifiedTableName::new("main", "t1")
}

impl Fixture {
    /// The table as a process that just started loads it, its partition
    /// indexes and its files' footers held in `cache`.
    fn load(&self, cache: &Arc<FileMetadataCache>) -> Table {
        let mut table = Table::load(
            &test_pool(),
            Arc::new(PivotlakeMetadataStore::new(self.location.clone())),
            self.location.clone(),
            &table_name(),
        )
        .unwrap();
        table.lineage = Arc::new(Lineage::new(Arc::clone(cache)));
        table
    }

    /// Delete the data manifests of `table` from the store.
    fn delete_data_manifests(&self, table: &Table) {
        for manifest in table.manifest_list().data_manifests() {
            self.delete_manifest(&manifest.manifest_path);
        }
    }

    fn delete_manifest(&self, manifest_path: &str) {
        let location = &self.location;
        let key = location.data_file_key(&location.file_path_for_uri(manifest_path).unwrap());
        location.store().delete(&key).unwrap();
    }

    /// Write `name`, a Parquet file of ten rows `v = value`, into the
    /// table's directory, and hand back its entry in the partition
    /// `v = value`.
    fn write_file(&self, name: &str, value: i64) -> DataFileEntry {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));
        let rows: ArrayRef = Arc::new(Int64Array::from(vec![value; 10]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![rows]).unwrap();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let mut written = entry(name, value, 0);
        written.file.size = bytes.len() as u64;
        let key = self.location.data_file_key(&written.file.path);
        self.location.store().put(&key, &bytes).unwrap();
        written
    }

    /// Put in the place of the data file `name` another file of its size
    /// that holds no footer, so whatever reads the footer of `name` from
    /// now on fails.
    fn replace_with_garbage(&self, name: &str) {
        let path = ObjectPath::new(format!("{name}.parquet"));
        let key = self.location.data_file_key(&path);
        let store = self.location.store();
        let size = store.get(&key).unwrap().unwrap().len();
        store.delete(&key).unwrap();
        store.put(&key, &vec![0; size]).unwrap();
    }
}

fn create_partitioned_table() -> (Fixture, Table) {
    let directory = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(directory.path()).unwrap());
    let location =
        TableLocation::in_datastore_store("main.t", store, ObjectPath::new("t1")).unwrap();
    let fixture = Fixture {
        _directory: directory,
        location,
    };
    let columns = [Column {
        name: "v".to_string(),
        col_type: Type::Int64,
    }];
    let table = Table::create(
        &test_pool(),
        fixture.location.clone(),
        &table_name(),
        TableCreation {
            table_uuid: Uuid::new_v4(),
            columns: &columns,
            partition_by: &["v".to_string()],
            sort_by: &[],
            adopted: Vec::new(),
        },
    )
    .unwrap();
    (fixture, table)
}

fn int_scalar(value: i64) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef)
}

/// An entry for `name`, a file of ten rows in the partition `v = value`, of
/// which `nulls` hold no value.
fn entry(name: &str, value: i64, nulls: i64) -> DataFileEntry {
    let bound = || -> ArrayRef { Arc::new(Int64Array::from(vec![value])) };
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new(format!("{name}.parquet")),
            size: 10,
        },
        partition: Some(PartitionValues::from([(
            "v".to_string(),
            int_scalar(value),
        )])),
        stats: Some(Arc::new(FileStats {
            num_records: Some(10),
            min_values: HashMap::from([("v".to_string(), bound())]),
            max_values: HashMap::from([("v".to_string(), bound())]),
            null_counts: HashMap::from([("v".to_string(), nulls)]),
        })),
        sort_order_id: None,
        first_row_id: None,
    }
}

fn append(table: &Table, entries: Vec<DataFileEntry>) -> Table {
    table
        .commit(&[], entries)
        .unwrap()
        .expect("no other writer")
}

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

/// The filter `v = value`.
fn v_equals(value: i64) -> Vec<Expression> {
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(int_scalar(value))),
        compare_type: CompareType::Equal,
        return_type: Type::Boolean,
    })]
}

/// The names of the files a scan of `table` over `filters` reads.
fn selected(table: &Table, filters: &[Expression]) -> Vec<String> {
    let files = table.select_scan(filters).unwrap().files;
    files
        .iter()
        .map(|file| {
            let path = file.file.path.as_str();
            path.rsplit('/').next().unwrap().to_string()
        })
        .collect()
}

/// The rows of each row group `plan` scans, in scan order.
fn row_group_rows(plan: &ScanPlan) -> Vec<i64> {
    plan.parquet
        .row_groups()
        .iter()
        .map(|row_group| row_group.num_rows)
        .collect()
}

/// What a cache counts the index of every partition of `table` for,
/// between them.
fn measure_indexes(table: &Table) -> u64 {
    table
        .partitions()
        .unwrap()
        .iter()
        .map(|partition| table.partition_files(partition).unwrap().heap_bytes())
        .sum()
}

/// What a cache counts the footer of every file of `table` for, between
/// them.
fn measure_footers(table: &Table) -> u64 {
    let files = table.select_scan(&[]).unwrap().files;
    table
        .open_footers(&files, FooterUse::Unrecorded)
        .unwrap()
        .iter()
        .map(|footer| measure_footer_entry(footer.file.path.as_str(), &footer.row_groups))
        .sum()
}

/// `table` as the catalog describes it.
fn describe(table: &Table) -> catalog::datastore::DatastoreTableMetadata {
    let name_file = |file: &IndexedFile| file.file.path.as_str().to_string();
    table
        .describe(table_name(), "t1".to_string(), &name_file)
        .unwrap()
}

#[test]
fn a_table_opens_without_reading_a_data_manifest() {
    let (fixture, created) = create_partitioned_table();
    let written = append(&created, vec![entry("seven", 7, 0), entry("nine", 9, 0)]);
    fixture.delete_data_manifests(&written);

    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));

    assert_eq!(opened.total_rows().unwrap(), 20);
}

#[test]
fn a_scan_filtered_on_the_partition_indexes_only_the_partitions_it_keeps() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![entry("seven", 7, 0)]);
    let second = append(&first, vec![entry("nine", 9, 0)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    fixture.delete_manifest(&find_joined_manifest(&first, &second));

    let files = selected(&opened, &v_equals(7));

    assert_eq!(files, ["seven.parquet"]);
}

#[test]
fn a_scan_that_keeps_only_partitions_without_nulls_reports_the_column_free_of_them() {
    let (fixture, created) = create_partitioned_table();
    let dense = append(&created, vec![entry("seven", 7, 0)]);
    let holed = append(&dense, vec![entry("nine", 9, 3)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    fixture.delete_manifest(&find_joined_manifest(&dense, &holed));

    let nullability = opened.select_scan(&v_equals(7)).unwrap().nullability;

    assert_eq!(nullability, [false]);
}

#[test]
fn a_scan_of_every_partition_reports_a_column_with_a_null_in_one_of_them_nullable() {
    let (fixture, created) = create_partitioned_table();
    let dense = append(&created, vec![entry("seven", 7, 0)]);
    append(&dense, vec![entry("nine", 9, 3)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));

    let nullability = opened.select_scan(&[]).unwrap().nullability;

    assert_eq!(nullability, [true]);
}

#[test]
fn a_scan_that_keeps_no_file_reports_every_column_nullable() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![entry("seven", 7, 0)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));

    let selected = opened.select_scan(&v_equals(8)).unwrap();

    assert!(selected.files.is_empty());
    assert_eq!(selected.nullability, [true]);
}

#[test]
fn a_partition_the_cache_let_go_of_is_indexed_again_when_asked_for() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![entry("seven", 7, 0)]);
    append(&first, vec![entry("nine", 9, 0)]);
    let opened = fixture.load(&FileMetadataCache::new(1));
    selected(&opened, &v_equals(7));
    selected(&opened, &v_equals(9));

    let files = selected(&opened, &v_equals(7));
    selected(&opened, &v_equals(9));
    fixture.delete_data_manifests(&first);
    let read_without_its_manifest = opened.select_scan(&v_equals(7));

    assert_eq!(files, ["seven.parquet"]);
    assert!(read_without_its_manifest.is_err());
}

#[test]
fn an_earlier_version_indexes_a_partition_the_cache_let_go_of_with_its_own_files() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![entry("early", 7, 0)]);
    let cache = FileMetadataCache::new(1);
    let earlier = fixture.load(&cache);
    selected(&earlier, &v_equals(7));
    let later = append(&earlier, vec![entry("late", 7, 0), entry("nine", 9, 0)]);
    selected(&later, &v_equals(9));

    let files_earlier = selected(&earlier, &v_equals(7));
    let files_later = selected(&later, &v_equals(7));

    assert_eq!(files_earlier, ["early.parquet"]);
    assert_eq!(files_later, ["early.parquet", "late.parquet"]);
}

#[test]
fn the_cache_keeps_two_partition_indexes_only_within_a_budget_with_room_for_both() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![entry("seven", 7, 0)]);
    let second = append(&first, vec![entry("nine", 9, 0)]);
    let measured = fixture.load(&FileMetadataCache::new(u64::MAX));
    let room_for_both: u64 = measured
        .partitions()
        .unwrap()
        .iter()
        .map(|partition| measured.partition_files(partition).unwrap().heap_bytes())
        .sum();
    let roomy = fixture.load(&FileMetadataCache::new(room_for_both));
    let tight = fixture.load(&FileMetadataCache::new(room_for_both - 1));
    for table in [&roomy, &tight] {
        selected(table, &v_equals(7));
        selected(table, &v_equals(9));
    }
    fixture.delete_data_manifests(&second);

    let kept = selected(&roomy, &v_equals(7));
    let let_go = tight.select_scan(&v_equals(7));

    assert_eq!(kept, ["seven.parquet"]);
    assert!(let_go.is_err());
}

#[test]
fn a_second_plan_over_a_file_with_another_filter_reads_no_footer() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    opened.plan_scan(&v_equals(7)).unwrap();
    fixture.replace_with_garbage("seven");

    let plan = opened.plan_scan(&[]).unwrap();

    assert_eq!(row_group_rows(&plan), [10]);
}

#[test]
fn a_footer_the_cache_let_go_of_is_read_again_when_a_plan_needs_it() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("seven", 7)]);
    append(&first, vec![fixture.write_file("nine", 9)]);
    let opened = fixture.load(&FileMetadataCache::new(1));
    opened.plan_scan(&v_equals(7)).unwrap();
    opened.plan_scan(&v_equals(9)).unwrap();

    let plan = opened.plan_scan(&v_equals(7)).unwrap();
    opened.plan_scan(&v_equals(9)).unwrap();
    fixture.replace_with_garbage("seven");
    let read_without_its_footer = opened.plan_scan(&v_equals(7));

    assert_eq!(row_group_rows(&plan), [10]);
    assert!(read_without_its_footer.is_err());
}

#[test]
fn the_cache_lets_go_of_a_scans_footers_before_the_index_it_selected_from() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("seven", 7)]);
    let second = append(&first, vec![fixture.write_file("nine", 9)]);
    let measured = fixture.load(&FileMetadataCache::new(u64::MAX));
    let room_for_all = measure_indexes(&measured) + measure_footers(&measured);
    let opened = fixture.load(&FileMetadataCache::new(room_for_all - 1));
    opened.plan_scan(&v_equals(7)).unwrap();
    opened.plan_scan(&v_equals(9)).unwrap();
    fixture.delete_data_manifests(&second);
    fixture.replace_with_garbage("seven");

    let selected_from_the_index = selected(&opened, &v_equals(7));
    let read_without_its_footer = opened.plan_scan(&v_equals(7));

    assert_eq!(selected_from_the_index, ["seven.parquet"]);
    assert!(read_without_its_footer.is_err());
}

#[test]
fn a_version_committed_on_top_reads_only_the_footer_of_the_file_it_added() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    opened.plan_scan(&[]).unwrap();
    fixture.replace_with_garbage("seven");

    let committed = append(&opened, vec![fixture.write_file("nine", 9)]);
    let plan = committed.plan_scan(&[]).unwrap();

    assert_eq!(row_group_rows(&plan), [10, 10]);
}

#[test]
fn footers_loaded_for_named_files_are_kept_for_the_next_plan() {
    let (fixture, created) = create_partitioned_table();
    let seven = fixture.write_file("seven", 7);
    append(&created, vec![seven.clone()]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));

    opened.load_footers(&[seven.file]).unwrap();
    fixture.replace_with_garbage("seven");
    let plan = opened.plan_scan(&[]).unwrap();

    assert_eq!(row_group_rows(&plan), [10]);
}

#[test]
fn a_second_plan_shares_the_row_groups_of_the_first() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    let first = opened.plan_scan(&[]).unwrap();

    let second = opened.plan_scan(&[]).unwrap();

    assert!(Arc::ptr_eq(
        &first.parquet.row_groups()[0],
        &second.parquet.row_groups()[0]
    ));
}

#[test]
fn a_forgotten_footer_is_read_again_and_its_file_no_longer_held_open() {
    let (fixture, created) = create_partitioned_table();
    let seven = fixture.write_file("seven", 7);
    append(&created, vec![seven.clone()]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    let held_open = Arc::downgrade(&opened.plan_scan(&[]).unwrap().parquet.row_groups()[0]);

    opened.forget_footers(&[seven.file.path]).unwrap();
    fixture.replace_with_garbage("seven");
    let read_without_its_footer = opened.plan_scan(&[]);

    assert!(held_open.upgrade().is_none());
    assert!(read_without_its_footer.is_err());
}

#[test]
fn describing_a_table_caches_none_of_the_footers_it_reads() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("seven", 7)]);
    let second = append(&first, vec![fixture.write_file("nine", 9)]);
    let measured = fixture.load(&FileMetadataCache::new(u64::MAX));
    let room_for_the_indexes = measure_indexes(&measured);
    let opened = fixture.load(&FileMetadataCache::new(room_for_the_indexes));

    let description = describe(&opened);
    fixture.delete_data_manifests(&second);
    let selected_from_the_indexes = selected(&opened, &[]);

    assert_eq!(description.files.len(), 2);
    assert_eq!(selected_from_the_indexes, ["seven.parquet", "nine.parquet"]);
}

#[test]
fn describing_a_table_reads_no_footer_the_cache_holds() {
    let (fixture, created) = create_partitioned_table();
    append(&created, vec![fixture.write_file("seven", 7)]);
    let opened = fixture.load(&FileMetadataCache::new(u64::MAX));
    opened.plan_scan(&[]).unwrap();
    fixture.replace_with_garbage("seven");

    let description = describe(&opened);

    assert_eq!(description.total_rows, 10);
}

#[test]
fn the_cache_keeps_an_index_and_a_footer_only_within_a_budget_with_room_for_both() {
    let (fixture, created) = create_partitioned_table();
    let first = append(&created, vec![fixture.write_file("seven", 7)]);
    let measured = fixture.load(&FileMetadataCache::new(u64::MAX));
    let room_for_both = measure_indexes(&measured) + measure_footers(&measured);
    let roomy = fixture.load(&FileMetadataCache::new(room_for_both));
    let tight = fixture.load(&FileMetadataCache::new(room_for_both - 1));
    for table in [&roomy, &tight] {
        table.plan_scan(&v_equals(7)).unwrap();
    }
    fixture.delete_data_manifests(&first);
    fixture.replace_with_garbage("seven");

    let kept = roomy.plan_scan(&v_equals(7)).unwrap();
    let let_go = tight.select_scan(&v_equals(7));

    assert_eq!(row_group_rows(&kept), [10]);
    assert!(let_go.is_err());
}
