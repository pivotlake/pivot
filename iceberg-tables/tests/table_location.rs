//! Where a table's objects are and how they are named: a key under the
//! table's store and the URI its metadata records convert both ways, in a
//! datastore's own store and in the store of a catalog table that spells
//! its root and names its paths itself.

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::spec::{
    FormatVersion, NestedField, PrimitiveType, Schema, SortOrder, TableMetadataBuilder, Type,
    UnboundPartitionSpec,
};
use iceberg_tables::metadata::layout::METADATA_PATH_PROPERTY;
use iceberg_tables::{Error, MetadataFile, TableLocation};
use object_storage::{DataFile, DataFileLocation, LocalStore, ObjectPath, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;

/// The location of a catalog table whose metadata names `table_uri` as the
/// table's own and sets `properties`, read through a store opened on the
/// filesystem root.
fn open_catalog_table(table_uri: &str, properties: HashMap<String, String>) -> TableLocation {
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "v", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    let metadata = TableMetadataBuilder::new(
        schema,
        UnboundPartitionSpec::builder().build(),
        SortOrder::unsorted_order(),
        table_uri.to_string(),
        FormatVersion::V3,
        properties,
    )
    .unwrap()
    .build()
    .unwrap()
    .metadata;
    let metadata_file = MetadataFile {
        location: format!("{table_uri}/metadata/v1.metadata.json"),
        metadata: Arc::new(metadata),
    };
    let at_root: Arc<dyn ObjectStore> = Arc::new(LocalStore::new("/").unwrap());
    TableLocation::of_catalog_table(
        &SchemaQualifiedTableName::new("db", "t"),
        at_root,
        &metadata_file,
    )
    .unwrap()
}

#[test]
fn a_datastores_own_store_addresses_files_from_its_root() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path().join("db")).unwrap());

    let location = TableLocation::in_datastore_store(
        &SchemaQualifiedTableName::new("main", "t"),
        store,
        ObjectPath::new("t"),
    )
    .unwrap();
    let file = location
        .locate_file_at_uri(
            &format!("file://{}/db/t/pivot-1.parquet", dir.path().display()),
            3,
        )
        .unwrap();

    assert_eq!(
        location.table_uri(),
        format!("file://{}/db/t", dir.path().display())
    );
    assert!(matches!(
        file.source,
        DataFileLocation::Local(path) if path == dir.path().join("db/t/pivot-1.parquet")
    ));
}

/// What a read of `file` is cached under and where it reads from.
fn identity_and_source(file: DataFile) -> (String, u64, std::path::PathBuf) {
    let DataFileLocation::Local(source) = file.source else {
        panic!("a local store reads from a path");
    };
    (file.file.path.as_str().to_string(), file.file.size, source)
}

#[test]
fn a_file_named_by_its_path_is_the_file_at_its_uri() {
    let dir = tempfile::tempdir().unwrap();
    let prefixed: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path().join("db")).unwrap());
    let directory_from_root = dir.path().join("db/t").display().to_string();
    let locations = [
        TableLocation::in_datastore_store(
            &SchemaQualifiedTableName::new("main", "t"),
            prefixed,
            ObjectPath::new("t"),
        )
        .unwrap(),
        open_catalog_table(&format!("file:{directory_from_root}"), HashMap::new()),
    ];
    let paths = [
        ObjectPath::new("data/pivot-1.parquet"),
        ObjectPath::new(format!("{}/shared/a.parquet", dir.path().display())),
    ];

    for location in &locations {
        for path in &paths {
            let by_path = location.locate_file_at_path(path, 7).unwrap();
            let by_uri = location
                .locate_file_at_uri(&location.build_uri_for_path(path).unwrap(), 7)
                .unwrap();

            assert_eq!(identity_and_source(by_path), identity_and_source(by_uri));
        }
    }
}

#[test]
fn a_data_file_uri_round_trips_to_the_same_path() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
    let location = TableLocation::in_datastore_store(
        &SchemaQualifiedTableName::new("main", "t"),
        store,
        ObjectPath::new("t1"),
    )
    .unwrap();
    let relative = ObjectPath::new("pivot-1.parquet");
    let adopted = ObjectPath::new(format!("{}/shared/a.parquet", dir.path().display()));

    let relative_uri = location.build_uri_for_path(&relative).unwrap();
    let adopted_uri = location.build_uri_for_path(&adopted).unwrap();

    assert_eq!(
        relative_uri,
        format!("file://{}/t1/pivot-1.parquet", dir.path().display())
    );
    assert_eq!(location.find_path_for_uri(&relative_uri).unwrap(), relative);
    assert_eq!(location.find_path_for_uri(&adopted_uri).unwrap(), adopted);
    assert_eq!(
        location.table_uri(),
        format!("file://{}/t1", dir.path().display())
    );
    assert_eq!(
        location.build_metadata_uri("m.avro"),
        format!("file://{}/t1/metadata/m.avro", dir.path().display())
    );
}

#[test]
fn a_table_spelling_its_root_another_way_names_its_files_that_way() {
    let dir = tempfile::tempdir().unwrap();
    let location = open_catalog_table(&format!("file:{}/t1", dir.path().display()), HashMap::new());
    let relative = ObjectPath::new("pivot-1.parquet");

    let uri = location.build_uri_for_path(&relative).unwrap();
    let path_of_other_spelling = location
        .find_path_for_uri(&format!(
            "file://{}/t1/pivot-1.parquet",
            dir.path().display()
        ))
        .unwrap();

    assert_eq!(
        uri,
        format!("file:{}/t1/pivot-1.parquet", dir.path().display())
    );
    assert_eq!(path_of_other_spelling, relative);
}

#[test]
fn a_uri_under_another_root_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
    let location = TableLocation::in_datastore_store(
        &SchemaQualifiedTableName::new("main", "t"),
        store,
        ObjectPath::new("t1"),
    )
    .unwrap();

    let error = location
        .find_path_for_uri("s3://elsewhere/t1/pivot-1.parquet")
        .unwrap_err();

    assert!(
        matches!(error, Error::FileOutsideTableStore { .. }),
        "{error}"
    );
}

#[test]
fn a_table_naming_its_own_paths_keeps_metadata_and_data_there() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().display().to_string();
    let location = open_catalog_table(
        &format!("file://{root}/t1"),
        HashMap::from([(
            METADATA_PATH_PROPERTY.to_string(),
            format!("file://{root}/elsewhere/meta"),
        )]),
    );
    // The store is opened on the filesystem root, so a key under it spells
    // the whole path.
    let key = |path: &str| ObjectPath::new(format!("{}/{path}", root.trim_start_matches('/')));

    let data_directory = location
        .data_directory_for(Some(&format!("file://{root}/t1/files/")))
        .unwrap();

    assert_eq!(
        location.build_metadata_uri("m.avro"),
        format!("file://{root}/elsewhere/meta/m.avro")
    );
    assert_eq!(
        location.metadata_key("m.avro"),
        key("elsewhere/meta/m.avro")
    );
    assert_eq!(data_directory, key("t1/files"));
    assert_eq!(location.data_directory_for(None).unwrap(), key("t1/data"));
}

#[test]
fn a_file_uploaded_under_the_table_is_named_relative_to_it() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
    let location = TableLocation::in_datastore_store(
        &SchemaQualifiedTableName::new("main", "t"),
        store,
        ObjectPath::new("t1"),
    )
    .unwrap();
    let file = ObjectPath::new("a.parquet");

    let in_data_directory = location
        .find_path_under(&location.data_directory_for(None).unwrap(), &file)
        .unwrap();
    let in_table_directory = location
        .find_path_under(location.directory(), &file)
        .unwrap();

    assert_eq!(in_data_directory, ObjectPath::new("data/a.parquet"));
    assert_eq!(in_table_directory, file);
    assert_eq!(
        location.build_uri_for_path(&in_data_directory).unwrap(),
        format!("file://{}/t1/data/a.parquet", dir.path().display())
    );
}

#[test]
fn a_file_uploaded_to_a_data_path_outside_the_table_is_named_by_its_absolute_key() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
    let location = TableLocation::in_datastore_store(
        &SchemaQualifiedTableName::new("main", "t"),
        store,
        ObjectPath::new("t1"),
    )
    .unwrap();
    let data_path = format!("file://{}/elsewhere/t1-data", dir.path().display());

    let data_directory = location.data_directory_for(Some(&data_path)).unwrap();
    let named = location
        .find_path_under(&data_directory, &ObjectPath::new("a.parquet"))
        .unwrap();

    assert!(data_directory.is_absolute(), "{data_directory}");
    assert_eq!(
        named,
        ObjectPath::new(format!(
            "{}/elsewhere/t1-data/a.parquet",
            dir.path().display()
        ))
    );
    assert_eq!(
        location.build_uri_for_path(&named).unwrap(),
        format!("{data_path}/a.parquet")
    );
}
