//! The table format a pivotlake datastore's configuration names for a
//! database that does not exist yet.

use std::path::{Path, PathBuf};
use std::time::Duration;

use catalog::metastore::Metastore;
use dispatch::Dispatch;
use metastore_disk::{DiskMetastore, Error, MetastoreConfig};

const REFRESH_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// Write `yaml` as a metastore file under `directory`.
fn write_metastore_file(directory: &Path, yaml: &str) -> PathBuf {
    let path = directory.join("metastore.yaml");
    std::fs::write(&path, yaml).unwrap();
    path
}

/// The table format the index of the database at `location` records.
fn read_recorded_table_format(location: &Path) -> String {
    let index = std::fs::read_to_string(location.join("_pivot_manifest.json")).unwrap();
    let index: serde_yaml_ng::Value = serde_yaml_ng::from_str(&index).unwrap();
    index["table_format"].as_str().unwrap().to_string()
}

#[test]
fn a_pivotlake_datastore_names_the_table_format_of_a_new_database() {
    let directory = tempfile::tempdir().unwrap();
    let location = |name: &str| directory.path().join(name);
    let yaml = format!(
        "datastores:\n  hot:\n    kind: pivotlake\n    location: {}\n    table_format: delta\n    default: true\n  lake:\n    kind: pivotlake\n    location: {}\n    table_format: deltalake\n  new:\n    kind: pivotlake\n    location: {}\n",
        location("hot").display(),
        location("lake").display(),
        location("new").display(),
    );
    let metastore_file = write_metastore_file(directory.path(), &yaml);
    let dispatch = Dispatch::spin_up(1, 32, None);
    let metastore = DiskMetastore::open(
        MetastoreConfig::default(),
        Some(&metastore_file),
        REFRESH_INTERVAL,
    )
    .unwrap();

    let datastores = metastore.open_datastores(dispatch.dispatcher()).unwrap();

    assert_eq!(datastores.len(), 3);
    assert_eq!(read_recorded_table_format(&location("hot")), "delta");
    assert_eq!(read_recorded_table_format(&location("lake")), "delta");
    assert_eq!(read_recorded_table_format(&location("new")), "iceberg");
}

#[test]
fn an_unknown_table_format_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let yaml = "datastores:\n  hot:\n    kind: pivotlake\n    location: /tmp/hot\n    table_format: hudi\n    default: true\n";
    let metastore_file = write_metastore_file(directory.path(), yaml);

    let result = DiskMetastore::open(
        MetastoreConfig::default(),
        Some(&metastore_file),
        REFRESH_INTERVAL,
    );

    assert!(matches!(result, Err(Error::Parse { .. })));
}
