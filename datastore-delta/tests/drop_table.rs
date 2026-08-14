//! Transaction and persistence behavior for `DROP TABLE`.

mod common;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

use common::{commit_datastore_transaction, shared_dispatcher};
use datastore::DatastoreTransaction;
use datastore_delta::store::{
    DataFileLocation, ListedObject, LocalStore, ObjectPath, ObjectStore, Result as StoreResult,
};
use datastore_delta::test_support as harness;
use datastore_delta::{DeltaDatastore, DeltaTransaction};
use delta_kernel::object_store::DynObjectStore;
use dispatch::DataFlowDispatcher;
use planner::catalog::{Column, CreateTableRequest, DropTableRequest, SchemaQualifiedTableName};
use planner::types::Type;
use tempfile::TempDir;

fn dispatcher() -> DataFlowDispatcher {
    shared_dispatcher(2, 32)
}

#[derive(Debug, Default)]
struct ManifestReadPause {
    state: Mutex<ManifestReadPauseState>,
    entered: Condvar,
    released: Condvar,
}

#[derive(Debug, Default)]
struct ManifestReadPauseState {
    armed: bool,
    blocked: bool,
    released: bool,
}

impl ManifestReadPause {
    fn arm(&self) {
        let mut state = self.state.lock().unwrap();
        assert!(!state.armed && !state.blocked);
        state.armed = true;
        state.released = false;
    }

    fn block_if_armed(&self) {
        let mut state = self.state.lock().unwrap();
        if !state.armed {
            return;
        }
        state.armed = false;
        state.blocked = true;
        self.entered.notify_all();
        while !state.released {
            state = self.released.wait(state).unwrap();
        }
        state.blocked = false;
    }

    fn wait_until_blocked(&self) {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .entered
            .wait_timeout_while(state, Duration::from_secs(5), |state| !state.blocked)
            .unwrap();
        assert!(state.blocked, "manifest GET did not reach the pause");
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        assert!(state.blocked);
        state.released = true;
        self.released.notify_all();
    }
}

#[derive(Debug)]
struct PausingManifestStore {
    inner: LocalStore,
    pause: Arc<ManifestReadPause>,
}

impl ObjectStore for PausingManifestStore {
    fn local_root(&self) -> Option<&Path> {
        self.inner.local_root()
    }

    fn get(&self, key: &ObjectPath) -> StoreResult<Option<Vec<u8>>> {
        let result = self.inner.get(key)?;
        if key.as_str() == "_pivot_manifest.json" {
            self.pause.block_if_armed();
        }
        Ok(result)
    }

    fn put(&self, key: &ObjectPath, data: &[u8]) -> StoreResult<()> {
        self.inner.put(key, data)
    }

    fn delete(&self, key: &ObjectPath) -> StoreResult<()> {
        self.inner.delete(key)
    }

    fn delete_dir(&self, prefix: &ObjectPath) -> StoreResult<()> {
        self.inner.delete_dir(prefix)
    }

    fn list(&self, prefix: &ObjectPath) -> StoreResult<Vec<ListedObject>> {
        self.inner.list(prefix)
    }

    fn source(&self, key: &ObjectPath) -> StoreResult<DataFileLocation> {
        self.inner.source(key)
    }

    fn sink(&self, key: &ObjectPath) -> StoreResult<DataFileLocation> {
        self.inner.sink(key)
    }

    fn prepare_write(&self) -> StoreResult<()> {
        self.inner.prepare_write()
    }

    fn create_dir(&self, prefix: &ObjectPath) -> StoreResult<()> {
        self.inner.create_dir(prefix)
    }

    fn absolute_key(&self, key: &ObjectPath) -> StoreResult<ObjectPath> {
        self.inner.absolute_key(key)
    }

    fn describe(&self) -> String {
        self.inner.describe()
    }

    fn location_uri(&self) -> String {
        self.inner.location_uri()
    }

    fn build_delta_object_store(&self) -> StoreResult<Arc<DynObjectStore>> {
        self.inner.build_delta_object_store()
    }
}

fn create_table(datastore: &Arc<DeltaDatastore>, name: &str) -> String {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                col_type: Type::Int64,
            }],
            options: HashMap::new(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    datastore
        .clone()
        .begin_transaction()
        .table_revision(&SchemaQualifiedTableName::in_default_schema(name))
        .unwrap()
        .identity
}

fn stage_drop(transaction: &Arc<DeltaTransaction>, name: &str, if_exists: bool) {
    transaction
        .bind_drop_table(DropTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            if_exists,
            cascade: false,
        })
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .execute()
        .collect()
        .unwrap();
}

fn drop_table(datastore: &Arc<DeltaDatastore>, name: &str, if_exists: bool) {
    let transaction = datastore.clone().begin_transaction();
    stage_drop(&transaction, name, if_exists);
    commit_datastore_transaction(transaction).unwrap();
}

#[test]
fn drop_is_visible_on_commit_and_survives_reopen() {
    let database = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(database.path().to_str().unwrap(), &dispatcher()).unwrap();
    let id = create_table(&datastore, "events");
    let table_directory = database.path().join(&id);

    drop_table(&datastore, "events", false);

    assert!(!datastore.contains_table(&SchemaQualifiedTableName::in_default_schema("events")));
    assert!(
        table_directory.exists(),
        "retention keeps the managed storage"
    );
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(database.path().join("_pivot_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["dropped_tables"][id.as_str()]["location"], id);
    drop(datastore);

    let reopened = DeltaDatastore::open(database.path().to_str().unwrap(), &dispatcher()).unwrap();
    assert!(!reopened.contains_table(&SchemaQualifiedTableName::in_default_schema("events")));
}

#[test]
fn drop_changes_nothing_until_its_dataflow_runs_and_transaction_commits() {
    let database = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(database.path().to_str().unwrap(), &dispatcher()).unwrap();
    create_table(&datastore, "events");
    let transaction = datastore.clone().begin_transaction();
    let compiled = transaction
        .bind_drop_table(DropTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "events".to_string(),
            if_exists: false,
            cascade: false,
        })
        .unwrap()
        .compile(&dispatcher())
        .unwrap();

    commit_datastore_transaction(transaction).unwrap();

    assert!(datastore.contains_table(&SchemaQualifiedTableName::in_default_schema("events")));
    drop(compiled);
}

#[test]
fn rolling_back_a_staged_drop_keeps_the_table() {
    let database = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(database.path().to_str().unwrap(), &dispatcher()).unwrap();
    create_table(&datastore, "events");
    let transaction = datastore.clone().begin_transaction();
    stage_drop(&transaction, "events", false);

    transaction.rollback();

    assert!(datastore.contains_table(&SchemaQualifiedTableName::in_default_schema("events")));
}

#[test]
fn if_exists_missing_in_the_snapshot_stays_a_noop_after_a_create() {
    let database = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(database.path().to_str().unwrap(), &dispatcher()).unwrap();
    let stale_drop = datastore.clone().begin_transaction();
    stage_drop(&stale_drop, "events", true);
    let created_id = create_table(&datastore, "events");

    commit_datastore_transaction(stale_drop).unwrap();

    let actual_id = datastore
        .clone()
        .begin_transaction()
        .table_revision(&SchemaQualifiedTableName::in_default_schema("events"))
        .unwrap()
        .identity;
    assert_eq!(actual_id, created_id);
}

#[test]
fn a_drop_cannot_remove_a_recreated_table_incarnation() {
    let database = TempDir::new().unwrap();
    let datastore = DeltaDatastore::open(database.path().to_str().unwrap(), &dispatcher()).unwrap();
    let original_id = create_table(&datastore, "events");
    let stale_drop = datastore.clone().begin_transaction();
    stage_drop(&stale_drop, "events", true);
    drop_table(&datastore, "events", false);
    let replacement_id = create_table(&datastore, "events");

    let error = commit_datastore_transaction(stale_drop).unwrap_err();

    assert_ne!(replacement_id, original_id);
    assert!(
        error
            .to_string()
            .contains("changed since the transaction began")
    );
    let actual_id = datastore
        .clone()
        .begin_transaction()
        .table_revision(&SchemaQualifiedTableName::in_default_schema("events"))
        .unwrap()
        .identity;
    assert_eq!(actual_id, replacement_id);
}

#[test]
fn a_manifest_refresh_does_not_hold_the_table_index_during_store_io() {
    let database = TempDir::new().unwrap();
    let pause = Arc::new(ManifestReadPause::default());
    let store = Arc::new(PausingManifestStore {
        inner: LocalStore::new(database.path()),
        pause: pause.clone(),
    });
    let datastore = DeltaDatastore::from_store(store, &dispatcher(), None).unwrap();
    pause.arm();

    let refreshing_datastore = datastore.clone();
    let refresh = std::thread::spawn(move || refreshing_datastore.refresh_from_store());
    pause.wait_until_blocked();
    let creating_datastore = datastore.clone();
    let (created_tx, created_rx) = mpsc::sync_channel(1);
    let creation = std::thread::spawn(move || {
        let id = create_table(&creating_datastore, "events");
        created_tx.send(id.clone()).unwrap();
        id
    });
    let created_while_refresh_was_blocked = created_rx.recv_timeout(Duration::from_secs(5));
    pause.release();
    let refresh_changed = refresh.join().unwrap().unwrap();
    let created_id = creation.join().unwrap();

    assert!(
        created_while_refresh_was_blocked.is_ok(),
        "catalog creation was blocked by the paused manifest GET"
    );
    assert_eq!(created_while_refresh_was_blocked.unwrap(), created_id);
    assert!(!refresh_changed, "the stale manifest was discarded");
    assert!(datastore.contains_table(&SchemaQualifiedTableName::in_default_schema("events")));
}

#[test]
fn refresh_observes_remote_drop_and_recreate_as_distinct_incarnations() {
    let Some(backend) = harness::s3("drop-refresh-between-processes") else {
        return;
    };
    let writer = DeltaDatastore::open(&backend.root, &dispatcher()).unwrap();
    let reader = DeltaDatastore::open(&backend.root, &dispatcher()).unwrap();
    let original_id = create_table(&writer, "events");
    assert!(reader.refresh_from_store().unwrap());
    assert_eq!(
        reader
            .clone()
            .begin_transaction()
            .table_revision(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .identity,
        original_id
    );

    drop_table(&writer, "events", false);
    assert!(reader.refresh_from_store().unwrap());
    assert!(!reader.contains_table(&SchemaQualifiedTableName::in_default_schema("events")));
    let replacement_id = create_table(&writer, "events");
    assert!(reader.refresh_from_store().unwrap());

    assert_ne!(replacement_id, original_id);
    assert_eq!(
        reader
            .clone()
            .begin_transaction()
            .table_revision(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .identity,
        replacement_id
    );
}
