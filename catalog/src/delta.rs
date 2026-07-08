//! The catalog's durable table metadata, as one Delta Lake table per pivot
//! table.
//!
//! Each table's transaction log (`_delta_log/` under the table's data
//! location) is the source of truth for its declared schema (each field also
//! carries the pivot column type in its metadata), its partition and sort
//! specs, and its committed data files with their partition tuples and
//! sort-key bounds. A commit is a compare-and-swap: delta-rs creates the next
//! log version with an atomic "create if absent" (a conditional PUT on S3, an
//! atomic rename on a local disk), fails if that version already exists, and
//! the caller reloads and retries on top of the winner. That is what lets two
//! processes register or replace files concurrently, exactly like any other
//! Delta writer.
//!
//! Two delta implementations share the log. The statement-facing hot paths
//! (opening a table, the per-query reload, committing freshly written files)
//! go through delta-kernel-rs, whose synchronous API the catalog control
//! plane calls directly; the kernel engine drives its IO on one shared
//! background thread. The cold paths (CREATE TABLE, compaction's swap, and
//! log maintenance) stay on delta-rs, which owns the operations kernel lacks
//! (vacuum, checkpointing with log cleanup); delta-rs is async, so those run
//! on one dedicated tokio runtime owned by this module, the calling thread
//! blocking on a channel for the result. Both sides write the same
//! compare-and-swap commits (an atomic "create if absent" of the next log
//! version), so they interleave safely with each other and with any other
//! Delta writer.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use delta_kernel::arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, MapArray, MapBuilder, MapFieldNames,
    RecordBatch, StringArray, StringBuilder, StructArray, UInt64Array,
};
use delta_kernel::arrow::buffer::NullBuffer;
use delta_kernel::arrow::datatypes::{DataType as ArrowDataType, Field, Fields};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::schema::MetadataValue as KernelMetadataValue;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::transaction::CommitResult;
use delta_kernel::{Engine, EngineData, Snapshot};
use delta_kernel_default_engine::DefaultEngine;
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::storage::store_from_url_opts;
use deltalake_core::kernel::transaction::{CommitBuilder, TransactionError};
use deltalake_core::kernel::{
    Action, Add, DataType as DeltaDataType, MetadataValue, PrimitiveType, Remove, StructField,
};
use deltalake_core::protocol::{DeltaOperation, SaveMode};
use deltalake_core::table::builder::DeltaTableBuilder;
use deltalake_core::{DeltaTable, DeltaTableError};
use planner::catalog::Column;
use planner::types::Type;
use serde::{Deserialize, Serialize};

use crate::store::{FileRef, ObjectPath};

/// Field-metadata key each schema field stores its exact pivot type under.
/// The Delta-native field type is a best-effort projection (Delta has no
/// unsigned integers, and pivot temporal values travel in arrow-json text
/// form); this key is what a reload trusts.
const COLUMN_TYPE_KEY: &str = "pivot.type";
/// Table-configuration key holding the sort spec as a JSON array of column
/// names. Delta has no native sort spec, so it rides in the table properties.
const SORT_BY_KEY: &str = "pivot.sortBy";
/// How long a superseded log version stays readable before the maintenance
/// sweep may clean it up, and how long a removed data file stays undeleted
/// after its tombstone. Every statement resolves the latest snapshot and
/// finishes in well under a second, so five minutes is ample headroom for a
/// reader while keeping the log directory and data directory from growing
/// without bound.
const RETENTION: std::time::Duration = std::time::Duration::from_secs(5 * 60);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Delta(#[from] DeltaTableError),
    #[error(transparent)]
    Kernel(#[from] delta_kernel::Error),
    #[error("arrow: {0}")]
    Arrow(#[from] delta_kernel::arrow::error::ArrowError),
    #[error("sort-key bound {value} is not the arrow-json form of a `{column_type:?}` value")]
    InvalidSortBound {
        value: serde_json::Value,
        column_type: Type,
    },
    #[error("delta scan row has no `{0}` column of the expected type")]
    MalformedScanRow(&'static str),
    #[error("table `{0}` is in the catalog index but has no delta log at its location")]
    MissingLog(String),
    #[error("delta table already exists")]
    TableExists,
    #[error(
        "column `{0}` has no `{COLUMN_TYPE_KEY}` field metadata; the table was not written by this catalog"
    )]
    MissingColumnType(String),
    #[error("metadata json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "data file `{0}` has an absolute path; a delta log only records files under the table's location"
    )]
    AbsoluteFilePath(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The sort key's range within one file: the `sort_by` columns at the file's
/// first and last row (the file is sorted, so these bound every row). Each is a
/// one-row arrow-json object, e.g. `{"Timestamp": 100}`. Lets a reader prune a
/// file on a sort-key range predicate without fetching its footer. Persisted as
/// the `minValues`/`maxValues` of the file's delta stats.
#[derive(Clone, Serialize, Deserialize)]
pub struct SortBounds {
    pub min: serde_json::Value,
    pub max: serde_json::Value,
}

/// One committed data file: its store identity ([`FileRef`]) plus the optional
/// partition tuple and sort-key bounds a partitioning/sorting writer sink
/// stamps on it. Both are `None` for files written without that metadata (an
/// unpartitioned/unsorted table, or files discovered at CREATE TABLE).
/// Persisted as one `add` action in the table's delta log.
#[derive(Clone)]
pub struct ManifestEntry {
    pub file: FileRef,
    pub partition: Option<serde_json::Value>,
    pub sort_bounds: Option<SortBounds>,
}

impl ManifestEntry {
    /// An entry with no partition/sort metadata (unpartitioned + unsorted table,
    /// or a writer that doesn't record it).
    pub fn new(file: FileRef) -> Self {
        Self {
            file,
            partition: None,
            sort_bounds: None,
        }
    }

    /// Whether this file *can* hold a row matching every partition filter, a
    /// soft test, so it never wrongly drops a file. A filter on a non-partition
    /// column, an entry with no recorded tuple, or a tuple missing the column all
    /// keep the entry (absence of a value is not proof of a mismatch). Only a
    /// recorded partition value that differs from the filter's excludes it. Both
    /// values are the JSON arrow-json produced (the sink for the tuple, the query
    /// for the constant), so they compare directly.
    pub fn maybe_matches_partition(
        &self,
        partition_by: &[String],
        filters: &[PartitionEqFilter],
    ) -> bool {
        let Some(tuple) = self.partition.as_ref().and_then(|v| v.as_object()) else {
            return true;
        };
        filters.iter().all(|filter| {
            !partition_by.iter().any(|c| c == &filter.column)
                || match tuple.get(&filter.column) {
                    Some(value) => *value == filter.value,
                    None => true,
                }
        })
    }
}

/// A `partition column = constant` predicate the query pushed down, with the
/// constant already encoded to the JSON shape a partition tuple records (via
/// arrow-json, the same encoder the sink uses). [`ManifestEntry::maybe_matches_partition`]
/// uses it to skip a file whose recorded partition value can't match *before*
/// its footer is fetched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionEqFilter {
    pub column: String,
    pub value: serde_json::Value,
}

/// One table's committed state at one delta log version: its declared schema,
/// its partition and sort specs, and its file list. Extracted whole from a
/// delta snapshot; a change swaps the whole value.
#[derive(Clone)]
pub struct TableState {
    pub version: u64,
    pub columns: Vec<Column>,
    /// Partition columns, in order (identity partitioning); empty = unpartitioned.
    pub partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    pub sort_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
}

impl TableState {
    fn contains_path(&self, path: &ObjectPath) -> bool {
        self.entries.iter().any(|e| &e.file.path == path)
    }

    /// The sort columns paired with their declared types (sort columns are
    /// always declared columns).
    fn sort_columns(&self) -> Vec<(String, Type)> {
        self.sort_by
            .iter()
            .map(|name| {
                let column = self
                    .columns
                    .iter()
                    .find(|column| &column.name == name)
                    .expect("sort columns are declared columns");
                (name.clone(), column.col_type.clone())
            })
            .collect()
    }
}

/// One table's delta transaction log, shared by every in-memory copy of the
/// table. Both cached handles only ever advance to newer versions; each
/// caller keeps its own [`TableState`] snapshot and reconciles against the
/// latest on refresh.
pub struct DeltaLog {
    /// The delta-rs handle behind the cold paths ([`create`](Self::create),
    /// [`replace`](Self::replace), [`maintain`](Self::maintain)), driven on
    /// the dedicated tokio runtime. Built unloaded at [`open`](Self::open);
    /// every use refreshes it first.
    table: Mutex<DeltaTable>,
    /// The kernel engine behind the synchronous hot paths
    /// ([`load_after`](Self::load_after), [`commit_added`](Self::commit_added)).
    engine: Arc<DefaultEngine<TokioBackgroundExecutor>>,
    /// The latest kernel snapshot seen.
    snapshot: Mutex<SnapshotRef>,
    /// The table's partition columns, immutable after CREATE TABLE.
    partition_by: Vec<String>,
}

/// What one round of [`DeltaLog::commit_retrying`] decided, planned against
/// the latest committed state.
enum CommitPlan {
    /// Commit these actions as the next log version.
    Commit(Vec<Action>, DeltaOperation),
    /// The plan's precondition no longer holds: stop without committing.
    Abort,
}

impl std::fmt::Debug for DeltaLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaLog").finish_non_exhaustive()
    }
}

/// The dedicated runtime every delta log operation runs on. Initialized once,
/// together with the handlers that teach delta-rs to open `s3://` and `gs://`
/// locations.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        deltalake_aws::register_handlers(None);
        deltalake_gcp::register_handlers(None);
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("delta-log")
            .build()
            .expect("delta log runtime builds")
    })
}

/// Run `future` on the delta runtime and block the calling thread for its
/// result. Blocking on a channel (rather than `Runtime::block_on`) keeps this
/// safe to call from any thread, including one inside another runtime.
fn run_blocking<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) -> T {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    runtime().spawn(async move {
        let _ = sender.send(future.await);
    });
    receiver.recv().expect("delta log task panicked")
}

/// The background executor every table's kernel engine drives its IO on:
/// kernel's API is synchronous, and this owns the one thread that runs its
/// object-store futures (blocking callers on a channel, so it too is safe to
/// call from inside another runtime).
fn kernel_executor() -> Arc<TokioBackgroundExecutor> {
    static EXECUTOR: OnceLock<Arc<TokioBackgroundExecutor>> = OnceLock::new();
    EXECUTOR
        .get_or_init(|| Arc::new(TokioBackgroundExecutor::new()))
        .clone()
}

/// The kernel engine for the table at `url`, its object store built from the
/// same storage options the delta-rs side opens with. The one option whose
/// spelling differs is translated: delta-rs reads `AWS_FORCE_PATH_STYLE`,
/// object_store the inverted `aws_virtual_hosted_style_request`.
fn build_kernel_engine(
    url: &url::Url,
    storage_options: &HashMap<String, String>,
) -> Result<Arc<DefaultEngine<TokioBackgroundExecutor>>> {
    let options = storage_options.iter().map(|(key, value)| {
        if key == "AWS_FORCE_PATH_STYLE" {
            (
                "aws_virtual_hosted_style_request".to_string(),
                (value != "true").to_string(),
            )
        } else {
            (key.clone(), value.clone())
        }
    });
    let store = store_from_url_opts(url, options)?;
    Ok(Arc::new(
        DefaultEngine::builder(store)
            .with_task_executor(kernel_executor())
            .build(),
    ))
}

/// Whether a kernel snapshot build failed because no delta log exists at the
/// location, as opposed to a real error.
fn is_missing_log(error: &delta_kernel::Error) -> bool {
    match error {
        delta_kernel::Error::Generic(message) => message.contains("No files in log segment"),
        delta_kernel::Error::FileNotFound(_) | delta_kernel::Error::MissingVersion => true,
        _ => false,
    }
}

impl DeltaLog {
    /// Create a brand-new delta table at `uri` recording the declared schema,
    /// the partition/sort specs, and the already-written initial files, and
    /// return its live log + first committed state. Fails with
    /// [`Error::TableExists`] if a delta log already exists there (including a
    /// concurrent creator winning the version-0 commit).
    pub fn create(
        uri: String,
        storage_options: HashMap<String, String>,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        entries: Vec<ManifestEntry>,
    ) -> Result<(Self, TableState)> {
        let fields: Vec<StructField> = columns.iter().map(build_field).collect();
        let adds: Vec<Action> = entries
            .iter()
            .map(|entry| build_add_action(entry, &partition_by, true))
            .collect::<Result<_>>()?;
        let configuration = [
            (
                SORT_BY_KEY.to_string(),
                Some(serde_json::to_string(&sort_by)?),
            ),
            // Superseded log versions only need to outlive a running statement;
            // the maintenance sweep enforces this horizon.
            (
                "delta.logRetentionDuration".to_string(),
                Some(format!("interval {} seconds", RETENTION.as_secs())),
            ),
            // Age tombstones out of snapshots/checkpoints on the same horizon
            // their files are vacuumed on. Left at the 7-day default, every
            // maintenance sweep would re-find the same expired tombstones and
            // commit a junk vacuum log version per sweep for a week.
            (
                "delta.deletedFileRetentionDuration".to_string(),
                Some(format!("interval {} seconds", RETENTION.as_secs())),
            ),
        ];
        let create_uri = uri.clone();
        let create_options = storage_options.clone();
        let create_partition_by = partition_by.clone();
        let table = run_blocking(async move {
            deltalake_core::operations::create::CreateBuilder::new()
                .with_location(create_uri)
                .with_storage_options(create_options)
                .with_columns(fields)
                .with_partition_columns(create_partition_by)
                .with_configuration(configuration)
                .with_raise_if_key_not_exists(false)
                .with_save_mode(SaveMode::ErrorIfExists)
                .with_actions(adds)
                .await
        })
        .map_err(map_create_error)?;
        let state = extract_state(&table)?;
        let url = url::Url::parse(&uri)
            .map_err(|e| DeltaTableError::InvalidTableLocation(format!("{uri}: {e}")))?;
        let engine = build_kernel_engine(&url, &storage_options)?;
        let snapshot = Snapshot::builder_for(uri).build(engine.as_ref())?;
        Ok((
            Self {
                table: Mutex::new(table),
                engine,
                snapshot: Mutex::new(snapshot),
                partition_by,
            },
            state,
        ))
    }

    /// Open the delta table at `uri` at its latest version. `name` is only for
    /// the error when no delta log exists there (a table the catalog index
    /// records must have one).
    pub fn open(
        uri: String,
        storage_options: HashMap<String, String>,
        name: &str,
    ) -> Result<(Self, TableState)> {
        let url = url::Url::parse(&uri)
            .map_err(|e| DeltaTableError::InvalidTableLocation(format!("{uri}: {e}")))?;
        let engine = build_kernel_engine(&url, &storage_options)?;
        let snapshot = Snapshot::builder_for(uri)
            .build(engine.as_ref())
            .map_err(|e| {
                if is_missing_log(&e) {
                    Error::MissingLog(name.to_string())
                } else {
                    Error::Kernel(e)
                }
            })?;
        let partition_by = read_partition_columns(&snapshot, engine.as_ref())?;
        let state = extract_snapshot_state(engine.as_ref(), &snapshot, &partition_by)?;
        // The delta-rs handle starts unloaded (no IO here); the cold paths it
        // serves refresh it to the latest version before every use.
        let table = DeltaTableBuilder::from_url(url)?
            .with_storage_options(storage_options)
            .build()?;
        Ok((
            Self {
                table: Mutex::new(table),
                engine,
                snapshot: Mutex::new(snapshot),
                partition_by,
            },
            state,
        ))
    }

    /// The latest committed state, but only if it is newer than `since`: the
    /// per-query reload. Returns `Ok(None)` when `since` is already current,
    /// without paying the per-file state extraction (no change is the common
    /// case at query bind).
    pub fn load_after(&self, since: u64) -> Result<Option<TableState>> {
        let snapshot = self.refresh_snapshot()?;
        if snapshot.version() <= since {
            return Ok(None);
        }
        extract_snapshot_state(self.engine.as_ref(), &snapshot, &self.partition_by).map(Some)
    }

    /// Advance the cached kernel snapshot to the latest committed version
    /// (incrementally: only log entries past the cached version are read) and
    /// return it. Concurrent refreshers race safely: the cache keeps
    /// whichever copy saw the newest version.
    fn refresh_snapshot(&self) -> Result<SnapshotRef> {
        let current = self.snapshot.lock().unwrap().clone();
        let latest = Snapshot::builder_from(current).build(self.engine.as_ref())?;
        Ok(self.store_snapshot(latest))
    }

    /// Cache `snapshot` if it is newer than the cached one; either way return
    /// the newest of the two.
    fn store_snapshot(&self, snapshot: SnapshotRef) -> SnapshotRef {
        let mut guard = self.snapshot.lock().unwrap();
        if snapshot.version() > guard.version() {
            *guard = snapshot;
        }
        guard.clone()
    }

    /// Commit `entries` as new files in **one** log version, retrying past
    /// concurrent writers. Entries whose path the log already holds are
    /// skipped, so a replayed commit can't double-count rows. Returns the
    /// state the commit landed at (or the current one if nothing was fresh).
    pub fn commit_added(&self, entries: Vec<ManifestEntry>) -> Result<TableState> {
        let engine = self.engine.as_ref();
        let mut snapshot = self.refresh_snapshot()?;
        loop {
            let state = extract_snapshot_state(engine, &snapshot, &self.partition_by)?;
            let fresh: Vec<&ManifestEntry> = entries
                .iter()
                .filter(|entry| !state.contains_path(&entry.file.path))
                .collect();
            if fresh.is_empty() {
                return Ok(state);
            }
            let sort_columns = state.sort_columns();
            let mut transaction = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), engine)?
                .with_operation("WRITE".to_string());
            transaction.add_files(build_add_files_data(
                &fresh,
                &self.partition_by,
                &sort_columns,
            )?);
            match transaction.commit(engine)? {
                CommitResult::CommittedTransaction(committed) => {
                    let landed = match committed.post_commit_snapshot() {
                        Some(snapshot) => self.store_snapshot(snapshot.clone()),
                        None => self.refresh_snapshot()?,
                    };
                    return extract_snapshot_state(engine, &landed, &self.partition_by);
                }
                // Another writer created our version first: reload and
                // re-plan on top of the winner (kernel, like this module's
                // delta-rs side, never transparently re-applies actions).
                CommitResult::ConflictedTransaction(_) => snapshot = self.refresh_snapshot()?,
                CommitResult::RetryableTransaction(retryable) => {
                    return Err(retryable.error.into());
                }
            }
        }
    }

    /// Atomically swap a set of files for another (the compaction commit):
    /// one log version holding the `removed` tombstones and the `added` files,
    /// retrying past concurrent commits. Returns the new state, or `None` if
    /// any of `removed` is no longer in the latest version: another writer
    /// already swapped those inputs out, so re-adding `added` (which holds
    /// their rows) would double-count. The caller must then discard its
    /// `added` files as orphans.
    pub fn replace(
        &self,
        removed: Vec<ObjectPath>,
        added: Vec<ManifestEntry>,
    ) -> Result<Option<TableState>> {
        self.commit_retrying(move |state| {
            if !removed.iter().all(|path| state.contains_path(path)) {
                return Ok(CommitPlan::Abort);
            }
            // The swap rewrites rows into new files without changing them, so
            // both sides carry `data_change: false`, like any compacting delta
            // writer.
            let mut actions: Vec<Action> = state
                .entries
                .iter()
                .filter(|entry| removed.contains(&entry.file.path))
                .map(|entry| build_remove_action(entry, &state.partition_by))
                .collect::<Result<_>>()?;
            for entry in &added {
                actions.push(build_add_action(entry, &state.partition_by, false)?);
            }
            Ok(CommitPlan::Commit(
                actions,
                DeltaOperation::Optimize {
                    predicate: None,
                    target_size: 0,
                },
            ))
        })
    }

    /// The delta-rs CAS-commit retry loop: refresh to the latest state, let
    /// `plan` derive the commit against it, and try to create the next log
    /// version; losing the race to a concurrent writer re-plans on top of
    /// the winner. Returns the state the commit landed at, or `None` when it
    /// aborted.
    fn commit_retrying<F>(&self, plan: F) -> Result<Option<TableState>>
    where
        F: Fn(&TableState) -> Result<CommitPlan> + Send + 'static,
    {
        self.mutate(move |table| {
            Box::pin(async move {
                loop {
                    table.update_state().await?;
                    let state = extract_state(table)?;
                    let (actions, operation) = match plan(&state)? {
                        CommitPlan::Commit(actions, operation) => (actions, operation),
                        CommitPlan::Abort => return Ok(None),
                    };
                    match try_commit(table, actions, operation).await {
                        Ok(()) => {
                            table.update_state().await?;
                            return extract_state(table).map(Some);
                        }
                        Err(e) if is_version_conflict(&e) => continue,
                        Err(e) => return Err(e.into()),
                    }
                }
            })
        })
    }

    /// Background upkeep of the log and the data directory: write a checkpoint
    /// at the current version, drop log entries older than the retention
    /// horizon, and physically delete data files whose tombstones have aged
    /// past it (no live reader can still hold a snapshot that old). Safe to
    /// run concurrently with commits and other sweeps.
    pub fn maintain(&self) -> Result<()> {
        self.mutate(|table| {
            Box::pin(async move {
                table.update_state().await?;
                deltalake_core::checkpoints::create_checkpoint(table, None).await?;
                deltalake_core::checkpoints::cleanup_metadata(table, None).await?;
                let (maintained, _metrics) = table
                    .clone()
                    .vacuum()
                    .with_retention_period(chrono::Duration::from_std(RETENTION).expect("fits"))
                    .with_enforce_retention_duration(false)
                    .await?;
                *table = maintained;
                Ok(())
            })
        })
    }

    /// Run one async `operation` against the shared [`DeltaTable`] on the delta
    /// runtime, blocking for its result. The table is cloned out under the lock
    /// and swapped back in afterwards, so the lock is never held across the
    /// operation itself: a query bind's reload never waits behind a slow commit
    /// or maintenance sweep. Concurrent operations race on clones, which is
    /// safe: commits are compare-and-swaps that retry on conflict, and the
    /// swap-back below keeps whichever copy saw the newest version.
    fn mutate<T, F>(&self, operation: F) -> T
    where
        T: Send + 'static,
        F: for<'a> FnOnce(&'a mut DeltaTable) -> Pin<Box<dyn Future<Output = T> + Send + 'a>>
            + Send
            + 'static,
    {
        let mut table = self.table.lock().unwrap().clone();
        let (table, result) = run_blocking(async move {
            let result = operation(&mut table).await;
            (table, result)
        });
        let mut guard = self.table.lock().unwrap();
        if table.version() > guard.version() {
            *guard = table;
        }
        result
    }
}

/// Try to commit `actions` as the table's next version. No transparent retry:
/// a concurrent commit surfaces as a version conflict for the caller's loop,
/// which must re-derive its actions against the newer state (delta-rs's own
/// conflict resolution would happily re-apply a stale action set).
async fn try_commit(
    table: &DeltaTable,
    actions: Vec<Action>,
    operation: DeltaOperation,
) -> Result<(), DeltaTableError> {
    CommitBuilder::default()
        .with_actions(actions)
        .with_max_retries(0)
        .build(Some(table.snapshot()?), table.log_store(), operation)
        .await?;
    Ok(())
}

/// Whether a commit failure means "another writer created this version first"
/// (reload and retry), as opposed to a real error.
fn is_version_conflict(error: &DeltaTableError) -> bool {
    match error {
        DeltaTableError::VersionAlreadyExists(_) => true,
        DeltaTableError::Transaction { source } => matches!(
            source,
            TransactionError::VersionAlreadyExists(_) | TransactionError::MaxCommitAttempts(_)
        ),
        _ => false,
    }
}

/// Map a CREATE failure: delta-rs reports "a table already exists" as a
/// generic error (its concrete create error is private), and a concurrent
/// creator winning version 0 as a version conflict. Both mean the same thing
/// to the caller.
fn map_create_error(error: DeltaTableError) -> Error {
    let already_exists = match &error {
        DeltaTableError::GenericError { source } => source.to_string().contains("already exists"),
        other => is_version_conflict(other),
    };
    if already_exists {
        Error::TableExists
    } else {
        Error::Delta(error)
    }
}

/// Milliseconds since the epoch, the timestamp delta actions carry.
fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_millis() as i64
}

/// The delta-native projection of a pivot column type. Delta has no unsigned
/// integers, so those widen (and `UInt64` lands on a decimal wide enough for
/// its full range); pivot timestamps are declared as strings because their
/// values travel everywhere in arrow-json text form, and the exact pivot type
/// rides in the field metadata regardless.
fn build_delta_type(col_type: &Type) -> DeltaDataType {
    match col_type {
        Type::Boolean => DeltaDataType::Primitive(PrimitiveType::Boolean),
        Type::Int8 => DeltaDataType::Primitive(PrimitiveType::Byte),
        Type::Int16 | Type::UInt8 => DeltaDataType::Primitive(PrimitiveType::Short),
        Type::Int32 | Type::UInt16 => DeltaDataType::Primitive(PrimitiveType::Integer),
        Type::Int64 | Type::UInt32 => DeltaDataType::Primitive(PrimitiveType::Long),
        Type::UInt64 => DeltaDataType::decimal(20, 0).expect("valid decimal"),
        Type::Int128 => DeltaDataType::decimal(38, 0).expect("valid decimal"),
        Type::Float64 | Type::Decimal => DeltaDataType::Primitive(PrimitiveType::Double),
        Type::Utf8 | Type::Timestamp => DeltaDataType::Primitive(PrimitiveType::String),
        Type::Date => DeltaDataType::Primitive(PrimitiveType::Date),
    }
}

/// The serde name of a pivot type (`"Int64"`), the form the field metadata
/// stores and [`parse_column`] reads back.
fn encode_type_name(col_type: &Type) -> String {
    match serde_json::to_value(col_type).expect("Type serializes") {
        serde_json::Value::String(name) => name,
        other => panic!("Type serializes as a string, got {other}"),
    }
}

/// One declared column as a delta schema field, carrying the exact pivot type
/// in its metadata.
fn build_field(column: &Column) -> StructField {
    StructField::new(
        column.name.clone(),
        build_delta_type(&column.col_type),
        false,
    )
    .with_metadata([(
        COLUMN_TYPE_KEY,
        MetadataValue::String(encode_type_name(&column.col_type)),
    )])
}

/// Read a declared column back from a delta schema field via its pivot type
/// metadata. A field without it was not written by this catalog: an error,
/// not a guess (the delta-native type is a lossy projection).
fn parse_column(field: &StructField) -> Result<Column> {
    let Some(MetadataValue::String(name)) = field.metadata().get(COLUMN_TYPE_KEY) else {
        return Err(Error::MissingColumnType(field.name().clone()));
    };
    let col_type = serde_json::from_value(serde_json::Value::String(name.clone()))?;
    Ok(Column {
        name: field.name().clone(),
        col_type,
    })
}

/// [`parse_column`] for the kernel's schema field type.
fn parse_kernel_column(field: &delta_kernel::schema::StructField) -> Result<Column> {
    let Some(KernelMetadataValue::String(name)) = field.metadata.get(COLUMN_TYPE_KEY) else {
        return Err(Error::MissingColumnType(field.name.clone()));
    };
    let col_type = serde_json::from_value(serde_json::Value::String(name.clone()))?;
    Ok(Column {
        name: field.name.clone(),
        col_type,
    })
}

/// Whether arrow-json encodes a value of `col_type` as a JSON string (rather
/// than a number or boolean). Decides how a delta partition-value string maps
/// back to the JSON shape the partition tuple records.
fn encodes_as_json_string(col_type: &Type) -> bool {
    matches!(col_type, Type::Utf8 | Type::Date | Type::Timestamp)
}

/// One partition value from the tuple's JSON to the string form a delta
/// `add` action records: strings are stored raw, everything else as its JSON
/// text, so the two forms convert back and forth without loss.
fn encode_partition_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// One partition value from its delta string form back to the JSON shape the
/// partition tuple records (the inverse of [`encode_partition_value`], keyed
/// by the column's type).
fn decode_partition_value(col_type: &Type, raw: &str) -> Result<serde_json::Value> {
    if encodes_as_json_string(col_type) {
        Ok(serde_json::Value::String(raw.to_string()))
    } else {
        Ok(serde_json::from_str(raw)?)
    }
}

/// A file's partition tuple as the `partitionValues` map of its `add` action:
/// every partition column present, `None` where the tuple records no value
/// (a file discovered at CREATE TABLE carries no tuple at all).
fn build_partition_values(
    entry: &ManifestEntry,
    partition_by: &[String],
) -> HashMap<String, Option<String>> {
    partition_by
        .iter()
        .map(|column| {
            let value = entry
                .partition
                .as_ref()
                .and_then(|tuple| tuple.get(column))
                .and_then(encode_partition_value);
            (column.clone(), value)
        })
        .collect()
}

/// The delta stats of one file: only the fields the catalog records (the
/// sort-key bounds as min/max), in the standard stats shape any delta reader
/// understands.
#[derive(Serialize, Deserialize, Default)]
struct DeltaStats {
    #[serde(rename = "minValues", default)]
    min_values: serde_json::Map<String, serde_json::Value>,
    #[serde(rename = "maxValues", default)]
    max_values: serde_json::Map<String, serde_json::Value>,
}

/// A file's sort-key bounds as the stats JSON of its `add` action, or `None`
/// when the writer recorded none.
fn encode_stats(sort_bounds: &Option<SortBounds>) -> Result<Option<String>> {
    let Some(bounds) = sort_bounds else {
        return Ok(None);
    };
    let as_map = |value: &serde_json::Value| {
        value
            .as_object()
            .cloned()
            .expect("sort bounds are one-row json objects")
    };
    Ok(Some(serde_json::to_string(&DeltaStats {
        min_values: as_map(&bounds.min),
        max_values: as_map(&bounds.max),
    })?))
}

/// Read a file's sort-key bounds back from its stats JSON: the min/max of
/// every sort column, or `None` when the stats don't cover them all (a file
/// written without bounds). Bounds are a soft pruning aid, so absence is not
/// an error.
fn decode_sort_bounds(stats: Option<&str>, sort_by: &[String]) -> Option<SortBounds> {
    if sort_by.is_empty() {
        return None;
    }
    let stats: DeltaStats = serde_json::from_str(stats?).ok()?;
    let project = |values: &serde_json::Map<String, serde_json::Value>| {
        sort_by
            .iter()
            .map(|column| Some((column.clone(), values.get(column)?.clone())))
            .collect::<Option<serde_json::Map<_, _>>>()
    };
    Some(SortBounds {
        min: serde_json::Value::Object(project(&stats.min_values)?),
        max: serde_json::Value::Object(project(&stats.max_values)?),
    })
}

/// One committed file as the `add` action recording it. `data_change` is
/// `false` when the commit only reorganizes existing rows (a compaction).
fn build_add_action(
    entry: &ManifestEntry,
    partition_by: &[String],
    data_change: bool,
) -> Result<Action> {
    if entry.file.path.is_absolute() {
        return Err(Error::AbsoluteFilePath(
            entry.file.path.as_str().to_string(),
        ));
    }
    Ok(Action::Add(Add {
        path: entry.file.path.as_str().to_string(),
        partition_values: build_partition_values(entry, partition_by),
        size: entry.file.size as i64,
        modification_time: now_millis(),
        data_change,
        stats: encode_stats(&entry.sort_bounds)?,
        ..Default::default()
    }))
}

/// The `remove` tombstone for one committed file a swap takes out. The file's
/// object is deleted later, by [`DeltaLog::maintain`], once the tombstone ages
/// past the retention horizon and no live reader can still reference it.
fn build_remove_action(entry: &ManifestEntry, partition_by: &[String]) -> Result<Action> {
    Ok(Action::Remove(Remove {
        path: entry.file.path.as_str().to_string(),
        data_change: false,
        deletion_timestamp: Some(now_millis()),
        extended_file_metadata: Some(true),
        partition_values: Some(build_partition_values(entry, partition_by)),
        size: Some(entry.file.size as i64),
        ..Default::default()
    }))
}

/// The table's partition columns out of a kernel snapshot. Kernel exposes
/// them only through a transaction (cheap here: for a table without
/// clustering, building one reads nothing).
fn read_partition_columns(snapshot: &SnapshotRef, engine: &dyn Engine) -> Result<Vec<String>> {
    let transaction = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine)?;
    Ok(transaction.logical_partition_columns().to_vec())
}

/// The add-files metadata batch for one kernel commit: one row per file, in
/// the shape [`add_files`](delta_kernel::transaction::Transaction::add_files)
/// expects. Kernel extends these rows into full `add` actions, serializing
/// the stats struct into the action's stats JSON.
fn build_add_files_data(
    entries: &[&ManifestEntry],
    partition_by: &[String],
    sort_columns: &[(String, Type)],
) -> Result<Box<dyn EngineData>> {
    for entry in entries {
        if entry.file.path.is_absolute() {
            return Err(Error::AbsoluteFilePath(
                entry.file.path.as_str().to_string(),
            ));
        }
    }
    let paths: ArrayRef = Arc::new(StringArray::from_iter_values(
        entries.iter().map(|entry| entry.file.path.as_str()),
    ));
    let partition_values = build_partition_values_array(entries, partition_by)?;
    let sizes: ArrayRef = Arc::new(Int64Array::from_iter_values(
        entries.iter().map(|entry| entry.file.size as i64),
    ));
    let modification_time = now_millis();
    let modification_times: ArrayRef = Arc::new(Int64Array::from_iter_values(
        entries.iter().map(|_| modification_time),
    ));
    let stats = build_stats_array(entries, sort_columns)?;
    let schema = delta_kernel::arrow::datatypes::Schema::new(vec![
        Field::new("path", ArrowDataType::Utf8, false),
        Field::new(
            "partitionValues",
            partition_values.data_type().clone(),
            false,
        ),
        Field::new("size", ArrowDataType::Int64, false),
        Field::new("modificationTime", ArrowDataType::Int64, false),
        Field::new("stats", stats.data_type().clone(), true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![paths, partition_values, sizes, modification_times, stats],
    )?;
    Ok(Box::new(ArrowEngineData::new(batch)))
}

/// The `partitionValues` column of an add-files batch: each file's partition
/// tuple as a string-to-string map, every partition column present, null
/// where the tuple records no value (mirroring
/// [`build_partition_values`], in the map layout kernel's add action schema
/// declares).
fn build_partition_values_array(
    entries: &[&ManifestEntry],
    partition_by: &[String],
) -> Result<ArrayRef> {
    let names = MapFieldNames {
        entry: "key_value".to_string(),
        key: "key".to_string(),
        value: "value".to_string(),
    };
    let mut builder = MapBuilder::new(Some(names), StringBuilder::new(), StringBuilder::new());
    for entry in entries {
        for column in partition_by {
            builder.keys().append_value(column);
            let value = entry
                .partition
                .as_ref()
                .and_then(|tuple| tuple.get(column))
                .and_then(encode_partition_value);
            match value {
                Some(value) => builder.values().append_value(value),
                None => builder.values().append_null(),
            }
        }
        builder.append(true)?;
    }
    Ok(Arc::new(builder.finish()))
}

/// The `stats` column of an add-files batch: each file's sort-key bounds as
/// a `{minValues, maxValues}` struct (null for a file without bounds), which
/// kernel's ToJson serializes into the same stats JSON [`encode_stats`]
/// produces on the delta-rs side.
fn build_stats_array(
    entries: &[&ManifestEntry],
    sort_columns: &[(String, Type)],
) -> Result<ArrayRef> {
    let with_bounds = NullBuffer::from(
        entries
            .iter()
            .map(|entry| entry.sort_bounds.is_some())
            .collect::<Vec<bool>>(),
    );
    if sort_columns.is_empty() {
        // Nothing to record: an all-null empty struct, which serializes to
        // no stats at all.
        return Ok(Arc::new(StructArray::new_empty_fields(
            entries.len(),
            Some(NullBuffer::new_null(entries.len())),
        )));
    }
    let min = build_bounds_struct(entries, sort_columns, |bounds| &bounds.min)?;
    let max = build_bounds_struct(entries, sort_columns, |bounds| &bounds.max)?;
    let fields = Fields::from(vec![
        Field::new("minValues", min.data_type().clone(), true),
        Field::new("maxValues", max.data_type().clone(), true),
    ]);
    Ok(Arc::new(StructArray::new(
        fields,
        vec![min, max],
        Some(with_bounds),
    )))
}

/// One side (min or max) of every file's sort-key bounds as a struct of the
/// sort columns.
fn build_bounds_struct(
    entries: &[&ManifestEntry],
    sort_columns: &[(String, Type)],
    select: impl Fn(&SortBounds) -> &serde_json::Value,
) -> Result<ArrayRef> {
    let mut fields = Vec::new();
    let mut children: Vec<ArrayRef> = Vec::new();
    for (name, column_type) in sort_columns {
        let values: Vec<Option<&serde_json::Value>> = entries
            .iter()
            .map(|entry| {
                entry
                    .sort_bounds
                    .as_ref()
                    .and_then(|bounds| select(bounds).get(name))
            })
            .collect();
        let child = build_bound_values(column_type, &values)?;
        fields.push(Field::new(name, child.data_type().clone(), true));
        children.push(child);
    }
    let with_bounds = NullBuffer::from(
        entries
            .iter()
            .map(|entry| entry.sort_bounds.is_some())
            .collect::<Vec<bool>>(),
    );
    Ok(Arc::new(StructArray::new(
        Fields::from(fields),
        children,
        Some(with_bounds),
    )))
}

/// One sort column's bound values as an arrow array whose JSON encoding is
/// the arrow-json form the bounds already carry: numbers stay numbers
/// (signed, unsigned, or float), text stays text. A value whose JSON shape
/// contradicts the declared column type is an error, not a guess.
fn build_bound_values(
    column_type: &Type,
    values: &[Option<&serde_json::Value>],
) -> Result<ArrayRef> {
    fn collect<'a, T>(
        values: &[Option<&'a serde_json::Value>],
        column_type: &Type,
        extract: impl Fn(&'a serde_json::Value) -> Option<T>,
    ) -> Result<Vec<Option<T>>> {
        values
            .iter()
            .map(|value| match value {
                None => Ok(None),
                Some(value) => extract(value)
                    .map(Some)
                    .ok_or_else(|| Error::InvalidSortBound {
                        value: (*value).clone(),
                        column_type: column_type.clone(),
                    }),
            })
            .collect()
    }
    Ok(match column_type {
        Type::Boolean => Arc::new(BooleanArray::from(collect(values, column_type, |v| {
            v.as_bool()
        })?)),
        Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128 => {
            Arc::new(Int64Array::from(collect(values, column_type, |v| {
                v.as_i64()
            })?))
        }
        Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 => {
            Arc::new(UInt64Array::from(collect(values, column_type, |v| {
                v.as_u64()
            })?))
        }
        Type::Float64 | Type::Decimal => {
            Arc::new(Float64Array::from(collect(values, column_type, |v| {
                v.as_f64()
            })?))
        }
        Type::Utf8 | Type::Timestamp | Type::Date => {
            Arc::new(StringArray::from(collect(values, column_type, |v| {
                v.as_str().map(str::to_string)
            })?))
        }
    })
}

/// The named top-level column of a scan-row batch, downcast to its concrete
/// array type. The scan row schema is documented and fixed, so a mismatch is
/// an error, not something to work around.
fn downcast_column<'a, T: 'static>(
    columns: &'a dyn ColumnLookup,
    name: &'static str,
) -> Result<&'a T> {
    columns
        .lookup(name)
        .and_then(|column| column.as_any().downcast_ref::<T>())
        .ok_or(Error::MalformedScanRow(name))
}

/// Name-based column access shared by a scan-row batch and its nested
/// structs.
trait ColumnLookup {
    fn lookup(&self, name: &str) -> Option<&dyn Array>;
}

impl ColumnLookup for RecordBatch {
    fn lookup(&self, name: &str) -> Option<&dyn Array> {
        self.column_by_name(name).map(|column| column.as_ref())
    }
}

impl ColumnLookup for StructArray {
    fn lookup(&self, name: &str) -> Option<&dyn Array> {
        self.column_by_name(name).map(|column| column.as_ref())
    }
}

/// One table's [`TableState`] out of a kernel snapshot: schema fields back
/// to declared columns, the sort spec from the table properties, and every
/// live file (via a metadata-only scan) back to a [`ManifestEntry`]. The
/// twin of [`extract_state`] for the kernel-driven paths.
fn extract_snapshot_state(
    engine: &dyn Engine,
    snapshot: &SnapshotRef,
    partition_by: &[String],
) -> Result<TableState> {
    let columns: Vec<Column> = snapshot
        .schema()
        .fields()
        .map(parse_kernel_column)
        .collect::<Result<_>>()?;
    let types: HashMap<&str, &Type> = columns
        .iter()
        .map(|c| (c.name.as_str(), &c.col_type))
        .collect();
    let sort_by: Vec<String> = match snapshot
        .table_properties()
        .unknown_properties
        .get(SORT_BY_KEY)
    {
        Some(raw) => serde_json::from_str(raw)?,
        None => Vec::new(),
    };

    let mut dated_entries = Vec::new();
    let scan = snapshot.clone().scan_builder().build()?;
    for metadata in scan.scan_metadata(engine)? {
        let (data, selection) = metadata?.scan_files.into_parts();
        let batch = ArrowEngineData::try_from_engine_data(data)?;
        let batch = batch.record_batch();
        let paths: &StringArray = downcast_column(batch, "path")?;
        let sizes: &Int64Array = downcast_column(batch, "size")?;
        let modification_times: &Int64Array = downcast_column(batch, "modificationTime")?;
        let stats: &StringArray = downcast_column(batch, "stats")?;
        let constants: &StructArray = downcast_column(batch, "fileConstantValues")?;
        let partitions: &MapArray = downcast_column(constants, "partitionValues")?;
        for row in 0..batch.num_rows() {
            if !selection.get(row).copied().unwrap_or(true) {
                continue;
            }
            let row_values = partitions.value(row);
            let row_values = row_values
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or(Error::MalformedScanRow("partitionValues"))?;
            let keys: &StringArray = downcast_column(row_values, "key")?;
            let values: &StringArray = downcast_column(row_values, "value")?;
            let recorded: HashMap<&str, &str> = (0..keys.len())
                .filter(|&i| !values.is_null(i))
                .map(|i| (keys.value(i), values.value(i)))
                .collect();
            let mut tuple = serde_json::Map::new();
            for column in partition_by {
                let Some(raw) = recorded.get(column.as_str()) else {
                    continue;
                };
                let col_type = types
                    .get(column.as_str())
                    .expect("partition columns are declared columns");
                tuple.insert(column.clone(), decode_partition_value(col_type, raw)?);
            }
            let raw_stats = (!stats.is_null(row)).then(|| stats.value(row));
            dated_entries.push((
                modification_times.value(row),
                ManifestEntry {
                    file: FileRef {
                        path: ObjectPath::new(paths.value(row)),
                        size: sizes.value(row) as u64,
                    },
                    partition: (!tuple.is_empty()).then_some(serde_json::Value::Object(tuple)),
                    sort_bounds: decode_sort_bounds(raw_stats, &sort_by),
                },
            ));
        }
    }
    // Log replay surfaces the newest commit's files first; present them in
    // commit order instead (oldest first, ties keeping replay order), the
    // order every derived view (the `metadata()` function's file numbering,
    // the flat scan view) presents files in.
    dated_entries.sort_by_key(|(modified, _)| *modified);
    let entries = dated_entries.into_iter().map(|(_, entry)| entry).collect();

    Ok(TableState {
        version: snapshot.version(),
        columns,
        partition_by: partition_by.to_vec(),
        sort_by,
        entries,
    })
}

/// One table's [`TableState`] out of the delta table's current snapshot:
/// schema fields back to declared columns, partition columns and the sort
/// spec from the table metadata, and every live file back to a
/// [`ManifestEntry`].
fn extract_state(table: &DeltaTable) -> Result<TableState> {
    let snapshot = table.snapshot()?;
    let columns: Vec<Column> = snapshot
        .schema()
        .fields()
        .map(parse_column)
        .collect::<Result<_>>()?;
    let types: HashMap<&str, &Type> = columns
        .iter()
        .map(|c| (c.name.as_str(), &c.col_type))
        .collect();
    let metadata = snapshot.metadata();
    let partition_by: Vec<String> = metadata.partition_columns().to_vec();
    let sort_by: Vec<String> = match metadata.configuration().get(SORT_BY_KEY) {
        Some(raw) => serde_json::from_str(raw)?,
        None => Vec::new(),
    };

    let mut dated_entries = Vec::new();
    for file in snapshot.log_data().iter() {
        #[allow(deprecated)] // the Add action is exactly the record we persist
        let add = file.add_action();
        let mut tuple = serde_json::Map::new();
        for column in &partition_by {
            let Some(Some(raw)) = add.partition_values.get(column) else {
                continue;
            };
            let col_type = types
                .get(column.as_str())
                .expect("partition columns are declared columns");
            tuple.insert(column.clone(), decode_partition_value(col_type, raw)?);
        }
        dated_entries.push((
            add.modification_time,
            ManifestEntry {
                file: FileRef {
                    path: ObjectPath::new(add.path),
                    size: add.size as u64,
                },
                partition: (!tuple.is_empty()).then_some(serde_json::Value::Object(tuple)),
                sort_bounds: decode_sort_bounds(add.stats.as_deref(), &sort_by),
            },
        ));
    }
    // Log replay surfaces the newest commit's files first; present them in
    // commit order instead (oldest first, ties keeping replay order), the
    // order every derived view (the `metadata()` function's file numbering,
    // the flat scan view) presents files in.
    dated_entries.sort_by_key(|(modified, _)| *modified);
    let entries = dated_entries.into_iter().map(|(_, entry)| entry).collect();

    Ok(TableState {
        version: snapshot.version(),
        columns,
        partition_by,
        sort_by,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the mapping of kernel's "no delta log here" errors (matched by
    /// message shape in [`is_missing_log`]) onto [`Error::MissingLog`].
    #[test]
    fn open_without_a_delta_log_reports_missing_log() {
        let dir = std::env::temp_dir().join(format!("pivot-missing-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let uri = url::Url::from_directory_path(&dir).unwrap().to_string();

        let result = DeltaLog::open(uri, HashMap::new(), "events");

        assert!(matches!(result, Err(Error::MissingLog(name)) if name == "events"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
