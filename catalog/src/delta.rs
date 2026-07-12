//! The catalog's durable table format: one Delta Lake transaction log per
//! table.
//!
//! The `_delta_log/` under a table's data location is the source of truth for
//! its declared schema (each field also carries the exact pivot type in its
//! metadata), its partition and sort specs, and its committed data files with
//! their partition tuples and sort-key bounds. The background catalog sync
//! loads the committed state here and materializes it as the in-memory
//! [`TableState`] everything downstream (snapshots, bindings, pruning) reads.
//!
//! The read paths go through delta-kernel-rs: each table shares one
//! [`DeltaLog`] handle whose kernel snapshot advances incrementally (a
//! refresh reads only the log entries past the version already held), and
//! kernel's API is synchronous, so the catalog calls it directly — the kernel
//! engine drives its object-store IO on one shared background thread. The one
//! write is [`DeltaLog::create`], the `CREATE TABLE` commit of the log's
//! first version: kernel cannot create a table, so that commit stays on
//! delta-rs (async, run on one small dedicated tokio runtime), an atomic
//! "create if absent" that fails if a concurrent creator won. The data write
//! paths (ingest appends, compaction swaps, log maintenance) are not
//! implemented yet; the catalog surfaces them as disabled.
//!
//! Tables written by an external Delta writer load the same way; delta
//! features the positional scan layer cannot honor are rejected at load with
//! an explicit error rather than serving wrong rows: deletion vectors (a scan
//! would resurrect deleted rows), column mapping (parquet columns are read
//! positionally by the declared schema), and absolute/URI file paths (a
//! [`FileRef`] path resolves under the table's location).

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::schema::MetadataValue as KernelMetadataValue;
use delta_kernel::snapshot::SnapshotRef;
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::{Engine, Snapshot};
use delta_kernel_default_engine::DefaultEngine;
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::storage::store_from_url_opts;
use deltalake_core::DeltaTableError;
use deltalake_core::kernel::transaction::TransactionError;
use deltalake_core::kernel::{
    Action, Add, DataType as DeltaDataType, MetadataValue, PrimitiveType, StructField,
};
use deltalake_core::protocol::SaveMode;
use planner::catalog::Column;
use planner::types::Type;

use crate::manifest::{ManifestEntry, TableState};
use crate::store::{FileRef, ObjectPath, StoreConfig};

/// Field-metadata key a schema field may store its exact pivot type under
/// (a table written by this catalog does). The Delta-native field type is a
/// lossy projection (Delta has no unsigned integers); when this key is
/// present it is what a load trusts.
const COLUMN_TYPE_KEY: &str = "pivot.type";
/// Table-configuration key holding the sort spec as a JSON array of column
/// names. Delta has no native sort spec, so it rides in the table properties;
/// a table without it is unsorted.
const SORT_BY_KEY: &str = "pivot.sortBy";
/// How long a superseded log version stays readable and how long a removed
/// data file stays undeleted after its tombstone, as recorded in a created
/// table's properties for the (future) maintenance sweep to enforce. Every
/// statement resolves the latest snapshot and finishes in well under a
/// second, so five minutes is ample headroom for a reader.
const RETENTION: std::time::Duration = std::time::Duration::from_secs(5 * 60);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Delta(#[from] DeltaTableError),
    #[error(transparent)]
    Kernel(#[from] delta_kernel::Error),
    #[error("delta metadata json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no delta log at `{0}` (the catalog index records a table there)")]
    MissingLog(url::Url),
    #[error("column `{column}` has delta type `{delta_type}`, which maps to no pivot type")]
    UnsupportedColumnType { column: String, delta_type: String },
    #[error("column `{0}` has a `{COLUMN_TYPE_KEY}` metadata value that is not a string")]
    MalformedColumnType(String),
    #[error("the table uses delta column mapping (`{0:?}`), which is not supported")]
    ColumnMappingUnsupported(ColumnMappingMode),
    #[error(
        "data file `{0}` carries a deletion vector, which is not supported (a scan would return deleted rows)"
    )]
    DeletionVectorUnsupported(String),
    #[error(
        "data file `{0}` has an absolute path; only paths relative to the table's location are supported"
    )]
    AbsoluteFilePath(String),
    #[error("data file `{path}` records invalid size {size}")]
    InvalidFileSize { path: String, size: i64 },
    #[error("delta table already exists")]
    TableExists,
    #[error("delta log task panicked")]
    TaskPanicked,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// One table's delta transaction log, shared (behind an `Arc`) by every
/// in-memory copy of the table. The cached kernel snapshot only ever advances
/// to newer versions; each caller keeps its own [`TableState`] and reconciles
/// against the latest on refresh.
pub(crate) struct DeltaLog {
    /// The table's root URL, kept to rebuild a snapshot from scratch when the
    /// incremental walk can't (external log cleanup past the cursor).
    url: url::Url,
    /// The kernel engine driving this table's log IO.
    engine: Arc<DefaultEngine<TokioBackgroundExecutor>>,
    /// The latest kernel snapshot seen.
    snapshot: Mutex<SnapshotRef>,
    /// The table's partition columns, immutable after CREATE TABLE.
    partition_by: Vec<String>,
}

impl DeltaLog {
    /// Open the delta table at `table_url` (within the store `config`
    /// describes), at its latest version. A location with no delta log is
    /// [`Error::MissingLog`] (a table the catalog index records must have
    /// one).
    pub(crate) fn open(config: &StoreConfig, table_url: url::Url) -> Result<(Self, TableState)> {
        let engine = build_kernel_engine(&table_url, &config.options)?;
        let snapshot = Snapshot::builder_for(table_url.clone())
            .build(engine.as_ref())
            .map_err(|e| {
                if is_missing_log(&e) {
                    Error::MissingLog(table_url.clone())
                } else {
                    Error::Kernel(e)
                }
            })?;
        let partition_by = read_partition_columns(&snapshot, engine.as_ref())?;
        let state = extract_snapshot_state(engine.as_ref(), &snapshot, &partition_by)?;
        Ok((
            Self {
                url: table_url,
                engine,
                snapshot: Mutex::new(snapshot),
                partition_by,
            },
            state,
        ))
    }

    /// Create a brand-new delta table at `target` recording the declared
    /// schema, the partition/sort specs, and the already-written initial
    /// files, and return its live log + first committed state. Fails with
    /// [`Error::TableExists`] if a delta log already exists there (including
    /// a concurrent creator winning the version-0 commit). The one delta-rs
    /// path: kernel cannot create a table.
    pub(crate) fn create(
        config: &StoreConfig,
        table_url: url::Url,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        entries: Vec<ManifestEntry>,
    ) -> Result<(Self, TableState)> {
        ensure_store_handlers_registered();
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
            // Superseded log versions only need to outlive a running
            // statement; the (future) maintenance sweep enforces this horizon.
            (
                "delta.logRetentionDuration".to_string(),
                Some(format!("interval {} seconds", RETENTION.as_secs())),
            ),
            // Age tombstones out of snapshots/checkpoints on the same horizon
            // their files would be vacuumed on.
            (
                "delta.deletedFileRetentionDuration".to_string(),
                Some(format!("interval {} seconds", RETENTION.as_secs())),
            ),
        ];
        let uri = table_url.to_string();
        let storage_options = config.options.clone();
        run_blocking(async move {
            deltalake_core::operations::create::CreateBuilder::new()
                .with_location(uri)
                .with_storage_options(storage_options)
                .with_columns(fields)
                .with_partition_columns(partition_by)
                .with_configuration(configuration)
                .with_raise_if_key_not_exists(false)
                .with_save_mode(SaveMode::ErrorIfExists)
                .with_actions(adds)
                .await
        })?
        .map_err(map_create_error)?;
        Self::open(config, table_url)
    }

    /// The latest committed state, but only if it is newer than `since`: the
    /// background refresh's reload. The cached snapshot advances
    /// incrementally (only log entries past the version already held are
    /// read), so the common tick (no new commit) is one small log listing.
    /// Returns `Ok(None)` when `since` is already current, without paying the
    /// per-file state extraction.
    pub(crate) fn load_after(&self, since: u64) -> Result<Option<TableState>> {
        let snapshot = self.refresh_snapshot()?;
        if snapshot.version() <= since {
            return Ok(None);
        }
        extract_snapshot_state(self.engine.as_ref(), &snapshot, &self.partition_by).map(Some)
    }

    /// Advance the cached kernel snapshot to the latest committed version and
    /// return it. Incremental first; when the incremental walk fails (the
    /// cursor's log entries were cleaned up past a checkpoint by an external
    /// writer), rebuild from scratch instead of erroring every tick until
    /// restart. Concurrent refreshers race safely: the cache keeps whichever
    /// copy saw the newest version.
    fn refresh_snapshot(&self) -> Result<SnapshotRef> {
        let current = self.snapshot.lock().unwrap().clone();
        let latest = match Snapshot::builder_from(current).build(self.engine.as_ref()) {
            Ok(latest) => latest,
            Err(_) => Snapshot::builder_for(self.url.clone()).build(self.engine.as_ref())?,
        };
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
}

/// The background executor every table's kernel engine drives its IO on:
/// kernel's API is synchronous, and this owns the one thread that runs its
/// object-store futures.
fn kernel_executor() -> Arc<TokioBackgroundExecutor> {
    static EXECUTOR: OnceLock<Arc<TokioBackgroundExecutor>> = OnceLock::new();
    EXECUTOR
        .get_or_init(|| Arc::new(TokioBackgroundExecutor::new()))
        .clone()
}

/// The kernel engine for the table at `url`, its object store built from the
/// same storage options the delta-rs create opens with.
fn build_kernel_engine(
    url: &url::Url,
    storage_options: &HashMap<String, String>,
) -> Result<Arc<DefaultEngine<TokioBackgroundExecutor>>> {
    let store = store_from_url_opts(url, storage_options.clone())?;
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

/// The table's partition columns out of a kernel snapshot.
fn read_partition_columns(snapshot: &SnapshotRef, engine: &dyn Engine) -> Result<Vec<String>> {
    // A workaround for a gap in kernel's read API: `Snapshot` has no
    // partition-columns accessor (the underlying metadata is crate-internal),
    // and the only public route is `Transaction::logical_partition_columns`.
    // So a throwaway transaction is built purely to read them and dropped,
    // never committed; the committer only satisfies the constructor's
    // signature (`FileSystemCommitter` is the one committer kernel ships) and
    // does no work. Building the transaction is cheap: it reads nothing for a
    // table like ours. If a later kernel exposes the columns on `Snapshot`,
    // this collapses to that one call.
    let transaction = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine)?;
    Ok(transaction.logical_partition_columns().to_vec())
}

/// One table's [`TableState`] out of a kernel snapshot: schema fields back to
/// declared columns, the sort spec from the table properties, and every live
/// file (via a metadata-only scan) back to a [`ManifestEntry`]. Rejects the
/// delta features the positional scan layer cannot honor.
fn extract_snapshot_state(
    engine: &dyn Engine,
    snapshot: &SnapshotRef,
    partition_by: &[String],
) -> Result<TableState> {
    // Column mapping renames the physical parquet columns away from the
    // declared field names; the positional scan cannot follow it.
    if let Some(mode) = snapshot.table_properties().column_mapping_mode
        && mode != ColumnMappingMode::None
    {
        return Err(Error::ColumnMappingUnsupported(mode));
    }
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
        let metadata = metadata?;
        // Kernel's visitor turns the columnar scan rows into plain per-file
        // structs.
        let scan_files = metadata
            .visit_scan_files(Vec::new(), |files: &mut Vec<ScanFile>, file| {
                files.push(file)
            })?;
        for file in scan_files {
            if file.dv_info.has_vector() {
                return Err(Error::DeletionVectorUnsupported(file.path));
            }
            // The Delta spec allows an add path to be an absolute URI (shallow
            // clones, converted tables); a FileRef path only resolves under
            // the table's location, so those tables are rejected, not
            // mis-read.
            if file.path.starts_with('/') || file.path.contains("://") {
                return Err(Error::AbsoluteFilePath(file.path));
            }
            let size = u64::try_from(file.size).map_err(|_| Error::InvalidFileSize {
                path: file.path.clone(),
                size: file.size,
            })?;
            let mut tuple = serde_json::Map::new();
            for column in partition_by {
                let Some(raw) = file.partition_values.get(column) else {
                    continue;
                };
                let col_type = types
                    .get(column.as_str())
                    .expect("partition columns are declared columns");
                if let Some(value) = decode_partition_value(col_type, raw) {
                    tuple.insert(column.clone(), value);
                }
            }
            dated_entries.push((
                file.modification_time,
                ManifestEntry {
                    file: FileRef {
                        path: ObjectPath::new(file.path),
                        size,
                    },
                    partition: (!tuple.is_empty()).then_some(serde_json::Value::Object(tuple)),
                    // Not decoded from the file's delta stats: nothing reads
                    // sort bounds while the write path that records them is
                    // disabled.
                    sort_bounds: None,
                },
            ));
        }
    }
    // Log replay surfaces the newest commit's files first; present them in
    // commit order instead (oldest first, ties keeping replay order), the
    // order every derived view presents files in.
    dated_entries.sort_by_key(|(modified, _)| *modified);
    let entries = dated_entries.into_iter().map(|(_, entry)| entry).collect();

    Ok(TableState {
        version: snapshot.version(),
        columns,
        partition_by: partition_by.to_vec(),
        sort_by,
        entries,
        // Footers are derived data the log doesn't record; the catalog
        // attaches them before installing the state.
        files: Vec::new(),
    })
}

/// Read a declared column back from a kernel schema field. The exact pivot
/// type in the field's metadata wins when present (a table this catalog
/// wrote); a field without it is mapped from its Delta-native type, erroring
/// on a type pivot has no equivalent for. A `pivot.type` value that is not a
/// string is malformed metadata, an error rather than a silent fall back to
/// the lossy native mapping.
fn parse_kernel_column(field: &delta_kernel::schema::StructField) -> Result<Column> {
    let col_type = match field.metadata.get(COLUMN_TYPE_KEY) {
        Some(KernelMetadataValue::String(name)) => {
            serde_json::from_value(serde_json::Value::String(name.clone()))?
        }
        Some(_) => return Err(Error::MalformedColumnType(field.name.clone())),
        None => kernel_native_column_type(field)?,
    };
    Ok(Column {
        name: field.name.clone(),
        col_type,
    })
}

/// The pivot type of a kernel field written by an external writer (no pivot
/// type metadata to trust), from its Delta-native type. Delta timestamps are
/// rejected: they are microsecond int64 in the parquet files, while pivot's
/// `Timestamp` scans second-unit int64, so mapping them would return values a
/// million times too large.
fn kernel_native_column_type(field: &delta_kernel::schema::StructField) -> Result<Type> {
    use delta_kernel::schema::{DataType as KernelDataType, PrimitiveType as KernelPrimitiveType};
    let unsupported = || Error::UnsupportedColumnType {
        column: field.name.clone(),
        delta_type: field.data_type().to_string(),
    };
    let KernelDataType::Primitive(primitive) = field.data_type() else {
        return Err(unsupported());
    };
    Ok(match primitive {
        KernelPrimitiveType::Boolean => Type::Boolean,
        KernelPrimitiveType::Byte => Type::Int8,
        KernelPrimitiveType::Short => Type::Int16,
        KernelPrimitiveType::Integer => Type::Int32,
        KernelPrimitiveType::Long => Type::Int64,
        KernelPrimitiveType::Float => Type::Float32,
        KernelPrimitiveType::Double => Type::Float64,
        KernelPrimitiveType::String => Type::Utf8,
        KernelPrimitiveType::Date => Type::Date,
        _ => return Err(unsupported()),
    })
}

/// One partition value from its delta string form to the JSON shape the
/// partition tuple records, or `None` when the delta form cannot faithfully
/// reproduce that shape.
///
/// The recorded shape must equal what arrow-json emits for the type's
/// physical arrow form: partition pruning compares the tuple against a query
/// constant encoded through arrow-json (see `scalar_to_json` in
/// `catalog::binding`), and [`maybe_matches_partition`] drops a file whose
/// *recorded* value differs from the constant. So recording a value in the
/// wrong spelling doesn't just miss the prune, it wrongly prunes every
/// matching file (a query silently returning zero rows), while recording
/// nothing is always safe: absence keeps the file, costing only the
/// optimization. That is why a type whose spelling can't be reproduced maps
/// to `None` rather than a best guess.
///
/// [`maybe_matches_partition`]: crate::manifest::ManifestEntry::maybe_matches_partition
fn decode_partition_value(col_type: &Type, raw: &str) -> Option<serde_json::Value> {
    match col_type {
        // The Delta protocol spells a timestamp partition value
        // `1970-01-01 00:00:00.000000` (space-separated, micros) while
        // arrow-json emits `1970-01-01T00:00:00`; the two never compare
        // equal, so the value is left unrecorded. The file is then always
        // kept, and the row-group min/max stats prune (footers are already
        // materialized) still trims the scan.
        Type::Timestamp => None,
        // Delta and arrow-json spell strings and dates identically.
        Type::Utf8 | Type::Date => Some(serde_json::Value::String(raw.to_string())),
        // Numbers and booleans parse as JSON; delta's non-JSON float
        // spellings (`NaN`, `Infinity`) are left unrecorded.
        _ => serde_json::from_str(raw).ok(),
    }
}

// --- the CREATE TABLE commit, the one delta-rs path -------------------------

/// Register delta-rs's S3 and GCS object-store factories, exactly once, before
/// its create builder resolves the URL scheme's factory. Registered here at
/// runtime: the `deltalake` meta crate would do it in a pre-main constructor,
/// which allocates before jemalloc is initialized and aborts.
fn ensure_store_handlers_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        deltalake_aws::register_handlers(None);
        deltalake_gcp::register_handlers(None);
    });
}

/// The dedicated runtime the delta-rs create commit runs on.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
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
/// safe to call from any thread, including one inside another runtime. A
/// panic inside the future surfaces as [`Error::TaskPanicked`] rather than
/// panicking the calling catalog thread.
fn run_blocking<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) -> Result<T> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    runtime().spawn(async move {
        let _ = sender.send(future.await);
    });
    receiver.recv().map_err(|_| Error::TaskPanicked)
}

/// Whether a commit failure means "another writer created this version first",
/// as opposed to a real error.
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
        Type::Float32 => DeltaDataType::Primitive(PrimitiveType::Float),
        Type::Float64 | Type::Decimal => DeltaDataType::Primitive(PrimitiveType::Double),
        Type::Utf8 | Type::Timestamp => DeltaDataType::Primitive(PrimitiveType::String),
        Type::Date => DeltaDataType::Primitive(PrimitiveType::Date),
    }
}

/// The serde name of a pivot type (`"Int64"`), the form the field metadata
/// stores and [`parse_kernel_column`] reads back.
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

/// One committed file as the `add` action recording it.
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
        // Sort bounds (delta stats) aren't recorded: CREATE TABLE's discovered
        // files never carry them, and the write path that would is disabled.
        ..Default::default()
    }))
}
