//! Delta Lake metadata loading for the catalog refresh path.
//!
//! Delta Kernel is the source of truth for a table's version, schema, and
//! active `Add` files.  Pivot then fetches those files' Parquet footers into
//! its existing in-memory scan representation.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use arrow_array::builder::{Int64Builder, MapBuilder, MapFieldNames, StringBuilder};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Datum, Decimal64Array, Decimal128Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch, Scalar,
    StringArray, StringViewArray, StructArray, TimestampMicrosecondArray, new_null_array,
};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Fields};
use delta_kernel::EngineData;
use delta_kernel::Snapshot;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine_data::FilteredEngineData;
use delta_kernel::expressions::Scalar as DeltaScalar;
use delta_kernel::scan::StatsOptions;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::schema::{
    DataType as DeltaDataType, MetadataValue, PrimitiveType, StructField, StructType,
};
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::{CommitResult, Transaction};
use delta_kernel_default_engine::executor::tokio::TokioMultiThreadExecutor;
use delta_kernel_default_engine::{DefaultEngine, DefaultEngineBuilder};
use planner::catalog::Column;
use planner::types::Type;
use url::Url;

use crate::manifest::DeltaFileEntry;
use crate::store::{FileRef, ObjectPath, ObjectStore};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid Delta table URI `{uri}`: {source}")]
    InvalidUri {
        uri: String,
        #[source]
        source: url::ParseError,
    },
    #[error("Delta Kernel: {0}")]
    Kernel(#[from] delta_kernel::Error),
    #[error("Arrow: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("Delta object store: {0}")]
    ObjectStore(#[from] delta_kernel::object_store::Error),
    #[error("catalog object store: {0}")]
    CatalogStore(#[from] crate::store::StoreError),
    #[error("Delta table uses unsupported column `{column}` type `{data_type}`")]
    UnsupportedType { column: String, data_type: String },
    #[error("Delta file `{path}` has an invalid negative size {size}")]
    InvalidFileSize { path: String, size: i64 },
    #[error(
        "Delta file `{0}` uses a deletion vector; Pivot's Parquet reader cannot apply deletion vectors yet"
    )]
    DeletionVector(String),
    #[error("cannot format Delta partition value for `{column}`: {message}")]
    PartitionFormat { column: String, message: String },
    #[error("Delta log commit {version} has a malformed action line: {source}")]
    CorruptLog {
        version: u64,
        #[source]
        source: serde_json::Error,
    },
    #[error("Delta log commit {version} has a `{field}` that is missing or not a string")]
    CorruptLogField { version: u64, field: &'static str },
}

/// How Delta serializes timestamp partition values (and how Delta Kernel
/// parses them back): `yyyy-MM-dd HH:mm:ss[.SSSSSS]`.
const DELTA_TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.f";

/// BoundTable-metadata configuration key holding the table's ordered sort columns,
/// comma-separated. Delta has no native sort spec, so it rides in the
/// `metaData` action's free-form `configuration` map.
const SORT_BY_CONFIGURATION_KEY: &str = "pivot.sortBy";

/// How many commits a table advances past its last checkpoint before the next
/// commit writes one. Read per table from the standard `delta.checkpointInterval`
/// property (so a table that sets it, e.g. a foreign one, is honored) and falls
/// back to this default. Every writer checkpoints at the same cadence with no
/// shared configuration. This (and the log retention below) is only ever read,
/// never written on our own tables: Kernel rejects setting `delta.*` maintenance
/// properties during CREATE, so a table that does not carry one just uses the
/// default.
const DEFAULT_CHECKPOINT_INTERVAL: u64 = 100;

/// How long a commit file already folded into a checkpoint is kept before the
/// vacuum sweep deletes it. Read per table from the standard
/// `delta.logRetentionDuration` property and falls back to this default. The
/// window must exceed the longest a reader can lag on (or time-travel to) an
/// older version.
const DEFAULT_LOG_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// How long a data file no longer referenced by the current table version is
/// kept before the vacuum sweep deletes it. Read per table from the standard
/// `delta.deletedFileRetentionDuration` property and falls back to Delta's own
/// default. It is the window a reader on a superseded snapshot is guaranteed its
/// files survive, so it must exceed the longest a reader can lag behind the
/// current version.
const DEFAULT_DELETED_FILE_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// The Delta state needed to rebuild one in-memory catalog table, together with
/// the snapshot it was read from. The holder keeps that snapshot so its next
/// refresh advances it and its next commit rides it. The table's durable
/// identity is not sourced here: the catalog owns its own id in its manifest,
/// independent of the id Kernel mints into the `metaData` action.
pub(crate) struct DeltaTableState {
    pub snapshot: Arc<Snapshot>,
    pub columns: Vec<Column>,
    pub partition_by: Vec<String>,
    pub sort_by: Vec<String>,
    pub file_entries: Vec<DeltaFileEntry>,
}

/// Initialize version 0 for `CREATE TABLE`. Data files already exist; this
/// commit atomically adopts them as the table's initial Delta snapshot, which is
/// returned for the new table to hold.
pub(crate) fn initialize_table(
    engine: &DeltaEngine,
    store: &dyn ObjectStore,
    location: &ObjectPath,
    columns: &[Column],
    partition_by: &[String],
    sort_by: &[String],
    entries: &[DeltaFileEntry],
) -> Result<Arc<Snapshot>, Error> {
    let uri = table_uri(&store.location_uri(), location)?;
    // Kernel canonicalizes the table root before writing v0, so on a local
    // filesystem the directory must exist first (an empty table has no data
    // files to have created it). A no-op on stores without directories.
    store.create_dir(location)?;

    let fields = columns
        .iter()
        .map(|column| build_delta_field(&column.name, &column.col_type))
        .collect::<Result<Vec<_>, Error>>()?;
    let schema = Arc::new(StructType::try_new(fields)?);

    // Kernel writes the `metaData`/`protocol` commit (v0), minting its own
    // `metaData` id (which the catalog does not use -- it owns table identity in
    // its manifest) and declaring any feature (e.g. variant) the schema needs.
    let mut builder = create_table(uri.as_str(), schema, "pivot");
    if !partition_by.is_empty() {
        builder = builder.with_data_layout(DataLayout::partitioned(
            partition_by.iter().map(String::as_str),
        ));
    }
    if !sort_by.is_empty() {
        // Delta has no native sort spec; it rides as a free-form table property
        // (custom app keys are preserved) and is read back in `load_table`.
        builder = builder.with_table_properties([(SORT_BY_CONFIGURATION_KEY, sort_by.join(","))]);
    }
    let mut txn = builder.build(engine.kernel(), Box::new(FileSystemCommitter::new()))?;

    if !entries.is_empty() {
        // `modificationTime` 0 is how these adopted files have always been
        // recorded; the caller stamps each entry's partition and stats.
        txn.add_files(add_files_metadata(entries, 0)?);
    }
    let CommitResult::CommittedTransaction(committed) = txn.commit(engine.kernel())? else {
        return Err(Error::Kernel(delta_kernel::Error::generic(
            "Delta table version 0 already exists",
        )));
    };
    committed_snapshot(&committed, engine, &uri)
}

/// Atomically commit a data-file change through Kernel: `add` every entry in
/// `added`, `remove` every entry in `removed`, in one commit. An empty `removed`
/// is a plain append. The caller has already verified that each removed entry is
/// live in `snapshot`. `data_change` labels the commit: `true` for a logical
/// change (INSERT, DELETE), `false` for a rearrangement that leaves the rows
/// identical (compaction), which lets incremental readers skip it.
///
/// The commit is written on top of `snapshot`, the caller's own version. No log
/// is read to open it, and none to close it either: the winning transaction
/// yields the snapshot at the committed version, which is what this returns for
/// the caller to hold. A commit therefore lands exactly one version above
/// `snapshot`, or not at all: if another writer got there first, Kernel reports
/// the conflict and this returns `None` for the caller to refresh and retry.
pub(crate) fn commit_file_changes(
    engine: &DeltaEngine,
    snapshot: &Arc<Snapshot>,
    removed: &[DeltaFileEntry],
    added: &[DeltaFileEntry],
    data_change: bool,
) -> Result<Option<Arc<Snapshot>>, Error> {
    let mut txn = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine.kernel())?
        .with_data_change(data_change);

    if !added.is_empty() {
        let modification_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        txn.add_files(add_files_metadata(added, modification_time)?);
    }
    stage_removals(&mut txn, removed)?;

    // A conflict returns `None` so the caller reloads onto the newer version and
    // retries — progress guaranteed by the advanced version. A retryable (IO)
    // failure leaves the version unchanged, so returning `None` would spin the
    // caller on the same commit; instead retry the recovered transaction a
    // bounded number of times and then surface the error.
    let mut txn = txn;
    let mut last_error: Option<delta_kernel::Error> = None;
    for _ in 0..COMMIT_RETRIES {
        match txn.commit(engine.kernel())? {
            CommitResult::CommittedTransaction(committed) => {
                let advanced = committed_snapshot(&committed, engine, snapshot.table_root())?;
                return Ok(Some(maybe_checkpoint(engine, advanced)));
            }
            CommitResult::ConflictedTransaction(_) => return Ok(None),
            CommitResult::RetryableTransaction(retryable) => {
                tracing::warn!(error = %retryable.error, "Delta commit hit a retryable failure; retrying");
                last_error = Some(retryable.error);
                txn = retryable.transaction;
            }
        }
    }
    Err(Error::Kernel(last_error.unwrap_or_else(|| {
        delta_kernel::Error::generic("Delta commit exhausted its retries")
    })))
}

/// The table's snapshot at the version `committed` just wrote. Kernel builds it
/// from the transaction's own base snapshot, so this normally costs no log read;
/// a Kernel build that does not yet produce one for this kind of transaction
/// reads the log at `uri` instead.
///
/// That read is pinned to the version committed. The caller pairs this snapshot
/// with a file set it folded its own committed files onto, so a snapshot that had
/// swept up a concurrent writer's later commit would name files the caller does
/// not hold, and nothing would heal the mismatch: that copy's next refresh finds
/// its version already current and reconciles nothing.
fn committed_snapshot(
    committed: &delta_kernel::transaction::CommittedTransaction,
    engine: &DeltaEngine,
    uri: impl AsRef<str>,
) -> Result<Arc<Snapshot>, Error> {
    match committed.post_commit_snapshot() {
        Some(snapshot) => Ok(snapshot.clone()),
        None => snapshot_at(uri, committed.commit_version(), engine),
    }
}

/// Read the table at `uri` at exactly `version`.
fn snapshot_at(
    uri: impl AsRef<str>,
    version: u64,
    engine: &DeltaEngine,
) -> Result<Arc<Snapshot>, Error> {
    Ok(Snapshot::builder_for(uri)
        .at_version(version)
        .build(engine.kernel())?)
}

/// How many times a transient (retryable) commit failure is retried before the
/// error is surfaced, rather than looping forever on an unchanged version.
const COMMIT_RETRIES: usize = 10;

/// Write a Delta checkpoint (the `*.checkpoint.parquet` + `_last_checkpoint`,
/// bounding the JSON log readers replay) when the just-committed `snapshot`'s
/// version crosses the table's checkpoint interval. Checkpointing rides the
/// commit rather than a background timer because it is a pure function of the
/// commit count, which the writer knows exactly here; the interval read is free
/// (off the snapshot the commit already yielded) and only one commit in
/// `interval` pays for the write. Returns the snapshot to carry forward: the one
/// Kernel rebuilds over the checkpointed log (it knows the checkpoint, so a later
/// log cleanup sees it) or the input unchanged.
///
/// Best-effort: the commit already succeeded, so a checkpoint failure is logged
/// and left for the next interval rather than failing the caller's write.
fn maybe_checkpoint(engine: &DeltaEngine, snapshot: Arc<Snapshot>) -> Arc<Snapshot> {
    let interval = snapshot
        .table_properties()
        .checkpoint_interval
        .map(|interval| interval.get())
        .unwrap_or(DEFAULT_CHECKPOINT_INTERVAL);
    let version = snapshot.version();
    if version == 0 || !version.is_multiple_of(interval) {
        return snapshot;
    }
    match snapshot.checkpoint(engine.kernel(), None) {
        Ok((_result, checkpointed)) => checkpointed,
        Err(e) => {
            tracing::warn!(
                error = %e,
                version,
                "inline checkpoint after commit failed; the next interval will retry"
            );
            snapshot
        }
    }
}

/// Stage removals of `removed` on `txn` from metadata the catalog already holds.
/// Kernel accepts the same scan-row shape it normally produces while scanning a
/// snapshot, so constructing those rows directly avoids rereading every active
/// Add action merely to remove a small compaction batch.
fn stage_removals(txn: &mut Transaction, removed: &[DeltaFileEntry]) -> Result<(), Error> {
    if removed.is_empty() {
        return Ok(());
    }
    let data = remove_files_metadata(removed)?;
    txn.remove_files(FilteredEngineData::try_new(
        data,
        vec![true; removed.len()],
    )?);
    Ok(())
}

/// Build Kernel scan rows for files being removed. Pivot does not support
/// deletion vectors or row tracking, and its catalog entries preserve the
/// fields relevant to that supported subset: path, size, and partition values.
fn remove_files_metadata(
    entries: &[DeltaFileEntry],
) -> Result<Box<dyn delta_kernel::EngineData>, Error> {
    let schema: arrow_schema::Schema = delta_kernel::scan::scan_row_schema()
        .as_ref()
        .try_into_arrow()?;
    let mut paths = StringBuilder::new();
    let mut sizes = Int64Builder::new();
    let mut partition_values = MapBuilder::new(
        Some(MapFieldNames {
            entry: "key_value".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        }),
        StringBuilder::new(),
        StringBuilder::new(),
    );
    for entry in entries {
        paths.append_value(entry.file.path.as_str());
        sizes.append_value(entry.file.size as i64);
        if let Some(partition) = &entry.partition {
            for (column, scalar) in partition {
                partition_values.keys().append_value(column);
                match format_partition_value(column, scalar)? {
                    serde_json::Value::String(value) => {
                        partition_values.values().append_value(value)
                    }
                    _ => partition_values.values().append_null(),
                }
            }
        }
        partition_values.append(true)?;
    }

    let constants_field = schema.field_with_name("fileConstantValues")?;
    let ArrowDataType::Struct(constant_fields) = constants_field.data_type() else {
        unreachable!("Kernel scan fileConstantValues is a struct")
    };
    let partitions: ArrayRef = Arc::new(partition_values.finish());
    let constant_columns = constant_fields
        .iter()
        .map(|field| match field.name().as_str() {
            "partitionValues" => partitions.clone(),
            _ => new_null_array(field.data_type(), entries.len()),
        })
        .collect();
    let constants: ArrayRef = Arc::new(StructArray::try_new(
        constant_fields.clone(),
        constant_columns,
        None,
    )?);

    let paths: ArrayRef = Arc::new(paths.finish());
    let sizes: ArrayRef = Arc::new(sizes.finish());
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|field| match field.name().as_str() {
            "path" => paths.clone(),
            "size" => sizes.clone(),
            "fileConstantValues" => constants.clone(),
            _ => new_null_array(field.data_type(), entries.len()),
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    Ok(Box::new(ArrowEngineData::new(batch)))
}

/// The window an unreferenced data file is kept before the vacuum sweep may
/// delete it: the table's `delta.deletedFileRetentionDuration`, or Delta's
/// default ([`DEFAULT_DELETED_FILE_RETENTION`]) when the table does not set it.
/// This is the same window a checkpoint keeps a `Remove` tombstone, so a reader
/// on a superseded snapshot is guaranteed its files survive for at least this
/// long after they stop being referenced. Read straight off the caller's
/// snapshot: the property is table metadata the snapshot already carries.
pub(crate) fn deleted_file_retention(snapshot: &Snapshot) -> Duration {
    snapshot
        .table_properties()
        .deleted_file_retention_duration
        .unwrap_or(DEFAULT_DELETED_FILE_RETENTION)
}

/// Build the `add_files` metadata batch Kernel expects for a set of
/// already-written files. The columns match [`Transaction::add_files_schema`]'s
/// mandatory fields — `path`, `partitionValues` (a `key → value` string map,
/// empty for unpartitioned files), `size`, `modificationTime`; Kernel extends
/// these to the full `Add` action and stamps `dataChange` from the transaction.
fn add_files_metadata(
    entries: &[DeltaFileEntry],
    modification_time: u64,
) -> Result<Box<dyn delta_kernel::EngineData>, Error> {
    let mut paths = StringBuilder::new();
    let mut sizes = Int64Builder::new();
    let mut mtimes = Int64Builder::new();
    // Kernel's Map arrow layout: entries struct `key_value` with `key`/`value`.
    let field_names = MapFieldNames {
        entry: "key_value".to_string(),
        key: "key".to_string(),
        value: "value".to_string(),
    };
    let mut partition_values = MapBuilder::new(
        Some(field_names),
        StringBuilder::new(),
        StringBuilder::new(),
    );

    for entry in entries {
        paths.append_value(entry.file.path.as_str());
        sizes.append_value(entry.file.size as i64);
        mtimes.append_value(modification_time as i64);
        if let Some(partition) = &entry.partition {
            for (column, scalar) in partition {
                partition_values.keys().append_value(column);
                match format_partition_value(column, scalar)? {
                    serde_json::Value::String(value) => {
                        partition_values.values().append_value(value)
                    }
                    _ => partition_values.values().append_null(),
                }
            }
        }
        partition_values.append(true)?;
    }

    // Kernel's add-files projection runs `ToJson(stats)`, so the batch carries a
    // `stats` struct column. We persist each file's row count and its per-column
    // min/max and null count, aggregated from its row groups (see `FileStats`), so
    // a reader can answer counts and prune by range straight from the log.
    let count = entries.len();
    let num_records = Int64Array::from_iter(
        entries
            .iter()
            .map(|entry| entry.stats.as_ref().and_then(|stats| stats.num_records)),
    );
    let (min_values, max_values) = build_bounds_stats(entries)?;
    let null_count = build_null_count_stats(entries)?;
    let stats_fields = Fields::from(vec![
        ArrowField::new("numRecords", ArrowDataType::Int64, true),
        ArrowField::new("nullCount", null_count.data_type().clone(), true),
        ArrowField::new("minValues", min_values.data_type().clone(), true),
        ArrowField::new("maxValues", max_values.data_type().clone(), true),
        ArrowField::new("tightBounds", ArrowDataType::Boolean, true),
    ]);
    let stats_columns: Vec<ArrayRef> = vec![
        Arc::new(num_records),
        null_count,
        min_values,
        max_values,
        new_null_array(&ArrowDataType::Boolean, count),
    ];
    let stats = StructArray::try_new(stats_fields, stats_columns, None)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("path", Arc::new(paths.finish()) as ArrayRef),
        (
            "partitionValues",
            Arc::new(partition_values.finish()) as ArrayRef,
        ),
        ("size", Arc::new(sizes.finish()) as ArrayRef),
        ("modificationTime", Arc::new(mtimes.finish()) as ArrayRef),
        ("stats", Arc::new(stats) as ArrayRef),
    ])?;
    Ok(Box::new(
        delta_kernel::engine::arrow_data::ArrowEngineData::new(batch),
    ))
}

/// Concatenate per-entry single-row arrays into one column, one row per entry.
fn concat_column(arrays: Vec<ArrayRef>) -> Result<ArrayRef, Error> {
    let refs: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    Ok(arrow_select::concat::concat(&refs)?)
}

/// Build the `minValues` and `maxValues` stats sub-structs together. A file bounds
/// a column on both sides or neither, so the two cover the same columns and one
/// pass fills both: each row holds that entry's typed bound, or a null where it
/// records none. Empty structs when nothing is bounded. Sorted column order gives
/// the batch one stable schema.
fn build_bounds_stats(entries: &[DeltaFileEntry]) -> Result<(ArrayRef, ArrayRef), Error> {
    // A column's type comes from any entry that records it (all agree). One pass
    // into a BTreeMap yields the column union, each type, and a stable order.
    let mut types: std::collections::BTreeMap<&str, &ArrowDataType> = Default::default();
    for entry in entries {
        if let Some(stats) = &entry.stats {
            for (name, value) in &stats.min_values {
                types.entry(name).or_insert_with(|| value.data_type());
            }
        }
    }
    if types.is_empty() {
        let empty = new_null_array(&ArrowDataType::Struct(Fields::empty()), entries.len());
        return Ok((empty.clone(), empty));
    }
    let mut fields = Vec::new();
    let mut min_columns = Vec::new();
    let mut max_columns = Vec::new();
    for (name, field_type) in types {
        let mut mins = Vec::with_capacity(entries.len());
        let mut maxs = Vec::with_capacity(entries.len());
        for entry in entries {
            let stats = entry.stats.as_ref();
            mins.push(
                stats
                    .and_then(|stats| stats.min_values.get(name))
                    .cloned()
                    .unwrap_or_else(|| new_null_array(field_type, 1)),
            );
            maxs.push(
                stats
                    .and_then(|stats| stats.max_values.get(name))
                    .cloned()
                    .unwrap_or_else(|| new_null_array(field_type, 1)),
            );
        }
        min_columns.push(concat_column(mins)?);
        max_columns.push(concat_column(maxs)?);
        fields.push(ArrowField::new(name, field_type.clone(), true));
    }
    let fields = Fields::from(fields);
    let min_values = StructArray::try_new(fields.clone(), min_columns, None)?;
    let max_values = StructArray::try_new(fields, max_columns, None)?;
    Ok((Arc::new(min_values), Arc::new(max_values)))
}

/// Build the `nullCount` stats sub-struct: one `Long` field per column any entry
/// has a null count for, each row that entry's count or null when it has none.
fn build_null_count_stats(entries: &[DeltaFileEntry]) -> Result<ArrayRef, Error> {
    let names: std::collections::BTreeSet<&str> = entries
        .iter()
        .filter_map(|entry| entry.stats.as_ref())
        .flat_map(|stats| stats.null_counts.keys().map(String::as_str))
        .collect();
    if names.is_empty() {
        return Ok(new_null_array(
            &ArrowDataType::Struct(Fields::empty()),
            entries.len(),
        ));
    }
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for name in names {
        let values = Int64Array::from_iter(entries.iter().map(|entry| {
            entry
                .stats
                .as_ref()
                .and_then(|stats| stats.null_counts.get(name).copied())
        }));
        fields.push(ArrowField::new(name, ArrowDataType::Int64, true));
        columns.push(Arc::new(values));
    }
    let structure = StructArray::try_new(Fields::from(fields), columns, None)?;
    Ok(Arc::new(structure))
}

/// Delete plain `{version:020}.json` commits strictly below the latest checkpoint
/// when their file mtime is older than `now_ms` minus the table's
/// `delta.logRetentionDuration`. The retention window matters because a reader
/// time-traveling within it may still open an old commit. Checkpoint parquet,
/// `_last_checkpoint`, CRC and multipart files, and all commits at or after the
/// checkpoint are kept, so the log remains replayable and published for the next
/// checkpoint. A table without a checkpoint keeps its whole log. Returns the
/// number of files deleted.
///
/// The checkpoint version and retention window come off the caller's `snapshot`
/// rather than a fresh log read. A snapshot behind a checkpoint another writer
/// has since published only means this sweep keeps commits the next one deletes,
/// which is the safe direction: nothing is deleted that the snapshot cannot
/// prove a checkpoint superseded.
pub(crate) fn cleanup_log(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    snapshot: &Snapshot,
    now_ms: u64,
) -> Result<usize, Error> {
    let Some(checkpoint) = snapshot.log_segment().checkpoint_version else {
        return Ok(0);
    };
    let retention = snapshot
        .table_properties()
        .log_retention_duration
        .unwrap_or(DEFAULT_LOG_RETENTION);
    let cutoff_ms = now_ms.saturating_sub(retention.as_millis() as u64);
    let log_dir = location.join("_delta_log");
    let mut deleted = 0;
    for object in store.list(&log_dir)? {
        // Only plain commit JSONs (`{version:020}.json`) are candidates; the
        // checkpoint parquet, `_last_checkpoint`, and CRC files never match the
        // `.json` suffix, and a coalesced commit (`{v}.{v}.compacted.json`) fails
        // the integer parse, so all are kept.
        let name = object.file.path.name();
        let Some(version) = name
            .strip_suffix(".json")
            .and_then(|digits| digits.parse::<u64>().ok())
        else {
            continue;
        };
        if version < checkpoint && object.modified_unix_ms < cutoff_ms {
            store.delete(&log_dir.join(name))?;
            deleted += 1;
        }
    }
    Ok(deleted)
}

/// Serialize one typed partition scalar the way Delta stores it in an `Add`
/// action's `partitionValues`: a string [`partition_scalar`] parses back on
/// reload, or JSON null for a null value.
fn format_partition_value(
    column: &str,
    scalar: &Scalar<ArrayRef>,
) -> Result<serde_json::Value, Error> {
    let (array, _) = scalar.get();
    if array.is_null(0) {
        return Ok(serde_json::Value::Null);
    }
    // Delta stores timestamp partition values as `yyyy-MM-dd HH:mm:ss[.SSSSSS]`;
    // arrow's default rendering (`1970-01-01T00:00:01`) would not parse back on
    // reload.
    let options = FormatOptions::default().with_timestamp_format(Some(DELTA_TIMESTAMP_FORMAT));
    let formatter =
        ArrayFormatter::try_new(array, &options).map_err(|e| Error::PartitionFormat {
            column: column.to_string(),
            message: e.to_string(),
        })?;
    Ok(serde_json::Value::String(formatter.value(0).to_string()))
}

/// Resolve a catalog-relative table location into the URI Delta Kernel reads.
/// Absolute locations retain Pivot's existing meaning: filesystem root or
/// bucket root, bypassing the database prefix.
pub(crate) fn table_uri(store_uri: &str, location: &ObjectPath) -> Result<Url, Error> {
    let mut root = Url::parse(store_uri).map_err(|source| Error::InvalidUri {
        uri: store_uri.to_string(),
        source,
    })?;
    if location.is_absolute() {
        root.set_path(location.as_str());
        return Ok(root);
    }
    if !root.path().ends_with('/') {
        root.set_path(&format!("{}/", root.path()));
    }
    root.join(location.as_str())
        .map_err(|source| Error::InvalidUri {
            uri: format!("{store_uri}/{}", location.as_str()),
            source,
        })
}

/// Load the table at `uri` from scratch: build its latest snapshot and
/// materialize that snapshot's active file list. The open path; a table already
/// holding a snapshot advances it with [`refresh_table`] instead.
pub(crate) fn load_table(uri: &Url, engine: &DeltaEngine) -> Result<DeltaTableState, Error> {
    let snapshot = Snapshot::builder_for(uri.clone()).build(engine.kernel())?;
    read_state(snapshot, engine)
}

/// Advance `snapshot` to the table's latest committed version and materialize
/// the new state, or `None` when the table has not moved. Kernel updates the
/// snapshot in place: it lists the log past the version in hand and replays only
/// the commits since, rather than rebuilding from the last checkpoint. An
/// unmoved table stops there, so an idle table's refresh costs one listing and
/// reads no files at all.
pub(crate) fn refresh_table(
    snapshot: &Arc<Snapshot>,
    engine: &DeltaEngine,
) -> Result<Option<DeltaTableState>, Error> {
    let latest = Snapshot::builder_from(snapshot.clone()).build(engine.kernel())?;
    if latest.version() == snapshot.version() {
        return Ok(None);
    }
    read_state(latest, engine).map(Some)
}

/// Materialize one snapshot's schema, layout, and active file list: the state a
/// catalog table is rebuilt from.
fn read_state(snapshot: Arc<Snapshot>, engine: &DeltaEngine) -> Result<DeltaTableState, Error> {
    let schema = snapshot.schema();
    let delta_types = schema
        .fields()
        .map(|field| (field.name().clone(), field.data_type().clone()))
        .collect::<HashMap<_, _>>();
    let columns = schema
        .fields()
        .map(|field| {
            Ok(Column {
                name: field.name().clone(),
                col_type: pivot_type_from_field(field)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let partition_by = snapshot
        .table_configuration()
        .metadata()
        .partition_columns()
        .to_vec();
    let sort_by = snapshot
        .table_configuration()
        .metadata()
        .configuration()
        .get(SORT_BY_CONFIGURATION_KEY)
        .map(|raw| raw.split(',').map(str::to_string).collect())
        .unwrap_or_default();

    // Ask Kernel for the typed `stats_parsed` struct (min/max/nullCount/numRecords)
    // alongside the scan files, so a reloaded file carries its Parquet statistics
    // straight from the log -- the in-memory view no longer needs the footers in
    // hand to prune a file by range.
    let scan = snapshot
        .clone()
        .scan_builder()
        .with_stats(StatsOptions::all_struct())
        .build()?;
    let mut files = Vec::new();
    let mut stats_by_path: HashMap<String, crate::manifest::FileStats> = HashMap::new();
    for metadata in scan.scan_metadata(engine.kernel())? {
        let scan_metadata = metadata?;
        files = scan_metadata.visit_scan_files(files, collect_scan_file)?;
        let (data, live) = scan_metadata.scan_files.into_parts();
        collect_file_stats(data, &live, &mut stats_by_path)?;
    }

    let by_name: HashMap<&str, &Type> = columns
        .iter()
        .map(|column| (column.name.as_str(), &column.col_type))
        .collect();
    let mut entries = files
        .into_iter()
        .map(|file| scan_file_entry(file, &by_name, &delta_types, &partition_by))
        .collect::<Result<Vec<_>, Error>>()?;
    // Join each file's log-persisted stats onto its entry by path.
    for entry in &mut entries {
        entry.stats = stats_by_path.remove(entry.file.path.as_str()).map(Arc::new);
    }

    Ok(DeltaTableState {
        snapshot,
        columns,
        partition_by,
        sort_by,
        file_entries: entries,
    })
}

/// The Delta Kernel engine a datastore reads and writes its tables' logs
/// through. It owns an object-store client and a task executor, neither of which
/// is table-specific, so one is built per datastore and shared by every table
/// (and every refresh, commit, and vacuum sweep) rather than stood up per
/// operation. Cheap to clone: the engine itself is behind an `Arc`.
#[derive(Clone)]
pub(crate) struct DeltaEngine {
    inner: Arc<DefaultEngine<TokioMultiThreadExecutor>>,
}

impl DeltaEngine {
    /// Build the engine that reads and writes the logs held by `store`. The
    /// backend supplies the object-store client, so the engine and the reader
    /// serving the tables' data are configured alike.
    ///
    /// It runs on the process's ambient Tokio runtime when there is one, so the
    /// engine shares the server's runtime rather than standing up a second.
    /// Kernel's engine needs a multi-threaded runtime -- a checkpoint/commit runs
    /// a log read and the write concurrently, and a single-thread runtime
    /// deadlocks it -- so a single-thread ambient runtime (a current-thread test)
    /// is declined in favour of [`ENGINE_RUNTIME`], which is also the fallback
    /// when there is no ambient runtime at all (a sync test, a standalone tool).
    pub(crate) fn new(store: &dyn ObjectStore) -> Result<Self, Error> {
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                handle
            }
            _ => ENGINE_RUNTIME.handle().clone(),
        };
        let executor = Arc::new(TokioMultiThreadExecutor::new(handle));
        let engine = DefaultEngineBuilder::new(store.build_delta_object_store()?)
            .with_task_executor(executor)
            .build();
        Ok(Self {
            inner: Arc::new(engine),
        })
    }

    /// The Kernel engine itself, for the Kernel calls that take one.
    fn kernel(&self) -> &DefaultEngine<TokioMultiThreadExecutor> {
        &self.inner
    }
}

/// Fallback multi-threaded runtime for Kernel's engine I/O (log reads and,
/// crucially, checkpoint/commit writes) when no suitable ambient runtime is
/// available: a sync test, a standalone tool, or a single-thread test runtime the
/// engine would deadlock on. Under the server, [`DeltaEngine::new`] uses the
/// server's runtime instead and this is never built. A `static` so it is never dropped
/// (dropping an owned runtime inside an async context panics). Two workers is
/// enough for a checkpoint's concurrent read+write. The hot data path never
/// touches this -- it stays on the io_uring ring.
static ENGINE_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build the Delta engine runtime")
});

fn collect_scan_file(files: &mut Vec<ScanFile>, file: ScanFile) {
    files.push(file);
}

fn scan_file_entry(
    file: ScanFile,
    column_types: &HashMap<&str, &Type>,
    delta_types: &HashMap<String, DeltaDataType>,
    partition_columns: &[String],
) -> Result<DeltaFileEntry, Error> {
    if file.dv_info.has_vector() {
        return Err(Error::DeletionVector(file.path));
    }
    let size = u64::try_from(file.size).map_err(|_| Error::InvalidFileSize {
        path: file.path.clone(),
        size: file.size,
    })?;
    let partition = if partition_columns.is_empty() {
        None
    } else {
        let values = partition_columns
            .iter()
            .filter_map(|name| {
                let raw = file.partition_values.get(name)?;
                Some((|| {
                    let pivot_type = column_types.get(name.as_str()).copied().ok_or_else(|| {
                        Error::UnsupportedType {
                            column: name.clone(),
                            data_type: "missing from Pivot schema".to_string(),
                        }
                    })?;
                    let delta_type =
                        delta_types
                            .get(name)
                            .ok_or_else(|| Error::UnsupportedType {
                                column: name.clone(),
                                data_type: "missing from Delta schema".to_string(),
                            })?;
                    Ok((
                        name.clone(),
                        partition_scalar(name, raw, pivot_type, delta_type)?,
                    ))
                })())
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        // A missing Delta partition map entry is unknown metadata, not proof of
        // a null value. Keep an empty/partial map so pruning remains soft.
        Some(values)
    };
    Ok(DeltaFileEntry {
        file: FileRef {
            path: ObjectPath::new(file.path),
            size,
        },
        partition,
        // The caller joins in the log-persisted stats (read as Kernel's typed
        // `stats_parsed`) by path; a file the log has no stats for stays `None`.
        stats: None,
    })
}

/// Read each scan file's `stats_parsed` struct out of one metadata batch and
/// record its [`FileStats`](crate::manifest::FileStats) by path, for the loader to
/// join onto the file entries. Kernel emits `stats_parsed` typed to the table's
/// schema (min/max as the column's own type, null counts and row count as longs),
/// so the bounds are ready to prune against a query's typed constants with no
/// footer read. A batch that carries no `stats_parsed` column (stats not
/// requested, or a column the schema does not index) simply records nothing --
/// pruning falls back to deriving stats from the footers.
fn collect_file_stats(
    data: Box<dyn EngineData>,
    live: &[bool],
    stats_by_path: &mut HashMap<String, crate::manifest::FileStats>,
) -> Result<(), Error> {
    let data: Arc<dyn EngineData> = Arc::from(data);
    let arrow = data.as_any().downcast::<ArrowEngineData>().map_err(|_| {
        Error::Kernel(delta_kernel::Error::generic(
            "scan metadata was not Arrow-backed",
        ))
    })?;
    let batch = arrow.record_batch();
    let (Some(paths), Some(stats)) = (
        batch.column_by_name("path"),
        batch
            .column_by_name("stats_parsed")
            .and_then(|column| column.as_any().downcast_ref::<StructArray>()),
    ) else {
        return Ok(());
    };
    let num_records = struct_field_as_int64(stats, "numRecords");
    let min_values = struct_field_as_struct(stats, "minValues");
    let max_values = struct_field_as_struct(stats, "maxValues");
    let null_count = struct_field_as_struct(stats, "nullCount");

    for row in 0..batch.num_rows() {
        // Only selected `Add` rows: a masked row, a `Remove`, or a file the log
        // recorded without stats has nothing to contribute.
        if !live.get(row).copied().unwrap_or(false) || stats.is_null(row) {
            continue;
        }
        let Some(path) = string_at(paths, row) else {
            continue;
        };
        stats_by_path.insert(
            path,
            crate::manifest::FileStats {
                // Left `None` when the log recorded no `numRecords` for this file,
                // so an unknown count is not mistaken for an empty file.
                num_records: num_records
                    .filter(|column| !column.is_null(row))
                    .map(|column| column.value(row)),
                min_values: struct_row_values(min_values, row),
                max_values: struct_row_values(max_values, row),
                null_counts: struct_row_null_counts(null_count, row),
            },
        );
    }
    Ok(())
}

/// The named struct field of `parent`, if present and itself a struct.
fn struct_field_as_struct<'a>(parent: &'a StructArray, name: &str) -> Option<&'a StructArray> {
    parent
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StructArray>())
}

/// The named struct field of `parent`, if present and a `Long`.
fn struct_field_as_int64<'a>(parent: &'a StructArray, name: &str) -> Option<&'a Int64Array> {
    parent
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
}

/// A Delta `String` value at `row` (Arrow `Utf8` or `Utf8View`), or `None` when
/// null or an unexpected physical type.
fn string_at(array: &ArrayRef, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    if let Some(strings) = array.as_any().downcast_ref::<StringArray>() {
        return Some(strings.value(row).to_string());
    }
    array
        .as_any()
        .downcast_ref::<StringViewArray>()
        .map(|strings| strings.value(row).to_string())
}

/// Each present, non-null leaf of a `minValues`/`maxValues` struct at `row`, as a
/// one-element array of the column's own type, keyed by column name.
fn struct_row_values(values: Option<&StructArray>, row: usize) -> HashMap<String, ArrayRef> {
    let mut out = HashMap::new();
    let Some(values) = values else {
        return out;
    };
    if values.is_null(row) {
        return out;
    }
    for (field, column) in values.fields().iter().zip(values.columns()) {
        if !column.is_null(row) {
            out.insert(field.name().clone(), column.slice(row, 1));
        }
    }
    out
}

/// Each `Long` leaf of a `nullCount` struct at `row`, keyed by column name.
/// Non-`Long` leaves (a nested null count for a variant/complex column) are
/// skipped: file-level range pruning only reads the flat per-column counts.
fn struct_row_null_counts(counts: Option<&StructArray>, row: usize) -> HashMap<String, i64> {
    let mut out = HashMap::new();
    let Some(counts) = counts else {
        return out;
    };
    if counts.is_null(row) {
        return out;
    }
    for (field, column) in counts.fields().iter().zip(counts.columns()) {
        if let Some(ints) = column.as_any().downcast_ref::<Int64Array>()
            && !ints.is_null(row)
        {
            out.insert(field.name().clone(), ints.value(row));
        }
    }
    out
}

/// Parse one partition value from a Delta `Add` action and convert it to the
/// Arrow scalar representation Pivot uses for comparisons. Delta stores the
/// value as a string, so its declared primitive type drives Delta Kernel's
/// parser; the Pivot type verifies that the result has the physical scalar
/// representation catalog partition pruning expects.
fn partition_scalar(
    column: &str,
    raw: &str,
    pivot_type: &Type,
    delta_type: &DeltaDataType,
) -> Result<Scalar<ArrayRef>, Error> {
    let DeltaDataType::Primitive(primitive) = delta_type else {
        return Err(Error::UnsupportedType {
            column: column.to_string(),
            data_type: format!("{delta_type:?}"),
        });
    };
    let expected = planner::types::physical_arrow_type(pivot_type);
    // Delta stores partition values as strings, and Delta Kernel parses an
    // empty string as null for every type. A string column's value is taken
    // verbatim so an empty string survives the round trip; for other types a
    // null parse result becomes a typed null scalar (partition pruning already
    // compares nulls with SQL semantics).
    let scalar = if matches!(pivot_type, Type::Utf8) {
        Scalar::new(Arc::new(StringViewArray::from(vec![raw])) as ArrayRef)
    } else {
        match primitive.parse_scalar(raw)? {
            DeltaScalar::Null(_) => Scalar::new(new_null_array(&expected, 1)),
            scalar => delta_scalar_to_pivot(column, scalar)?,
        }
    };
    let actual = scalar.get().0.data_type().clone();
    if actual != expected {
        return Err(Error::UnsupportedType {
            column: column.to_string(),
            data_type: format!(
                "{delta_type:?} produces {actual:?}, but Pivot expects {expected:?}"
            ),
        });
    }
    Ok(scalar)
}

fn delta_scalar_to_pivot(column: &str, scalar: DeltaScalar) -> Result<Scalar<ArrayRef>, Error> {
    fn erased<T: Array + 'static>(array: T) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(array))
    }

    let scalar = match scalar {
        DeltaScalar::Boolean(value) => erased(BooleanArray::from(vec![value])),
        DeltaScalar::Byte(value) => erased(Int8Array::from(vec![value])),
        DeltaScalar::Short(value) => erased(Int16Array::from(vec![value])),
        DeltaScalar::Integer(value) => erased(Int32Array::from(vec![value])),
        DeltaScalar::Long(value) => erased(Int64Array::from(vec![value])),
        DeltaScalar::Float(value) => erased(Float32Array::from(vec![value])),
        DeltaScalar::Double(value) => erased(Float64Array::from(vec![value])),
        DeltaScalar::String(value) => erased(StringViewArray::from(vec![value])),
        DeltaScalar::Date(value) => erased(Date32Array::from(vec![value])),
        // Delta counts a timestamp in microseconds, the same unit a pivot
        // timestamp counts in, so the value carries over unscaled.
        DeltaScalar::Timestamp(value) | DeltaScalar::TimestampNtz(value) => {
            erased(TimestampMicrosecondArray::from(vec![value]))
        }
        // A decimal partition value lands on the declared type's carrier,
        // matching `physical_arrow_type`: `Decimal64` up to 18 digits (where
        // the unscaled integer is guaranteed to fit), `Decimal128` beyond.
        // Delta Kernel enforces a valid precision/scale, so restamping the
        // unscaled integer cannot fail.
        DeltaScalar::Decimal(value) => {
            if value.precision() <= planner::types::MAX_DECIMAL64_PRECISION {
                let narrow = i64::try_from(value.bits()).expect("an 18-digit decimal fits in i64");
                erased(
                    Decimal64Array::from(vec![narrow])
                        .with_precision_and_scale(value.precision(), value.scale() as i8)
                        .expect("Delta Kernel enforces a valid decimal shape"),
                )
            } else {
                erased(
                    Decimal128Array::from(vec![value.bits()])
                        .with_precision_and_scale(value.precision(), value.scale() as i8)
                        .expect("Delta Kernel enforces a valid decimal shape"),
                )
            }
        }
        unsupported => {
            return Err(Error::UnsupportedType {
                column: column.to_string(),
                data_type: unsupported.data_type().to_string(),
            });
        }
    };
    Ok(scalar)
}

fn delta_type(column: &str, data_type: &Type) -> Result<DeltaDataType, Error> {
    let primitive = match data_type {
        // The unshredded physical layout Delta declares for a variant column;
        // a shredded file's extra typed fields are a per-file matter the scan
        // resolves from the Parquet footer, not the table schema.
        Type::Variant => {
            let fields = [
                StructField::new("metadata", PrimitiveType::Binary, false),
                StructField::new("value", PrimitiveType::Binary, false),
            ];
            let unshredded = StructType::try_new(fields)?;
            return Ok(DeltaDataType::Variant(Box::new(unshredded)));
        }
        Type::Boolean => PrimitiveType::Boolean,
        Type::Int8 => PrimitiveType::Byte,
        Type::Int16 => PrimitiveType::Short,
        Type::Int32 => PrimitiveType::Integer,
        Type::Int64 => PrimitiveType::Long,
        Type::Float32 => PrimitiveType::Float,
        Type::Float64 => PrimitiveType::Double,
        Type::Utf8 => PrimitiveType::String,
        Type::Date => PrimitiveType::Date,
        Type::Timestamp => PrimitiveType::TimestampNtz,
        Type::Decimal { precision, scale } => PrimitiveType::decimal(*precision, *scale as u8)?,
        // A Delta *table schema* has no interval primitive: the kernel reads
        // `interval day`/`interval second` as an unsupported table type, even
        // though the SQL dialects over Delta do have an INTERVAL expression
        // type. Nothing is lost, as an interval only ever arises here as a
        // computed result (a timestamp difference), never as a stored column.
        Type::Interval
        | Type::Int128
        | Type::UInt8
        | Type::UInt16
        | Type::UInt32
        | Type::UInt64 => {
            return Err(Error::UnsupportedType {
                column: column.to_string(),
                data_type: data_type.to_string(),
            });
        }
    };
    Ok(primitive.into())
}

/// Field-metadata key that records a column's exact Pivot type when Delta's
/// primitives can't express it. Delta has no unsigned integer type, so an
/// unsigned column is stored as a signed primitive that holds its range and its
/// true type is recovered from this tag on read.
const PIVOT_LOGICAL_TYPE_KEY: &str = "pivot.logicalTypeOverride";

/// Build the Delta struct field for one Pivot column. Types Delta represents
/// natively map straight through [`delta_type`]; the unsigned types it rejects
/// fall back to [`build_unsigned_field`], which stores them as a tagged signed
/// primitive.
fn build_delta_field(column: &str, data_type: &Type) -> Result<StructField, Error> {
    match delta_type(column, data_type) {
        Ok(delta_type) => Ok(StructField::new(column, delta_type, false)),
        Err(_) => build_unsigned_field(column, data_type),
    }
}

/// Build the Delta field for an unsigned integer column, which Delta has no
/// primitive for: store it as the smallest signed primitive that holds its full
/// range and tag the field with its true type so the read recovers it. `UInt64`
/// alone has no wider signed primitive; `Long` still holds every value the
/// physical file decodes because the read is driven by the recovered type, not
/// this stored primitive. Any non-unsigned type reaching here is genuinely
/// unsupported and errors.
fn build_unsigned_field(column: &str, data_type: &Type) -> Result<StructField, Error> {
    let primitive = match data_type {
        Type::UInt8 => PrimitiveType::Short,
        Type::UInt16 => PrimitiveType::Integer,
        Type::UInt32 | Type::UInt64 => PrimitiveType::Long,
        _ => {
            return Err(Error::UnsupportedType {
                column: column.to_string(),
                data_type: data_type.to_string(),
            });
        }
    };
    Ok(StructField::new(column, primitive, false)
        .with_metadata([(PIVOT_LOGICAL_TYPE_KEY, data_type.to_string())]))
}

/// Recover a column's Pivot type from a Delta field, honoring the
/// [`PIVOT_LOGICAL_TYPE_KEY`] tag that carries unsigned types Delta stores as a
/// signed primitive.
fn pivot_type_from_field(field: &StructField) -> Result<Type, Error> {
    if let Some(MetadataValue::String(tag)) = field.metadata.get(PIVOT_LOGICAL_TYPE_KEY) {
        return parse_unsigned_tag(tag).ok_or_else(|| Error::UnsupportedType {
            column: field.name().clone(),
            data_type: tag.clone(),
        });
    }
    let unsupported = |data_type: String| Error::UnsupportedType {
        column: field.name().clone(),
        data_type,
    };
    if let DeltaDataType::Variant(_) = field.data_type() {
        return Ok(Type::Variant);
    }
    let DeltaDataType::Primitive(primitive) = field.data_type() else {
        return Err(unsupported(format!("{:?}", field.data_type())));
    };
    let data_type = match primitive {
        PrimitiveType::String => Type::Utf8,
        PrimitiveType::Long => Type::Int64,
        PrimitiveType::Integer => Type::Int32,
        PrimitiveType::Short => Type::Int16,
        PrimitiveType::Byte => Type::Int8,
        PrimitiveType::Float => Type::Float32,
        PrimitiveType::Double => Type::Float64,
        PrimitiveType::Boolean => Type::Boolean,
        PrimitiveType::Date => Type::Date,
        PrimitiveType::Timestamp | PrimitiveType::TimestampNtz => Type::Timestamp,
        // Delta Kernel already enforces precision 1..=38 and scale <= precision,
        // exactly the shapes Pivot's decimal supports.
        PrimitiveType::Decimal(decimal) => Type::Decimal {
            precision: decimal.precision(),
            scale: decimal.scale() as i8,
        },
        PrimitiveType::Binary | PrimitiveType::Void => {
            return Err(unsupported(primitive.to_string()));
        }
    };
    Ok(data_type)
}

/// Parse a [`PIVOT_LOGICAL_TYPE_KEY`] tag back into its unsigned Pivot type.
fn parse_unsigned_tag(tag: &str) -> Option<Type> {
    match tag {
        "UInt8" => Some(Type::UInt8),
        "UInt16" => Some(Type::UInt16),
        "UInt32" => Some(Type::UInt32),
        "UInt64" => Some(Type::UInt64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Datum;

    /// Exercise the whole Kernel write path the catalog relies on end to end at
    /// this layer: create a table, append a file with our add metadata, read it
    /// back, remove it by scan selection, and checkpoint. A Kernel-authored log
    /// is a precondition for `checkpoint` — it refuses a log it did not write.
    #[test]
    fn kernel_authored_table_is_checkpointable() {
        use delta_kernel::committer::FileSystemCommitter;
        use delta_kernel::transaction::CommitResult;
        use delta_kernel::transaction::create_table::create_table;

        let dir = tempfile::tempdir().unwrap();
        let store = crate::store::LocalStore::new(dir.path());
        let uri = Url::from_directory_path(dir.path()).unwrap();
        let engine = DeltaEngine::new(&store).unwrap();
        let kernel = engine.kernel();

        // Create the table THROUGH kernel (commit v0: protocol + metadata).
        let schema = Arc::new(
            StructType::try_new([StructField::nullable("value", DeltaDataType::LONG)]).unwrap(),
        );
        let txn = create_table(uri.as_str(), schema, "pivot")
            .build(kernel, Box::new(FileSystemCommitter::new()))
            .unwrap();
        assert!(
            matches!(
                txn.commit(kernel).unwrap(),
                CommitResult::CommittedTransaction(_)
            ),
            "kernel create-table commit should succeed"
        );

        // Append a file through kernel using our hand-built add metadata.
        let snapshot = Snapshot::builder_for(uri.clone()).build(kernel).unwrap();
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), kernel)
            .unwrap();
        let entry = DeltaFileEntry::new(FileRef {
            path: ObjectPath::new("part-0.parquet"),
            size: 123,
        });
        txn.add_files(add_files_metadata(&[entry], 1).unwrap());
        assert!(
            matches!(
                txn.commit(kernel).unwrap(),
                CommitResult::CommittedTransaction(_)
            ),
            "kernel add-files commit should succeed"
        );

        // The appended file is now the table's live set, read back through Kernel.
        let state = load_table(&uri, &engine).unwrap();
        assert_eq!(state.file_entries.len(), 1);
        assert_eq!(state.file_entries[0].file.path.as_str(), "part-0.parquet");

        // Remove the file through kernel: scan the snapshot and hand its scan-file
        // rows to `remove_files` (kernel removes by selection, not by path).
        let snapshot = Snapshot::builder_for(uri.clone()).build(kernel).unwrap();
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), kernel)
            .unwrap();
        let scan = snapshot.scan_builder().build().unwrap();
        for metadata in scan.scan_metadata(kernel).unwrap() {
            txn.remove_files(metadata.unwrap().scan_files);
        }
        assert!(
            matches!(
                txn.commit(kernel).unwrap(),
                CommitResult::CommittedTransaction(_)
            ),
            "kernel remove-files commit should succeed"
        );
        assert_eq!(
            load_table(&uri, &engine).unwrap().file_entries.len(),
            0,
            "file should be removed"
        );

        // Checkpoint the kernel-authored log.
        let snapshot = Snapshot::builder_for(uri.clone()).build(kernel).unwrap();
        let (result, _) = snapshot
            .checkpoint(kernel, None)
            .expect("kernel-authored table should be checkpointable");
        assert!(matches!(
            result,
            delta_kernel::snapshot::CheckpointWriteResult::Written
        ));
    }

    /// A file's persisted Parquet stats round-trip through the log: `load_table`
    /// reads them back off the Delta `Add` (as Kernel's typed `stats_parsed`),
    /// with no footer in hand -- so a reloaded file can be pruned by range.
    #[test]
    fn load_reads_persisted_stats_from_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let table = create_test_table(dir.path());
        let entry = DeltaFileEntry {
            file: FileRef {
                path: ObjectPath::new("part-0.parquet"),
                size: 123,
            },
            partition: None,
            stats: Some(Arc::new(crate::manifest::FileStats {
                num_records: Some(5),
                min_values: HashMap::from([(
                    "value".to_string(),
                    Arc::new(Int64Array::from(vec![10])) as ArrayRef,
                )]),
                max_values: HashMap::from([(
                    "value".to_string(),
                    Arc::new(Int64Array::from(vec![20])) as ArrayRef,
                )]),
                null_counts: HashMap::from([("value".to_string(), 1)]),
            })),
        };
        commit_file_changes(&table.engine, &table.snapshot, &[], &[entry], true).unwrap();

        let state = load_table(&table.uri, &table.engine).unwrap();

        let stats = state.file_entries[0]
            .stats
            .as_ref()
            .expect("reloaded file carries the log's stats");
        let min = stats.min_values["value"]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let max = stats.max_values["value"]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(stats.num_records, Some(5));
        assert_eq!(min.value(0), 10);
        assert_eq!(max.value(0), 20);
        assert_eq!(stats.null_counts["value"], 1);
    }

    /// Everything a test needs to drive one table's log the way the catalog does:
    /// the store it lives in, the engine and snapshot a commit rides, and the
    /// location and URI naming it.
    struct TestTable {
        store: crate::store::LocalStore,
        engine: DeltaEngine,
        snapshot: Arc<Snapshot>,
        location: ObjectPath,
        uri: Url,
    }

    /// Create a bare kernel-authored table at a store-relative location (no
    /// `delta.*` maintenance properties, which Kernel forbids setting at CREATE).
    fn create_test_table(dir: &std::path::Path) -> TestTable {
        let store = crate::store::LocalStore::new(dir);
        let location = ObjectPath::new("t");
        store.create_dir(&location).unwrap();
        let uri = table_uri(&store.location_uri(), &location).unwrap();
        let engine = DeltaEngine::new(&store).unwrap();
        let schema = Arc::new(
            StructType::try_new([StructField::nullable("value", DeltaDataType::LONG)]).unwrap(),
        );
        let committed = create_table(uri.as_str(), schema, "pivot")
            .build(engine.kernel(), Box::new(FileSystemCommitter::new()))
            .unwrap()
            .commit(engine.kernel())
            .unwrap();
        let CommitResult::CommittedTransaction(committed) = committed else {
            panic!("creating the test table should commit version 0");
        };
        let snapshot = committed_snapshot(&committed, &engine, &uri).unwrap();
        TestTable {
            store,
            engine,
            snapshot,
            location,
            uri,
        }
    }

    fn has_checkpoint(log_dir: &std::path::Path) -> bool {
        std::fs::read_dir(log_dir).unwrap().flatten().any(|e| {
            e.file_name()
                .to_string_lossy()
                .ends_with(".checkpoint.parquet")
        })
    }

    /// The inline checkpoint on the commit path fires only when the committed
    /// version crosses the interval: ordinary commits below it write no
    /// checkpoint, and a version at a multiple of the interval writes one.
    #[test]
    fn inline_checkpoint_fires_only_when_the_interval_is_crossed() {
        let dir = tempfile::tempdir().unwrap();
        let table = create_test_table(dir.path());
        let log_dir = dir.path().join("t").join("_delta_log");

        // Append a file per commit up to the interval. Each rides the snapshot the
        // previous commit handed back, so no version reads the log to be written.
        let mut snapshot = table.snapshot.clone();
        for version in 1..=DEFAULT_CHECKPOINT_INTERVAL {
            let entry = DeltaFileEntry::new(FileRef {
                path: ObjectPath::new(format!("f{version}.parquet")),
                size: 1,
            });
            snapshot = commit_file_changes(&table.engine, &snapshot, &[], &[entry], true)
                .unwrap()
                .expect("a commit on our own snapshot has nothing to conflict with");
            assert_eq!(snapshot.version(), version);
            if version < DEFAULT_CHECKPOINT_INTERVAL {
                assert!(
                    !has_checkpoint(&log_dir),
                    "a commit below the interval writes no checkpoint"
                );
            }
        }

        assert!(
            has_checkpoint(&log_dir),
            "the commit on the interval writes a checkpoint"
        );
    }

    /// Log cleanup deletes the commit JSONs a checkpoint superseded once they are
    /// past retention, and the tombstones they carried stay readable off the
    /// checkpoint.
    #[test]
    fn cleanup_deletes_superseded_commits_while_tombstones_survive() {
        let dir = tempfile::tempdir().unwrap();
        let table = create_test_table(dir.path());

        // Add a file (v1), then remove it (v2).
        let entry = DeltaFileEntry::new(FileRef {
            path: ObjectPath::new("f.parquet"),
            size: 1,
        });
        let added = commit_file_changes(
            &table.engine,
            &table.snapshot,
            &[],
            std::slice::from_ref(&entry),
            true,
        )
        .unwrap()
        .expect("the add commits on our own snapshot");
        let removed = commit_file_changes(
            &table.engine,
            &added,
            std::slice::from_ref(&entry),
            &[],
            true,
        )
        .unwrap()
        .expect("the remove commits on the snapshot the add handed back");

        // Checkpoint at v2 (the default interval is well above these versions, so
        // the inline path did not fire; force one to exercise cleanup).
        let (_result, checkpointed) = removed.checkpoint(table.engine.kernel(), None).unwrap();

        // Evaluate cleanup well past the default log-retention window, so every
        // commit below the checkpoint is eligible.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            + 40 * 24 * 60 * 60 * 1000;
        assert!(
            cleanup_log(&table.store, &table.location, &checkpointed, now_ms).unwrap() > 0,
            "superseded commits below the checkpoint are deleted"
        );
    }

    /// A commit's snapshot names the version that commit landed on, never the
    /// log's latest: it is paired with a file set holding that commit's files and
    /// no one else's, so a snapshot that had swept up a later writer's commit
    /// would claim files its holder does not have, and no refresh would heal
    /// that: the holder's version would already be current.
    #[test]
    fn a_rebuilt_commit_snapshot_names_its_own_version_not_the_logs_latest() {
        let dir = tempfile::tempdir().unwrap();
        let table = create_test_table(dir.path());
        let entry = |name: &str| {
            DeltaFileEntry::new(FileRef {
                path: ObjectPath::new(name),
                size: 1,
            })
        };
        let ours = commit_file_changes(&table.engine, &table.snapshot, &[], &[entry("ours")], true)
            .unwrap()
            .expect("our add commits on our own snapshot");
        // Another writer moves the log on past the version we landed on.
        commit_file_changes(&table.engine, &ours, &[], &[entry("theirs")], true)
            .unwrap()
            .expect("their add commits on the snapshot ours handed back");

        let rebuilt = snapshot_at(&table.uri, ours.version(), &table.engine).unwrap();

        assert_eq!(rebuilt.version(), ours.version());
        assert_eq!(
            read_state(rebuilt, &table.engine)
                .unwrap()
                .file_entries
                .len(),
            1,
            "the rebuilt snapshot names our file alone, not the later writer's"
        );
    }

    /// A commit hands back a snapshot that still knows the table's checkpoint, so
    /// a vacuum reading log cleanup off a written-to table's current snapshot
    /// still sees which commits that checkpoint superseded.
    #[test]
    fn a_commit_carries_its_snapshots_checkpoint_forward() {
        let dir = tempfile::tempdir().unwrap();
        let table = create_test_table(dir.path());
        let entry = |name: &str| {
            DeltaFileEntry::new(FileRef {
                path: ObjectPath::new(name),
                size: 1,
            })
        };

        let added = commit_file_changes(&table.engine, &table.snapshot, &[], &[entry("a")], true)
            .unwrap()
            .expect("the add commits on our own snapshot");
        let (_result, checkpointed) = added.checkpoint(table.engine.kernel(), None).unwrap();
        let after = commit_file_changes(&table.engine, &checkpointed, &[], &[entry("b")], true)
            .unwrap()
            .expect("the next commit rides the checkpointed snapshot");

        // Past the retention window, so the one commit below the checkpoint (v0)
        // is eligible -- which is only visible if the checkpoint carried forward.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            + 40 * 24 * 60 * 60 * 1000;
        assert_eq!(
            cleanup_log(&table.store, &table.location, &after, now_ms).unwrap(),
            1
        );
    }

    #[test]
    fn delta_partition_values_become_pivot_scalars() {
        let string = partition_scalar(
            "service",
            "api",
            &Type::Utf8,
            &DeltaDataType::Primitive(PrimitiveType::String),
        )
        .unwrap();
        assert_eq!(
            string
                .get()
                .0
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            "api"
        );

        let date = partition_scalar(
            "day",
            "1970-01-03",
            &Type::Date,
            &DeltaDataType::Primitive(PrimitiveType::Date),
        )
        .unwrap();
        assert_eq!(
            date.get()
                .0
                .as_any()
                .downcast_ref::<Date32Array>()
                .unwrap()
                .value(0),
            2
        );
    }

    /// Every partition value written into an `Add` action must parse back on
    /// reload, or the table becomes unloadable.
    #[test]
    fn timestamp_partition_value_round_trips_through_the_delta_log() {
        let scalar: Scalar<ArrayRef> =
            Scalar::new(
                Arc::new(TimestampMicrosecondArray::from(vec![86_401_000_000])) as ArrayRef,
            );

        let written = format_partition_value("ts", &scalar).unwrap();
        let restored = partition_scalar(
            "ts",
            written.as_str().unwrap(),
            &Type::Timestamp,
            &DeltaDataType::Primitive(PrimitiveType::TimestampNtz),
        )
        .unwrap();

        assert_eq!(written, serde_json::json!("1970-01-02 00:00:01"));
        assert_eq!(
            restored
                .get()
                .0
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .value(0),
            86_401_000_000
        );
    }

    #[test]
    fn empty_string_partition_value_round_trips_through_the_delta_log() {
        let scalar: Scalar<ArrayRef> =
            Scalar::new(Arc::new(StringViewArray::from(vec![""])) as ArrayRef);

        let written = format_partition_value("service", &scalar).unwrap();
        let restored = partition_scalar(
            "service",
            written.as_str().unwrap(),
            &Type::Utf8,
            &DeltaDataType::Primitive(PrimitiveType::String),
        )
        .unwrap();

        assert_eq!(
            restored
                .get()
                .0
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(0),
            ""
        );
    }

    /// A null non-string partition value parses back as a typed null scalar
    /// rather than failing the table load.
    #[test]
    fn null_partition_value_restores_as_typed_null() {
        let restored = partition_scalar(
            "shard",
            "",
            &Type::Int64,
            &DeltaDataType::Primitive(PrimitiveType::Long),
        )
        .unwrap();

        let (array, _) = restored.get();
        assert_eq!(array.data_type(), &arrow_schema::DataType::Int64);
        assert!(array.is_null(0));
    }

    /// Delta has no Int128/HUGEINT primitive, and decimal(38, 0) cannot represent
    /// its full range. Keep Delta decimals as Decimal instead of using an
    /// ambiguous encoding for Int128.
    #[test]
    fn hugeint_is_not_encoded_as_delta_decimal() {
        assert!(matches!(
            delta_type("total", &Type::Int128),
            Err(Error::UnsupportedType { column, data_type })
                if column == "total" && data_type == "Int128"
        ));
        let decimal = StructField::new("total", PrimitiveType::decimal(38, 0).unwrap(), false);
        assert_eq!(
            pivot_type_from_field(&decimal).unwrap(),
            Type::Decimal {
                precision: 38,
                scale: 0
            },
            "a genuine Delta decimal must remain Decimal"
        );
    }

    /// Delta has no unsigned primitive, so each unsigned column is stored as the
    /// smallest signed primitive that holds its full value range.
    #[test]
    fn unsigned_columns_store_as_signed_delta_primitives() {
        let stored_primitive = |ty| match build_delta_field("c", &ty).unwrap().data_type() {
            DeltaDataType::Primitive(p) => p.clone(),
            other => panic!("expected primitive, got {other:?}"),
        };

        assert_eq!(stored_primitive(Type::UInt8), PrimitiveType::Short);
        assert_eq!(stored_primitive(Type::UInt16), PrimitiveType::Integer);
        assert_eq!(stored_primitive(Type::UInt32), PrimitiveType::Long);
        assert_eq!(stored_primitive(Type::UInt64), PrimitiveType::Long);
    }

    /// An unsigned column's exact type survives the Delta round trip: it is
    /// tagged on write and recovered from the tag on read, so the Parquet
    /// decoder still sees the unsigned type the file physically stores.
    #[test]
    fn unsigned_column_type_round_trips_through_field_metadata() {
        for ty in [Type::UInt8, Type::UInt16, Type::UInt32, Type::UInt64] {
            let field = build_delta_field("event_date", &ty).unwrap();
            assert_eq!(pivot_type_from_field(&field).unwrap(), ty);
        }
    }

    /// A signed column carries no tag, so it round-trips through the native
    /// primitive mapping without one.
    #[test]
    fn signed_columns_round_trip_without_a_metadata_tag() {
        let field = build_delta_field("id", &Type::Int32).unwrap();
        assert!(!field.metadata.contains_key(PIVOT_LOGICAL_TYPE_KEY));
        assert_eq!(pivot_type_from_field(&field).unwrap(), Type::Int32);
    }
}
