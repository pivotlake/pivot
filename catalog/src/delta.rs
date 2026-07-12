//! The catalog's durable table format: one Delta Lake transaction log per
//! table.
//!
//! The `_delta_log/` under a table's data location is the source of truth for
//! its declared schema, partition and sort specs, and committed data files with
//! their partition tuples and sort-key bounds. The background catalog sync
//! loads the committed state here and materializes it as the in-memory
//! [`TableManifest`] everything downstream (snapshots, bindings, pruning)
//! reads.
//!
//! Tables written by an external Delta writer load the same way; delta
//! features the positional scan layer cannot honor are rejected at load with
//! an explicit error rather than serving wrong rows: deletion vectors (a scan
//! would resurrect deleted rows), column mapping (parquet columns are read
//! positionally by the declared schema), and absolute/URI file paths (a
//! [`FileRef`] path resolves under the table's location).
//!
//! delta-rs is async, so every log operation runs on one small tokio runtime
//! owned by this module, the calling thread blocking on a channel for the
//! result.

use std::future::Future;
use std::sync::OnceLock;

use deltalake_core::kernel::{
    DataType as DeltaDataType, MetadataValue, PrimitiveType, StructField,
};
use deltalake_core::table::builder::DeltaTableBuilder;
use deltalake_core::{DeltaTable, DeltaTableError};
use planner::catalog::Column;
use planner::types::Type;
use serde::{Deserialize, Serialize};

use crate::manifest::{ManifestEntry, SortBounds, TableManifest};
use crate::store::{DeltaTableTarget, FileRef, ObjectPath};

/// Field-metadata key an external writer may use to preserve a pivot type that
/// has no exact Delta-native representation.
const COLUMN_TYPE_KEY: &str = "pivot.type";
/// Table-configuration key holding the sort spec as a JSON array of column
/// names. Delta has no native sort spec, so it rides in the table properties;
/// a table without it is unsorted.
const SORT_BY_KEY: &str = "pivot.sortBy";
/// Table-configuration key for delta column mapping. Any mode other than
/// `none` renames/reorders the parquet columns away from the declared schema,
/// which the positional scan cannot follow.
const COLUMN_MAPPING_KEY: &str = "delta.columnMapping.mode";
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Delta(#[from] DeltaTableError),
    #[error("delta metadata json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("table `{0}` is in the catalog index but has no delta log at its location")]
    MissingLog(String),
    #[error("column `{column}` has delta type `{delta_type}`, which maps to no pivot type")]
    UnsupportedColumnType { column: String, delta_type: String },
    #[error("column `{0}` has a `{COLUMN_TYPE_KEY}` metadata value that is not a string")]
    MalformedColumnType(String),
    #[error(
        "the table uses delta column mapping (`{COLUMN_MAPPING_KEY}` = `{0}`), which is not supported"
    )]
    ColumnMappingUnsupported(String),
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
    #[error("delta log task panicked")]
    TaskPanicked,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Register delta-rs's S3 and GCS object-store factories, exactly once, before
/// any table handle is built (building resolves the URL scheme's factory).
/// Registered here at runtime: the `deltalake` meta crate would do it in a
/// pre-main constructor, which allocates before jemalloc is initialized and
/// aborts.
fn ensure_store_handlers_registered() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        deltalake_aws::register_handlers(None);
        deltalake_gcp::register_handlers(None);
    });
}

/// The dedicated runtime every delta log read runs on.
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

/// Load the delta table at `target` at its latest version and extract its
/// committed state as a [`TableManifest`]. `name` is only for the error when
/// no delta table is readable there.
pub(crate) fn load_manifest(target: &DeltaTableTarget, name: &str) -> Result<TableManifest> {
    let table = open_table(target).map_err(|e| match e {
        Error::Delta(DeltaTableError::NotATable(_)) => Error::MissingLog(name.to_string()),
        other => other,
    })?;
    extract_manifest(&table)
}

/// Load the latest committed state **only if it is newer than `since`**, the
/// background refresh's reload. The latest version is peeked from the log
/// store first, so the common tick (no new external commit) pays one small
/// version probe instead of a full log replay; only a table whose version
/// moved is loaded and extracted.
pub(crate) fn load_manifest_after(
    target: &DeltaTableTarget,
    since: u64,
) -> Result<Option<TableManifest>> {
    let log_store = build_table(target)?.log_store();
    match run_blocking(async move { log_store.get_latest_version(since).await })? {
        Ok(latest) if latest <= since => Ok(None),
        Ok(_) => extract_manifest(&open_table(target)?).map(Some),
        // The commit at the cursor no longer exists (external log cleanup past
        // a checkpoint, or a drop-and-recreate that reset versions): the peek
        // can't walk forward from `since`, but a full load from the latest
        // checkpoint still can. Refresh from scratch instead of erroring every
        // tick until restart.
        Err(DeltaTableError::InvalidVersion(_)) => {
            let manifest = extract_manifest(&open_table(target)?)?;
            Ok((manifest.version != since).then_some(manifest))
        }
        Err(e) => Err(e.into()),
    }
}

/// Build an unloaded delta-rs handle for the table at `target`.
fn build_table(target: &DeltaTableTarget) -> Result<DeltaTable> {
    ensure_store_handlers_registered();
    let url = url::Url::parse(&target.uri)
        .map_err(|e| DeltaTableError::InvalidTableLocation(format!("{}: {e}", target.uri)))?;
    Ok(DeltaTableBuilder::from_url(url)?
        .with_storage_options(target.storage_options.clone())
        .build()?)
}

/// Open the delta table at `target`, loaded to its latest version.
fn open_table(target: &DeltaTableTarget) -> Result<DeltaTable> {
    let mut table = build_table(target)?;
    Ok(run_blocking(async move {
        table.load().await?;
        Ok::<_, DeltaTableError>(table)
    })??)
}

/// The loaded table's version, the catalog's per-table version cursor.
fn current_version(table: &DeltaTable) -> Result<u64> {
    Ok(table.version().ok_or(DeltaTableError::NotInitialized)?)
}

/// Extract one table's committed state (declared schema, partition/sort
/// specs, and file list with partition tuples and sort-key bounds) from a
/// loaded delta table, as the in-memory [`TableManifest`].
fn extract_manifest(table: &DeltaTable) -> Result<TableManifest> {
    let version = current_version(table)?;
    let snapshot = table.snapshot()?;
    let metadata = snapshot.metadata();
    // Column mapping renames the physical parquet columns away from the
    // declared field names; the positional scan cannot follow it.
    if let Some(mode) = metadata.configuration().get(COLUMN_MAPPING_KEY)
        && mode != "none"
    {
        return Err(Error::ColumnMappingUnsupported(mode.clone()));
    }
    let columns: Vec<Column> = snapshot
        .schema()
        .fields()
        .map(parse_column)
        .collect::<Result<_>>()?;
    let partition_by: Vec<String> = metadata.partition_columns().to_vec();
    let sort_by: Vec<String> = match metadata.configuration().get(SORT_BY_KEY) {
        Some(raw) => serde_json::from_str(raw)?,
        None => Vec::new(),
    };
    let types: std::collections::HashMap<&str, &Type> = columns
        .iter()
        .map(|c| (c.name.as_str(), &c.col_type))
        .collect();

    let mut dated_entries = Vec::new();
    for file in snapshot.log_data().iter() {
        #[allow(deprecated)] // the Add action is exactly the record we read
        let add = file.add_action();
        if add.deletion_vector.is_some() {
            return Err(Error::DeletionVectorUnsupported(add.path));
        }
        // The Delta spec allows an add path to be an absolute URI (shallow
        // clones, converted tables); a FileRef path only resolves under the
        // table's location, so those tables are rejected, not mis-read.
        if add.path.starts_with('/') || add.path.contains("://") {
            return Err(Error::AbsoluteFilePath(add.path));
        }
        let size = u64::try_from(add.size).map_err(|_| Error::InvalidFileSize {
            path: add.path.clone(),
            size: add.size,
        })?;
        let mut tuple = serde_json::Map::new();
        for column in &partition_by {
            let Some(Some(raw)) = add.partition_values.get(column) else {
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
            add.modification_time,
            ManifestEntry {
                file: FileRef {
                    path: ObjectPath::new(add.path),
                    size,
                },
                partition: (!tuple.is_empty()).then_some(serde_json::Value::Object(tuple)),
                sort_bounds: decode_sort_bounds(add.stats.as_deref(), &sort_by),
            },
        ));
    }
    // Log replay surfaces the newest commit's files first; present them in
    // commit order instead (oldest first, ties keeping replay order), the
    // order every derived view presents files in.
    dated_entries.sort_by_key(|(modified, _)| *modified);
    let entries = dated_entries.into_iter().map(|(_, entry)| entry).collect();

    Ok(TableManifest {
        version,
        columns,
        partition_by,
        sort_by,
        entries,
    })
}

/// Read a declared column back from a delta schema field. The exact pivot
/// type in the field's metadata wins when present (a table this catalog
/// wrote); a field without it is mapped from its Delta-native type, erroring
/// on a type pivot has no equivalent for. A `pivot.type` value that is not a
/// string is malformed metadata, an error rather than a silent fall back to
/// the lossy native mapping.
fn parse_column(field: &StructField) -> Result<Column> {
    let col_type = match field.metadata().get(COLUMN_TYPE_KEY) {
        Some(MetadataValue::String(name)) => {
            serde_json::from_value(serde_json::Value::String(name.clone()))?
        }
        Some(_) => return Err(Error::MalformedColumnType(field.name().clone())),
        None => native_column_type(field)?,
    };
    Ok(Column {
        name: field.name().clone(),
        col_type,
    })
}

/// The pivot type of a delta field written by an external writer (no pivot
/// type metadata to trust), from its Delta-native type. Delta timestamps are
/// rejected: they are microsecond int64 in the parquet files, while pivot's
/// `Timestamp` scans second-unit int64, so mapping them would return values a
/// million times too large.
fn native_column_type(field: &StructField) -> Result<Type> {
    let unsupported = || Error::UnsupportedColumnType {
        column: field.name().clone(),
        delta_type: field.data_type().to_string(),
    };
    let DeltaDataType::Primitive(primitive) = field.data_type() else {
        return Err(unsupported());
    };
    Ok(match primitive {
        PrimitiveType::Boolean => Type::Boolean,
        PrimitiveType::Byte => Type::Int8,
        PrimitiveType::Short => Type::Int16,
        PrimitiveType::Integer => Type::Int32,
        PrimitiveType::Long => Type::Int64,
        PrimitiveType::Float => Type::Float32,
        PrimitiveType::Double => Type::Float64,
        PrimitiveType::String => Type::Utf8,
        PrimitiveType::Date => Type::Date,
        _ => return Err(unsupported()),
    })
}

/// One partition value from its delta string form to the JSON shape the
/// partition tuple records, or `None` when the delta form cannot faithfully
/// reproduce that shape. The tuple is a soft pruning aid (an absent value
/// keeps the file), so `None` never loses rows, it only skips the prune.
///
/// The recorded shape must equal what arrow-json emits for the type's
/// physical arrow form: partition pruning compares the tuple against a query
/// constant encoded through arrow-json (see `scalar_to_json` in
/// `catalog::binding`), and a value that differs in spelling silently prunes
/// files that match. Strings and dates pass through verbatim (delta and
/// arrow-json spell them identically); timestamps do not (delta's
/// space-separated micros form vs arrow-json's `T`-separated form), so they
/// are never recorded; numbers and booleans parse as JSON, with delta's
/// non-JSON float spellings (`NaN`, `Infinity`) left unrecorded.
fn decode_partition_value(col_type: &Type, raw: &str) -> Option<serde_json::Value> {
    match col_type {
        Type::Timestamp => None,
        Type::Utf8 | Type::Date => Some(serde_json::Value::String(raw.to_string())),
        _ => serde_json::from_str(raw).ok(),
    }
}

/// The delta stats of one file, only the fields the catalog reads (the
/// sort-key bounds as min/max), in the standard stats shape any delta writer
/// records.
#[derive(Serialize, Deserialize, Default)]
struct DeltaStats {
    #[serde(rename = "minValues", default)]
    min_values: serde_json::Map<String, serde_json::Value>,
    #[serde(rename = "maxValues", default)]
    max_values: serde_json::Map<String, serde_json::Value>,
}

/// Read a file's sort-key bounds from its stats JSON: the min/max of every
/// sort column, or `None` when the stats don't cover them all (a file written
/// without bounds). Bounds are a soft pruning aid, so absence is not an error.
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
