//! Delta Lake metadata loading for the catalog refresh path.
//!
//! Delta Kernel is the source of truth for a table's version, schema, and
//! active `Add` files.  Pivot then fetches those files' Parquet footers into
//! its existing in-memory scan representation.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Datum, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, Scalar, StringViewArray, TimestampSecondArray,
    new_null_array,
};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use delta_kernel::Snapshot;
use delta_kernel::expressions::Scalar as DeltaScalar;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::object_store::aws::AmazonS3Builder;
use delta_kernel::object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use delta_kernel::object_store::local::LocalFileSystem;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::schema::{DataType as DeltaDataType, PrimitiveType, StructField, StructType};
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::{DefaultEngine, DefaultEngineBuilder};
use planner::catalog::Column;
use planner::types::Type;
use url::Url;

use crate::manifest::{ManifestEntry, SortBounds};
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
    #[error("Delta table URI scheme `{0}` is not supported by the catalog")]
    UnsupportedScheme(String),
    #[error("cannot format Delta partition value for `{column}`: {message}")]
    PartitionFormat { column: String, message: String },
    #[error("Delta table has a non-UUID table id `{id}`: {source}")]
    InvalidTableId {
        id: String,
        #[source]
        source: uuid::Error,
    },
}

/// How Delta serializes timestamp partition values (and how Delta Kernel
/// parses them back): `yyyy-MM-dd HH:mm:ss[.SSSSSS]`.
const DELTA_TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.f";

/// Table-metadata configuration key holding the table's ordered sort columns,
/// comma-separated. Delta has no native sort spec, so it rides in the
/// `metaData` action's free-form `configuration` map.
const SORT_BY_CONFIGURATION_KEY: &str = "pivot.sortBy";

/// The Delta state needed to rebuild one in-memory catalog table.
pub(crate) struct DeltaTableState {
    /// The table's durable identity, from the Delta `metaData` action's `id`
    /// (minted once at creation, stable across every commit).
    pub id: uuid::Uuid,
    pub version: u64,
    pub columns: Vec<Column>,
    pub partition_by: Vec<String>,
    pub sort_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
}

/// Initialize version 0 for `CREATE TABLE`. Data files already exist; this
/// commit atomically adopts them as the table's initial Delta snapshot.
pub(crate) fn initialize_table(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    columns: &[Column],
    partition_by: &[String],
    sort_by: &[String],
    files: &[FileRef],
) -> Result<(Url, uuid::Uuid), Error> {
    let uri = table_uri(&store.describe(), location)?;
    let fields = columns
        .iter()
        .map(|column| {
            Ok(StructField::new(
                column.name.clone(),
                delta_type(&column.name, &column.col_type)?,
                false,
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let schema = StructType::try_new(fields)?;
    let schema = serde_json::to_string(&schema).expect("Delta Kernel schema is serializable");
    let mut configuration = serde_json::Map::new();
    if !sort_by.is_empty() {
        configuration.insert(
            SORT_BY_CONFIGURATION_KEY.to_string(),
            serde_json::Value::String(sort_by.join(",")),
        );
    }
    let id = uuid::Uuid::new_v4();
    let metadata = serde_json::json!({
        "metaData": {
            "id": id.to_string(),
            "format": {"provider": "parquet", "options": {}},
            "schemaString": schema,
            "partitionColumns": partition_by,
            "configuration": configuration,
        }
    });
    let mut actions = vec![
        serde_json::json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}),
        metadata,
    ];
    actions.extend(files.iter().map(|file| {
        serde_json::json!({
            "add": {
                "path": file.path.as_str(),
                "partitionValues": {},
                "size": file.size,
                "modificationTime": 0,
                "dataChange": true,
            }
        })
    }));
    let mut commit = actions
        .into_iter()
        .map(|action| serde_json::to_string(&action).expect("JSON value is serializable"))
        .collect::<Vec<_>>()
        .join("\n");
    commit.push('\n');
    let key = location
        .join("_delta_log")
        .join("00000000000000000000.json");
    if !store.put_if_absent(&key, commit.as_bytes())? {
        return Err(Error::Kernel(delta_kernel::Error::generic(
            "Delta table version 0 already exists",
        )));
    }
    Ok((uri, id))
}

/// Atomically append data-file Add actions at `version`. A false return means
/// another writer already committed that version; callers refresh and retry at
/// the next one.
pub(crate) fn append_files(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    version: u64,
    entries: &[ManifestEntry],
) -> Result<bool, Error> {
    let modification_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let lines = entries
        .iter()
        .map(|entry| add_action(entry, modification_time))
        .collect::<Result<Vec<_>, Error>>()?;
    commit_actions(store, location, version, lines)
}

/// Atomically swap data files at `version` — the compaction commit: one Delta
/// commit holding a Remove action per swapped-out input and an Add action per
/// merged output. A false return means another writer already committed that
/// version; callers refresh and retry at the next one.
pub(crate) fn replace_files(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    version: u64,
    removed: &[ObjectPath],
    added: &[ManifestEntry],
) -> Result<bool, Error> {
    let modification_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut lines = removed
        .iter()
        .map(|path| {
            serde_json::to_string(&serde_json::json!({
                "remove": {
                    "path": path.as_str(),
                    "deletionTimestamp": modification_time,
                    "dataChange": true,
                }
            }))
            .expect("Delta Remove action is serializable")
        })
        .collect::<Vec<_>>();
    for entry in added {
        lines.push(add_action(entry, modification_time)?);
    }
    commit_actions(store, location, version, lines)
}

/// Serialize one committed file as a Delta `Add` action line.
fn add_action(entry: &ManifestEntry, modification_time: u64) -> Result<String, Error> {
    let mut partition_values = serde_json::Map::new();
    if let Some(partition) = &entry.partition {
        for (column, scalar) in partition {
            partition_values.insert(column.clone(), format_partition_value(column, scalar)?);
        }
    }
    Ok(serde_json::to_string(&serde_json::json!({
        "add": {
            "path": entry.file.path.as_str(),
            "partitionValues": partition_values,
            "size": entry.file.size,
            "modificationTime": modification_time,
            "dataChange": true,
        }
    }))
    .expect("Delta Add action is serializable"))
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

/// Atomically write `lines` as the commit at `version`. A false return means
/// another writer already committed that version.
fn commit_actions(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    version: u64,
    lines: Vec<String>,
) -> Result<bool, Error> {
    let mut commit = lines.join("\n");
    commit.push('\n');
    let version_file = format!("{version:020}.json");
    let key = location.join("_delta_log").join(&version_file);
    Ok(store.put_if_absent(&key, commit.as_bytes())?)
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

/// Load the latest Delta snapshot and materialize its active file list.
pub(crate) fn load_table(uri: &Url) -> Result<DeltaTableState, Error> {
    let engine = build_engine(uri)?;
    let snapshot = Snapshot::builder_for(uri.clone()).build(&engine)?;
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
                col_type: pivot_type(field.name(), field.data_type())?,
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

    let scan = snapshot.clone().scan_builder().build()?;
    let mut files = Vec::new();
    for metadata in scan.scan_metadata(&engine)? {
        files = metadata?.visit_scan_files(files, collect_scan_file)?;
    }

    let by_name: HashMap<&str, &Type> = columns
        .iter()
        .map(|column| (column.name.as_str(), &column.col_type))
        .collect();
    let entries = files
        .into_iter()
        .map(|file| scan_file_entry(file, &by_name, &delta_types, &partition_by))
        .collect::<Result<Vec<_>, Error>>()?;

    let raw_id = snapshot.table_configuration().metadata().id();
    let id = uuid::Uuid::parse_str(raw_id).map_err(|source| Error::InvalidTableId {
        id: raw_id.to_string(),
        source,
    })?;

    Ok(DeltaTableState {
        id,
        version: snapshot.version(),
        columns,
        partition_by,
        sort_by,
        entries,
    })
}

fn build_engine(uri: &Url) -> Result<DefaultEngine<TokioBackgroundExecutor>, Error> {
    let store: Arc<DynObjectStore> = match uri.scheme() {
        "file" => Arc::new(LocalFileSystem::new()),
        "s3" | "s3a" => {
            let mut builder = AmazonS3Builder::from_env().with_url(uri.as_str());
            if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
                builder = builder
                    .with_endpoint(endpoint)
                    .with_allow_http(true)
                    .with_virtual_hosted_style_request(false);
            }
            Arc::new(builder.build()?)
        }
        "gs" => {
            let mut builder = GoogleCloudStorageBuilder::from_env().with_url(uri.as_str());
            if let Ok(endpoint) = std::env::var("STORAGE_EMULATOR_HOST") {
                let endpoint = if endpoint.contains("://") {
                    endpoint
                } else {
                    format!("http://{endpoint}")
                };
                builder = builder
                    .with_config(GoogleConfigKey::BaseUrl, endpoint)
                    .with_config(GoogleConfigKey::SkipSignature, "true");
            }
            Arc::new(builder.build()?)
        }
        scheme => return Err(Error::UnsupportedScheme(scheme.to_string())),
    };
    Ok(DefaultEngineBuilder::new(store).build())
}

fn collect_scan_file(files: &mut Vec<ScanFile>, file: ScanFile) {
    files.push(file);
}

fn scan_file_entry(
    file: ScanFile,
    column_types: &HashMap<&str, &Type>,
    delta_types: &HashMap<String, DeltaDataType>,
    partition_columns: &[String],
) -> Result<ManifestEntry, Error> {
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
    Ok(ManifestEntry {
        file: FileRef {
            path: ObjectPath::new(file.path),
            size,
        },
        partition,
        sort_bounds: None::<SortBounds>,
    })
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
            scalar => delta_scalar_to_pivot(column, scalar, pivot_type)?,
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

fn delta_scalar_to_pivot(
    column: &str,
    scalar: DeltaScalar,
    pivot_type: &Type,
) -> Result<Scalar<ArrayRef>, Error> {
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
        DeltaScalar::Timestamp(value) | DeltaScalar::TimestampNtz(value) => {
            erased(TimestampSecondArray::from(vec![value / 1_000_000]))
        }
        // An Int128 (HUGEINT) column travels through Delta as decimal(38, 0);
        // restore its exact Decimal128 physical representation. Every other
        // decimal executes as Float64 in Pivot, so its value lands there.
        DeltaScalar::Decimal(value) if matches!(pivot_type, Type::Int128) => erased(
            Decimal128Array::from(vec![value.bits()])
                .with_precision_and_scale(38, 0)
                .expect("decimal(38, 0) is a valid Decimal128 precision/scale"),
        ),
        DeltaScalar::Decimal(value) => {
            let divisor = 10_f64.powi(i32::from(value.scale()));
            erased(Float64Array::from(vec![value.bits() as f64 / divisor]))
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

fn pivot_type(column: &str, data_type: &DeltaDataType) -> Result<Type, Error> {
    let DeltaDataType::Primitive(primitive) = data_type else {
        return Err(Error::UnsupportedType {
            column: column.to_string(),
            data_type: format!("{data_type:?}"),
        });
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
        // decimal(38, 0) is how Int128 (HUGEINT) is written to Delta (see
        // `delta_type`); mapping it back keeps the column's physical
        // Decimal128(38, 0) layout intact across a reload.
        PrimitiveType::Decimal(decimal) if decimal.precision() == 38 && decimal.scale() == 0 => {
            Type::Int128
        }
        PrimitiveType::Decimal(_) => Type::Decimal,
        PrimitiveType::Binary | PrimitiveType::Void => {
            return Err(Error::UnsupportedType {
                column: column.to_string(),
                data_type: primitive.to_string(),
            });
        }
    };
    Ok(data_type)
}

fn delta_type(column: &str, data_type: &Type) -> Result<DeltaDataType, Error> {
    let primitive = match data_type {
        Type::Boolean => PrimitiveType::Boolean,
        Type::Int8 => PrimitiveType::Byte,
        Type::Int16 => PrimitiveType::Short,
        Type::Int32 => PrimitiveType::Integer,
        Type::Int64 => PrimitiveType::Long,
        Type::Float32 => PrimitiveType::Float,
        Type::Float64 => PrimitiveType::Double,
        Type::Utf8 => PrimitiveType::String,
        Type::Date => PrimitiveType::Date,
        Type::Timestamp => PrimitiveType::Timestamp,
        Type::Int128 => PrimitiveType::decimal(38, 0)?,
        Type::Decimal => PrimitiveType::decimal(38, 18)?,
        Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 => {
            return Err(Error::UnsupportedType {
                column: column.to_string(),
                data_type: data_type.to_string(),
            });
        }
    };
    Ok(primitive.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Datum;

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
            Scalar::new(Arc::new(TimestampSecondArray::from(vec![86_401])) as ArrayRef);

        let written = format_partition_value("ts", &scalar).unwrap();
        let restored = partition_scalar(
            "ts",
            written.as_str().unwrap(),
            &Type::Timestamp,
            &DeltaDataType::Primitive(PrimitiveType::Timestamp),
        )
        .unwrap();

        assert_eq!(written, serde_json::json!("1970-01-02 00:00:01"));
        assert_eq!(
            restored
                .get()
                .0
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .unwrap()
                .value(0),
            86_401
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

    /// Int128 is written to Delta as decimal(38, 0); the reload must restore
    /// the same type, not a generic Decimal (whose physical layout differs).
    #[test]
    fn hugeint_column_type_round_trips_through_delta() {
        let written = delta_type("total", &Type::Int128).unwrap();

        assert_eq!(pivot_type("total", &written).unwrap(), Type::Int128);
        let value = partition_scalar("total", "12", &Type::Int128, &written).unwrap();
        assert_eq!(
            value
                .get()
                .0
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .unwrap()
                .value(0),
            12
        );
    }
}
