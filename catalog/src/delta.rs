//! Delta Lake metadata loading for the catalog refresh path.
//!
//! Delta Kernel is the source of truth for a table's version, schema, and
//! active `Add` files.  Pivot then fetches those files' Parquet footers into
//! its existing in-memory scan representation.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Datum, Decimal64Array, Decimal128Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, Scalar,
    StringViewArray, TimestampSecondArray, new_null_array,
};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use delta_kernel::Snapshot;
use delta_kernel::expressions::Scalar as DeltaScalar;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::object_store::aws::AmazonS3Builder;
use delta_kernel::object_store::local::LocalFileSystem;
use delta_kernel::scan::state::ScanFile;
use delta_kernel::schema::{
    DataType as DeltaDataType, MetadataValue, PrimitiveType, StructField, StructType,
};
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::{DefaultEngine, DefaultEngineBuilder};
use planner::catalog::{Column, GenerationExpression};
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
    let uri = table_uri(&store.location_uri(), location)?;
    let fields = columns
        .iter()
        .map(build_delta_field)
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
    // A variant column is a Delta table feature: readers and writers must
    // declare `variantType`, which requires the feature-listing protocol
    // versions. Generated columns are a writer-only feature, since their values
    // are materialized into the data files like any other column and readers
    // need to know nothing. Tables with neither keep the plain legacy protocol
    // so any reader can open them.
    let has_variant = columns.iter().any(|c| matches!(c.col_type, Type::Variant));
    let has_generated = columns.iter().any(|c| c.generated.is_some());
    let protocol = if has_variant {
        let mut writer_features = vec!["variantType"];
        if has_generated {
            writer_features.push("generatedColumns");
        }
        serde_json::json!({"protocol": {
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": ["variantType"],
            "writerFeatures": writer_features,
        }})
    } else if has_generated {
        // Writer version 4 is the legacy version that introduced generated
        // columns, so no feature list is needed to declare them.
        serde_json::json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 4}})
    } else {
        serde_json::json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}})
    };
    let mut actions = vec![protocol, metadata];
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

/// Atomically commit a data-file change at `version`: a Remove action per path
/// in `removed` and an Add action per entry in `added`, in one Delta commit. An
/// empty `removed` is a plain append. `data_change` is stamped on every action:
/// `true` for a logical change (INSERT, DELETE), `false` for a rearrangement
/// that leaves the table's rows identical (compaction), which lets incremental
/// log readers skip it. A false return means another writer already committed
/// that version; callers refresh and retry at the next one.
pub(crate) fn commit_file_changes(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    version: u64,
    removed: &[ObjectPath],
    added: &[ManifestEntry],
    data_change: bool,
) -> Result<bool, Error> {
    let modification_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut lines = removed
        .iter()
        .map(|path| remove_action(path, modification_time, data_change))
        .collect::<Vec<_>>();
    for entry in added {
        lines.push(add_action(entry, modification_time, data_change)?);
    }
    commit_actions(store, location, version, lines)
}

/// Serialize one removed file as a Delta `Remove` action line.
fn remove_action(path: &ObjectPath, modification_time: u64, data_change: bool) -> String {
    serde_json::to_string(&serde_json::json!({
        "remove": {
            "path": path.as_str(),
            "deletionTimestamp": modification_time,
            "dataChange": data_change,
        }
    }))
    .expect("Delta Remove action is serializable")
}

/// Serialize one committed file as a Delta `Add` action line.
fn add_action(
    entry: &ManifestEntry,
    modification_time: u64,
    data_change: bool,
) -> Result<String, Error> {
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
            "dataChange": data_change,
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
            let col_type = pivot_type_from_field(field)?;
            Ok(match generation_expression_from_field(field) {
                Some(expression) => Column::generated(field.name().clone(), col_type, expression),
                None => Column::new(field.name().clone(), col_type),
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
        DeltaScalar::Timestamp(value) | DeltaScalar::TimestampNtz(value) => {
            erased(TimestampSecondArray::from(vec![value / 1_000_000]))
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
        Type::Timestamp => PrimitiveType::Timestamp,
        Type::Decimal { precision, scale } => PrimitiveType::decimal(*precision, *scale as u8)?,
        Type::Int128 | Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64 => {
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

/// Field-metadata key holding a generated column's expression. This is Delta's
/// own key for the generated-columns writer feature, so a table pivot writes
/// stays readable by any Delta engine that understands it.
const DELTA_GENERATION_EXPRESSION_KEY: &str = "delta.generationExpression";

/// Build the Delta struct field for one Pivot column. Types Delta represents
/// natively map straight through [`delta_type`]; the unsigned types it rejects
/// fall back to [`build_unsigned_field`], which stores them as a tagged signed
/// primitive. A generated column carries its expression in field metadata.
fn build_delta_field(column: &Column) -> Result<StructField, Error> {
    let field = match delta_type(&column.name, &column.col_type) {
        Ok(delta_type) => StructField::new(&column.name, delta_type, false),
        Err(_) => build_unsigned_field(&column.name, &column.col_type)?,
    };
    Ok(match &column.generated {
        Some(expression) => {
            field.add_metadata([(DELTA_GENERATION_EXPRESSION_KEY, expression.sql())])
        }
        None => field,
    })
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

/// Recover a generated column's expression from its Delta field metadata.
fn generation_expression_from_field(field: &StructField) -> Option<GenerationExpression> {
    match field.metadata.get(DELTA_GENERATION_EXPRESSION_KEY) {
        Some(MetadataValue::String(expression)) => Some(GenerationExpression::new(expression)),
        _ => None,
    }
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
        let stored_primitive = |ty| match build_delta_field(&Column::new("c", ty))
            .unwrap()
            .data_type()
        {
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
            let field = build_delta_field(&Column::new("event_date", ty.clone())).unwrap();
            assert_eq!(pivot_type_from_field(&field).unwrap(), ty);
        }
    }

    /// A signed column carries no tag, so it round-trips through the native
    /// primitive mapping without one.
    #[test]
    fn signed_columns_round_trip_without_a_metadata_tag() {
        let field = build_delta_field(&Column::new("id", Type::Int32)).unwrap();
        assert!(!field.metadata.contains_key(PIVOT_LOGICAL_TYPE_KEY));
        assert_eq!(pivot_type_from_field(&field).unwrap(), Type::Int32);
    }

    /// A generated column's expression is stored under Delta's own key and comes
    /// back with the column, so a table reopened from its log still knows which
    /// columns it computes on write.
    #[test]
    fn generation_expression_round_trips_through_field_metadata() {
        let column = Column::generated(
            "day",
            Type::Int32,
            GenerationExpression::new("date_trunc('day', ts)"),
        );

        let field = build_delta_field(&column).unwrap();

        assert_eq!(
            generation_expression_from_field(&field),
            Some(GenerationExpression::new("date_trunc('day', ts)"))
        );
    }

    /// The generation expression rides alongside the type tag rather than
    /// replacing it, so an unsigned generated column keeps both.
    #[test]
    fn a_generated_unsigned_column_keeps_its_type_tag() {
        let column = Column::generated(
            "bucket",
            Type::UInt32,
            GenerationExpression::new("id % 100"),
        );

        let field = build_delta_field(&column).unwrap();

        assert_eq!(pivot_type_from_field(&field).unwrap(), Type::UInt32);
        assert_eq!(
            generation_expression_from_field(&field),
            Some(GenerationExpression::new("id % 100"))
        );
    }
}
