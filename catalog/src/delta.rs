//! Delta Lake metadata loading for the catalog refresh path.
//!
//! Delta Kernel is the source of truth for a table's version, schema, and
//! active `Add` files.  Pivot then fetches those files' Parquet footers into
//! its existing in-memory scan representation.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Datum, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, Scalar, StringViewArray, TimestampSecondArray,
};
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
}

/// The Delta state needed to rebuild one in-memory catalog table.
pub(crate) struct DeltaTableState {
    pub version: u64,
    pub columns: Vec<Column>,
    pub partition_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
}

/// Initialize version 0 for `CREATE TABLE`. Data files already exist; this
/// commit atomically adopts them as the table's initial Delta snapshot.
pub(crate) fn initialize_table(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    columns: &[Column],
    partition_by: &[String],
    files: &[FileRef],
) -> Result<Url, Error> {
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
    let metadata = serde_json::json!({
        "metaData": {
            "id": uuid::Uuid::new_v4().to_string(),
            "format": {"provider": "parquet", "options": {}},
            "schemaString": schema,
            "partitionColumns": partition_by,
            "configuration": {},
        }
    });
    // A variant column is a Delta table feature: readers and writers must
    // declare `variantType`, which requires the feature-listing protocol
    // versions. Tables without one keep the plain legacy protocol so any
    // reader can open them.
    let protocol = if columns.iter().any(|c| matches!(c.col_type, Type::Variant)) {
        serde_json::json!({"protocol": {
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": ["variantType"],
            "writerFeatures": ["variantType"],
        }})
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
    Ok(uri)
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

    Ok(DeltaTableState {
        version: snapshot.version(),
        columns,
        partition_by,
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
    let scalar = primitive.parse_scalar(raw)?;
    let scalar = delta_scalar_to_pivot(column, scalar)?;
    let actual = scalar.get().0.data_type().clone();
    let expected = planner::types::physical_arrow_type(pivot_type);
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
        // Pivot currently executes DECIMAL as Float64; retain that physical
        // representation while preserving the logical type in the table schema.
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
    if let DeltaDataType::Variant(_) = data_type {
        return Ok(Type::Variant);
    }
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
}
