//! What can go wrong reading, writing or sweeping an Iceberg table's own
//! metadata.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error from the spec crate while building, writing or decoding this
    /// backend's own metadata.
    #[error("Iceberg metadata: {0}")]
    Spec(#[source] Box<iceberg::Error>),
    #[error("metadata json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    /// An error from the Iceberg reading crate: locating a file, mapping the
    /// schema's field ids onto the declared columns.
    #[error(transparent)]
    Reading(#[from] datastore_iceberg::Error),
    #[error("column `{column}` has Pivot type `{pivot_type}`, which Iceberg cannot store")]
    UnsupportedPivotType { column: String, pivot_type: String },
    #[error("`{option}` column `{column}` is not a declared column of the table")]
    UnknownSpecColumn { option: String, column: String },
    #[error(
        "table `{table}` carries delete files in its current snapshot (manifest `{manifest}`); row-level deletes are not applied, so the table is not served"
    )]
    DeleteFiles { table: String, manifest: String },
    #[error("table `{table}` data file `{file}` is {format}; only Parquet data files are read")]
    NonParquetFile {
        table: String,
        file: String,
        format: String,
    },
    #[error("table property `{property}` is malformed: {message}")]
    MalformedProperty { property: String, message: String },
    #[error("version hint `{content}` is not a padded metadata version")]
    MalformedVersionHint { content: String },
    #[error("table `{table}` metadata version {version} is not in the store")]
    MetadataVersionMissing { table: String, version: u64 },
    #[error("table `{table}` at `{location}` has no metadata file")]
    TableMetadataMissing { table: String, location: String },
    #[error("table `{table}` at `{location}` already has a first metadata version")]
    TableAlreadyInitialized { table: String, location: String },
    #[error("table `{table}` metadata object `{path}` is not in the store")]
    MissingMetadataObject { table: String, path: String },
    #[error("table `{table}` metadata object `{path}` does not parse: {source}")]
    MalformedMetadataObject {
        table: String,
        path: String,
        #[source]
        source: Box<iceberg::Error>,
    },
    #[error(
        "table `{table}` file `{file}` is outside `{root}`, where the table's metadata lives; a table's files must all live in one bucket"
    )]
    FileOutsideTableStore {
        table: String,
        file: String,
        root: String,
    },
    #[error("file `{file}` has no statistics, which a manifest entry requires")]
    MissingFileStats { file: String },
    #[error(
        "file `{file}` cannot be adopted into a partitioned table: its partition columns are not constant across the file"
    )]
    PartitionNotConstant { file: String },
    #[error("file `{file}` cannot be recorded in a manifest: {message}")]
    InvalidDataFile { file: String, message: String },
    #[error("file `{file}` was never committed, so it cannot be removed")]
    UncommittedRemoval { file: String },
    #[error("manifest `{manifest}` is not one of the current snapshot's")]
    UnknownManifest { manifest: String },
    #[error("a commit write failed {attempts} times; last: {source}")]
    CommitRetriesExhausted {
        attempts: usize,
        #[source]
        source: object_storage::StoreError,
    },
}

impl Error {
    pub(crate) fn spec(error: iceberg::Error) -> Self {
        Self::Spec(Box::new(error))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
