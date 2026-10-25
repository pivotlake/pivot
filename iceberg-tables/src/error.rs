//! What can go wrong reading, writing or sweeping an Iceberg table's
//! metadata and data files.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error from the spec crate while building, writing or decoding
    /// metadata.
    #[error("Iceberg metadata: {0}")]
    Spec(#[source] Box<iceberg::Error>),
    /// An error from a catalog service while loading a table or publishing
    /// a version of it.
    #[error("Iceberg catalog: {0}")]
    Catalog(#[source] Box<iceberg::Error>),
    #[error(
        "table `{table}` is created through its catalog's own create call, not by publishing a first version"
    )]
    CatalogCreatesTables { table: String },
    #[error("metadata json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    /// A read of metadata objects over the worker pool failed.
    #[error(transparent)]
    Load(#[from] dispatch::DataFlowError),
    #[error("column `{column}` has Pivot type `{pivot_type}`, which Iceberg cannot store")]
    UnsupportedPivotType { column: String, pivot_type: String },
    #[error(
        "table `{table}` column `{column}` has Iceberg type `{iceberg_type}`, which Pivot cannot represent"
    )]
    UnsupportedColumnType {
        table: String,
        column: String,
        iceberg_type: String,
    },
    #[error(
        "table `{table}` column `{column}` has initial default {default}; a column absent from a file is only read as NULL, so the table is not served"
    )]
    UnsupportedColumnDefault {
        table: String,
        column: String,
        default: String,
    },
    #[error("table `{table}` is format version {version}; only versions 2 and 3 are written")]
    UnsupportedFormatVersion { table: String, version: u8 },
    #[error(
        "table `{table}` partition field `{field}` has no source among the table's columns, so its files cannot be partitioned"
    )]
    UnsupportedPartitionSource { table: String, field: String },
    #[error(
        "table `{table}` partition field `{field}` is named like a column, so its key could not be told from the column's values"
    )]
    PartitionFieldNamedLikeColumn { table: String, field: String },
    #[error(
        "table `{table}` partition field `{field}` applies `{transform}`, which cannot be computed for its source column: {message}"
    )]
    UnsupportedPartitionTransform {
        table: String,
        field: String,
        transform: String,
        message: String,
    },
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
    #[error("metadata location `{location}` is not a numbered metadata file of a directory table")]
    UnnumberedMetadataLocation { location: String },
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
    #[error("file location `{path}` names no supported store: {message}")]
    UnsupportedLocation { path: String, message: String },
    #[error(
        "table `{table}` file `{file}` is outside `{root}`, where the table's metadata lives; a table's files must all live in one bucket"
    )]
    FileOutsideTableStore {
        table: String,
        file: String,
        root: String,
    },
    #[error("table `{table}` was vended an S3 access key without its secret key, or the reverse")]
    IncompleteVendedCredentials { table: String },
    #[error(
        "table `{table}` was vended keys whose expiry `{value}` is not a Unix millisecond timestamp"
    )]
    MalformedVendedExpiry { table: String, value: String },
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
    #[error("the commit lost the race {attempts} times; another writer keeps landing first")]
    CommitRacesExhausted { attempts: usize },
    #[error("table `{table}` is encrypted; encrypted tables are not read")]
    EncryptedTable { table: String },
    #[error("table `{table}` partition metadata is invalid: {source}")]
    InvalidPartitionMetadata {
        table: String,
        #[source]
        source: Box<iceberg::Error>,
    },
    #[error(
        "delete file `{file}` is not a deletion vector; only deletion vectors are applied, so the table is not read"
    )]
    DeleteFile { file: String },
    #[error("the deletion vector for `{file}` is invalid: {message}")]
    DeletionVector { file: String, message: String },
}

impl Error {
    pub fn spec(error: iceberg::Error) -> Self {
        Self::Spec(Box::new(error))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
