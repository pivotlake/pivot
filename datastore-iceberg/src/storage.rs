//! Adapts Pivot's object-store I/O to Apache Iceberg's storage interfaces.
//!
//! Iceberg uses this layer to read metadata through Dispatch while preserving
//! catalog-vended S3 or GCS credentials for the corresponding data files.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use dispatch::DataFlowDispatcher;
use dispatch::io::{DispatchReader, FileRange, OpenFile};
use futures::stream::BoxStream;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, OutputFile, Storage, StorageConfig,
    StorageFactory,
};
use iceberg::{Error, ErrorKind, Result};
use iceberg_catalog_rest::StorageCredential;
use object_storage::{
    DataFile, ExternalStoreFactory, FileRef, GcsStore, ObjectPath, ObjectStore, S3Credentials,
    S3Store,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

const GCS_TOKEN: &str = "gcs.oauth2.token";
const S3_ACCESS_KEY_ID: &str = "s3.access-key-id";
const S3_SECRET_ACCESS_KEY: &str = "s3.secret-access-key";
const S3_SESSION_TOKEN: &str = "s3.session-token";
const S3_ENDPOINT: &str = "s3.endpoint";
const S3_REGION: &str = "s3.region";
const CLIENT_REGION: &str = "client.region";

/// Builds Apache Iceberg `FileIO` storage over Pivot's object-store layer.
///
/// The factory deliberately cannot be deserialized. Its credential resolver is
/// process-local authority supplied by the metastore, and silently replacing it
/// with ambient credentials after deserialization would cross that boundary.
#[derive(Clone)]
pub struct PivotStorageFactory {
    stores: Arc<dyn ExternalStoreFactory>,
    reader: DispatchReader,
    table_access: Arc<TableAccess>,
}

#[derive(Default)]
struct TableAccess {
    props: HashMap<String, String>,
    credentials: Arc<[StorageCredential]>,
}

impl PivotStorageFactory {
    pub fn new(stores: Arc<dyn ExternalStoreFactory>, dispatcher: DataFlowDispatcher) -> Self {
        Self {
            stores,
            reader: DispatchReader::new(dispatcher),
            table_access: Arc::new(TableAccess::default()),
        }
    }

    pub(crate) fn with_table_access(
        &self,
        props: HashMap<String, String>,
        credentials: Vec<StorageCredential>,
    ) -> Self {
        Self {
            stores: self.stores.clone(),
            reader: self.reader.clone(),
            table_access: Arc::new(TableAccess {
                props,
                credentials: credentials.into(),
            }),
        }
    }

    pub(crate) fn data_file(
        &self,
        path: &str,
        size: u64,
        config: &StorageConfig,
    ) -> Result<DataFile> {
        let storage = PivotStorage {
            stores: self.stores.clone(),
            reader: self.reader.clone(),
            config: config.clone(),
            table_access: self.table_access.clone(),
        };
        let (store, key) = storage.resolve(path)?;
        Ok(DataFile {
            file: FileRef {
                path: ObjectPath::new(path),
                size,
            },
            source: store.source(&key).map_err(storage_error)?,
            immutable: true,
        })
    }
}

impl Debug for PivotStorageFactory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PivotStorageFactory").finish()
    }
}

impl Serialize for PivotStorageFactory {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_unit_struct("PivotStorageFactory")
    }
}

impl<'de> Deserialize<'de> for PivotStorageFactory {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Err(de::Error::custom(
            "PivotStorageFactory requires a metastore credential resolver",
        ))
    }
}

#[typetag::serde]
impl StorageFactory for PivotStorageFactory {
    fn build(&self, config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(PivotStorage {
            stores: self.stores.clone(),
            reader: self.reader.clone(),
            config: config.clone(),
            table_access: self.table_access.clone(),
        }))
    }
}

#[derive(Clone)]
struct PivotStorage {
    stores: Arc<dyn ExternalStoreFactory>,
    reader: DispatchReader,
    config: StorageConfig,
    table_access: Arc<TableAccess>,
}

impl Debug for PivotStorage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PivotStorage").finish()
    }
}

impl Serialize for PivotStorage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_unit_struct("PivotStorage")
    }
}

impl<'de> Deserialize<'de> for PivotStorage {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Err(de::Error::custom(
            "PivotStorage requires a metastore credential resolver",
        ))
    }
}

impl PivotStorage {
    fn resolve(&self, path: &str) -> Result<(Arc<dyn ObjectStore>, ObjectPath)> {
        let mut effective_config = self
            .config
            .clone()
            .with_props(self.table_access.props.clone());
        if let Some(credential) = self
            .table_access
            .credentials
            .iter()
            .filter(|credential| path.starts_with(&credential.prefix))
            .max_by_key(|credential| credential.prefix.len())
        {
            effective_config = effective_config.with_props(credential.config.clone());
        }
        let (scheme, below_scheme) = path.split_once("://").ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Iceberg object path `{path}` has no URI scheme"),
            )
        })?;
        let (bucket, object) = below_scheme.split_once('/').unwrap_or((below_scheme, ""));
        if bucket.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Iceberg object path `{path}` has no bucket"),
            ));
        }

        let root = match scheme {
            "s3" | "s3a" => format!("s3://{bucket}"),
            "gs" | "gcs" => format!("gs://{bucket}"),
            _ => {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!("Iceberg v1 supports only s3:// and gs:// paths, got `{path}`"),
                ));
            }
        };

        let store: Arc<dyn ObjectStore> = match scheme {
            "s3" | "s3a" if has_s3_credentials(&effective_config) => {
                let access_key = required(&effective_config, S3_ACCESS_KEY_ID)?;
                let secret_key = required(&effective_config, S3_SECRET_ACCESS_KEY)?;
                Arc::new(
                    S3Store::with_credentials(
                        &root,
                        S3Credentials {
                            region: effective_config
                                .get(CLIENT_REGION)
                                .or_else(|| effective_config.get(S3_REGION))
                                .cloned(),
                            access_key,
                            secret_key,
                            session_token: effective_config.get(S3_SESSION_TOKEN).cloned(),
                            endpoint: effective_config.get(S3_ENDPOINT).cloned(),
                        },
                    )
                    .map_err(storage_error)?,
                )
            }
            "gs" | "gcs" if effective_config.get(GCS_TOKEN).is_some() => Arc::new(
                GcsStore::with_access_token(
                    &root,
                    required(&effective_config, GCS_TOKEN)?.as_str(),
                )
                .map_err(storage_error)?,
            ),
            _ => self.stores.open(&root).map_err(storage_error)?,
        };

        Ok((store, ObjectPath::new(format!("/{object}"))))
    }

    fn open_existing(&self, path: &str) -> Result<(OpenFile, u64)> {
        let (store, key) = self.resolve(path)?;
        let size = store
            .file_size(&key)
            .map_err(storage_error)?
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Iceberg object `{path}` does not exist"),
                )
            })?;
        let open_file = store
            .source(&key)
            .map_err(storage_error)?
            .open_immutable_read(size)
            .map_err(|error| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("opening Iceberg object `{path}` for Dispatch"),
                )
                .with_source(error)
            })?;
        Ok((open_file, size))
    }

    async fn open_existing_async(&self, path: &str) -> Result<(OpenFile, u64)> {
        let storage = self.clone();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || storage.open_existing(&path))
            .await
            .map_err(join_error)?
    }

    async fn size_async(&self, path: &str) -> Result<Option<u64>> {
        let storage = self.clone();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || {
            let (store, key) = storage.resolve(&path)?;
            store.file_size(&key).map_err(storage_error)
        })
        .await
        .map_err(join_error)?
    }

    async fn read_open_file(
        &self,
        path: &str,
        open_file: OpenFile,
        range: Range<u64>,
        size: u64,
    ) -> Result<Bytes> {
        if range.start > range.end || range.end > size {
            return Err(invalid_range(path, &range));
        }
        let offset = usize::try_from(range.start).map_err(|_| invalid_range(path, &range))?;
        let len =
            usize::try_from(range.end - range.start).map_err(|_| invalid_range(path, &range))?;
        let reader = self.reader.clone();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || reader.read(open_file, FileRange::new(offset, len)))
            .await
            .map_err(join_error)?
            .map(Bytes::from)
            .map_err(|error| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Dispatch read failed for Iceberg object `{path}`"),
                )
                .with_source(error)
            })
    }

    fn read_only_error(operation: &str) -> Error {
        Error::new(
            ErrorKind::FeatureUnsupported,
            format!("Iceberg datastore is read-only and cannot {operation}"),
        )
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for PivotStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        Ok(self.size_async(path).await?.is_some())
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        let size = self.size_async(path).await?.ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Iceberg object `{path}` does not exist"),
            )
        })?;
        Ok(FileMetadata { size })
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        let (open_file, size) = self.open_existing_async(path).await?;
        self.read_open_file(path, open_file, 0..size, size).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        let (open_file, size) = self.open_existing_async(path).await?;
        Ok(Box::new(PivotFileRead {
            storage: self.clone(),
            open_file,
            size,
            path: path.to_string(),
        }))
    }

    async fn write(&self, _path: &str, _bytes: Bytes) -> Result<()> {
        Err(Self::read_only_error("write objects"))
    }

    async fn writer(&self, _path: &str) -> Result<Box<dyn FileWrite>> {
        Err(Self::read_only_error("open an object writer"))
    }

    async fn delete(&self, _path: &str) -> Result<()> {
        Err(Self::read_only_error("delete objects"))
    }

    async fn delete_prefix(&self, _path: &str) -> Result<()> {
        Err(Self::read_only_error("delete object prefixes"))
    }

    async fn delete_stream(&self, _paths: BoxStream<'static, String>) -> Result<()> {
        Err(Self::read_only_error("delete objects"))
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, _path: &str) -> Result<OutputFile> {
        Err(Self::read_only_error("create an output file"))
    }
}

struct PivotFileRead {
    storage: PivotStorage,
    open_file: OpenFile,
    size: u64,
    path: String,
}

#[async_trait]
impl FileRead for PivotFileRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        self.storage
            .read_open_file(&self.path, self.open_file.clone(), range, self.size)
            .await
    }
}

fn invalid_range(path: &str, range: &Range<u64>) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!(
            "Iceberg object `{path}` cannot satisfy byte range {}..{}",
            range.start, range.end
        ),
    )
}

fn storage_error(error: object_storage::StoreError) -> Error {
    Error::new(ErrorKind::Unexpected, "Iceberg object-store access failed").with_source(error)
}

fn join_error(error: tokio::task::JoinError) -> Error {
    Error::new(ErrorKind::Unexpected, "Iceberg object-store task failed").with_source(error)
}

fn has_s3_credentials(config: &StorageConfig) -> bool {
    config.get(S3_ACCESS_KEY_ID).is_some()
        || config.get(S3_SECRET_ACCESS_KEY).is_some()
        || config.get(S3_SESSION_TOKEN).is_some()
}

fn required(config: &StorageConfig, key: &'static str) -> Result<String> {
    config.get(key).cloned().ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Iceberg storage configuration is missing `{key}`"),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use object_storage::{StoreConnection, StoreError};

    use super::*;

    #[derive(Debug)]
    struct RejectAmbientCredentials;

    impl ExternalStoreFactory for RejectAmbientCredentials {
        fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
            Err(StoreError::Config(format!(
                "unexpected ambient credential lookup for `{root_uri}`"
            )))
        }
    }

    struct TestDispatch(Option<dispatch::Dispatch>);

    impl TestDispatch {
        fn new() -> Self {
            Self(Some(dispatch::Dispatch::spin_up(1, 8, None)))
        }

        fn dispatcher(&self) -> DataFlowDispatcher {
            self.0.as_ref().unwrap().dispatcher().clone()
        }
    }

    impl Drop for TestDispatch {
        fn drop(&mut self) {
            self.0.take().unwrap().exit();
        }
    }

    fn create_storage(
        dispatch: &TestDispatch,
        credentials: Vec<StorageCredential>,
    ) -> PivotStorage {
        PivotStorage {
            stores: Arc::new(RejectAmbientCredentials),
            reader: DispatchReader::new(dispatch.dispatcher()),
            config: StorageConfig::new(),
            table_access: Arc::new(TableAccess {
                props: HashMap::new(),
                credentials: credentials.into(),
            }),
        }
    }

    fn create_credential(prefix: &str, entries: &[(&str, &str)]) -> StorageCredential {
        StorageCredential {
            prefix: prefix.to_string(),
            config: entries
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>(),
        }
    }

    #[test]
    fn longest_matching_s3_credential_wins_and_keeps_the_session_token() {
        let dispatch = TestDispatch::new();
        let storage = create_storage(
            &dispatch,
            vec![
                create_credential(
                    "s3://warehouse/",
                    &[
                        (S3_ACCESS_KEY_ID, "broad-key"),
                        (S3_SECRET_ACCESS_KEY, "broad-secret"),
                        (S3_REGION, "us-east-1"),
                    ],
                ),
                create_credential(
                    "s3://warehouse/private/",
                    &[
                        (S3_ACCESS_KEY_ID, "narrow-key"),
                        (S3_SECRET_ACCESS_KEY, "narrow-secret"),
                        (S3_SESSION_TOKEN, "temporary-token"),
                        (S3_REGION, "eu-west-1"),
                    ],
                ),
            ],
        );

        let (store, key) = storage
            .resolve("s3://warehouse/private/data.parquet")
            .unwrap();

        assert_eq!(key, ObjectPath::new("/private/data.parquet"));
        let StoreConnection::S3 { credentials, .. } = store.connection() else {
            panic!("expected S3 connection");
        };
        let credentials = credentials.expect("vended credentials must remain signed");
        assert_eq!(credentials.access_key, "narrow-key");
        assert_eq!(credentials.secret_key, "narrow-secret");
        assert_eq!(credentials.region.as_deref(), Some("eu-west-1"));
        assert_eq!(
            credentials.session_token.as_deref(),
            Some("temporary-token")
        );
    }

    #[test]
    fn gcs_vended_token_avoids_ambient_credentials() {
        let dispatch = TestDispatch::new();
        let storage = create_storage(
            &dispatch,
            vec![create_credential(
                "gs://warehouse/",
                &[(GCS_TOKEN, "vended-token")],
            )],
        );

        let (store, key) = storage
            .resolve("gs://warehouse/events/data.parquet")
            .unwrap();

        assert_eq!(key, ObjectPath::new("/events/data.parquet"));
        let StoreConnection::Gcs { access_token, .. } = store.connection() else {
            panic!("expected GCS connection");
        };
        assert_eq!(access_token.as_deref(), Some("vended-token"));
    }

    #[test]
    fn incomplete_vended_s3_credentials_fail_without_ambient_fallback() {
        let dispatch = TestDispatch::new();
        let storage = create_storage(
            &dispatch,
            vec![create_credential(
                "s3://warehouse/",
                &[(S3_ACCESS_KEY_ID, "key-without-secret")],
            )],
        );

        let error = storage.resolve("s3://warehouse/data.parquet").unwrap_err();

        assert!(error.to_string().contains(S3_SECRET_ACCESS_KEY), "{error}");
    }

    #[test]
    fn table_access_is_scoped_to_each_factory() {
        let dispatch = TestDispatch::new();
        let factory =
            PivotStorageFactory::new(Arc::new(RejectAmbientCredentials), dispatch.dispatcher())
                .with_table_access(
                    HashMap::new(),
                    vec![create_credential(
                        "s3://warehouse/",
                        &[
                            (S3_ACCESS_KEY_ID, "old-key"),
                            (S3_SECRET_ACCESS_KEY, "old-secret"),
                            (S3_REGION, "us-east-1"),
                        ],
                    )],
                );
        let storage = PivotStorage {
            stores: factory.stores.clone(),
            reader: factory.reader.clone(),
            config: StorageConfig::new(),
            table_access: factory.table_access.clone(),
        };

        let next_factory = factory.with_table_access(
            HashMap::new(),
            vec![create_credential(
                "s3://warehouse/",
                &[
                    (S3_ACCESS_KEY_ID, "new-key"),
                    (S3_SECRET_ACCESS_KEY, "new-secret"),
                    (S3_REGION, "us-east-1"),
                ],
            )],
        );

        let (store, _) = storage
            .resolve("s3://warehouse/events/data.parquet")
            .unwrap();
        let StoreConnection::S3 { credentials, .. } = store.connection() else {
            panic!("expected S3 connection");
        };
        assert_eq!(credentials.unwrap().access_key, "old-key");

        let next_storage = PivotStorage {
            stores: next_factory.stores.clone(),
            reader: next_factory.reader.clone(),
            config: StorageConfig::new(),
            table_access: next_factory.table_access.clone(),
        };
        let (store, _) = next_storage
            .resolve("s3://warehouse/events/data.parquet")
            .unwrap();
        let StoreConnection::S3 { credentials, .. } = store.connection() else {
            panic!("expected S3 connection");
        };
        assert_eq!(credentials.unwrap().access_key, "new-key");
    }
}
