//! Loads Iceberg tables from a REST catalog.
//!
//! This module owns catalog authentication and turns each load-table response
//! into an Apache Iceberg table plus the storage credentials needed to read it.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::{Duration, Instant};

use iceberg::io::FileIOBuilder;
use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Runtime, TableIdent};
use iceberg_catalog_rest::{ErrorResponse, LoadTableResult};
use reqwest::{RequestBuilder, StatusCode};
use serde::Deserialize;
use tokio::sync::{Mutex, OnceCell};

use crate::{IcebergAuth, IcebergConfig, PivotStorageFactory};

const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(30);

pub(crate) struct RestTableLoader {
    catalog_uri: String,
    warehouse: Option<String>,
    auth: IcebergAuth,
    client: reqwest::Client,
    token: Mutex<Option<CachedToken>>,
    context: OnceCell<RestContext>,
    storage: Arc<PivotStorageFactory>,
    runtime: Runtime,
}

pub(crate) struct LoadedTable {
    pub(crate) table: Arc<Table>,
    pub(crate) storage: Arc<PivotStorageFactory>,
}

impl Debug for RestTableLoader {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RestTableLoader")
            .field("catalog_uri", &self.catalog_uri)
            .field("warehouse", &self.warehouse)
            .field("auth", &self.auth)
            .finish_non_exhaustive()
    }
}

struct RestContext {
    uri: String,
    prefix: Option<String>,
    props: HashMap<String, String>,
}

struct CachedToken {
    value: String,
    refresh_at: Option<Instant>,
}

#[derive(Deserialize)]
struct CatalogConfigResponse {
    #[serde(default)]
    defaults: HashMap<String, String>,
    #[serde(default)]
    overrides: HashMap<String, String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

impl RestTableLoader {
    pub(crate) fn new(
        config: &IcebergConfig,
        storage: Arc<PivotStorageFactory>,
        runtime: Runtime,
    ) -> Self {
        Self {
            catalog_uri: config.catalog_uri.trim_end_matches('/').to_string(),
            warehouse: config.warehouse.clone(),
            auth: config.auth.clone(),
            client: reqwest::Client::new(),
            token: Mutex::new(None),
            context: OnceCell::new(),
            storage,
            runtime,
        }
    }

    pub(crate) async fn load_table(&self, ident: &TableIdent) -> iceberg::Result<LoadedTable> {
        let context = self.context().await?;
        let endpoint = build_table_endpoint(context, ident)?;
        let response = self.send(self.client.get(endpoint)).await?;
        let response = match response.status() {
            StatusCode::OK => response
                .json::<LoadTableResult>()
                .await
                .map_err(http_error)?,
            StatusCode::NOT_FOUND => {
                return Err(Error::new(
                    ErrorKind::TableNotFound,
                    format!("Iceberg table `{ident}` does not exist"),
                ));
            }
            _ => return Err(response_error(response).await),
        };

        let mut props = context.props.clone();
        props.extend(response.config);
        let credentials = response.storage_credentials.unwrap_or_default();

        let storage = Arc::new(self.storage.with_table_access(props.clone(), credentials));
        let file_io = FileIOBuilder::new(storage.clone())
            .with_props(props.clone())
            .build();
        let metadata_location = response.metadata_location;
        let builder = Table::builder()
            .identifier(ident.clone())
            .file_io(file_io)
            .metadata(response.metadata)
            .runtime(self.runtime.clone())
            .readonly(true);
        let table = match metadata_location {
            Some(location) => builder.metadata_location(location).build(),
            None => builder.build(),
        }?;
        Ok(LoadedTable {
            table: Arc::new(table),
            storage,
        })
    }

    async fn context(&self) -> iceberg::Result<&RestContext> {
        self.context
            .get_or_try_init(|| async {
                let mut request = self.client.get(format!("{}/v1/config", self.catalog_uri));
                if let Some(warehouse) = &self.warehouse {
                    request = request.query(&[("warehouse", warehouse)]);
                }
                let response = self.send(request).await?;
                if response.status() != StatusCode::OK {
                    return Err(response_error(response).await);
                }
                let config = response
                    .json::<CatalogConfigResponse>()
                    .await
                    .map_err(http_error)?;
                let mut props = config.defaults;
                props.extend(config.overrides.clone());
                let uri = config
                    .overrides
                    .get("uri")
                    .cloned()
                    .unwrap_or_else(|| self.catalog_uri.clone());
                let prefix = props.get("prefix").cloned();
                Ok(RestContext { uri, prefix, props })
            })
            .await
    }

    async fn send(&self, request: RequestBuilder) -> iceberg::Result<reqwest::Response> {
        let request = match &self.auth {
            IcebergAuth::None => request,
            IcebergAuth::Bearer { token } => request.bearer_auth(token),
            IcebergAuth::OAuth2ClientCredentials { .. } => {
                request.bearer_auth(self.oauth_token().await?)
            }
        };
        request.send().await.map_err(http_error)
    }

    async fn oauth_token(&self) -> iceberg::Result<String> {
        let mut cached = self.token.lock().await;
        if let Some(token) = cached.as_ref()
            && token
                .refresh_at
                .is_none_or(|refresh_at| refresh_at > Instant::now())
        {
            return Ok(token.value.clone());
        }
        let IcebergAuth::OAuth2ClientCredentials {
            client_id,
            client_secret,
            scope,
            token_endpoint,
        } = &self.auth
        else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "OAuth token requested for non-OAuth Iceberg catalog auth",
            ));
        };
        let endpoint = token_endpoint
            .clone()
            .unwrap_or_else(|| format!("{}/v1/oauth/tokens", self.catalog_uri));
        let mut form = vec![
            ("grant_type", "client_credentials".to_string()),
            ("client_id", client_id.clone()),
            ("client_secret", client_secret.clone()),
        ];
        form.push((
            "scope",
            scope.clone().unwrap_or_else(|| "catalog".to_string()),
        ));
        let response = self
            .client
            .post(endpoint)
            .form(&form)
            .send()
            .await
            .map_err(http_error)?;
        if response.status() != StatusCode::OK {
            return Err(response_error(response).await);
        }
        let response = response.json::<TokenResponse>().await.map_err(http_error)?;
        let refresh_at = response.expires_in.map(|lifetime| {
            Instant::now() + Duration::from_secs(lifetime).saturating_sub(TOKEN_REFRESH_MARGIN)
        });
        let value = response.access_token;
        *cached = Some(CachedToken {
            value: value.clone(),
            refresh_at,
        });
        Ok(value)
    }
}

async fn response_error(response: reqwest::Response) -> Error {
    let status = response.status();
    match response.json::<ErrorResponse>().await {
        Ok(response) => response.into(),
        Err(source) => Error::new(
            ErrorKind::Unexpected,
            format!("Iceberg REST catalog returned HTTP {status}"),
        )
        .with_source(source),
    }
}

fn http_error(error: reqwest::Error) -> Error {
    Error::new(ErrorKind::Unexpected, "Iceberg REST request failed").with_source(error)
}

fn invalid_uri(uri: &str, error: url::ParseError) -> Error {
    Error::new(
        ErrorKind::DataInvalid,
        format!("invalid Iceberg REST catalog URI `{uri}`"),
    )
    .with_source(error)
}

fn build_table_endpoint(
    context: &RestContext,
    ident: &TableIdent,
) -> iceberg::Result<reqwest::Url> {
    let mut endpoint = reqwest::Url::parse(context.uri.trim_end_matches('/'))
        .map_err(|error| invalid_uri(&context.uri, error))?;
    {
        let mut segments = endpoint.path_segments_mut().map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Iceberg REST catalog URI `{}` cannot be a base URL",
                    context.uri
                ),
            )
        })?;
        segments.pop_if_empty().push("v1");
        if let Some(prefix) = context.prefix.as_deref() {
            for segment in prefix.split('/').filter(|segment| !segment.is_empty()) {
                segments.push(segment);
            }
        }
        segments
            .push("namespaces")
            .push(&ident.namespace.to_url_string())
            .push("tables")
            .push(&ident.name);
    }
    Ok(endpoint)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use catalog::datastore::Datastore;
    use iceberg::{NamespaceIdent, Runtime, TableIdent};
    use mockito::Matcher;
    use object_storage::{AmbientExternalStoreFactory, DataFileLocation};
    use planner::catalog::SchemaQualifiedTableName;

    use super::*;
    use crate::IcebergDatastore;

    struct Harness {
        tokio: tokio::runtime::Runtime,
        storage: Arc<PivotStorageFactory>,
        dispatch: Option<dispatch::Dispatch>,
    }

    impl Harness {
        fn new() -> Self {
            let dispatch = dispatch::Dispatch::spin_up(1, 8, None);
            Self {
                tokio: tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap(),
                storage: Arc::new(PivotStorageFactory::new(
                    Arc::new(AmbientExternalStoreFactory),
                    dispatch.dispatcher().clone(),
                )),
                dispatch: Some(dispatch),
            }
        }

        fn build_loader(&self, catalog_uri: &str, auth: IcebergAuth) -> RestTableLoader {
            let mut config = IcebergConfig::new(catalog_uri);
            config.auth = auth;
            RestTableLoader::new(&config, self.storage.clone(), Runtime::new(&self.tokio))
        }

        fn send(&self, loader: &RestTableLoader, url: &str) -> iceberg::Result<StatusCode> {
            self.tokio
                .block_on(loader.send(loader.client.get(url)))
                .map(|response| response.status())
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            self.dispatch.take().unwrap().exit();
        }
    }

    fn table_response(metadata_location: &str, access_key: &str, session_token: &str) -> String {
        serde_json::json!({
            "metadata-location": metadata_location,
            "metadata": {
                "format-version": 2,
                "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
                "location": "s3://warehouse/events",
                "last-sequence-number": 0,
                "last-updated-ms": 1602638573590_u64,
                "last-column-id": 1,
                "current-schema-id": 0,
                "schemas": [{
                    "type": "struct",
                    "schema-id": 0,
                    "fields": [{"id": 1, "name": "id", "required": true, "type": "long"}]
                }],
                "default-spec-id": 0,
                "partition-specs": [{"spec-id": 0, "fields": []}],
                "last-partition-id": 999,
                "default-sort-order-id": 0,
                "sort-orders": [{"order-id": 0, "fields": []}]
            },
            "config": {},
            "storage-credentials": [{
                "prefix": "s3://warehouse/",
                "config": {
                    "s3.access-key-id": access_key,
                    "s3.secret-access-key": "vended-secret",
                    "s3.session-token": session_token,
                    "s3.region": "us-east-1"
                }
            }]
        })
        .to_string()
    }

    #[test]
    fn table_endpoint_encodes_namespace_and_table_segments() {
        let context = RestContext {
            uri: "https://catalog.example/root".to_string(),
            prefix: Some("warehouse/path".to_string()),
            props: HashMap::new(),
        };
        let ident = TableIdent::new(
            NamespaceIdent::from_vec(vec!["team space".into(), "events".into()]).unwrap(),
            "daily/report".to_string(),
        );

        let endpoint = build_table_endpoint(&context, &ident).unwrap();

        assert_eq!(
            endpoint.as_str(),
            "https://catalog.example/root/v1/warehouse/path/namespaces/team%20space%1Fevents/tables/daily%2Freport"
        );
    }

    #[test]
    fn no_auth_sends_no_authorization_header() {
        let mut server = mockito::Server::new();
        let request = server
            .mock("GET", "/resource")
            .match_header("authorization", Matcher::Missing)
            .with_status(204)
            .create();
        let harness = Harness::new();
        let loader = harness.build_loader(&server.url(), IcebergAuth::None);

        assert_eq!(
            harness
                .send(&loader, &format!("{}/resource", server.url()))
                .unwrap(),
            StatusCode::NO_CONTENT
        );
        request.assert();
    }

    #[test]
    fn bearer_auth_sends_configured_token() {
        let mut server = mockito::Server::new();
        let request = server
            .mock("GET", "/resource")
            .match_header("authorization", "Bearer catalog-token")
            .with_status(204)
            .create();
        let harness = Harness::new();
        let loader = harness.build_loader(
            &server.url(),
            IcebergAuth::Bearer {
                token: "catalog-token".to_string(),
            },
        );

        assert_eq!(
            harness
                .send(&loader, &format!("{}/resource", server.url()))
                .unwrap(),
            StatusCode::NO_CONTENT
        );
        request.assert();
    }

    #[test]
    fn oauth_client_credentials_exchanges_and_caches_a_token() {
        let mut server = mockito::Server::new();
        let token = server
            .mock("POST", "/token")
            .match_body(Matcher::AllOf(vec![
                Matcher::UrlEncoded("grant_type".into(), "client_credentials".into()),
                Matcher::UrlEncoded("client_id".into(), "client".into()),
                Matcher::UrlEncoded("client_secret".into(), "secret".into()),
                Matcher::UrlEncoded("scope".into(), "catalog".into()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"access_token":"oauth-token","expires_in":3600}"#)
            .expect(1)
            .create();
        let resource = server
            .mock("GET", "/resource")
            .match_header("authorization", "Bearer oauth-token")
            .with_status(204)
            .expect(2)
            .create();
        let harness = Harness::new();
        let loader = harness.build_loader(
            &server.url(),
            IcebergAuth::OAuth2ClientCredentials {
                client_id: "client".to_string(),
                client_secret: "secret".to_string(),
                scope: None,
                token_endpoint: Some(format!("{}/token", server.url())),
            },
        );
        let resource_url = format!("{}/resource", server.url());

        assert_eq!(
            harness.send(&loader, &resource_url).unwrap(),
            StatusCode::NO_CONTENT
        );
        assert_eq!(
            harness.send(&loader, &resource_url).unwrap(),
            StatusCode::NO_CONTENT
        );
        token.assert();
        resource.assert();
    }

    #[test]
    fn each_transaction_loads_the_table_once_and_disables_plan_reuse() {
        let mut server = mockito::Server::new();
        let config = server
            .mock("GET", "/v1/config")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"defaults":{},"overrides":{}}"#)
            .create();
        let table = server
            .mock("GET", "/v1/namespaces/main/tables/events")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(table_response(
                "s3://warehouse/events/metadata/00001.json",
                "vended-key",
                "vended-session",
            ))
            .expect(2)
            .create();
        let harness = Harness::new();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let datastore = Arc::new(runtime.block_on(async {
            IcebergDatastore::new(IcebergConfig::new(server.url()), harness.storage.clone())
                .unwrap()
        }));
        let name = SchemaQualifiedTableName::new("main", "events");
        let transaction = datastore.clone().begin_transaction();
        let binding = transaction.bind_table("lake", &name).unwrap().unwrap();
        assert!(!binding.is_plan_cacheable());
        let rebound = transaction.bind_table("lake", &name).unwrap().unwrap();
        assert_eq!(binding.table_revision(), rebound.table_revision());

        let next_transaction = datastore.begin_transaction();
        let next_binding = next_transaction.bind_table("lake", &name).unwrap().unwrap();
        assert!(!next_binding.is_plan_cacheable());
        assert_eq!(binding.table_revision(), next_binding.table_revision());
        config.assert();
        table.assert();
    }

    #[test]
    fn each_load_builds_a_fresh_table_with_its_own_credentials() {
        let mut server = mockito::Server::new();
        let config = server
            .mock("GET", "/v1/config")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"defaults":{},"overrides":{}}"#)
            .create();
        let first_table = server
            .mock("GET", "/v1/namespaces/main/tables/events")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(table_response(
                "s3://warehouse/events/metadata/00001.json",
                "vended-key",
                "vended-session",
            ))
            .expect(1)
            .create();
        let harness = Harness::new();
        let loader = harness.build_loader(&server.url(), IcebergAuth::None);
        let ident = TableIdent::new(
            NamespaceIdent::new("main".to_string()),
            "events".to_string(),
        );

        let loaded = harness.tokio.block_on(loader.load_table(&ident)).unwrap();
        let data_file = loaded
            .storage
            .data_file(
                "s3://warehouse/events/data.parquet",
                10,
                loaded.table.file_io().config(),
            )
            .unwrap();

        let DataFileLocation::Remote { url, auth } = data_file.source else {
            panic!("expected remote S3 source");
        };
        assert!(auth.is_none());
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(
            query.get("X-Amz-Security-Token").map(String::as_str),
            Some("vended-session")
        );
        assert!(
            query
                .get("X-Amz-Credential")
                .is_some_and(|credential| credential.starts_with("vended-key/"))
        );
        first_table.assert();
        first_table.remove();

        let refreshed_table = server
            .mock("GET", "/v1/namespaces/main/tables/events")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(table_response(
                "s3://warehouse/events/metadata/00001.json",
                "refreshed-key",
                "refreshed-session",
            ))
            .expect(1)
            .create();
        let reloaded = harness.tokio.block_on(loader.load_table(&ident)).unwrap();
        assert!(!Arc::ptr_eq(&loaded.table, &reloaded.table));
        assert!(!Arc::ptr_eq(&loaded.storage, &reloaded.storage));
        assert_eq!(
            loaded.table.metadata_location(),
            reloaded.table.metadata_location()
        );
        let refreshed_file = reloaded
            .storage
            .data_file(
                "s3://warehouse/events/data.parquet",
                10,
                reloaded.table.file_io().config(),
            )
            .unwrap();
        let DataFileLocation::Remote { url, .. } = refreshed_file.source else {
            panic!("expected remote S3 source");
        };
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(
            query.get("X-Amz-Security-Token").map(String::as_str),
            Some("refreshed-session")
        );
        assert!(
            query
                .get("X-Amz-Credential")
                .is_some_and(|credential| credential.starts_with("refreshed-key/"))
        );

        // Loading again does not alter credentials held by an earlier transaction.
        let original_file = loaded
            .storage
            .data_file(
                "s3://warehouse/events/data.parquet",
                10,
                loaded.table.file_io().config(),
            )
            .unwrap();
        let DataFileLocation::Remote { url, .. } = original_file.source else {
            panic!("expected remote S3 source");
        };
        let query: HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(
            query.get("X-Amz-Security-Token").map(String::as_str),
            Some("vended-session")
        );
        assert!(
            query
                .get("X-Amz-Credential")
                .is_some_and(|credential| credential.starts_with("vended-key/"))
        );
        refreshed_table.assert();
        refreshed_table.remove();

        let changed_table = server
            .mock("GET", "/v1/namespaces/main/tables/events")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(table_response(
                "s3://warehouse/events/metadata/00002.json",
                "next-key",
                "next-session",
            ))
            .expect(1)
            .create();
        let changed = harness.tokio.block_on(loader.load_table(&ident)).unwrap();
        assert!(!Arc::ptr_eq(&loaded.table, &changed.table));
        assert_eq!(
            changed.table.metadata_location(),
            Some("s3://warehouse/events/metadata/00002.json")
        );
        config.assert();
        changed_table.assert();
    }
}
