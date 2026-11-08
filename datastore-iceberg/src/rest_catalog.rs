//! What reaches the REST catalog: its configuration, the one client built
//! from it for the life of the datastore, the OAuth2 session that keeps that
//! client's token current, and blocking on the client's calls.
//!
//! The Iceberg REST client is async. Every caller of this datastore runs on a
//! blocking thread (a query is planned on the blocking pool), so each catalog
//! call blocks on the ambient runtime, or on a runtime of this crate's own when
//! there is none (a sync test, a standalone tool).

use std::collections::HashMap;
use std::future::Future;
use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use iceberg::io::StorageFactory;
use iceberg::{CatalogBuilder, ErrorKind};
use iceberg_catalog_rest::{
    AuthManager, AuthSession, HttpClient, HttpRequest, RestCatalog, RestCatalogBuilder,
};
use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::Result;

/// The name the client registers the catalog under: it requires one and
/// never uses it, so every catalog gets this one.
const CATALOG_NAME: &str = "pivot_catalog";

/// How to reach the REST catalog. The fields mirror the properties the
/// Iceberg REST client is configured with; `properties` passes any further
/// ones (a `prefix`, a `header.*`) through verbatim.
#[derive(Clone, Debug, Default)]
pub struct IcebergCatalogConfig {
    /// The catalog's base URI, such as `https://catalog.example.com/api`.
    pub uri: String,
    /// The warehouse the catalog serves, when it serves several.
    pub warehouse: Option<String>,
    /// What the catalog is authenticated to with; `None` for a catalog that
    /// requires nothing.
    pub auth: Option<IcebergCatalogAuth>,
    /// Further client properties, passed through as written.
    pub properties: HashMap<String, String>,
}

/// What a REST catalog is authenticated to with.
#[derive(Clone)]
pub enum IcebergCatalogAuth {
    /// A bearer token sent on every request.
    Token(String),
    /// An OAuth2 client credential (`client_id:client_secret`, or a bare
    /// secret) exchanged for a token at `server_uri`, or at the catalog's own
    /// token endpoint when there is none, for `scope` when one is given.
    OAuth2 {
        credential: String,
        server_uri: Option<String>,
        scope: Option<String>,
    },
}

/// By hand with the secret redacted: a token or credential must not leak
/// into a log through a `{:?}` of the config that holds it.
impl std::fmt::Debug for IcebergCatalogAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Token(_) => f.write_str("Token(redacted)"),
            Self::OAuth2 {
                server_uri, scope, ..
            } => f
                .debug_struct("OAuth2")
                .field("credential", &"redacted")
                .field("server_uri", server_uri)
                .field("scope", scope)
                .finish(),
        }
    }
}

impl IcebergCatalogConfig {
    /// The client for the catalog this config describes, built once per
    /// datastore: an OAuth2 credential is exchanged for a token by
    /// [`RenewingOAuth2`], which exchanges it again before the token expires,
    /// so the client never has to be rebuilt. Nothing is sent yet: the client
    /// fetches the catalog's own configuration, and obtains its first token,
    /// on its first call, so an unreachable catalog fails there, not here.
    /// `storage_factory` builds the file IO of every table the client loads,
    /// which a commit writes its manifests through; a query never uses it,
    /// every file it reads going through Pivot's ring.
    pub(crate) fn build_client(
        &self,
        storage_factory: Arc<dyn StorageFactory>,
    ) -> Result<RestCatalog> {
        let mut properties = self.properties.clone();
        properties.insert("uri".to_string(), self.uri.clone());
        let mut builder = RestCatalogBuilder::default().with_storage_factory(storage_factory);
        // Ask the catalog to vend storage credentials with each table it loads. A
        // catalog that cannot ignores the header; a configured `header.*`
        // property of the same name wins, so vending can be switched off per
        // catalog.
        properties
            .entry("header.X-Iceberg-Access-Delegation".to_string())
            .or_insert_with(|| "vended-credentials".to_string());
        if let Some(warehouse) = &self.warehouse {
            properties.insert("warehouse".to_string(), warehouse.clone());
        }
        match &self.auth {
            Some(IcebergCatalogAuth::Token(token)) => {
                properties.insert("token".to_string(), token.clone());
            }
            Some(IcebergCatalogAuth::OAuth2 {
                credential,
                server_uri,
                scope,
            }) => {
                builder = builder.with_auth_manager(RenewingOAuth2::new(
                    credential,
                    server_uri.clone(),
                    scope.clone(),
                ));
            }
            None => {}
        }
        // `load` captures the current runtime before returning its future.
        let catalog = block_on(builder.load(CATALOG_NAME, properties)).map_err(Box::new)?;
        Ok(catalog)
    }
}

/// The OAuth2 client-credentials flow with the token exchanged again before
/// it expires, so one client serves the catalog for as long as the datastore
/// lives. The REST client's own flow exchanges the credential once and keeps
/// that token for good.
struct RenewingOAuth2 {
    client_id: Option<String>,
    client_secret: String,
    /// The token endpoint when one is configured; else the catalog's own,
    /// under the URI the catalog's configuration settles on.
    server_uri: Option<String>,
    scope: Option<String>,
    /// The token every session this manager builds shares, so the one the
    /// configuration handshake obtained serves the catalog's requests too.
    token: Arc<Mutex<Option<IssuedToken>>>,
}

/// The scope requested when the configuration names none, as the REST
/// client requests it.
const DEFAULT_SCOPE: &str = "catalog";

impl RenewingOAuth2 {
    /// A manager for `credential`, a `client_id:client_secret` pair or a bare
    /// secret, exchanged at `server_uri` or the catalog's own token endpoint.
    fn new(credential: &str, server_uri: Option<String>, scope: Option<String>) -> Self {
        let (client_id, client_secret) = match credential.split_once(':') {
            Some((client_id, client_secret)) => (Some(client_id.to_string()), client_secret),
            None => (None, credential),
        };
        Self {
            client_id,
            client_secret: client_secret.to_string(),
            server_uri,
            scope,
            token: Arc::new(Mutex::new(None)),
        }
    }

    /// A session over `client`, which the catalog hands over unauthenticated,
    /// exchanging at the endpoint `props` settle on and sending the `header.*`
    /// properties with the exchange, as the REST client does.
    fn build_session(
        &self,
        client: &HttpClient,
        props: &HashMap<String, String>,
    ) -> iceberg::Result<RenewingOAuth2Session> {
        let token_endpoint = match &self.server_uri {
            Some(server_uri) => server_uri.clone(),
            None => {
                let catalog_uri = props.get("uri").ok_or_else(|| {
                    iceberg::Error::new(
                        ErrorKind::DataInvalid,
                        "the catalog has no URI to derive its token endpoint from",
                    )
                })?;
                format!("{catalog_uri}/v1/oauth/tokens")
            }
        };
        let mut headers = HeaderMap::new();
        for (key, value) in props {
            let Some(name) = key.strip_prefix("header.") else {
                continue;
            };
            let invalid_header = |error: &dyn std::error::Error| {
                iceberg::Error::new(
                    ErrorKind::DataInvalid,
                    format!("invalid catalog header `{name}`: {error}"),
                )
            };
            headers.insert(
                HeaderName::from_str(name).map_err(|error| invalid_header(&error))?,
                HeaderValue::from_str(value).map_err(|error| invalid_header(&error))?,
            );
        }
        Ok(RenewingOAuth2Session {
            client: client.clone(),
            token_endpoint,
            headers,
            client_id: self.client_id.clone(),
            client_secret: self.client_secret.clone(),
            scope: self
                .scope
                .clone()
                .unwrap_or_else(|| DEFAULT_SCOPE.to_string()),
            token: self.token.clone(),
        })
    }
}

/// By hand with the secret left out: a credential must not leak into a log
/// through a `{:?}` of the client that holds it.
impl std::fmt::Debug for RenewingOAuth2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenewingOAuth2")
            .field("client_id", &self.client_id)
            .field("server_uri", &self.server_uri)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl AuthManager for RenewingOAuth2 {
    async fn init_session(
        &self,
        client: &HttpClient,
        props: &HashMap<String, String>,
    ) -> iceberg::Result<Box<dyn AuthSession>> {
        Ok(Box::new(self.build_session(client, props)?))
    }

    async fn catalog_session(
        &self,
        client: &HttpClient,
        props: &HashMap<String, String>,
    ) -> iceberg::Result<Arc<dyn AuthSession>> {
        Ok(Arc::new(self.build_session(client, props)?))
    }
}

/// Signs every request with the current token, exchanging the credential for
/// a new one whenever the held token is due for renewal.
struct RenewingOAuth2Session {
    client: HttpClient,
    token_endpoint: String,
    headers: HeaderMap,
    client_id: Option<String>,
    client_secret: String,
    scope: String,
    token: Arc<Mutex<Option<IssuedToken>>>,
}

impl std::fmt::Debug for RenewingOAuth2Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenewingOAuth2Session")
            .field("token_endpoint", &self.token_endpoint)
            .finish_non_exhaustive()
    }
}

/// What the token endpoint answers with, as far as a session needs it.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    /// How long the token stays valid, in seconds, when the server says.
    expires_in: Option<u64>,
}

impl RenewingOAuth2Session {
    /// Exchange the credential for a token at the token endpoint.
    async fn exchange_credential(&self) -> iceberg::Result<IssuedToken> {
        let mut form = HashMap::from([
            ("grant_type", "client_credentials"),
            ("client_secret", self.client_secret.as_str()),
            ("scope", self.scope.as_str()),
        ]);
        if let Some(client_id) = &self.client_id {
            form.insert("client_id", client_id);
        }
        let response = self
            .client
            .post_form(&self.token_endpoint, &self.headers, &form)
            .await?;
        if response.status() != StatusCode::OK {
            return Err(iceberg::Error::new(
                ErrorKind::Unexpected,
                "the catalog's token endpoint refused the credential",
            )
            .with_context("code", response.status().to_string())
            .with_context("url", self.token_endpoint.clone())
            .with_context("body", String::from_utf8_lossy(response.body())));
        }
        let issued_at = Instant::now();
        let token: TokenResponse = serde_json::from_slice(response.body()).map_err(|error| {
            iceberg::Error::new(
                ErrorKind::Unexpected,
                "the catalog's token endpoint answered with no token",
            )
            .with_context("url", self.token_endpoint.clone())
            .with_source(error)
        })?;
        Ok(IssuedToken::new(
            token.access_token,
            token.expires_in.map(Duration::from_secs),
            issued_at,
        ))
    }
}

#[async_trait]
impl AuthSession for RenewingOAuth2Session {
    /// The lock is held across the exchange: concurrent requests at renewal
    /// time share the one new token rather than each exchanging for its own.
    async fn authenticate(&self, request: &mut HttpRequest) -> iceberg::Result<()> {
        let token = {
            let mut held = self.token.lock().await;
            match &*held {
                Some(issued) if issued.is_current_at(Instant::now()) => issued.token.clone(),
                _ => {
                    let issued = self.exchange_credential().await?;
                    let token = issued.token.clone();
                    *held = Some(issued);
                    token
                }
            }
        };
        let mut value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|error| {
            iceberg::Error::new(
                ErrorKind::DataInvalid,
                "the catalog issued a token that cannot be sent as a header",
            )
            .with_source(error)
        })?;
        value.set_sensitive(true);
        request.headers_mut().insert(AUTHORIZATION, value);
        Ok(())
    }
}

/// A token and the moment it is due for renewal: before its lifetime ends,
/// by a [`RENEWAL_LEAD_DIVISOR`]th of it, so a request signed as the lifetime
/// runs out is not refused.
struct IssuedToken {
    token: String,
    renew_at: Instant,
}

/// The lifetime assumed for a token whose server states none: the hour the
/// reference client assumes.
const LIFETIME_WHEN_UNSTATED: Duration = Duration::from_secs(3600);

/// The share of a token's lifetime it is renewed ahead of its end.
const RENEWAL_LEAD_DIVISOR: u32 = 10;

impl IssuedToken {
    fn new(token: String, lifetime: Option<Duration>, issued_at: Instant) -> Self {
        let lifetime = lifetime.unwrap_or(LIFETIME_WHEN_UNSTATED);
        Self {
            token,
            renew_at: issued_at + lifetime - lifetime / RENEWAL_LEAD_DIVISOR,
        }
    }

    /// Whether the token still signs requests at `now`, or is due for renewal.
    fn is_current_at(&self, now: Instant) -> bool {
        now < self.renew_at
    }
}

/// Drive `future` to completion from a blocking thread: on the ambient
/// multi-thread runtime when there is one, else on [`CATALOG_RUNTIME`]. A
/// single-thread ambient runtime (a current-thread test) cannot be blocked on
/// from inside itself, so it is declined in favour of the crate's own.
pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(future))
        }
        _ => CATALOG_RUNTIME.block_on(future),
    }
}

/// The runtime the catalog client runs on when the process has no
/// multi-thread runtime of its own. A `static` so it is never dropped
/// (dropping an owned runtime inside an async context panics). Two workers:
/// the client's requests are few and sequential.
static CATALOG_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build the Iceberg catalog runtime")
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_renewed_a_tenth_of_its_lifetime_before_it_expires() {
        let issued_at = Instant::now();
        let token = IssuedToken::new("t".to_string(), Some(Duration::from_secs(20)), issued_at);

        assert!(token.is_current_at(issued_at + Duration::from_secs(17)));
        assert!(!token.is_current_at(issued_at + Duration::from_secs(18)));
    }

    #[test]
    fn a_token_whose_lifetime_the_server_leaves_unstated_lasts_most_of_an_hour() {
        let issued_at = Instant::now();
        let token = IssuedToken::new("t".to_string(), None, issued_at);

        assert!(token.is_current_at(issued_at + Duration::from_secs(50 * 60)));
        assert!(!token.is_current_at(issued_at + Duration::from_secs(55 * 60)));
    }

    #[test]
    fn a_credential_splits_into_client_id_and_secret_when_it_has_both() {
        let paired = RenewingOAuth2::new("id:secret", None, None);
        let bare = RenewingOAuth2::new("secret", None, None);

        assert_eq!(paired.client_id.as_deref(), Some("id"));
        assert_eq!(paired.client_secret, "secret");
        assert_eq!(bare.client_id, None);
        assert_eq!(bare.client_secret, "secret");
    }
}
