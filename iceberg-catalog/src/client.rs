//! The Iceberg REST protocol client: `/v1/config` at connect and `loadTable`
//! per resolve, over blocking HTTP ([`ureq`]) so no async runtime enters the
//! query path.

use catalog::store::percent_encode;

use crate::metadata::TableMetadata;
use crate::{Error, Result};

/// A connected REST catalog endpoint. Built once by
/// [`connect`](RestClient::connect) (which fetches `/v1/config` and adopts the
/// server's route prefix) and shared by every table resolve.
pub(crate) struct RestClient {
    /// Endpoint root, no trailing slash, e.g. `http://localhost:8181`.
    base: String,
    /// The server-assigned route prefix from `/v1/config` (multi-tenant
    /// catalogs use it to scope routes), spliced between `v1/` and
    /// `namespaces/...`. Empty when the server assigns none.
    prefix: Option<String>,
    /// Optional bearer token sent as `Authorization` on every request.
    token: Option<String>,
    agent: ureq::Agent,
}

impl std::fmt::Debug for RestClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestClient")
            .field("base", &self.base)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

/// The `/v1/config` response: property maps the server asks clients to adopt.
/// Only `prefix` matters to us; storage properties are resolved from the
/// environment by `catalog::store`.
#[derive(Debug, serde::Deserialize)]
struct ConfigResponse {
    #[serde(default)]
    defaults: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    overrides: serde_json::Map<String, serde_json::Value>,
}

/// The `loadTable` response. `config` (per-table storage properties) is
/// intentionally not modeled: data files are read with the environment's
/// credentials, like every other store access in pivot.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct LoadTableResponse {
    pub metadata: TableMetadata,
}

impl RestClient {
    /// Connect to the REST catalog at `uri`: fetch `/v1/config` (passing
    /// `warehouse` through when given, as the spec requires) and adopt the
    /// server's route prefix. Fails if the endpoint is unreachable or answers
    /// with anything but a config document, so a misconfigured catalog is
    /// caught at startup rather than at the first query.
    pub(crate) fn connect(
        uri: &str,
        warehouse: Option<&str>,
        token: Option<String>,
    ) -> Result<Self> {
        let base = uri.trim_end_matches('/').to_string();
        let mut config_url = format!("{base}/v1/config");
        if let Some(warehouse) = warehouse {
            config_url = format!("{config_url}?warehouse={}", percent_encode(warehouse));
        }

        // Bounded timeouts: table resolution runs on a query's bind path, so a
        // hung catalog endpoint must fail the query, not block it forever.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(10))
            .timeout_read(std::time::Duration::from_secs(30))
            .build();
        let response = send_get(&agent, &config_url, token.as_deref())?.ok_or_else(|| {
            Error::UnexpectedResponse {
                url: config_url.clone(),
                message: "config endpoint returned 404".to_string(),
            }
        })?;
        let config: ConfigResponse = parse_json(&config_url, response)?;

        // An override wins over a default, per the config endpoint's contract.
        let prefix = config
            .overrides
            .get("prefix")
            .or_else(|| config.defaults.get("prefix"))
            .and_then(|value| value.as_str())
            .map(|prefix| prefix.to_string());

        Ok(Self {
            base,
            prefix,
            token,
            agent,
        })
    }

    /// Load the current metadata of `table` in `namespace`. `Ok(None)` means
    /// the catalog has no such table (or namespace); errors are transport or
    /// decode failures.
    pub(crate) fn load_table(
        &self,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<LoadTableResponse>> {
        // Multi-level namespaces travel as their levels joined by the unit
        // separator (0x1F), percent-encoded - the REST spec's convention. The
        // prefix is spliced in verbatim: the server described its own route
        // with it, so it may legitimately contain `/` or be pre-encoded.
        let namespace_path = percent_encode(&namespace.join("\u{1f}"));
        let prefix_segment = match &self.prefix {
            Some(prefix) => format!("{prefix}/"),
            None => String::new(),
        };
        let url = format!(
            "{}/v1/{prefix_segment}namespaces/{namespace_path}/tables/{}",
            self.base,
            percent_encode(table)
        );
        match send_get(&self.agent, &url, self.token.as_deref())? {
            Some(response) => Ok(Some(parse_json(&url, response)?)),
            None => Ok(None),
        }
    }
}

/// GET `url`, `Ok(None)` on a 404, `Err` on any other failure.
fn send_get(agent: &ureq::Agent, url: &str, token: Option<&str>) -> Result<Option<ureq::Response>> {
    let mut request = agent.get(url);
    if let Some(token) = token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    match request.call() {
        Ok(response) => Ok(Some(response)),
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(e) => Err(Error::Http {
            url: url.to_string(),
            message: e.to_string(),
        }),
    }
}

/// Decode a JSON response body into `T`, wrapping decode failures with the URL
/// they came from.
fn parse_json<T: serde::de::DeserializeOwned>(url: &str, response: ureq::Response) -> Result<T> {
    response
        .into_json::<T>()
        .map_err(|e| Error::UnexpectedResponse {
            url: url.to_string(),
            message: e.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::percent_encode;

    // The REST spec's multi-level namespace convention: levels joined by the
    // unit separator, which must survive encoding as `%1F`.
    #[test]
    fn encodes_namespace_levels_with_the_unit_separator() {
        let encoded = percent_encode(&["a", "b"].join("\u{1f}"));

        assert_eq!(encoded, "a%1Fb");
    }
}
