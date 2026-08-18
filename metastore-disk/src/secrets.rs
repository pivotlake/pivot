//! The `secrets` section: the credentials an object store is opened with, held
//! apart from the datastores that use them.
//!
//! A secret is written once and reaches every datastore whose location it
//! covers, so a bucket's keys are stated in one place however many datastores
//! sit in it. Which secret authenticates a location is decided by the secret's
//! `scope`, a URI prefix; the most specific scope covering the location wins,
//! and no two secrets may claim the same scope, so that "most specific" is
//! never a coin toss.
//!
//! ```yaml
//! metastore:
//!   secrets:
//!     analytics:
//!       type: s3
//!       scope: s3://analytics/          # this bucket, whatever the prefix
//!       region: us-east-1
//!       access_key_id: AKIA...
//!       secret_access_key: "..."
//!     analytics-archive:
//!       type: s3
//!       scope: s3://analytics/archive/  # more specific: wins under archive/
//!       region: us-east-1
//!       access_key_id: AKIA...
//!       secret_access_key: "..."
//!     google:
//!       type: gcs                       # no scope: every gs:// location
//!       credentials_file: /etc/pivot/gcs-key.json
//! ```

use std::collections::HashMap;

use datastore_delta::store::S3Credentials;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// One secret as a file spells it. `type` names the backend it authenticates
/// to and carries exactly that backend's fields, so an S3 secret cannot name a
/// Google key file and a GCS secret cannot carry an access key.
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub(crate) enum SecretConfig {
    #[serde(rename = "s3")]
    S3 {
        /// The `s3://bucket/prefix` this secret covers. Omitted, it covers
        /// every S3 location.
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        region: String,
        access_key_id: String,
        secret_access_key: String,
        /// A path-style S3-compatible endpoint (e.g. MinIO). Omitted, requests
        /// go to AWS virtual-hosted style.
        #[serde(skip_serializing_if = "Option::is_none")]
        endpoint: Option<String>,
    },
    #[serde(rename = "gcs")]
    Gcs {
        /// The `gs://bucket/prefix` this secret covers. Omitted, it covers
        /// every GCS location.
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        /// A Google service-account (or authorized-user) JSON key file.
        credentials_file: String,
    },
}

/// By hand with the S3 keys redacted: a secret must not leak into a log
/// through a `{:?}` of some struct that holds one.
impl std::fmt::Debug for SecretConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::S3 {
                scope,
                region,
                endpoint,
                ..
            } => f
                .debug_struct("S3")
                .field("scope", scope)
                .field("region", region)
                .field("access_key_id", &"redacted")
                .field("secret_access_key", &"redacted")
                .field("endpoint", endpoint)
                .finish(),
            Self::Gcs {
                scope,
                credentials_file,
            } => f
                .debug_struct("Gcs")
                .field("scope", scope)
                .field("credentials_file", credentials_file)
                .finish(),
        }
    }
}

/// The storage backend a URI addresses, taken from its scheme. A location is
/// classified with it, and so is a secret's scope: the two must agree for the
/// secret to cover the location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StoreScheme {
    S3,
    Gcs,
    Local,
}

impl StoreScheme {
    /// The backend `uri` addresses. `s3://` and its `s3a://` spelling are the
    /// same backend over the same buckets, so they share one scope space;
    /// anything that is not a recognised object-store URI is a local path.
    pub(crate) fn of(uri: &str) -> Self {
        if uri.starts_with("s3://") || uri.starts_with("s3a://") {
            Self::S3
        } else if uri.starts_with("gs://") {
            Self::Gcs
        } else {
            Self::Local
        }
    }

    /// The URI prefix a location of this backend is written with, and the form
    /// a scope of it is printed back in.
    fn uri_prefix(self) -> &'static str {
        match self {
            Self::S3 => "s3://",
            Self::Gcs => "gs://",
            Self::Local => "file://",
        }
    }
}

/// The paths a secret covers: a backend, then the segments pinned down below
/// it (the bucket, then any prefix parts). No segments at all is every location
/// the backend addresses, which is what a secret written without a `scope`
/// gets.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SecretScope {
    scheme: StoreScheme,
    segments: Vec<String>,
}

impl SecretScope {
    /// The scope `written` spells, for a secret named `name` of `scheme`. A
    /// scope addressing another backend is refused rather than quietly covering
    /// nothing.
    fn parse(name: &str, written: &Option<String>, scheme: StoreScheme) -> Result<Self> {
        let Some(written) = written else {
            return Ok(Self {
                scheme,
                segments: Vec::new(),
            });
        };
        if StoreScheme::of(written) != scheme {
            return Err(Error::SecretScope {
                name: name.to_string(),
                scope: written.clone(),
                expected: scheme.uri_prefix(),
            });
        }
        Ok(Self {
            scheme,
            segments: split_segments(written),
        })
    }

    /// Whether this scope covers `location`: the same backend, and every one of
    /// the scope's segments matching `location`'s in turn. Whole segments only,
    /// so `s3://logs/eu` covers `s3://logs/eu/2024` but not `s3://logs/europe`.
    fn covers(&self, location: &str) -> bool {
        StoreScheme::of(location) == self.scheme
            && split_segments(location).starts_with(&self.segments)
    }

    /// How much of a location this scope pins down. The most specific scope
    /// covering a location is the one that authenticates it.
    fn specificity(&self) -> usize {
        self.segments.len()
    }
}

impl std::fmt::Display for SecretScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.scheme.uri_prefix(), self.segments.join("/"))
    }
}

/// The segments of an object-store URI below its scheme: the bucket, then the
/// key prefix's parts. Empty parts are dropped, so a trailing slash and a
/// doubled one make no difference to what a scope covers or matches.
fn split_segments(uri: &str) -> Vec<String> {
    let below_scheme = uri.split_once("://").map_or(uri, |(_, rest)| rest);
    below_scheme
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

/// An S3 secret as resolved: what signing a request needs.
#[derive(Clone)]
struct S3Secret {
    region: String,
    access_key_id: String,
    secret_access_key: String,
    endpoint: Option<String>,
}

/// By hand with the keys redacted, for the same reason as [`SecretConfig`]'s.
impl std::fmt::Debug for S3Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Secret")
            .field("region", &self.region)
            .field("access_key_id", &"redacted")
            .field("secret_access_key", &"redacted")
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

/// A GCS secret as resolved: the key file tokens are minted from.
#[derive(Clone, Debug)]
struct GcsSecret {
    credentials_file: String,
}

/// A secret paired with the name it was written under and the scope it covers.
/// The name is kept for the error a duplicate scope raises.
#[derive(Debug)]
struct Scoped<T> {
    name: String,
    scope: SecretScope,
    secret: T,
}

/// Every secret of a metastore, split by the backend it authenticates to and
/// ready to match a location against.
///
/// Scopes are unique within a backend (checked by [`build`](Self::build)), so
/// the most specific scope covering a location is unambiguous.
#[derive(Debug, Default)]
pub(crate) struct Secrets {
    s3: Vec<Scoped<S3Secret>>,
    gcs: Vec<Scoped<GcsSecret>>,
}

impl Secrets {
    /// Split the secrets of the config file's section and of the metastore
    /// file into per-backend lists, refusing a scope two of them claim.
    ///
    /// Names are visited in sorted order, so a file with two faults is reported
    /// the same way on every run.
    pub(crate) fn build(sections: [&HashMap<String, SecretConfig>; 2]) -> Result<Self> {
        let mut written: Vec<(&String, &SecretConfig)> = sections.into_iter().flatten().collect();
        written.sort_by_key(|(name, _)| *name);

        let mut secrets = Self::default();
        for (name, config) in written {
            match config {
                SecretConfig::S3 {
                    scope,
                    region,
                    access_key_id,
                    secret_access_key,
                    endpoint,
                } => secrets.s3.push(Scoped {
                    name: name.clone(),
                    scope: SecretScope::parse(name, scope, StoreScheme::S3)?,
                    secret: S3Secret {
                        region: region.clone(),
                        access_key_id: access_key_id.clone(),
                        secret_access_key: secret_access_key.clone(),
                        endpoint: endpoint.clone(),
                    },
                }),
                SecretConfig::Gcs {
                    scope,
                    credentials_file,
                } => secrets.gcs.push(Scoped {
                    name: name.clone(),
                    scope: SecretScope::parse(name, scope, StoreScheme::Gcs)?,
                    secret: GcsSecret {
                        credentials_file: credentials_file.clone(),
                    },
                }),
            }
        }
        check_unique_scopes(&secrets.s3)?;
        check_unique_scopes(&secrets.gcs)?;
        Ok(secrets)
    }

    /// The credentials an S3 `location` is opened with, from the most specific
    /// scope covering it.
    pub(crate) fn resolve_s3(&self, location: &str) -> Option<S3Credentials> {
        let secret = &most_specific(&self.s3, location)?.secret;
        Some(S3Credentials {
            region: secret.region.clone(),
            access_key: secret.access_key_id.clone(),
            secret_key: secret.secret_access_key.clone(),
            endpoint: secret.endpoint.clone(),
        })
    }

    /// The Google key file a GCS `location` is opened with, from the most
    /// specific scope covering it. Nothing covering it leaves the store on the
    /// ambient Application Default Credentials chain.
    pub(crate) fn resolve_gcs(&self, location: &str) -> Option<&str> {
        Some(
            most_specific(&self.gcs, location)?
                .secret
                .credentials_file
                .as_str(),
        )
    }
}

/// The secret whose scope covers `location` and pins down the most of it.
/// Scopes being unique, no two covering scopes can be equally specific, so this
/// is the one secret the location resolves to.
fn most_specific<'a, T>(secrets: &'a [Scoped<T>], location: &str) -> Option<&'a Scoped<T>> {
    secrets
        .iter()
        .filter(|scoped| scoped.scope.covers(location))
        .max_by_key(|scoped| scoped.scope.specificity())
}

/// Refuse two secrets scoped to the same paths: which one signed a request
/// there would be arbitrary.
fn check_unique_scopes<T>(secrets: &[Scoped<T>]) -> Result<()> {
    for (index, secret) in secrets.iter().enumerate() {
        let claimed = secrets[..index]
            .iter()
            .find(|earlier| earlier.scope == secret.scope);
        if let Some(claimed) = claimed {
            return Err(Error::DuplicateScope {
                first: claimed.name.clone(),
                second: secret.name.clone(),
                scope: secret.scope.to_string(),
            });
        }
    }
    Ok(())
}
