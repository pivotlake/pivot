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
//! An S3 secret says how the requests to its locations are authenticated, in
//! one of three ways: with the access keys it carries, with the identity the
//! server itself runs as when it carries none, or not at all when it says
//! `anonymous: true`. An S3 location no secret covers is opened like one
//! covered by a secret without keys.
//!
//! ```yaml
//! secrets:
//!   analytics:
//!     type: s3
//!     scope: s3://analytics/          # this bucket, whatever the prefix
//!     region: us-east-1
//!     access_key_id: AKIA...
//!     secret_access_key: "..."
//!   analytics-archive:
//!     type: s3
//!     scope: s3://analytics/archive/  # more specific: wins under archive/
//!     region: us-east-1               # no keys: the server's own AWS identity
//!   open-data:
//!     type: s3
//!     scope: s3://open-data/
//!     anonymous: true                 # a public bucket, read unsigned
//!   google:
//!     type: gcs                       # no scope: every gs:// location
//!     credentials_file: /etc/pivot/gcs-key.json
//! ```

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use datastore_iceberg::IcebergCatalogAuth;
use object_storage::{AwsCredentialChain, S3Auth, S3Credentials, S3Keys, StoreScheme};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// One secret as a file spells it. `type` names what it authenticates to and
/// carries exactly that type's fields, so an S3 secret cannot name a Google
/// key file and a GCS secret cannot carry an access key. A storage secret is
/// matched to the locations its `scope` covers; an `iceberg` secret has no
/// scope, since the datastore over the catalog names it.
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum SecretConfig {
    #[serde(rename = "s3")]
    S3 {
        /// The `s3://bucket/prefix` this secret covers. Omitted, it covers
        /// every S3 location.
        #[serde(skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        /// Optional signing region. For AWS S3, an unspecified region is
        /// discovered from the bucket before the store is opened.
        #[serde(skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        /// The access key requests are signed with, written together with
        /// `secret_access_key`. Both omitted, requests are signed as the
        /// identity the server runs as, found through its AWS credential chain.
        #[serde(skip_serializing_if = "Option::is_none")]
        access_key_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        secret_access_key: Option<String>,
        /// Send requests unsigned, for a public bucket. Written without keys.
        #[serde(default, skip_serializing_if = "is_false")]
        anonymous: bool,
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
    #[serde(rename = "iceberg")]
    Iceberg {
        /// A bearer token sent on every catalog request. Exactly one of
        /// `token` and `credential` is written.
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// An OAuth2 client credential (`client_id:client_secret`) exchanged
        /// for a token at the catalog's token endpoint.
        #[serde(skip_serializing_if = "Option::is_none")]
        credential: Option<String>,
        /// The OAuth2 token endpoint, when it is not the catalog's own.
        #[serde(skip_serializing_if = "Option::is_none")]
        oauth2_server_uri: Option<String>,
        /// The OAuth2 scope requested with `credential`.
        #[serde(skip_serializing_if = "Option::is_none")]
        oauth2_scope: Option<String>,
    },
}

/// Serde's `skip_serializing_if` for a flag that is off by default.
fn is_false(flag: &bool) -> bool {
    !flag
}

/// By hand with the S3 keys redacted: a secret must not leak into a log
/// through a `{:?}` of some struct that holds one.
impl std::fmt::Debug for SecretConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::S3 {
                scope,
                region,
                access_key_id,
                secret_access_key,
                anonymous,
                endpoint,
            } => f
                .debug_struct("S3")
                .field("scope", scope)
                .field("region", region)
                .field("access_key_id", &access_key_id.as_ref().map(|_| "redacted"))
                .field(
                    "secret_access_key",
                    &secret_access_key.as_ref().map(|_| "redacted"),
                )
                .field("anonymous", anonymous)
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
            Self::Iceberg {
                token,
                credential,
                oauth2_server_uri,
                oauth2_scope,
            } => f
                .debug_struct("Iceberg")
                .field("token", &token.as_ref().map(|_| "redacted"))
                .field("credential", &credential.as_ref().map(|_| "redacted"))
                .field("oauth2_server_uri", oauth2_server_uri)
                .field("oauth2_scope", oauth2_scope)
                .finish(),
        }
    }
}

/// The paths a secret covers: a backend, then the segments pinned down below
/// it (the bucket, then any prefix parts). No segments at all is every location
/// the backend addresses, which is what a secret written without a `scope`
/// gets.
///
/// A scope is classified by the same [`StoreScheme`] a datastore's location is,
/// so the two agree on what `s3a://` and a `file://` path mean.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SecretScope {
    scheme: StoreScheme,
    segments: Vec<String>,
}

impl SecretScope {
    /// The scope `written` spells, for a secret named `name` of `scheme`. A
    /// scope addressing another backend, or one whose scheme addresses no
    /// backend at all, is refused rather than quietly covering nothing.
    fn parse(name: &str, written: &Option<String>, scheme: StoreScheme) -> Result<Self> {
        let Some(written) = written else {
            return Ok(Self {
                scheme,
                segments: Vec::new(),
            });
        };
        if !matches!(StoreScheme::of(written), Ok(written_scheme) if written_scheme == scheme) {
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

/// An S3 secret as resolved: where its locations are and how requests to them
/// are authenticated. Its `{:?}` shows no key: [`S3Keys`] redacts its own.
#[derive(Clone, Debug)]
struct S3Secret {
    region: Option<String>,
    auth: S3SecretAuth,
    endpoint: Option<String>,
}

/// How an S3 secret says requests are authenticated.
#[derive(Clone, Debug)]
enum S3SecretAuth {
    /// Signed with the keys the secret carries.
    Keys(S3Keys),
    /// Signed as the identity the server runs as.
    ServerIdentity,
    Anonymous,
}

impl S3SecretAuth {
    /// What the secret `name` says, from the fields it was written with. A
    /// combination that says two things at once is refused rather than read as
    /// either.
    fn from_fields(
        name: &str,
        access_key_id: &Option<String>,
        secret_access_key: &Option<String>,
        anonymous: bool,
    ) -> Result<Self> {
        match (access_key_id, secret_access_key) {
            (Some(access_key), Some(secret_key)) if !anonymous => Ok(Self::Keys(S3Keys {
                access_key: access_key.clone(),
                secret_key: secret_key.clone(),
                session_token: None,
            })),
            (None, None) => Ok(if anonymous {
                Self::Anonymous
            } else {
                Self::ServerIdentity
            }),
            (Some(_), None) | (None, Some(_)) if !anonymous => Err(Error::S3SecretLoneKey {
                name: name.to_string(),
            }),
            _ => Err(Error::S3SecretAnonymousWithKeys {
                name: name.to_string(),
            }),
        }
    }
}

/// A GCS secret as resolved: the key file tokens are minted from.
#[derive(Clone, Debug)]
struct GcsSecret {
    credentials_file: String,
}

/// A secret and the name it was written under, which the error a second secret
/// on the same scope raises needs. The scope is the key this is held at, so it
/// is not repeated here.
#[derive(Debug)]
struct Named<T> {
    name: String,
    secret: T,
}

/// Every secret of a metastore: the storage secrets split by the backend they
/// authenticate to and keyed within each by the scope they cover, and the
/// catalog secrets keyed by name.
///
/// A scope keying at most one secret is what makes "the most specific scope
/// covering a location" one secret rather than a race between two, and it is
/// the map itself that holds a second secret out.
#[derive(Debug, Default)]
pub(crate) struct Secrets {
    s3: HashMap<SecretScope, Named<S3Secret>>,
    gcs: HashMap<SecretScope, Named<GcsSecret>>,
    iceberg: HashMap<String, IcebergCatalogAuth>,
}

impl Secrets {
    /// Key the secrets of the config file's section and of the metastore file
    /// by the scope each covers, under the backend it authenticates to.
    ///
    /// Names are visited in sorted order, so which of two secrets sharing a
    /// scope is named as the one already holding it is the same on every run.
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
                    anonymous,
                    endpoint,
                } => insert_secret(
                    &mut secrets.s3,
                    SecretScope::parse(name, scope, StoreScheme::S3)?,
                    name,
                    S3Secret {
                        region: region.clone(),
                        auth: S3SecretAuth::from_fields(
                            name,
                            access_key_id,
                            secret_access_key,
                            *anonymous,
                        )?,
                        endpoint: endpoint.clone(),
                    },
                )?,
                SecretConfig::Gcs {
                    scope,
                    credentials_file,
                } => insert_secret(
                    &mut secrets.gcs,
                    SecretScope::parse(name, scope, StoreScheme::Gcs)?,
                    name,
                    GcsSecret {
                        credentials_file: credentials_file.clone(),
                    },
                )?,
                SecretConfig::Iceberg {
                    token,
                    credential,
                    oauth2_server_uri,
                    oauth2_scope,
                } => {
                    let auth = match (token, credential) {
                        (Some(token), None) => IcebergCatalogAuth::Token(token.clone()),
                        (None, Some(credential)) => IcebergCatalogAuth::OAuth2 {
                            credential: credential.clone(),
                            server_uri: oauth2_server_uri.clone(),
                            scope: oauth2_scope.clone(),
                        },
                        _ => {
                            return Err(Error::IcebergSecretCredential { name: name.clone() });
                        }
                    };
                    secrets.iceberg.insert(name.clone(), auth);
                }
            }
        }
        Ok(secrets)
    }

    /// What the `iceberg` secret `secret` authenticates to a catalog with, for
    /// the datastore `datastore` that names it.
    pub(crate) fn find_iceberg_secret(
        &self,
        datastore: &str,
        secret: &str,
    ) -> Result<IcebergCatalogAuth> {
        self.iceberg
            .get(secret)
            .cloned()
            .ok_or_else(|| Error::UnknownIcebergSecret {
                datastore: datastore.to_string(),
                secret: secret.to_string(),
            })
    }

    /// What an S3 `location` is opened with, from the most specific scope
    /// covering it. Nothing covering it opens it as the identity the server
    /// runs as, at AWS's own endpoint.
    pub(crate) fn resolve_s3(&self, location: &str) -> S3Credentials {
        let Some(covering) = find_most_specific(&self.s3, StoreScheme::S3, location) else {
            return S3Credentials {
                region: None,
                auth: S3Auth::Chain(AwsCredentialChain::ambient()),
                endpoint: None,
            };
        };
        let secret = &covering.secret;
        S3Credentials {
            region: secret.region.clone(),
            auth: match &secret.auth {
                S3SecretAuth::Keys(keys) => S3Auth::Keys(keys.clone()),
                S3SecretAuth::ServerIdentity => S3Auth::Chain(AwsCredentialChain::ambient()),
                S3SecretAuth::Anonymous => S3Auth::Anonymous,
            },
            endpoint: secret.endpoint.clone(),
        }
    }

    /// The Google key file a GCS `location` is opened with, from the most
    /// specific scope covering it. Nothing covering it leaves the store on the
    /// ambient Application Default Credentials chain.
    pub(crate) fn resolve_gcs(&self, location: &str) -> Option<&str> {
        Some(
            find_most_specific(&self.gcs, StoreScheme::Gcs, location)?
                .secret
                .credentials_file
                .as_str(),
        )
    }
}

/// The secret whose scope covers `location` and pins down the most of it.
///
/// A scope covers a location when it names some prefix of the location's
/// segments, so the scopes that could cover this one are known outright: there
/// is a candidate per prefix, and no others. Asking for each in turn, longest
/// first, finds the most specific without searching what is held. `scheme` is
/// the backend the caller is resolving credentials for, and is part of the key,
/// so a secret held for another backend matches nothing.
fn find_most_specific<'a, T>(
    secrets: &'a HashMap<SecretScope, T>,
    scheme: StoreScheme,
    location: &str,
) -> Option<&'a T> {
    let segments = split_segments(location);
    (0..=segments.len()).rev().find_map(|depth| {
        secrets.get(&SecretScope {
            scheme,
            segments: segments[..depth].to_vec(),
        })
    })
}

/// Add a secret to the map under the scope it covers, refusing a scope another
/// secret already holds: which of the two signed a request there would be
/// arbitrary.
fn insert_secret<T>(
    secrets: &mut HashMap<SecretScope, Named<T>>,
    scope: SecretScope,
    name: &str,
    secret: T,
) -> Result<()> {
    match secrets.entry(scope) {
        Entry::Occupied(held) => Err(Error::DuplicateScope {
            first: held.get().name.clone(),
            second: name.to_string(),
            scope: held.key().to_string(),
        }),
        Entry::Vacant(free) => {
            free.insert(Named {
                name: name.to_string(),
                secret,
            });
            Ok(())
        }
    }
}
