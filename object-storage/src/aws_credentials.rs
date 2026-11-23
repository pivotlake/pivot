//! The AWS identity a process runs as: where an S3 store's signing keys come
//! from when nobody handed it any.
//!
//! An [`AwsCredentialChain`] looks where the AWS SDKs look, and signs with the
//! keys of the first source that has some:
//!
//! 1. the environment: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and, for a
//!    session's keys, `AWS_SESSION_TOKEN`;
//! 2. the container credentials endpoint:
//!    `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` or
//!    `AWS_CONTAINER_CREDENTIALS_FULL_URI`, as an ECS task or EKS Pod Identity
//!    sets them;
//! 3. the EC2 instance metadata service: the keys of the IAM role attached to
//!    the instance.
//!
//! A source that is configured and broken (a key variable without its pair, an
//! endpoint that refuses) is an error rather than a reason to try the next
//! one, which would sign as someone else.
//!
//! The keys of the last two sources are a session's and expire within hours.
//! Like the GCS store's token they are cached, and read from their source
//! again once they are within [`KEYS_REFRESH_MARGIN`] seconds of expiring.
//! That read is a blocking `ureq` round trip kept off the ring: `refresh_keys`
//! runs on the control thread, for every request a store signs itself and
//! every file it hands out, so a worker only ever reads the keys held
//! (`current_keys`).
//!
//! [`AwsCredentialChain::ambient`] is the chain over the process's own
//! environment, shared so that every store signs with the same keys and the
//! sources are asked once for all of them.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use crate::instance_bandwidth::{
    INSTANCE_METADATA_ENDPOINT, METADATA_CONNECT_TIMEOUT, METADATA_TIMEOUT,
};
use crate::s3::S3Keys;
use crate::{Result, StoreError, parse_iso8601_millis, unix_now};

/// Read a session's keys again once they are within this many seconds of
/// expiry. Generous so keys read at query start stay valid for the whole
/// query's worker reads. The instance metadata service publishes the next keys
/// at least this long before the current ones expire.
const KEYS_REFRESH_MARGIN: u64 = 300;

/// How many seconds an answer that calls for another look stands before the
/// sources are asked again: that no source has keys, or keys their source has
/// not replaced yet. Asking costs a connection attempt to the instance
/// metadata service, which off EC2 can only time out.
const SOURCES_RECHECK_INTERVAL: u64 = 60;

/// How long a container's credentials endpoint gets to answer.
const CONTAINER_ENDPOINT_TIMEOUT: Duration = Duration::from_secs(10);

/// Where the ECS agent answers from inside a task.
const ECS_AGENT_ENDPOINT: &str = "http://169.254.170.2";

/// Where the instance metadata service lists the role attached to the
/// instance, and under which it serves that role's keys.
const INSTANCE_ROLE_KEYS_PATH: &str = "/latest/meta-data/iam/security-credentials/";

/// Where a chain looks for keys, as an error about finding none names them.
pub(crate) const KEY_SOURCES: &str = "the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY variables, \
     a container credentials endpoint, and the role of the EC2 instance";

/// The keys of the identity one environment gives a process: those in its
/// variables, of a container's role, or of the EC2 instance's role, whichever
/// is found first. One cell is shared by every store opened with the chain, so
/// keys read again before they expire reach all of them.
pub struct AwsCredentialChain {
    /// The environment variables the sources are configured by.
    variables: HashMap<String, String>,
    /// What the sources last answered; `None` until they are first asked.
    /// Workers take the lock for read and never wait on a source: the blocking
    /// round trip runs outside it.
    cached_keys: RwLock<Option<CachedKeys>>,
}

/// Without the variables: among them are secret keys and tokens.
impl std::fmt::Debug for AwsCredentialChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsCredentialChain").finish_non_exhaustive()
    }
}

/// What the sources answered when they were last asked.
#[derive(Clone)]
struct CachedKeys {
    /// `None` when no source had keys.
    keys: Option<Arc<S3Keys>>,
    /// The Unix second from which the sources are asked again.
    stale_at: u64,
}

/// Keys a source handed over.
struct IssuedKeys {
    keys: S3Keys,
    /// The Unix second the keys expire at; `None` for long-lived ones.
    expires_at: Option<u64>,
}

impl AwsCredentialChain {
    /// The chain over this process's own environment, shared by every caller.
    pub fn ambient() -> Arc<Self> {
        static AMBIENT: LazyLock<Arc<AwsCredentialChain>> = LazyLock::new(|| {
            let variables = std::env::vars_os().filter_map(|(name, value)| {
                Some((name.into_string().ok()?, value.into_string().ok()?))
            });
            AwsCredentialChain::new(variables.collect())
        });
        Arc::clone(&AMBIENT)
    }

    /// The chain over exactly the environment `variables`, whatever the
    /// process's own are. Nothing is looked up until a store is opened with
    /// it.
    pub fn new(variables: HashMap<String, String>) -> Arc<Self> {
        Arc::new(Self {
            variables,
            cached_keys: RwLock::new(None),
        })
    }

    /// The keys to sign with now, or `None` when no source has any. Asks the
    /// sources when they were not asked yet, or when what they last answered
    /// is stale. Blocking (`ureq`); call only off the ring, whose workers use
    /// [`current_keys`](Self::current_keys) instead.
    pub(crate) fn refresh_keys(&self) -> Result<Option<Arc<S3Keys>>> {
        let now = unix_now();
        let cached = self.cached_keys.read().unwrap().clone();
        if let Some(cached) = &cached
            && cached.stale_at > now
        {
            return Ok(cached.keys.clone());
        }
        let held_keys = cached.and_then(|cached| cached.keys);

        // Ask without holding the lock: a concurrent ask is harmless (last
        // writer wins) and a worker's `current_keys` never waits on the
        // network round trip.
        let answer = match (self.find_keys()?, held_keys) {
            (Some(issued), held_keys) => CachedKeys {
                stale_at: choose_stale_at(&issued, held_keys.as_deref(), now),
                keys: Some(Arc::new(issued.keys)),
            },
            (None, None) => CachedKeys {
                keys: None,
                stale_at: now + SOURCES_RECHECK_INTERVAL,
            },
            (None, Some(_)) => {
                return Err(StoreError::Config(format!(
                    "the AWS credentials requests are signed with are due to be read again, \
                     and none are left in {KEY_SOURCES}"
                )));
            }
        };
        *self.cached_keys.write().unwrap() = Some(answer.clone());
        Ok(answer.keys)
    }

    /// The keys the sources last handed over; `None` when they had none, or
    /// were not asked yet. The read-only path for workers: it never asks a
    /// source and never waits on one.
    pub(crate) fn current_keys(&self) -> Option<Arc<S3Keys>> {
        self.cached_keys
            .read()
            .unwrap()
            .as_ref()
            .and_then(|cached| cached.keys.clone())
    }

    /// The value of the environment variable `name`. One set to nothing counts
    /// as not set, as it does to the AWS SDKs.
    fn read_variable(&self, name: &str) -> Option<&str> {
        self.variables
            .get(name)
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
    }

    /// Look for keys where the AWS SDKs look, in their order: the environment,
    /// a container's credentials endpoint, then the role of the EC2 instance.
    fn find_keys(&self) -> Result<Option<IssuedKeys>> {
        if let Some(keys) = self.read_environment_keys()? {
            return Ok(Some(IssuedKeys {
                keys,
                expires_at: None,
            }));
        }
        if let Some(issued) = self.fetch_container_keys()? {
            return Ok(Some(issued));
        }
        self.fetch_instance_role_keys()
    }

    /// The keys handed over in the environment. They are never read again: a
    /// session's keys set this way last until the session does. A key without
    /// its pair is an error rather than no keys at all, which would quietly
    /// sign as whoever the next source names.
    fn read_environment_keys(&self) -> Result<Option<S3Keys>> {
        let access_key = self.read_variable("AWS_ACCESS_KEY_ID");
        let secret_key = self.read_variable("AWS_SECRET_ACCESS_KEY");
        match (access_key, secret_key) {
            (Some(access_key), Some(secret_key)) => Ok(Some(S3Keys {
                access_key: access_key.to_string(),
                secret_key: secret_key.to_string(),
                session_token: self.read_variable("AWS_SESSION_TOKEN").map(str::to_string),
            })),
            (None, None) => Ok(None),
            _ => Err(StoreError::Config(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together".to_string(),
            )),
        }
    }

    /// The keys of the role a container platform gives its workload: an ECS
    /// task's, or the one EKS Pod Identity associates with a pod. The platform
    /// names the endpoint as `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, a path
    /// on the ECS agent's address, or as `AWS_CONTAINER_CREDENTIALS_FULL_URI`.
    /// A request presents the token in the file
    /// `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE` names, read every time because
    /// the platform rotates it, else the one in
    /// `AWS_CONTAINER_AUTHORIZATION_TOKEN`.
    fn fetch_container_keys(&self) -> Result<Option<IssuedKeys>> {
        let relative_uri = self.read_variable("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");
        let full_uri = self.read_variable("AWS_CONTAINER_CREDENTIALS_FULL_URI");
        let url = match (relative_uri, full_uri) {
            (Some(relative_uri), _) => format!("{ECS_AGENT_ENDPOINT}{relative_uri}"),
            (None, Some(full_uri)) => full_uri.to_string(),
            (None, None) => return Ok(None),
        };
        let issuer = format!("the container credentials endpoint {url}");

        let mut request = ureq::get(&url).timeout(CONTAINER_ENDPOINT_TIMEOUT);
        if let Some(path) = self.read_variable("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
            let token = std::fs::read_to_string(path).map_err(|source| StoreError::Io {
                key: path.to_string(),
                source,
            })?;
            request = request.set("Authorization", token.trim());
        } else if let Some(token) = self.read_variable("AWS_CONTAINER_AUTHORIZATION_TOKEN") {
            request = request.set("Authorization", token);
        }
        let document: SessionKeysDocument = request
            .call()
            .map_err(|error| StoreError::Http(format!("asking {issuer} for keys: {error}")))?
            .into_json()
            .map_err(|error| {
                StoreError::Http(format!("{issuer} answered with no keys: {error}"))
            })?;
        document.into_issued_keys(&issuer).map(Some)
    }

    /// The keys of the IAM role attached to the EC2 instance the process runs
    /// on, which the instance metadata service hands out and rotates.
    /// `AWS_EC2_METADATA_SERVICE_ENDPOINT` names another endpoint than the
    /// link-local default, and `AWS_EC2_METADATA_DISABLED=true` skips the
    /// service altogether.
    ///
    /// A service that hands out no session token is no keys rather than an
    /// error: off EC2 nothing is there to answer, or something else is. An
    /// instance without a role has no keys either. Once the service names a
    /// role, failing to read its keys is an error.
    fn fetch_instance_role_keys(&self) -> Result<Option<IssuedKeys>> {
        let is_disabled = self
            .read_variable("AWS_EC2_METADATA_DISABLED")
            .is_some_and(|value| value.eq_ignore_ascii_case("true"));
        if is_disabled {
            return Ok(None);
        }
        let endpoint = self
            .read_variable("AWS_EC2_METADATA_SERVICE_ENDPOINT")
            .unwrap_or(INSTANCE_METADATA_ENDPOINT)
            .trim_end_matches('/');
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(METADATA_CONNECT_TIMEOUT)
            .timeout(METADATA_TIMEOUT)
            .build();

        // The service hands out a session token first and answers only
        // requests that carry it.
        let session_token = agent
            .put(&format!("{endpoint}/latest/api/token"))
            .set("X-aws-ec2-metadata-token-ttl-seconds", "60")
            .call()
            .ok()
            .and_then(|response| response.into_string().ok());
        let Some(session_token) = session_token else {
            return Ok(None);
        };
        let roles_url = format!("{endpoint}{INSTANCE_ROLE_KEYS_PATH}");
        let roles = agent
            .get(&roles_url)
            .set("X-aws-ec2-metadata-token", &session_token)
            .call();
        let roles = match roles {
            Ok(response) => response.into_string().map_err(|error| {
                StoreError::Http(format!(
                    "reading the instance's role from {roles_url}: {error}"
                ))
            })?,
            Err(ureq::Error::Status(404, _)) => return Ok(None),
            Err(error) => {
                return Err(StoreError::Http(format!(
                    "asking the instance metadata service for the instance's role: {error}"
                )));
            }
        };
        let Some(role) = roles.lines().map(str::trim).find(|line| !line.is_empty()) else {
            return Ok(None);
        };

        let issuer = format!("the instance role `{role}`");
        let document: SessionKeysDocument = agent
            .get(&format!("{roles_url}{role}"))
            .set("X-aws-ec2-metadata-token", &session_token)
            .call()
            .map_err(|error| {
                StoreError::Http(format!(
                    "asking the instance metadata service for the keys of {issuer}: {error}"
                ))
            })?
            .into_json()
            .map_err(|error| {
                StoreError::Http(format!(
                    "the instance metadata service answered for {issuer} with no keys: {error}"
                ))
            })?;
        document.into_issued_keys(&issuer).map(Some)
    }
}

/// The Unix second from which `issued` keys are due to be read again:
/// [`KEYS_REFRESH_MARGIN`] before they expire, and never for long-lived ones.
/// Keys the source handed out before, the `held_keys`, it has not replaced
/// yet, and asking again at once would only return them again, so those wait
/// [`SOURCES_RECHECK_INTERVAL`].
fn choose_stale_at(issued: &IssuedKeys, held_keys: Option<&S3Keys>, now: u64) -> u64 {
    let Some(expires_at) = issued.expires_at else {
        return u64::MAX;
    };
    let refresh_at = expires_at.saturating_sub(KEYS_REFRESH_MARGIN);
    let are_keys_unchanged =
        held_keys.is_some_and(|held| held.access_key == issued.keys.access_key);
    if are_keys_unchanged {
        refresh_at.max(now + SOURCES_RECHECK_INTERVAL)
    } else {
        refresh_at
    }
}

/// A session's keys as the container endpoint and the instance metadata
/// service both spell them.
#[derive(serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct SessionKeysDocument {
    access_key_id: String,
    secret_access_key: String,
    token: String,
    expiration: String,
}

impl SessionKeysDocument {
    /// The keys of this document, which `issuer` issued.
    fn into_issued_keys(self, issuer: &str) -> Result<IssuedKeys> {
        let expires_at_millis = parse_iso8601_millis(&self.expiration).ok_or_else(|| {
            StoreError::Config(format!(
                "{issuer} issued keys with an unparseable expiration `{}`",
                self.expiration
            ))
        })?;
        Ok(IssuedKeys {
            keys: S3Keys {
                access_key: self.access_key_id,
                secret_key: self.secret_access_key,
                session_token: Some(self.token),
            },
            expires_at: Some(expires_at_millis / 1000),
        })
    }
}
