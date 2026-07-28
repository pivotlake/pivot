//! Authenticating a connection against the metastore's users.
//!
//! The metastore's users are an allowlist. Each user explicitly either uses
//! `trust` (the supplied name is accepted without proof) or SCRAM-SHA-256, where
//! the client proves it knows the password without sending it and the server
//! proves it holds the verifier without being able to recover the password from
//! it. Neither the password nor verifier crosses the wire during SCRAM, so it is
//! safe on a plaintext socket, and it is what libpq negotiates by default.
//!
//! The first message of every connection carries the user name. That login
//! consults the metastore for its current authentication method and retains the
//! selected trust or SCRAM handler only for the lifetime of that connection.
//! Consequently users added and verifiers changed while the process is running
//! apply to subsequent logins.

use std::fmt::Debug;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use futures::Sink;
use metastore::{Metastore, SCRAM_ITERATIONS, ScramVerifier, UserAuth};
use pgwire::api::auth::sasl::SASLAuthStartupHandler;
use pgwire::api::auth::sasl::scram::ScramAuth;
use pgwire::api::auth::{AuthSource, DefaultServerParameterProvider, LoginInfo, Password};
use pgwire::api::auth::{StartupHandler, noop::NoopStartupHandler};
use pgwire::api::{ClientInfo, ConnectionManager};
use pgwire::error::{PgWireError, PgWireResult};
use pgwire::messages::{PgWireBackendMessage, PgWireFrontendMessage};
use sha2::{Digest, Sha256};
use tracing::info;

/// The salt width PostgreSQL itself uses for a SCRAM verifier. The salt is
/// public (it goes to the client in the server-first message); its job is to
/// make one precomputed table useless against every other user's verifier.
const SALT_LEN: usize = 16;

/// Width of the per-process secret behind [`mock_verifier`], matching the
/// SHA-256 block output it is hashed with.
const MOCK_SECRET_LEN: usize = 32;

/// A fake SCRAM credential returned for a user the metastore does not define.
///
/// Without it, an unknown user would fail before the server-first message while
/// a known user with a wrong password would complete the SCRAM exchange and fail
/// at its proof. That difference lets an attacker enumerate account names. The
/// mock verifier gives an unknown user a normal-looking challenge so both cases
/// fail at the same stage. It does not create a user or grant access: producing
/// a password proof for its synthetic salted password is computationally
/// infeasible.
///
/// For `H(x) = SHA256(secret || x || username)`, the public salt is the first
/// [`SALT_LEN`] bytes of `H(0)` and the synthetic salted password is all of
/// `H(1)`. The distinct domain bytes keep those values independent. The secret
/// is generated once per process, making the result unpredictable but stable
/// for a username across repeated attempts, like a real stored verifier. It
/// changes when the server restarts. PostgreSQL calls this mock authentication.
fn mock_verifier(secret: &[u8; MOCK_SECRET_LEN], username: &str) -> ScramVerifier {
    let derive = |domain: u8| {
        let mut digest = Sha256::new();
        digest.update(secret);
        digest.update([domain]);
        digest.update(username.as_bytes());
        digest.finalize()
    };
    ScramVerifier {
        salt: derive(0)[..SALT_LEN].to_vec(),
        salted_password: derive(1).to_vec(),
    }
}

/// Looks a login's user up in the metastore and hands pgwire the stored SCRAM
/// verifier to check the client's proof against.
pub(crate) struct MetastoreAuthSource {
    /// The live source consulted for each login.
    metastore: Arc<dyn Metastore>,
    /// Keys [`mock_verifier`]. Generated per process and never leaves it, so an
    /// unknown user's answer cannot be recomputed by whoever asked for it.
    // TODO: Persist the mock-verifier secret instead of regenerating it for
    // each process. Otherwise, an unknown user's derived salt changes across
    // restarts while a real user's stored salt remains stable, creating a
    // restart oracle that reveals whether the account exists.
    mock_secret: [u8; MOCK_SECRET_LEN],
}

/// The metastore holds credentials, so it is not printable; pgwire requires
/// [`Debug`] on an [`AuthSource`] only to include it in its own diagnostics.
impl Debug for MetastoreAuthSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MetastoreAuthSource")
    }
}

impl MetastoreAuthSource {
    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        self.metastore.user_auth(username)
    }
}

#[async_trait]
impl AuthSource for MetastoreAuthSource {
    async fn get_password(&self, login: &LoginInfo) -> PgWireResult<Password> {
        let user = login.user().ok_or(PgWireError::UserNameRequired)?;
        // An unknown user is answered with a verifier no password satisfies, so
        // it fails at the same point in the handshake a wrong SCRAM password
        // does. A trusted user never reaches the SCRAM handler; treating one as
        // unknown here is a defensive failure if routing and lookup ever drift.
        let verifier = match self.user_auth(user) {
            Some(UserAuth::ScramSha256(verifier)) => verifier,
            Some(UserAuth::Trust) | None => mock_verifier(&self.mock_secret, user),
        };
        Ok(Password::new(Some(verifier.salt), verifier.salted_password))
    }
}

/// State shared by all logins. The selected authentication method and any SCRAM
/// exchange state remain per connection.
pub struct Authenticator {
    auth_source: Arc<MetastoreAuthSource>,
    parameters: Arc<DefaultServerParameterProvider>,
    manager: Arc<ConnectionManager>,
}

impl Authenticator {
    /// Build authentication over the live metastore.
    pub fn new(metastore: Arc<dyn Metastore>, manager: Arc<ConnectionManager>) -> Self {
        info!("authenticating connections against the metastore");
        Self {
            auth_source: Arc::new(MetastoreAuthSource {
                metastore,
                mock_secret: rand::random(),
            }),
            parameters: Arc::new(DefaultServerParameterProvider::default()),
            manager,
        }
    }

    /// A new handler for one connection. Its first startup message performs a
    /// fresh metastore lookup; a SCRAM handshake then retains its own state.
    pub fn startup_handler(&self) -> Arc<UserStartupHandler> {
        Arc::new(UserStartupHandler {
            auth_source: self.auth_source.clone(),
            parameters: self.parameters.clone(),
            manager: self.manager.clone(),
            selected: OnceLock::new(),
        })
    }
}

/// One configured-user connection before and after its user name selects an
/// authentication method.
pub struct UserStartupHandler {
    auth_source: Arc<MetastoreAuthSource>,
    parameters: Arc<DefaultServerParameterProvider>,
    manager: Arc<ConnectionManager>,
    selected: OnceLock<SelectedUserAuth>,
}

impl UserStartupHandler {
    fn select(&self, username: &str) -> SelectedUserAuth {
        match self.auth_source.user_auth(username) {
            Some(UserAuth::Trust) => SelectedUserAuth::Trust(UnauthenticatedStartupHandler {
                manager: self.manager.clone(),
            }),
            // Unknown users take the SCRAM path with the auth source's mock
            // verifier, preserving the existing unknown/wrong-password shape.
            Some(UserAuth::ScramSha256(_)) | None => {
                let verifiers: Arc<dyn AuthSource> = self.auth_source.clone();
                let mut scram = ScramAuth::new(verifiers);
                // The count the client is told to derive its proof with. Every
                // stored verifier was derived with the same one, and a metastore
                // rejects a verifier that disagrees, so the two cannot drift.
                scram.set_iterations(SCRAM_ITERATIONS);
                SelectedUserAuth::Scram(
                    SASLAuthStartupHandler::new(self.parameters.clone())
                        .with_scram(scram)
                        .with_connection_manager(self.manager.clone()),
                )
            }
        }
    }
}

#[async_trait]
impl StartupHandler for UserStartupHandler {
    async fn on_startup<C>(
        &self,
        client: &mut C,
        message: PgWireFrontendMessage,
    ) -> PgWireResult<()>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let selected = match self.selected.get() {
            Some(selected) => selected,
            None => {
                let PgWireFrontendMessage::Startup(startup) = &message else {
                    return Err(PgWireError::InvalidStartupMessage);
                };
                let username = startup
                    .parameters
                    .get("user")
                    .ok_or(PgWireError::UserNameRequired)?;
                self.selected.get_or_init(|| self.select(username))
            }
        };
        match selected {
            SelectedUserAuth::Trust(handler) => handler.on_startup(client, message).await,
            SelectedUserAuth::Scram(handler) => handler.on_startup(client, message).await,
        }
    }
}

/// The authentication handler retained after a configured connection's first
/// startup message.
enum SelectedUserAuth {
    Trust(UnauthenticatedStartupHandler),
    Scram(SASLAuthStartupHandler<DefaultServerParameterProvider>),
}

/// No-auth startup handling for an explicitly trusted user, plus the
/// [`ConnectionManager`] registration that cancellation needs.
struct UnauthenticatedStartupHandler {
    manager: Arc<ConnectionManager>,
}

impl NoopStartupHandler for UnauthenticatedStartupHandler {
    fn connection_manager(&self) -> Option<Arc<ConnectionManager>> {
        Some(self.manager.clone())
    }
}
