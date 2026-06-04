//! rustls client configuration for the http backends.
//!
//! rustls is sans-IO: it owns the TLS state machine and produces/consumes
//! ciphertext through in-memory buffers, which each backend shuttles over its
//! socket (io_uring `send`/`recv` on Linux, blocking reads/writes elsewhere).
//! We pin the `ring` crypto provider explicitly (rather than relying on a
//! process-global default) and trust the bundled webpki Mozilla root set.

use rustls::ClientConfig;
use std::sync::{Arc, OnceLock};

/// Build a client config trusting the given roots, using the `ring` provider and
/// safe default protocol versions.
pub fn client_config_with_roots(roots: rustls::RootCertStore) -> Arc<ClientConfig> {
    let config = ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("ring provider supports the default protocol versions")
    .with_root_certificates(roots)
    .with_no_client_auth();
    Arc::new(config)
}

/// The process-wide default client config, trusting webpki's bundled roots.
///
/// Built once and shared (cloning an `Arc`), so every worker's http backend uses
/// the same config without rebuilding the root store per core.
pub fn default_client_config() -> Arc<ClientConfig> {
    static CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let mut roots = rustls::RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            client_config_with_roots(roots)
        })
        .clone()
}
