//! Encrypting the PostgreSQL endpoint.
//!
//! pgwire runs the `SSLRequest` handshake itself; all this module does is build
//! the [`TlsAcceptor`] it needs. Without one, a connection asking to upgrade is
//! turned down, and either way a client asking for plaintext gets plaintext: a
//! certificate makes SSL available rather than compulsory.
//!
//! The rustls types come from pgwire's re-export because the acceptor goes
//! straight back to [`process_socket`](pgwire::tokio::process_socket), so it has
//! to be built by the same rustls that runs the handshake.
//!
//! Channel binding (`SCRAM-SHA-256-PLUS`) is not offered, so a client passing
//! `channel_binding=require` is turned away. Encrypted sessions authenticate
//! with plain `SCRAM-SHA-256`, as plaintext ones do.

use std::sync::Arc;

use pgwire::tokio::TlsAcceptor;
use pgwire::tokio::tokio_rustls::rustls::crypto::aws_lc_rs;
use pgwire::tokio::tokio_rustls::rustls::pki_types::pem::{self, PemObject};
use pgwire::tokio::tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use pgwire::tokio::tokio_rustls::rustls::{self, ServerConfig as TlsServerConfig};

use crate::config::TlsConfig;

/// The protocol a client names in the TLS handshake when it skips the
/// `SSLRequest` round trip and starts with the handshake itself (direct SSL
/// negotiation, offered by PostgreSQL 17 and later clients). pgwire drops such a
/// connection unless the handshake agreed on this name, so advertising it is
/// what makes those clients work.
const POSTGRES_ALPN: &[u8] = b"postgresql";

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading the TLS certificate chain `{path}`: {source}")]
    Certificate {
        path: String,
        #[source]
        source: pem::Error,
    },
    #[error("reading the TLS private key `{path}`: {source}")]
    PrivateKey {
        path: String,
        #[source]
        source: pem::Error,
    },
    #[error(
        "the TLS certificate `{cert}` and the private key `{key}` are not a usable pair: {source}"
    )]
    Pair {
        cert: String,
        key: String,
        #[source]
        source: rustls::Error,
    },
}

/// Build the acceptor that upgrades a connection asking for SSL, from the
/// certificate chain and private key `config` names.
///
/// Both files are read once, here, so a certificate that is missing, malformed
/// or paired with the wrong key stops startup instead of surfacing on whichever
/// client first asks to encrypt.
pub fn build_acceptor(config: &TlsConfig) -> Result<TlsAcceptor> {
    let cert_path = config.cert.display().to_string();
    let key_path = config.key.display().to_string();

    // Every section of the file, not just the first: a certificate signed by an
    // intermediate authority is only verifiable if the intermediates travel with
    // it up to a root the client already trusts.
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&config.cert)
        .and_then(|sections| sections.collect())
        .map_err(|source| Error::Certificate {
            path: cert_path.clone(),
            source,
        })?;
    if chain.is_empty() {
        return Err(Error::Certificate {
            path: cert_path,
            source: pem::Error::NoItemsFound,
        });
    }
    // Whichever of PKCS#8, PKCS#1 and SEC1 the file holds: each is a PEM section
    // kind a `PrivateKeyDer` decodes, and which one a key is written in depends
    // on the tool that generated it rather than on anything we choose.
    let key = PrivateKeyDer::from_pem_file(&config.key).map_err(|source| Error::PrivateKey {
        path: key_path.clone(),
        source,
    })?;

    // Name the cryptography provider rather than leaving rustls to its default.
    // The default is only defined when exactly one provider is compiled into the
    // process, and this workspace's dependencies bring in more than one; naming
    // aws-lc-rs also keeps the handshake on the provider pgwire itself selects.
    let mut server_config =
        TlsServerConfig::builder_with_provider(Arc::new(aws_lc_rs::default_provider()))
            .with_safe_default_protocol_versions()
            .expect("aws-lc-rs supports every protocol version rustls enables by default")
            .with_no_client_auth()
            .with_single_cert(chain, key)
            .map_err(|source| Error::Pair {
                cert: cert_path,
                key: key_path,
                source,
            })?;
    server_config.alpn_protocols = vec![POSTGRES_ALPN.to_vec()];

    Ok(TlsAcceptor::from(Arc::new(server_config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A self-signed certificate for `localhost` and the key that goes with it,
    /// each written to its own PEM file in a directory the caller keeps alive.
    fn self_signed(dir: &tempfile::TempDir) -> TlsConfig {
        let issued = rcgen::generate_simple_self_signed(["localhost".to_string()]).unwrap();
        let config = TlsConfig {
            cert: dir.path().join("server.crt"),
            key: dir.path().join("server.key"),
        };
        std::fs::write(&config.cert, issued.cert.pem()).unwrap();
        std::fs::write(&config.key, issued.key_pair.serialize_pem()).unwrap();
        config
    }

    #[test]
    fn a_certificate_and_its_key_build_an_acceptor() {
        let dir = tempfile::tempdir().unwrap();
        let config = self_signed(&dir);

        build_acceptor(&config).expect("a matching certificate and key should be accepted");
    }

    #[test]
    fn a_certificate_belonging_to_another_key_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let config = self_signed(&dir);
        let other = tempfile::tempdir().unwrap();

        let error = build_acceptor(&TlsConfig {
            cert: config.cert,
            key: self_signed(&other).key,
        })
        .err()
        .expect("a key that signed nothing in the chain should be rejected");

        assert!(matches!(error, Error::Pair { .. }), "{error}");
    }

    #[test]
    fn a_file_holding_no_certificate_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let config = self_signed(&dir);
        let empty = dir.path().join("empty.crt");
        std::fs::write(&empty, "").unwrap();
        let named = empty.display().to_string();

        let error = build_acceptor(&TlsConfig {
            cert: empty,
            key: config.key,
        })
        .err()
        .expect("a certificate file with nothing in it should be rejected");

        assert!(
            matches!(&error, Error::Certificate { path, .. } if *path == named),
            "{error}"
        );
    }

    #[test]
    fn a_missing_certificate_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let config = self_signed(&dir);

        let error = build_acceptor(&TlsConfig {
            cert: PathBuf::from("/nonexistent/server.crt"),
            key: config.key,
        })
        .err()
        .expect("a certificate path that does not exist should be rejected");

        assert!(
            matches!(&error, Error::Certificate { path, .. } if path == "/nonexistent/server.crt"),
            "{error}"
        );
    }
}
