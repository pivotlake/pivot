//! End-to-end blackbox test of the encrypted PostgreSQL endpoint, isolated in
//! its own test binary because it needs a server holding a certificate.
//!
//! The server presents a throwaway self-signed certificate. The client insists
//! on SSL and trusts that one certificate and nothing else, so a session it
//! opens at all is proof the configured certificate reached the wire. The
//! plaintext client next to it proves turning SSL on did not take the
//! unencrypted endpoint away.

mod common;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bin::server::config::TlsConfig;
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use common::{CatalogFixture, pivot_metastore, select_rows, start_server_with_tls};
use datastore_delta::DeltaDatastore;
use pgwire::tokio::tokio_rustls::rustls::crypto::aws_lc_rs;
use pgwire::tokio::tokio_rustls::rustls::pki_types::CertificateDer;
use pgwire::tokio::tokio_rustls::rustls::pki_types::pem::PemObject;
use pgwire::tokio::tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tempfile::TempDir;
use tokio_postgres::Client;
use tokio_postgres::config::SslMode;
use tokio_postgres_rustls::MakeRustlsConnect;

/// The name the certificate is issued for and the address the server binds. An
/// IP subject alternative name keeps the client's certificate check off DNS.
const HOST: &str = "127.0.0.1";

/// The one server this binary runs, and the PEM of the certificate it presents.
/// `Dispatch::spin_up` is process-global, so both tests share it.
fn server() -> &'static (u16, Vec<u8>) {
    static SERVER: OnceLock<(u16, Vec<u8>)> = OnceLock::new();
    SERVER.get_or_init(|| {
        let certificate_dir = TempDir::new().unwrap();
        let issued = rcgen::generate_simple_self_signed([HOST.to_string()]).unwrap();
        let config = TlsConfig {
            cert: certificate_dir.path().join("server.crt"),
            key: certificate_dir.path().join("server.key"),
        };
        let pem = issued.cert.pem().into_bytes();
        std::fs::write(&config.cert, &pem).unwrap();
        std::fs::write(&config.key, issued.key_pair.serialize_pem()).unwrap();
        let acceptor = bin::server::tls::build_acceptor(&config).unwrap();

        let port = start_server_with_tls(32, Some(acceptor), |dispatch| {
            let data_dir = TempDir::new().unwrap();
            let datastore: Arc<dyn Datastore> =
                DeltaDatastore::open(&data_dir.path().to_string_lossy(), dispatch.dispatcher())
                    .unwrap();
            let catalog = Arc::new(
                PivotCatalog::new(
                    HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
                    DEFAULT_DATASTORE_NAME.to_string(),
                    pivot_metastore(),
                )
                .unwrap(),
            );
            (
                CatalogFixture::with_data_dir(catalog, data_dir),
                pivot_metastore(),
            )
        });
        (port, pem)
    })
}

/// Connect insisting on SSL, trusting only the certificate the server was
/// configured with. Any other certificate, or none at all, fails the connect.
async fn connect_over_ssl() -> Client {
    let (port, pem) = server();

    let mut roots = RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_slice(pem).unwrap())
        .unwrap();
    // Name the provider for the same reason the server does: rustls has no
    // default while more than one is compiled into the process.
    let client_config =
        ClientConfig::builder_with_provider(Arc::new(aws_lc_rs::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();

    let (client, connection) = tokio_postgres::Config::new()
        .host(HOST)
        .port(*port)
        .user("pivot")
        .dbname("test")
        .ssl_mode(SslMode::Require)
        .connect(MakeRustlsConnect::new(client_config))
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_insisting_on_ssl_gets_an_encrypted_session() {
    let client = connect_over_ssl().await;

    let rows = select_rows(&client, "SELECT 1 + 1").await;

    assert_eq!(rows, vec![vec![Some("2".to_string())]]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_that_declines_ssl_still_gets_a_plaintext_session() {
    let (port, _) = server();

    let client = common::connect_client(*port).await;

    assert_eq!(
        select_rows(&client, "SELECT 1 + 1").await,
        vec![vec![Some("2".to_string())]]
    );
}
