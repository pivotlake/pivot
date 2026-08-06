//! Blackbox tests of the Postgres-backed metastore against a real PostgreSQL
//! container. Every test gets its own database (the schema name is fixed);
//! environments without Docker skip.

use metastore::{DEFAULT_USER_NAME, Metastore, ScramVerifier, UserAuth, format_scram_verifier};
use metastore_postgres::{Error, PostgresMetastore, PostgresMetastoreConfig, test_support};
use postgres::error::SqlState;

fn admin_client(url: &str) -> postgres::Client {
    postgres::Client::connect(url, postgres::NoTls).unwrap()
}

fn connect(url: &str) -> PostgresMetastore {
    PostgresMetastore::connect(PostgresMetastoreConfig::new(url)).unwrap()
}

/// A fresh database whose `pivot_metastore` schema exists: the first connection
/// creates it, and reports the still-empty `datastores` table as the missing
/// default it is.
fn prepare_database(name: &str) -> Option<String> {
    let url = test_support::fresh_database_url(name)?;
    let error = PostgresMetastore::connect(PostgresMetastoreConfig::new(&url)).unwrap_err();
    assert!(matches!(error, Error::MissingDefault), "{error}");
    Some(url)
}

fn insert_datastore(admin: &mut postgres::Client, name: &str, is_default: bool) {
    admin
        .execute(
            "INSERT INTO pivot_metastore.datastores (name, location, is_default) \
             VALUES ($1, $2, $3)",
            &[&name, &format!("/tmp/{name}"), &is_default],
        )
        .unwrap();
}

fn verifier_for(password: &str) -> String {
    format_scram_verifier(&ScramVerifier {
        salt: vec![1; 16],
        salted_password: password
            .as_bytes()
            .iter()
            .cycle()
            .take(32)
            .copied()
            .collect(),
    })
}

#[test]
fn the_default_datastore_is_the_flagged_row() {
    let Some(url) = prepare_database("default_flag") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);
    insert_datastore(&mut admin, "warm", false);

    let metastore = connect(&url);

    assert_eq!(metastore.default_datastore_name(), "hot");
}

#[test]
fn a_second_default_datastore_is_rejected_by_the_shared_database() {
    let Some(url) = prepare_database("two_defaults") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);

    let error = admin
        .execute(
            "INSERT INTO pivot_metastore.datastores (name, location, is_default) \
             VALUES ('warm', '/tmp/warm', true)",
            &[],
        )
        .unwrap_err();

    assert_eq!(error.code(), Some(&SqlState::UNIQUE_VIOLATION));
}

#[test]
fn a_second_instance_prepares_the_same_schema_without_conflict() {
    let Some(url) = prepare_database("two_instances") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);

    let first = connect(&url);
    let second = connect(&url);

    assert_eq!(first.default_datastore_name(), "hot");
    assert_eq!(second.default_datastore_name(), "hot");
}

#[test]
fn an_empty_users_table_provides_the_builtin_pivot_user() {
    let Some(url) = prepare_database("builtin_pivot") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);

    let metastore = connect(&url);

    assert!(matches!(
        metastore.user_auth(DEFAULT_USER_NAME).unwrap(),
        Some(UserAuth::Trust)
    ));
    assert!(metastore.user_auth("analytics").unwrap().is_none());
}

#[test]
fn a_user_inserted_after_startup_replaces_the_builtin_allowlist() {
    let Some(url) = prepare_database("live_users") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);
    let metastore = connect(&url);

    admin
        .execute(
            "INSERT INTO pivot_metastore.users (name, auth_method) VALUES ('reader', 'trust')",
            &[],
        )
        .unwrap();

    assert!(matches!(
        metastore.user_auth("reader").unwrap(),
        Some(UserAuth::Trust)
    ));
    assert!(metastore.user_auth(DEFAULT_USER_NAME).unwrap().is_none());
}

#[test]
fn a_scram_user_keeps_its_verifier_and_rotation_applies_to_the_next_lookup() {
    let Some(url) = prepare_database("scram_rotation") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);
    admin
        .execute(
            "INSERT INTO pivot_metastore.users (name, auth_method, scram_verifier) \
             VALUES ('analytics', 'scram-sha-256', $1)",
            &[&verifier_for("a")],
        )
        .unwrap();
    let metastore = connect(&url);

    let Some(UserAuth::ScramSha256(before)) = metastore.user_auth("analytics").unwrap() else {
        panic!("analytics should use SCRAM-SHA-256");
    };
    admin
        .execute(
            "UPDATE pivot_metastore.users SET scram_verifier = $1 WHERE name = 'analytics'",
            &[&verifier_for("b")],
        )
        .unwrap();
    let Some(UserAuth::ScramSha256(after)) = metastore.user_auth("analytics").unwrap() else {
        panic!("analytics should use SCRAM-SHA-256");
    };

    assert_eq!(before.salted_password, [b'a'; 32]);
    assert_eq!(after.salted_password, [b'b'; 32]);
}

#[test]
fn a_malformed_verifier_fails_the_lookup_rather_than_denying_in_silence() {
    let Some(url) = prepare_database("malformed_verifier") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);
    admin
        .execute(
            "INSERT INTO pivot_metastore.users (name, auth_method, scram_verifier) \
             VALUES ('broken', 'scram-sha-256', 'hunter2')",
            &[],
        )
        .unwrap();
    let metastore = connect(&url);

    let error = metastore.user_auth("broken").unwrap_err();

    assert!(error.to_string().contains("pivot-scram-sha-256"), "{error}");
}

#[test]
fn inconsistent_user_rows_are_rejected_by_the_database() {
    let Some(url) = prepare_database("inconsistent_users") else {
        return;
    };
    let mut admin = admin_client(&url);
    insert_datastore(&mut admin, "hot", true);

    for insert in [
        "INSERT INTO pivot_metastore.users (name, auth_method, scram_verifier) \
         VALUES ('reader', 'trust', 'irrelevant')",
        "INSERT INTO pivot_metastore.users (name, auth_method) VALUES ('analytics', 'scram-sha-256')",
        "INSERT INTO pivot_metastore.users (name, auth_method) VALUES ('spy', 'kerberos')",
    ] {
        let error = admin.execute(insert, &[]).unwrap_err();

        assert_eq!(error.code(), Some(&SqlState::CHECK_VIOLATION), "{insert}");
    }
}
