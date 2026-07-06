//! Blackbox tests against a real local Iceberg stack: tables are created and
//! snapshots committed by the upstream `iceberg` crate (the oracle) through an
//! `apache/iceberg-rest-fixture` service over MinIO, and read back through
//! [`IcebergRestCatalog`] - resolve, compile, and scan over the engine. Skips
//! (green) when Docker is unavailable.

mod common;
use common::*;

use dispatch::Projection;
use iceberg_catalog::{IcebergRestCatalog, IcebergRestConfig};
use planner::catalog::{Catalog, Table};

fn connect(dispatch: &DispatchGuard, config: IcebergRestConfig) -> IcebergRestCatalog {
    IcebergRestCatalog::connect(config, (*dispatch).clone()).expect("connect to rest catalog")
}

/// Scan `table`'s two columns through a fresh query context, returning
/// `(id, name)` rows sorted by id (workers emit batches in nondeterministic
/// order).
fn scan_rows(
    dispatch: &DispatchGuard,
    catalog: &IcebergRestCatalog,
    table: &dyn Table,
) -> Vec<(i64, String)> {
    let ctx = catalog.query_context();
    let spec = table
        .compile(dispatch, Projection::all(2), vec![], false, ctx.as_ref())
        .expect("compile scan");
    let batches = spec.collect().expect("run scan");

    let ids = collect_i64s(&batches, 0);
    let names = collect_strings(&batches, 1);
    let mut rows: Vec<(i64, String)> = ids.into_iter().zip(names).collect();
    rows.sort();
    rows
}

#[test]
fn scans_rows_committed_by_an_external_writer() {
    let Some(harness) = harness() else { return };
    let dispatch = start_dispatch(2);
    let oracle = Oracle::connect(harness);
    oracle.create_table("db_scan", "events");
    oracle.append_rows("db_scan", "events", &[(1, "a"), (2, "b"), (3, "c")]);

    let catalog = connect(&dispatch, IcebergRestConfig::new(&harness.rest_uri));
    let table = catalog.table("db_scan.events").expect("table resolves");

    let columns: Vec<(String, String)> = table
        .columns()
        .iter()
        .map(|c| (c.name.clone(), c.col_type.to_string()))
        .collect();
    assert_eq!(
        columns,
        vec![
            ("id".to_string(), "Int64".to_string()),
            ("name".to_string(), "Utf8".to_string()),
        ]
    );
    assert_eq!(
        scan_rows(&dispatch, &catalog, table.as_ref()),
        vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string())
        ]
    );
}

#[test]
fn a_reused_binding_sees_snapshots_committed_after_it_was_bound() {
    let Some(harness) = harness() else { return };
    let dispatch = start_dispatch(2);
    let oracle = Oracle::connect(harness);
    oracle.create_table("db_snapshots", "events");
    oracle.append_rows("db_snapshots", "events", &[(1, "a")]);
    let catalog = connect(&dispatch, IcebergRestConfig::new(&harness.rest_uri));
    let table = catalog.table("db_snapshots.events").expect("resolves");

    let before = scan_rows(&dispatch, &catalog, table.as_ref());
    oracle.append_rows("db_snapshots", "events", &[(2, "b"), (3, "c")]);
    let after = scan_rows(&dispatch, &catalog, table.as_ref());

    // The binding carries no snapshot; each query's context resolves the
    // current one, so even a cached plan's reused binding (the server caches
    // plans by SQL text) sees rows committed after it was bound.
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 3);
}

#[test]
fn an_unqualified_name_resolves_in_the_configured_default_namespace() {
    let Some(harness) = harness() else { return };
    let dispatch = start_dispatch(2);
    let oracle = Oracle::connect(harness);
    oracle.create_table("db_unqualified", "orders");
    oracle.append_rows("db_unqualified", "orders", &[(7, "x")]);

    let mut config = IcebergRestConfig::new(&harness.rest_uri);
    config.default_namespace = "db_unqualified".to_string();
    let catalog = connect(&dispatch, config);
    let table = catalog
        .table("orders")
        .expect("resolves in the default namespace");

    assert_eq!(
        scan_rows(&dispatch, &catalog, table.as_ref()),
        vec![(7, "x".to_string())]
    );
}

#[test]
fn a_missing_table_resolves_to_none() {
    let Some(harness) = harness() else { return };
    let dispatch = start_dispatch(1);

    let catalog = connect(&dispatch, IcebergRestConfig::new(&harness.rest_uri));

    assert!(catalog.table("db_missing.nope").is_none());
    assert!(catalog.table("nope").is_none());
}

#[test]
fn a_table_with_no_snapshot_scans_empty() {
    let Some(harness) = harness() else { return };
    let dispatch = start_dispatch(1);
    let oracle = Oracle::connect(harness);
    oracle.create_table("db_empty", "events");

    let catalog = connect(&dispatch, IcebergRestConfig::new(&harness.rest_uri));
    let table = catalog.table("db_empty.events").expect("resolves");

    assert_eq!(scan_rows(&dispatch, &catalog, table.as_ref()), vec![]);
}

#[test]
fn create_table_is_rejected() {
    let Some(harness) = harness() else { return };
    let dispatch = start_dispatch(1);

    let catalog = connect(&dispatch, IcebergRestConfig::new(&harness.rest_uri));
    let result = catalog.create_table(
        planner::catalog::CreateTableRequest {
            name: "t".to_string(),
            columns: vec![],
            options: Default::default(),
            if_not_exists: false,
        },
        &dispatch,
    );

    let error = match result {
        Ok(_) => panic!("create_table unexpectedly succeeded"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("read-only"), "unexpected error: {error}");
}
