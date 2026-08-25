//! Planner lifecycle and revision-aware plan caching.

use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;

use super::{Error, Result};

thread_local! {
    /// One planner and its non-Send DuckDB context per blocking-pool thread.
    static PLANNER: RefCell<Option<planner::Planner>> = const { RefCell::new(None) };
}

const PLAN_CACHE_QUERY_CAPACITY: usize = 128;

pub(super) struct PlanCache {
    inner: Mutex<LruCache<String, Arc<planner::Plan>>>,
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(PLAN_CACHE_QUERY_CAPACITY)
    }
}

impl PlanCache {
    fn new(query_capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(
                NonZeroUsize::new(query_capacity).expect("plan cache capacity must be non-zero"),
            )),
        }
    }

    fn get(
        &self,
        query: &str,
        transaction: &dyn planner::catalog::CatalogTransaction,
    ) -> Option<Arc<planner::Plan>> {
        let mut inner = self.inner.lock().unwrap();
        let plan = inner.get(query)?.clone();
        if plan.has_matching_table_revisions(transaction) {
            Some(plan)
        } else {
            let _ = inner.pop(query);
            None
        }
    }

    fn insert(&self, query: String, plan: Arc<planner::Plan>) {
        debug_assert!(plan.is_cacheable());
        self.inner.lock().unwrap().put(query, plan);
    }
}

fn with_planner<R>(
    catalog: &Arc<catalog::PivotCatalog>,
    function: impl FnOnce(&mut planner::Planner) -> R,
) -> Result<R, planner::Error> {
    PLANNER.with_borrow_mut(|slot| {
        let planner = match slot {
            Some(planner) => planner,
            None => slot.insert(planner::Planner::from_datastore_names(
                catalog.datastore_names(),
                catalog.default_datastore_name().to_string(),
            )?),
        };
        Ok(function(planner))
    })
}

pub(super) async fn plan_query(
    catalog: &Arc<catalog::PivotCatalog>,
    transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    plan_cache: &PlanCache,
    query: &str,
) -> Result<Arc<planner::Plan>> {
    if let Some(plan) = plan_cache.get(query, transaction.as_ref()) {
        return Ok(plan);
    }

    let catalog = catalog.clone();
    let cache_key = query.to_string();
    let query = cache_key.clone();
    let plan = tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
        with_planner(&catalog, |planner| {
            Ok(Arc::new(planner.plan(&query, transaction)?))
        })?
    })
    .await
    .map_err(Error::PlannerPanic)??;
    if plan.is_cacheable() {
        plan_cache.insert(cache_key, plan.clone());
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use planner::DEFAULT_DATASTORE_NAME;
    use planner::catalog::{BoundTable, Column, TableReference, TableRevision};

    use super::PlanCache;

    #[derive(Clone, Debug)]
    struct RevisionTable {
        reference: TableReference,
        revision: TableRevision,
    }

    impl BoundTable for RevisionTable {
        fn table_reference(&self) -> TableReference {
            self.reference.clone()
        }

        fn table_revision(&self) -> TableRevision {
            self.revision.clone()
        }

        fn compile_scan(
            &self,
            _dispatcher: &dispatch::DataFlowDispatcher,
            _projection: dispatch::Projection,
            _dynamic_filters: Vec<planner::catalog::DynamicScanPredicate>,
            _emit_row_group_metadata: bool,
        ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
            unreachable!("cache tests do not compile plans")
        }

        fn columns(&self) -> Vec<Column> {
            Vec::new()
        }

        fn clone_box(&self) -> Box<dyn BoundTable> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct RevisionTransaction {
        revisions: HashMap<TableReference, TableRevision>,
    }

    #[async_trait]
    impl planner::catalog::CatalogTransaction for RevisionTransaction {
        fn does_schema_exist(&self, _datastore: &str, schema: &str) -> bool {
            schema == planner::DEFAULT_SCHEMA_NAME
        }

        fn bind_table(&self, _reference: &TableReference) -> Option<Box<dyn BoundTable>> {
            None
        }

        fn table_revision(&self, reference: &TableReference) -> Option<TableRevision> {
            self.revisions.get(reference).cloned()
        }
    }

    fn table() -> TableReference {
        TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: planner::DEFAULT_SCHEMA_NAME.to_string(),
            table: "events".to_string(),
        }
    }

    fn revision(identity: &str, version: u64) -> TableRevision {
        TableRevision {
            identity: identity.to_string(),
            version,
        }
    }

    fn transaction(identity: &str, version: u64) -> RevisionTransaction {
        RevisionTransaction {
            revisions: HashMap::from([(table(), revision(identity, version))]),
        }
    }

    fn plan(identity: &str, version: u64) -> Arc<planner::Plan> {
        Arc::new(planner::Plan {
            root: planner::PlanNode {
                name: "input".to_string(),
                inputs: Vec::new(),
                operator: planner::Operator::Input(planner::operator::Input {
                    table: Box::new(RevisionTable {
                        reference: table(),
                        revision: revision(identity, version),
                    }),
                    columns: Vec::new(),
                    dynamic_filters: Vec::new(),
                    duckdb_table_binding_index: None,
                    join_filter_info: None,
                    emit_row_group_metadata: false,
                }),
            },
            output_names: Vec::new(),
            requires_rebind: false,
        })
    }

    #[test]
    fn cache_hit_requires_the_same_identity_and_version() {
        let cache = PlanCache::new(2);
        let cached = plan("table-id", 7);
        cache.insert("SELECT * FROM events".to_string(), cached.clone());

        let hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 7))
            .unwrap();
        let changed_version = cache.get("SELECT * FROM events", &transaction("table-id", 8));
        cache.insert("SELECT * FROM events".to_string(), cached.clone());
        let changed_identity = cache.get("SELECT * FROM events", &transaction("new-table-id", 7));

        assert!(Arc::ptr_eq(&hit, &cached));
        assert!(changed_version.is_none());
        assert!(changed_identity.is_none());
        assert!(cache.inner.lock().unwrap().is_empty());
    }

    #[test]
    fn inserting_a_new_revision_replaces_the_previous_plan() {
        let cache = PlanCache::new(2);
        cache.insert("SELECT * FROM events".to_string(), plan("table-id", 7));
        let current = plan("table-id", 8);

        cache.insert("SELECT * FROM events".to_string(), current.clone());
        let hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 8))
            .unwrap();

        assert!(Arc::ptr_eq(&hit, &current));
        assert_eq!(cache.inner.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_revision_from_another_schema_does_not_validate_a_cached_plan() {
        let analytics = TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: "analytics".to_string(),
            table: "events".to_string(),
        };
        let cached = Arc::new(planner::Plan {
            root: planner::PlanNode {
                name: "input".to_string(),
                inputs: Vec::new(),
                operator: planner::Operator::Input(planner::operator::Input {
                    table: Box::new(RevisionTable {
                        reference: analytics.clone(),
                        revision: revision("table-id", 7),
                    }),
                    columns: Vec::new(),
                    dynamic_filters: Vec::new(),
                    duckdb_table_binding_index: None,
                    join_filter_info: None,
                    emit_row_group_metadata: false,
                }),
            },
            output_names: Vec::new(),
            requires_rebind: false,
        });
        let cache = PlanCache::new(2);
        cache.insert("SELECT * FROM analytics.events".to_string(), cached);
        let other_schema = RevisionTransaction {
            revisions: HashMap::from([(table(), revision("table-id", 7))]),
        };

        let hit = cache.get("SELECT * FROM analytics.events", &other_schema);

        assert!(hit.is_none());
    }

    #[test]
    fn inserting_over_capacity_evicts_the_least_recently_used_query() {
        let cache = PlanCache::new(2);
        let first = plan("first", 1);
        cache.insert("first query".to_string(), first.clone());
        cache.insert("second query".to_string(), plan("second", 1));
        cache.get("first query", &transaction("first", 1)).unwrap();

        let third = plan("third", 1);
        cache.insert("third query".to_string(), third.clone());

        assert!(
            cache
                .get("second query", &transaction("second", 1))
                .is_none()
        );
        assert!(Arc::ptr_eq(
            &cache.get("first query", &transaction("first", 1)).unwrap(),
            &first
        ));
        assert!(Arc::ptr_eq(
            &cache.get("third query", &transaction("third", 1)).unwrap(),
            &third
        ));
    }
}
