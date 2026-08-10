//! A bounded, shared cache of planned read queries.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;

/// Maximum distinct SQL strings retained.
const PLAN_CACHE_QUERY_CAPACITY: usize = 128;

/// A bounded, shared cache of planned read queries.
///
/// The map is an LRU over exact SQL strings and retains one plan per query.
pub struct PlanCache {
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

    /// Find the cached plan if its revisions match `transaction`'s frozen
    /// snapshots. A revision mismatch discards the entry.
    pub(crate) fn get(
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

    /// Insert one cacheable plan, replacing any plan for the same SQL.
    pub(crate) fn insert(&self, query: String, plan: Arc<planner::Plan>) {
        debug_assert!(plan.is_cacheable());
        self.inner.lock().unwrap().put(query, plan);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use planner::DEFAULT_DATASTORE_NAME;
    use planner::catalog::{BoundTable, Column, TableReference, TableRevision};
    use std::collections::HashMap;

    #[derive(Clone, Debug)]
    struct CacheTestTable {
        reference: TableReference,
        revision: TableRevision,
    }

    impl BoundTable for CacheTestTable {
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
            unreachable!("cache unit test does not compile its plans")
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
        table_in_schema(planner::DEFAULT_SCHEMA_NAME)
    }

    /// The same table name in `schema`. Two schemas can each hold an `events`,
    /// so these are references to different tables.
    fn table_in_schema(schema: &str) -> TableReference {
        TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: schema.to_string(),
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
        transaction_over(table(), identity, version)
    }

    /// A transaction whose only known table is `reference`, at `identity` and
    /// `version`.
    fn transaction_over(
        reference: TableReference,
        identity: &str,
        version: u64,
    ) -> RevisionTransaction {
        RevisionTransaction {
            revisions: HashMap::from([(reference, revision(identity, version))]),
        }
    }

    fn plan(identity: &str, version: u64) -> Arc<planner::Plan> {
        plan_over(table(), identity, version)
    }

    /// A one-input plan that reads `reference` at `identity` and `version`.
    fn plan_over(reference: TableReference, identity: &str, version: u64) -> Arc<planner::Plan> {
        Arc::new(planner::Plan {
            root: planner::PlanNode {
                name: "input".to_string(),
                inputs: Vec::new(),
                operator: planner::Operator::Input(planner::operator::Input {
                    table: Box::new(CacheTestTable {
                        reference,
                        revision: revision(identity, version),
                    }),
                    columns: Vec::new(),
                    dynamic_filters: Vec::new(),
                    emit_row_group_metadata: false,
                }),
            },
            output_names: Vec::new(),
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

    /// A cached plan stays valid only while the tables it bound are unchanged,
    /// and a table is identified by its schema as much as by its name: two
    /// schemas can each hold an `events`, and they are different tables. A
    /// revision recorded for one of them must therefore not vouch for a plan
    /// built over the other, however alike the two revisions look.
    #[test]
    fn a_revision_from_another_schema_does_not_validate_a_cached_plan() {
        const QUERY: &str = "SELECT * FROM analytics.events";

        let cache = PlanCache::new(2);
        let cached = plan_over(table_in_schema("analytics"), "table-id", 7);
        cache.insert(QUERY.to_string(), cached.clone());

        let hit = cache
            .get(
                QUERY,
                &transaction_over(table_in_schema("analytics"), "table-id", 7),
            )
            .unwrap();
        let other_schema = cache.get(QUERY, &transaction_over(table(), "table-id", 7));

        assert!(Arc::ptr_eq(&hit, &cached));
        assert!(other_schema.is_none());
    }

    #[test]
    fn inserting_a_new_revision_replaces_the_previous_plan() {
        let cache = PlanCache::new(2);
        let first = plan("table-id", 7);
        let second = plan("table-id", 8);
        cache.insert("SELECT * FROM events".to_string(), first);
        cache.insert("SELECT * FROM events".to_string(), second.clone());

        let second_hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 8))
            .unwrap();

        assert!(Arc::ptr_eq(&second_hit, &second));
        assert_eq!(cache.inner.lock().unwrap().len(), 1);
    }

    #[test]
    fn inserting_over_capacity_evicts_the_least_recently_used_query() {
        let cache = PlanCache::new(2);
        let first = plan("first", 1);
        let second = plan("second", 1);
        let third = plan("third", 1);
        cache.insert("first query".to_string(), first.clone());
        cache.insert("second query".to_string(), second);
        cache.get("first query", &transaction("first", 1)).unwrap();

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
