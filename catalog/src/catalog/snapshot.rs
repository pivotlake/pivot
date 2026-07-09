//! An immutable, point-in-time view of the whole catalog: every table with all
//! its files' row groups materialized. Built by the background refresher
//! ([`ParquetCatalog::refresh_snapshot`](super::ParquetCatalog::refresh_snapshot))
//! and published whole. A statement pins one for its lifetime through its
//! [`QueryContext`](planner::catalog::QueryContext), so the refresher can
//! publish a newer snapshot without disturbing a query already reading an older
//! one — the older snapshot lives exactly as long as some transaction still
//! holds its `Arc`.

use std::collections::HashMap;
use std::sync::Arc;

use super::table::CatalogTable;

/// An immutable snapshot of all tables at one refresh tick. Shared by `Arc`;
/// each refresh builds a new one, reusing the `Arc<CatalogTable>` of every table
/// whose manifest did not advance, so an unchanged table costs no extra memory
/// or footer I/O from one tick to the next.
pub struct CatalogSnapshot {
    tables: HashMap<String, Arc<CatalogTable>>,
}

impl CatalogSnapshot {
    /// Assemble a snapshot from a fully-loaded table set.
    pub(super) fn new(tables: HashMap<String, Arc<CatalogTable>>) -> Self {
        Self { tables }
    }

    /// The empty starting snapshot, before the first refresh — and the entire
    /// content of an ephemeral database with no tables yet.
    pub(super) fn empty() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    /// The fully-loaded table `name` as of this refresh, or `None` if it did not
    /// exist yet (created since the last refresh) or has been dropped.
    pub(super) fn table(&self, name: &str) -> Option<&Arc<CatalogTable>> {
        self.tables.get(name)
    }
}
