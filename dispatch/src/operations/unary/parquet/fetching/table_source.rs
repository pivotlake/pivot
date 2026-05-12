//! Work-stealing source that feeds row groups into the fetching pipeline.
//!
//! [`RowGroupInjectorFactory`] pre-loads every row group from a
//! [`ParquetTable`] into a shared [`Injector`] queue. Each worker gets its own
//! [`RowGroupInjector`] (via [`RootChannelFactory`]) that steals row groups on
//! demand, wrapping them in a [`RowGroupRequest`] with the target projection.

use crate::operations::channels::{Receiver, RootChannelFactory};
use crate::operations::unary::parquet::RowGroupRequest;
use crate::operations::unary::parquet::types::metadata::{QueryRowGroupMetadata, RowGroupMetadata};
use crate::operations::unary::parquet::types::projection::Projection;
use crate::operations::unary::parquet::types::table::ParquetTable;
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;

/// A per-row-group predicate evaluated at steal time. Returns `true` to keep
/// the row group, `false` to skip it. Cheaply cloneable so each worker's
/// [`RowGroupInjector`] holds the same `Arc` and the closure is evaluated
/// lazily (e.g. against a dynamic-filter slot whose value isn't published
/// until execution starts).
pub type RowGroupFilter = Arc<dyn Fn(&RowGroupMetadata) -> bool + Send + Sync>;

/// Factory that populates a shared [`Injector`] with every row group in a
/// table and produces [`RowGroupInjector`] receivers for each worker.
#[derive(Clone)]
pub struct RowGroupInjectorFactory {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl RowGroupInjectorFactory {
    /// Creates a new factory, pushing all row groups from `table` into the
    /// shared work-stealing queue. With `filter == Some(_)`, each row group
    /// is offered to the closure when stolen and skipped if it returns
    /// `false`.
    pub fn new(
        table: &Arc<ParquetTable>,
        projection: Projection,
        filter: Option<RowGroupFilter>,
    ) -> Self {
        let injector = Arc::new(Injector::new());
        for row_group_idx in 0..table.row_groups.len() {
            injector.push(QueryRowGroupMetadata::new(table, row_group_idx, None));
        }
        Self {
            row_groups: injector,
            projection,
            filter,
        }
    }
}

impl RootChannelFactory<RowGroupRequest> for RowGroupInjectorFactory {
    type Receiver = RowGroupInjector;

    fn build(self) -> Self::Receiver {
        RowGroupInjector {
            row_groups: self.row_groups,
            projection: self.projection,
            filter: self.filter,
        }
    }
}

/// A [`Receiver`] that steals row groups from the shared [`Injector`] queue.
///
/// Work is only consumed through [`steal`](Self::steal); `try_recv` always
/// returns `None`.
pub struct RowGroupInjector {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    projection: Projection,
    filter: Option<RowGroupFilter>,
}

impl Receiver<RowGroupRequest> for RowGroupInjector {
    fn is_empty(&self) -> bool {
        self.row_groups.is_empty()
    }

    fn try_recv(&self) -> Option<RowGroupRequest> {
        None
    }

    fn steal(&self) -> Option<RowGroupRequest> {
        loop {
            match self.row_groups.steal() {
                Steal::Empty => return None,
                Steal::Success(s) => {
                    if let Some(filter) = &self.filter
                        && !filter(s.get_metadata())
                    {
                        // Row group rejected by the predicate — try the next.
                        continue;
                    }
                    return Some(RowGroupRequest::from(s, &self.projection));
                }
                Steal::Retry => continue,
            }
        }
    }
}
