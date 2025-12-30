use crate::table::{RowGroupMetadataHandle, Table};
use crossbeam_deque::{Injector, Steal};
use std::sync::Arc;

/// A consumable source representing a particular logical table.
pub struct TableSource {
    row_groups: Injector<RowGroupMetadataHandle>,
}

impl From<&Arc<Table>> for TableSource {
    fn from(table: &Arc<Table>) -> Self {
        let injector = Injector::new();
        for row_group_idx in 0..table.row_groups.len() {
            injector.push(RowGroupMetadataHandle {
                table: table.clone(),
                row_group_index: row_group_idx,
            })
        }
        TableSource {
            row_groups: injector,
        }
    }
}

impl TableSource {
    pub fn is_empty(&self) -> bool {
        self.row_groups.is_empty()
    }

    pub fn pop_row_group(&self) -> Option<RowGroupMetadataHandle> {
        loop {
            return match self.row_groups.steal() {
                Steal::Empty => None,
                Steal::Success(s) => Some(s),
                Steal::Retry => {
                    continue;
                }
            };
        }
    }
}
