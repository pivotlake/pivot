//! The columns a table declares, and how each file's columns are matched to
//! them when its footer is read.

use planner::catalog::Column;
use std::sync::Arc;

/// The columns a table declares, paired with the rule that matches a file's
/// columns to them. Handed to every footer read of the table, so each row group
/// comes out reconciled with the declared schema the same way.
#[derive(Clone, Debug)]
pub struct TableColumns {
    columns: Arc<[Column]>,
    resolution: ColumnResolution,
}

/// How a file's columns are matched to the declared ones.
#[derive(Clone, Debug)]
pub enum ColumnResolution {
    /// By name, case-insensitively. A row group keeps its file's own column
    /// order and every column the file holds, declared or not; a declared
    /// column the file lacks is simply not there.
    ByName,
    /// By Parquet field id: `field_ids[i]` identifies declared column `i`, allowing
    /// columns to be tracked across renames. File columns are mapped to the declared
    /// schema by field id, renamed to their declared names, and reordered accordingly.
    /// Unknown file columns are dropped, while missing declared columns are read as
    /// NULL. If the file has no field ids, columns are matched by name instead.
    ByFieldId(Arc<[i32]>),
}

impl TableColumns {
    /// Declared columns matched by name (see [`ColumnResolution::ByName`]).
    pub fn by_name(columns: impl Into<Arc<[Column]>>) -> Self {
        Self {
            columns: columns.into(),
            resolution: ColumnResolution::ByName,
        }
    }

    /// Declared columns matched by field id (see
    /// [`ColumnResolution::ByFieldId`]); `field_ids[i]` is column `i`'s id.
    pub fn by_field_id(
        columns: impl Into<Arc<[Column]>>,
        field_ids: impl Into<Arc<[i32]>>,
    ) -> Self {
        let columns = columns.into();
        let field_ids = field_ids.into();
        assert_eq!(
            columns.len(),
            field_ids.len(),
            "every declared column needs exactly one field id"
        );
        Self {
            columns,
            resolution: ColumnResolution::ByFieldId(field_ids),
        }
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    pub fn resolution(&self) -> &ColumnResolution {
        &self.resolution
    }
}
