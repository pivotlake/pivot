//! [`Window`] — a `row_number()` window, appended as a new `Int64` column.

use std::fmt;

use dispatch::{OrderBy as DispatchOrderBy, RecordBatchOperatorSpec};

use crate::compile::Error;

/// `row_number() OVER (PARTITION BY … ORDER BY …)`: numbers rows from 1 within
/// each partition (in the order-key order), appended as one `Int64` column
/// after all the input columns.
#[derive(Debug)]
pub struct Window {
    /// Partition-key column indices, into the input's output.
    pub partition_keys: Vec<usize>,
    /// Order keys as `(column index, descending)`.
    pub order_keys: Vec<(usize, bool)>,
}

impl fmt::Display for Window {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Window(row_number, partition: {:?}, order: {:?})",
            self.partition_keys, self.order_keys
        )
    }
}

impl Window {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Sort by the partition keys (ascending) then the order keys, so equal
        // partitions are contiguous and ordered within; the dispatch operator
        // numbers each partition run.
        let mut order_by: Vec<DispatchOrderBy> = self
            .partition_keys
            .iter()
            .map(|&col| DispatchOrderBy::new(col, false, false))
            .collect();
        order_by.extend(
            self.order_keys
                .iter()
                .map(|&(col, descending)| DispatchOrderBy::new(col, descending, false)),
        );
        Ok(input.window_row_number(order_by, self.partition_keys.len()))
    }
}
