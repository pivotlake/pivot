//! Shared JSON shapes and the Arrow-to-JSON renderer used by more than one
//! endpoint (`/api/query` and the row-group pages).

use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::{Array, RecordBatch};
use serde::Serialize;

/// A result column: its name and its Arrow type rendered as text.
#[derive(Serialize)]
pub(super) struct ColumnOut {
    pub(super) name: String,
    pub(super) col_type: String,
}

/// Render result batches (copied out on dispatch workers; see [`crate::query_handler`])
/// as JSON columns + text cells - the universally-safe representation, like the
/// wire protocol's text format. `null` for SQL NULL.
pub(super) fn batches_to_json(
    batches: &[RecordBatch],
) -> (Vec<ColumnOut>, Vec<Vec<Option<String>>>) {
    let Some(first) = batches.first() else {
        return (Vec::new(), Vec::new());
    };
    let columns = first
        .schema()
        .fields()
        .iter()
        .map(|f| ColumnOut {
            name: f.name().clone(),
            col_type: f.data_type().to_string(),
        })
        .collect();

    let opts = FormatOptions::default();
    let mut rows = Vec::new();
    for batch in batches {
        let formatters: Vec<Option<ArrayFormatter>> = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).ok())
            .collect();
        for r in 0..batch.num_rows() {
            let row = (0..batch.num_columns())
                .map(|c| {
                    let arr = batch.column(c);
                    if arr.is_null(r) {
                        None
                    } else {
                        formatters[c].as_ref().map(|f| f.value(r).to_string())
                    }
                })
                .collect();
            rows.push(row);
        }
    }
    (columns, rows)
}
