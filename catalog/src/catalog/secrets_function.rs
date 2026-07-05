//! The `pivot_secrets()` table function: one row per registered secret, with
//! sensitive option values redacted - pivot's `duckdb_secrets()`.
//!
//! It lives here, not in the planner, because secrets are catalog state. There
//! is deliberately no unredacted mode: the clear-text values never leave the
//! registry and the object store.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, StringViewArray};
use arrow_schema::{Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use planner::ScalarValue;
use planner::catalog::{Column, QueryContext};
use planner::compile::Error;
use planner::types::{Type, physical_arrow_type};
use planner::{TableFunction, TableFunctionSignature};

use crate::secrets::SecretsRegistry;

/// The output columns, in declared order - the single source of truth for both
/// the binding signature and the emitted batch schema.
const COLUMNS: [(&str, Type); 6] = [
    ("name", Type::Utf8),
    ("type", Type::Utf8),
    ("provider", Type::Utf8),
    ("persistent", Type::Boolean),
    ("scope", Type::Utf8),
    ("secret_string", Type::Utf8),
];

pub(super) struct SecretsTableFunction {
    pub(super) secrets: Arc<SecretsRegistry>,
}

impl TableFunction for SecretsTableFunction {
    fn name(&self) -> &str {
        "pivot_secrets"
    }

    fn signature(&self) -> TableFunctionSignature {
        TableFunctionSignature {
            arguments: vec![],
            columns: COLUMNS
                .iter()
                .map(|(name, col_type)| Column {
                    name: name.to_string(),
                    col_type: col_type.clone(),
                })
                .collect(),
        }
    }

    fn compile(
        &self,
        args: &[ScalarValue],
        dispatcher: &DataFlowDispatcher,
        _ctx: &dyn QueryContext,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if !args.is_empty() {
            return Err(Error::InvalidTableFunctionArgument {
                function: "pivot_secrets".to_string(),
                message: format!("expected no arguments, got {}", args.len()),
            });
        }

        let entries = self.secrets.list();
        fn string_column<S: AsRef<str>>(values: impl Iterator<Item = S>) -> ArrayRef {
            Arc::new(StringViewArray::from_iter_values(values))
        }
        let columns: Vec<ArrayRef> = vec![
            string_column(entries.iter().map(|e| e.secret.name.as_str())),
            string_column(entries.iter().map(|e| e.secret.secret_type.as_str())),
            string_column(entries.iter().map(|e| e.secret.provider.as_str())),
            Arc::new(BooleanArray::from(
                entries.iter().map(|e| !e.temporary).collect::<Vec<_>>(),
            )),
            string_column(entries.iter().map(|e| e.secret.scope.join(";"))),
            string_column(entries.iter().map(|e| e.secret.redacted_options())),
        ];
        let schema = Arc::new(Schema::new(
            COLUMNS
                .iter()
                .map(|(name, col_type)| Field::new(*name, physical_arrow_type(col_type), false))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema, columns)
            .expect("secret columns are equal-length single arrays");
        Ok(dispatch::values_input(dispatcher, [batch]).record_batches())
    }
}
