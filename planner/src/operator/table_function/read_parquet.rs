//! Planner-owned definition of `read_parquet(path)`. The planner owns its SQL
//! signature and behavior as a table function; the current catalog transaction
//! supplies the storage-specific external parquet binding.

use super::{TableFunction, TableFunctionSignature, invalid_argument};
use crate::catalog::{BoundTable, CatalogTransaction, Result as CatalogResult};
use crate::types::Type;
use duckdb_planner::ScalarValue;

pub(super) struct ReadParquetTableFunction;

impl TableFunction for ReadParquetTableFunction {
    fn name(&self) -> &str {
        "read_parquet"
    }

    fn signatures(&self) -> Vec<TableFunctionSignature> {
        vec![TableFunctionSignature {
            arguments: vec![Type::Utf8],
        }]
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    fn bind(
        &self,
        arguments: &[ScalarValue],
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<Box<dyn BoundTable>> {
        let [ScalarValue::Utf8(location)] = arguments else {
            return Err(crate::catalog::Error::Other(Box::new(invalid_argument(
                self.name(),
                "expected one VARCHAR path".to_string(),
            ))));
        };
        transaction.bind_read_parquet(location)
    }
}
