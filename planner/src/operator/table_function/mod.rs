//! Globally registered, Pivot-executed table functions.
//!
//! Each [`TableFunction`] publishes every positional overload DuckDB may bind.
//! Binding one invocation produces a regular [`BoundTable`] that carries its
//! argument-dependent schema and execution state. Fixed-schema generators and
//! dynamic external-file readers therefore follow the same planning path.

mod read_parquet;
mod series;

use crate::catalog::{BoundTable, CatalogTransaction};
use crate::compile::{DynamicFilterSlots, Error};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::Expression;
use crate::operator::input::{build_dynamic_scan_predicates, plan_scan_projection};
use crate::types::{Type, logical_from_type};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::ScalarValue;
use duckdb_planner::catalog_provider::TableFunctionDef;
use std::collections::BTreeSet;
use std::fmt;

/// One positional overload exposed to DuckDB's binder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableFunctionSignature {
    pub arguments: Vec<Type>,
}

/// One globally registered table-valued function. Its overloads are registered
/// when the planner context is created; binding receives DuckDB-coerced constant
/// arguments and returns the table-like scan for that invocation.
pub trait TableFunction: Send + Sync {
    fn name(&self) -> &str;

    fn signatures(&self) -> Vec<TableFunctionSignature>;

    /// Whether every invocation of this function can tag a narrow scan with
    /// row-group metadata and fetch deferred columns for its surviving rows.
    /// DuckDB needs this before it binds any particular invocation, so this is
    /// a property of the function rather than the returned [`BoundTable`].
    fn supports_late_materialization(&self) -> bool {
        false
    }

    fn bind(
        &self,
        arguments: &[ScalarValue],
        transaction: &dyn CatalogTransaction,
    ) -> crate::catalog::Result<Box<dyn BoundTable>>;
}

static READ_PARQUET: read_parquet::ReadParquetTableFunction =
    read_parquet::ReadParquetTableFunction;
static RANGE: series::SeriesTableFunction = series::SeriesTableFunction::range();
static GENERATE_SERIES: series::SeriesTableFunction =
    series::SeriesTableFunction::generate_series();

static BUILT_IN_TABLE_FUNCTIONS: [&dyn TableFunction; 3] =
    [&READ_PARQUET, &RANGE, &GENERATE_SERIES];

pub(crate) fn built_in_table_function(name: &str) -> Option<&'static dyn TableFunction> {
    BUILT_IN_TABLE_FUNCTIONS
        .iter()
        .copied()
        .find(|function| function.name().eq_ignore_ascii_case(name))
}

pub(crate) fn built_in_table_function_defs() -> Result<Vec<TableFunctionDef>, String> {
    table_function_defs(BUILT_IN_TABLE_FUNCTIONS)
}

fn table_function_defs<'a>(
    functions: impl IntoIterator<Item = &'a dyn TableFunction>,
) -> Result<Vec<TableFunctionDef>, String> {
    let mut names = BTreeSet::new();
    let mut definitions = Vec::new();
    for function in functions {
        let name = function.name().to_ascii_lowercase();
        if !names.insert(name.clone()) {
            return Err(format!("duplicate table function `{name}`"));
        }
        definitions.extend(function.signatures().into_iter().map(|signature| {
            TableFunctionDef {
                name: name.clone(),
                arg_type_ids: signature
                    .arguments
                    .iter()
                    .map(|argument_type| logical_from_type(argument_type).id as u8)
                    .collect(),
                supports_late_materialization: function.supports_late_materialization(),
            }
        }));
    }
    Ok(definitions)
}

/// One invocation after DuckDB has selected an overload and Pivot has bound its
/// arguments to a table-like scan.
#[derive(Debug)]
pub struct TableFunctionScan {
    function_name: String,
    arguments: Vec<ScalarValue>,
    columns: Vec<Expression>,
    table: Box<dyn BoundTable>,
    dynamic_filters: Vec<DynamicFilter>,
    emit_row_group_metadata: bool,
}

impl TableFunctionScan {
    pub(crate) fn new(
        function_name: String,
        arguments: Vec<ScalarValue>,
        columns: Vec<Expression>,
        table: Box<dyn BoundTable>,
        dynamic_filters: Vec<DynamicFilter>,
    ) -> Self {
        Self {
            function_name,
            arguments,
            columns,
            table,
            dynamic_filters,
            emit_row_group_metadata: false,
        }
    }

    pub(crate) fn output_types(&self) -> Result<Vec<Type>, Error> {
        self.columns.iter().map(Expression::result_type).collect()
    }

    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        slots: &mut DynamicFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let (projection, extract_projection) =
            plan_scan_projection(&self.columns, self.table.as_ref())?;
        let dynamic_filters = build_dynamic_scan_predicates(&self.dynamic_filters, slots);
        let scan = self
            .table
            .compile_scan(
                dispatcher,
                projection,
                dynamic_filters,
                self.emit_row_group_metadata,
            )
            .map_err(Error::TableScan)?;
        match extract_projection {
            Some(extract_projection) => extract_projection.compile(scan),
            None => Ok(scan),
        }
    }

    /// Turn this invocation into the narrow side of late materialization and
    /// return an independent table binding for the deferred-column fetch.
    pub(crate) fn prepare_late_materialization(&mut self) -> Box<dyn BoundTable> {
        self.emit_row_group_metadata = true;
        self.table.clone_box()
    }
}

impl fmt::Display for TableFunctionScan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arguments: Vec<String> = self.arguments.iter().map(ToString::to_string).collect();
        write!(
            formatter,
            "TableFunctionScan({}({}))",
            self.function_name,
            arguments.join(", ")
        )
    }
}

/// Build an [`Error::InvalidTableFunctionArgument`] for one function.
pub(crate) fn invalid_argument(function: &str, message: String) -> Error {
    Error::InvalidTableFunctionArgument {
        function: function.to_string(),
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DuplicateRange;

    impl TableFunction for DuplicateRange {
        fn name(&self) -> &str {
            "RANGE"
        }

        fn signatures(&self) -> Vec<TableFunctionSignature> {
            vec![TableFunctionSignature {
                arguments: vec![Type::Int64],
            }]
        }

        fn bind(
            &self,
            _arguments: &[ScalarValue],
            _transaction: &dyn CatalogTransaction,
        ) -> crate::catalog::Result<Box<dyn BoundTable>> {
            unreachable!("a duplicate function is rejected before binding")
        }
    }

    #[test]
    fn table_function_names_are_unique_case_insensitively() {
        let duplicate = DuplicateRange;
        let functions = BUILT_IN_TABLE_FUNCTIONS
            .into_iter()
            .chain(std::iter::once(&duplicate as &dyn TableFunction));

        let result = table_function_defs(functions);
        assert_eq!(result.err().unwrap(), "duplicate table function `range`");
    }
}
