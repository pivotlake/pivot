//! Constructs Pivot [`Operator`](crate::operator::Operator)s by walking DuckDB's
//! borrowed logical-operator handles. Each operator's `from_handle` is implemented
//! here (the walk in [`super`] dispatches to them), so the `crate::operator`
//! modules hold only Pivot IR (the operator types, their `Display`, and their
//! `compile`). Expression construction is the sibling
//! [`expression`](super::expression).

use std::any::Any;

use duckdb_planner::DuckDBTable;
use duckdb_planner::catalog_provider::OptionalTableWrapper;
use duckdb_planner::duckdb_bridge::duckdb_types::LimitNodeType;
use duckdb_planner::handle::{
    Aggregate as AggregateView, CreateSecret as CreateSecretView, CreateTable as CreateTableView,
    DropSecret as DropSecretView, Filter as FilterView, Limit as LimitView, OrderBy as OrderByView,
    OrderKey, Projection as ProjectionView, Reset as ResetView, SecretPersistMode, Set as SetView,
    TableFunctionScan as TableFunctionScanView, TableScan as TableScanView, TopN as TopNView,
};

use super::{BuildCtx, build_scan_columns};
use crate::catalog::{
    Column, CreateSecretRequest, CreateTableRequest, DropSecretRequest, DuckDBTableAdapter, Table,
};
use crate::expression::{Error as ExpressionError, Expression};
use crate::operator::{
    Aggregate, CreateSecret, CreateTable, DropSecret, Error as OperatorError, Filter, Input, Limit,
    OrderBy, OrderByNode, Projection, SetVariable, TableFunctionScan, TopN,
};
use crate::types::type_from_logical;

impl Projection {
    pub(crate) fn from_handle(view: ProjectionView<'_>) -> Result<Projection, OperatorError> {
        Ok(Projection {
            projections: view
                .exprs()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Filter {
    pub(crate) fn from_handle(view: FilterView<'_>) -> Result<Filter, OperatorError> {
        Ok(Filter {
            conditions: view
                .exprs()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Aggregate {
    pub(crate) fn from_handle(view: AggregateView<'_>) -> Result<Aggregate, OperatorError> {
        Ok(Aggregate {
            output_limit: None,
            groups: view
                .groups()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: view
                .expressions()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl OrderBy {
    pub(crate) fn from_handle(view: OrderByView<'_>) -> Result<OrderBy, OperatorError> {
        Ok(OrderBy {
            order_bys: build_orders(view.keys())?,
        })
    }
}

impl TopN {
    pub(crate) fn from_handle(
        view: TopNView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<TopN, OperatorError> {
        let produces_dynamic_filter = match view.dynamic_filter() {
            Some(df) => Some(ctx.dynamic_filter(df)?),
            None => None,
        };
        Ok(TopN {
            order_bys: build_orders(view.keys())?,
            limit: view.limit(),
            offset: view.offset(),
            produces_dynamic_filter,
        })
    }
}

impl Limit {
    pub(crate) fn from_handle(view: LimitView<'_>) -> Result<Limit, OperatorError> {
        Ok(Limit {
            limit: limit_bound(view.value_kind(), || view.value(), "value")?,
            offset: limit_bound(view.offset_kind(), || view.offset(), "offset")?.unwrap_or(0),
        })
    }
}

impl Input {
    pub(crate) fn from_handle(
        scan: TableScanView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<Input, OperatorError> {
        Ok(Input {
            table: resolve_table(*scan.take_table()),
            columns: build_scan_columns(scan.output_columns())?,
            dynamic_filters: scan
                .dynamic_filters()
                .map(|df| ctx.dynamic_filter(df))
                .collect::<Result<Vec<_>, _>>()?,
            emit_row_group_metadata: false,
        })
    }
}

impl TableFunctionScan {
    pub(crate) fn from_handle(
        view: TableFunctionScanView<'_>,
    ) -> Result<TableFunctionScan, OperatorError> {
        // Only positional parameters are supported; reject named parameters rather
        // than silently drop a bound argument.
        if view.has_named_params() {
            return Err(OperatorError::Unsupported(format!(
                "table function {} with named parameters is not supported",
                view.function_name()
            )));
        }
        Ok(TableFunctionScan::new(
            view.function_name(),
            view.params().collect(),
            build_scan_columns(view.output_columns())?,
        ))
    }
}

impl CreateTable {
    pub(crate) fn from_handle(view: CreateTableView<'_>) -> Result<CreateTable, OperatorError> {
        Ok(CreateTable {
            request: CreateTableRequest {
                name: view.name(),
                columns: view
                    .columns()
                    .map(|(name, col_type)| {
                        Ok(Column {
                            name,
                            col_type: type_from_logical(col_type)?,
                        })
                    })
                    .collect::<Result<Vec<_>, OperatorError>>()?,
                options: view.options().collect(),
                if_not_exists: view.if_not_exists(),
            },
            or_replace: view.or_replace(),
            temporary: view.temporary(),
            has_query: view.has_query(),
            constraint_count: view.constraint_count(),
        })
    }
}

impl CreateSecret {
    pub(crate) fn from_handle(view: CreateSecretView<'_>) -> Result<CreateSecret, OperatorError> {
        if let Some(storage) = view.storage() {
            return Err(OperatorError::Unsupported(format!(
                "CREATE SECRET IN {storage} is not supported: pivot stores secrets in its catalog"
            )));
        }
        let secret_type = view.secret_type();
        // An unnamed secret gets the type's default name, as in DuckDB.
        let name = match view.name() {
            name if name.is_empty() => format!("__default_{secret_type}"),
            name => name,
        };
        // The binder leaves the provider empty when the statement omitted it;
        // the registered types all default to explicit config values.
        let provider = match view.provider() {
            provider if provider.is_empty() => "config".to_string(),
            provider => provider,
        };
        let scope = match view.scope().collect::<Vec<String>>() {
            scope if scope.is_empty() => default_secret_scope(&secret_type)?,
            scope => scope,
        };
        Ok(CreateSecret {
            request: CreateSecretRequest {
                name,
                secret_type,
                provider,
                scope,
                options: view
                    .options()
                    .map(|(key, value)| match value {
                        Some(value) => Ok((key, value)),
                        None => Err(OperatorError::Unsupported(format!(
                            "secret option `{key}` cannot be NULL"
                        ))),
                    })
                    .collect::<Result<_, _>>()?,
                // Pivot's catalog is a durable shared database, so an
                // unqualified CREATE SECRET persists (DuckDB, embedded,
                // defaults to temporary instead).
                temporary: view.persist_mode() == SecretPersistMode::Temporary,
                or_replace: view.or_replace(),
                if_not_exists: view.if_not_exists(),
            },
        })
    }
}

/// The scope a secret of `secret_type` applies to when the statement gave
/// none, matching DuckDB's per-type defaults. Lives beside the other
/// per-statement defaults (name, provider) so every secret request leaves the
/// planner fully resolved. A type registered with the binder but missing here
/// is rejected rather than stored with a scope that matches nothing.
fn default_secret_scope(secret_type: &str) -> Result<Vec<String>, OperatorError> {
    match secret_type {
        "s3" => Ok(vec![
            "s3://".to_string(),
            "s3n://".to_string(),
            "s3a://".to_string(),
        ]),
        other => Err(OperatorError::Unsupported(format!(
            "secret type {other:?} has no default scope; give the secret an explicit SCOPE"
        ))),
    }
}

impl DropSecret {
    pub(crate) fn from_handle(view: DropSecretView<'_>) -> Result<DropSecret, OperatorError> {
        if let Some(storage) = view.storage() {
            return Err(OperatorError::Unsupported(format!(
                "DROP SECRET FROM {storage} is not supported: pivot stores secrets in its catalog"
            )));
        }
        Ok(DropSecret {
            request: DropSecretRequest {
                // DuckDB's parser lowercases a CREATE SECRET name but passes a
                // DROP SECRET name through verbatim (its own catalog set is
                // case-insensitive); pivot's registry matches exactly, so
                // mirror the CREATE-side lowering here.
                name: view.name().to_ascii_lowercase(),
                if_exists: view.if_exists(),
                temporary: match view.persist_mode() {
                    SecretPersistMode::Default => None,
                    SecretPersistMode::Temporary => Some(true),
                    SecretPersistMode::Persistent => Some(false),
                },
            },
        })
    }
}

impl SetVariable {
    pub(crate) fn from_set(view: SetView<'_>) -> SetVariable {
        SetVariable {
            name: view.name(),
            value: Some(view.value()),
        }
    }

    /// `RESET name` is modelled as a `SET` with no value (the consumer reads "no
    /// value" as "off / default").
    pub(crate) fn from_reset(view: ResetView<'_>) -> SetVariable {
        SetVariable {
            name: view.name(),
            value: None,
        }
    }
}

/// Lower a sequence of handle sort keys into Pivot [`OrderByNode`]s. Shared by the
/// `OrderBy` and `TopN` constructors.
fn build_orders<'a>(
    keys: impl Iterator<Item = OrderKey<'a>>,
) -> Result<Vec<OrderByNode>, ExpressionError> {
    keys.map(|key| {
        Ok(OrderByNode {
            direction: key.direction.into(),
            expression: Expression::from_handle(key.expression)?,
        })
    })
    .collect()
}

/// Resolve a scan's catalog entry into the Pivot [`Table`] it wraps: the bound
/// DuckDB table is a [`DuckDBTableAdapter`] holding the `Box<dyn Table>`.
fn resolve_table(wrapper: OptionalTableWrapper) -> Box<dyn Table> {
    let duck: Box<dyn DuckDBTable> = wrapper.table.expect("planner returned an unbound table");
    let any: Box<dyn Any> = duck;
    let adapter: Box<DuckDBTableAdapter> = any
        .downcast::<DuckDBTableAdapter>()
        .expect("scan table should be a DuckDBTableAdapter");
    adapter.table
}

/// Interpret a `LIMIT`/`OFFSET` bound's [`LimitNodeType`] into an `Option<usize>`:
/// an unset bound is unbounded (`None`), a constant is the row count, and a
/// percentage/expression bound is rejected (pivot only handles a fixed row count).
///
/// `value` is read lazily: DuckDB's constant accessor throws for a non-constant
/// bound, so it must only be called once the kind is known to be a constant.
fn limit_bound(
    kind: LimitNodeType,
    value: impl FnOnce() -> usize,
    what: &str,
) -> Result<Option<usize>, OperatorError> {
    match kind {
        LimitNodeType::UNSET => Ok(None),
        LimitNodeType::CONSTANT_VALUE => Ok(Some(value())),
        _ => Err(OperatorError::Unsupported(format!(
            "Unsupported non-constant LIMIT {what}"
        ))),
    }
}
