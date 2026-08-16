//! PostgreSQL compatibility scalar functions used by `psql` catalog queries.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::builder::StringViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt32Type};
use arrow_array::{Array, BooleanArray, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// PostgreSQL's `pg_get_userbyid(oid)` compatibility function.
#[derive(Debug, Clone)]
pub struct PgGetUserById {
    pub role_oid: Box<Expression>,
}

impl PgGetUserById {
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        visit(&self.role_oid);
    }

    pub fn result_type(&self) -> Type {
        Type::Utf8
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let role_oid_builder = self.role_oid.compile()?;
        Ok(Box::new(move || {
            let mut role_oid = role_oid_builder();
            Box::new(move |batch: &RecordBatch| {
                let role_oids = role_oid(batch).into_array(batch.num_rows());
                let role_oids = role_oids.as_primitive::<UInt32Type>();
                let mut names = StringViewBuilder::with_capacity(batch.num_rows());
                for row in 0..batch.num_rows() {
                    if role_oids.is_null(row)
                        || role_oids.value(row) != crate::pg_catalog::PIVOT_OWNER_OID
                    {
                        names.append_null();
                    } else {
                        names.append_value(crate::pg_catalog::PIVOT_OWNER_NAME);
                    }
                }
                ExprResult::Array(Arc::new(names.finish()))
            }) as ExprEvalFn
        }))
    }
}

impl Display for PgGetUserById {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pg_get_userbyid({})", self.role_oid)
    }
}

/// PostgreSQL's `pg_table_is_visible(oid)` compatibility function.
#[derive(Debug, Clone)]
pub struct PgTableIsVisible {
    pub relation_oid: Box<Expression>,
}

impl PgTableIsVisible {
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        visit(&self.relation_oid);
    }

    pub fn result_type(&self) -> Type {
        Type::Boolean
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let relation_oid_builder = self.relation_oid.compile()?;
        Ok(Box::new(move || {
            let mut relation_oid = relation_oid_builder();
            Box::new(move |batch: &RecordBatch| {
                let relation_ids = relation_oid(batch).into_array(batch.num_rows());
                let relation_ids = relation_ids.as_primitive::<UInt32Type>();
                let visible: BooleanArray = (0..batch.num_rows())
                    .map(|row| {
                        (!relation_ids.is_null(row)).then(|| {
                            crate::pg_catalog::relation_oid_is_visible(relation_ids.value(row))
                        })
                    })
                    .collect();
                ExprResult::Array(Arc::new(visible))
            }) as ExprEvalFn
        }))
    }
}

impl Display for PgTableIsVisible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pg_table_is_visible({})", self.relation_oid)
    }
}

/// PostgreSQL's `format_type(oid, typmod)` compatibility function.
#[derive(Debug, Clone)]
pub struct PgFormatType {
    pub type_oid: Box<Expression>,
    pub type_modifier: Box<Expression>,
}

impl PgFormatType {
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        visit(&self.type_oid);
        visit(&self.type_modifier);
    }

    pub fn result_type(&self) -> Type {
        Type::Utf8
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let type_oid_builder = self.type_oid.compile()?;
        let type_modifier_builder = self.type_modifier.compile()?;
        Ok(Box::new(move || {
            let mut type_oid = type_oid_builder();
            let mut type_modifier = type_modifier_builder();
            Box::new(move |batch: &RecordBatch| {
                let type_ids = type_oid(batch).into_array(batch.num_rows());
                let type_ids = type_ids.as_primitive::<UInt32Type>();
                let modifiers = type_modifier(batch).into_array(batch.num_rows());
                let modifiers = modifiers.as_primitive::<Int64Type>();
                let mut formatted = StringViewBuilder::with_capacity(batch.num_rows());
                for row in 0..batch.num_rows() {
                    if type_ids.is_null(row) {
                        formatted.append_null();
                        continue;
                    }
                    let modifier = if modifiers.is_null(row) {
                        -1
                    } else {
                        modifiers.value(row)
                    };
                    formatted.append_value(crate::pg_catalog::format_type(
                        type_ids.value(row),
                        modifier,
                    ));
                }
                ExprResult::Array(Arc::new(formatted.finish()))
            }) as ExprEvalFn
        }))
    }
}

impl Display for PgFormatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "format_type({}, {})", self.type_oid, self.type_modifier)
    }
}

/// PostgreSQL's `pg_get_expr(pg_node_tree, relation_oid [, pretty])` function.
#[derive(Debug, Clone)]
pub struct PgGetExpr {
    pub expression: Box<Expression>,
    pub relation_oid: Box<Expression>,
    pub pretty: Option<Box<Expression>>,
}

impl PgGetExpr {
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        visit(&self.expression);
        visit(&self.relation_oid);
        if let Some(pretty) = &self.pretty {
            visit(pretty);
        }
    }

    pub fn result_type(&self) -> Type {
        Type::Utf8
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let expression_builder = self.expression.compile()?;
        let relation_oid_builder = self.relation_oid.compile()?;
        let pretty_builder = self
            .pretty
            .as_ref()
            .map(|pretty| pretty.compile())
            .transpose()?;
        Ok(Box::new(move || {
            let mut expression = expression_builder();
            let mut relation_oid = relation_oid_builder();
            let mut pretty = pretty_builder.as_ref().map(|builder| builder());
            Box::new(move |batch: &RecordBatch| {
                let expression = expression(batch).into_array(batch.num_rows());
                let _ = relation_oid(batch).into_array(batch.num_rows());
                if let Some(pretty) = &mut pretty {
                    let _ = pretty(batch).into_array(batch.num_rows());
                }
                ExprResult::Array(expression)
            }) as ExprEvalFn
        }))
    }
}

impl Display for PgGetExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pg_get_expr({}, {}", self.expression, self.relation_oid)?;
        if let Some(pretty) = &self.pretty {
            write!(f, ", {pretty}")?;
        }
        write!(f, ")")
    }
}

/// PostgreSQL's `pg_relation_is_publishable(oid)` compatibility function.
#[derive(Debug, Clone)]
pub struct PgRelationIsPublishable {
    pub relation_oid: Box<Expression>,
}

impl PgRelationIsPublishable {
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        visit(&self.relation_oid);
    }

    pub fn result_type(&self) -> Type {
        Type::Boolean
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let relation_oid_builder = self.relation_oid.compile()?;
        Ok(Box::new(move || {
            let mut relation_oid = relation_oid_builder();
            Box::new(move |batch: &RecordBatch| {
                let relation_ids = relation_oid(batch).into_array(batch.num_rows());
                let relation_ids = relation_ids.as_primitive::<UInt32Type>();
                let publishable: BooleanArray = (0..batch.num_rows())
                    .map(|row| (!relation_ids.is_null(row)).then_some(false))
                    .collect();
                ExprResult::Array(Arc::new(publishable))
            }) as ExprEvalFn
        }))
    }
}

impl Display for PgRelationIsPublishable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pg_relation_is_publishable({})", self.relation_oid)
    }
}

/// PostgreSQL's `pg_get_statisticsobjdef_columns(oid)` compatibility function.
#[derive(Debug, Clone)]
pub struct PgGetStatisticsObjDefColumns {
    pub statistics_oid: Box<Expression>,
}

impl PgGetStatisticsObjDefColumns {
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        visit(&self.statistics_oid);
    }

    pub fn result_type(&self) -> Type {
        Type::Utf8
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let statistics_oid_builder = self.statistics_oid.compile()?;
        Ok(Box::new(move || {
            let mut statistics_oid = statistics_oid_builder();
            Box::new(move |batch: &RecordBatch| {
                let _ = statistics_oid(batch).into_array(batch.num_rows());
                let mut columns = StringViewBuilder::with_capacity(batch.num_rows());
                for _ in 0..batch.num_rows() {
                    columns.append_null();
                }
                ExprResult::Array(Arc::new(columns.finish()))
            }) as ExprEvalFn
        }))
    }
}

impl Display for PgGetStatisticsObjDefColumns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pg_get_statisticsobjdef_columns({})",
            self.statistics_oid
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::pg_catalog::{PIVOT_OWNER_OID, VIRTUAL_OID_PREFIX, mark_relation_oid_visible};
    use crate::test_support::*;
    use arrow_array::Array;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    fn pg_get_userbyid_compiles_and_returns_the_pivot_owner(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            &format!("SELECT pg_catalog.pg_get_userbyid({PIVOT_OWNER_OID}::UINTEGER)"),
        );

        assert_eq!(only_column(&rows[0]), &json!("pivot"));
    }

    #[rstest]
    fn pg_get_userbyid_does_not_call_an_unknown_oid_pivot(mut testing_planner: TestingPlanner) {
        let batches = run_batches(
            &mut testing_planner,
            "SELECT pg_catalog.pg_get_userbyid(1::UINTEGER)",
        );

        assert!(batches[0].column(0).is_null(0));
    }

    #[rstest]
    fn pg_table_is_visible_compiles_and_reads_the_oid_flag(mut testing_planner: TestingPlanner) {
        let visible = mark_relation_oid_visible(VIRTUAL_OID_PREFIX | 1);
        let rows = run(
            &mut testing_planner,
            &format!("SELECT pg_catalog.pg_table_is_visible({visible}::UINTEGER)"),
        );

        assert_eq!(only_column(&rows[0]), &json!(true));
    }

    #[rstest]
    fn format_type_compiles_and_formats_the_postgres_type_oid(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT pg_catalog.format_type(20::UINTEGER, -1::BIGINT)",
        );

        assert_eq!(only_column(&rows[0]), &json!("bigint"));
    }

    #[rstest]
    fn pg_get_expr_compiles_and_returns_the_stored_expression(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT pg_catalog.pg_get_expr('id > 0', 1::UINTEGER, true)",
        );

        assert_eq!(only_column(&rows[0]), &json!("id > 0"));
    }

    #[rstest]
    fn pg_relation_is_publishable_compiles_and_returns_false(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT pg_catalog.pg_relation_is_publishable(1::UINTEGER)",
        );

        assert_eq!(only_column(&rows[0]), &json!(false));
    }

    #[rstest]
    fn pg_get_statisticsobjdef_columns_compiles_and_returns_null(
        mut testing_planner: TestingPlanner,
    ) {
        let batches = run_batches(
            &mut testing_planner,
            "SELECT pg_catalog.pg_get_statisticsobjdef_columns(1::UINTEGER)",
        );

        assert_eq!(batches[0].num_columns(), 1);
        assert!(batches[0].column(0).is_null(0));
    }
}
