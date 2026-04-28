//! The [`PlanNode`] tree that represents a deserialized DuckDB logical plan.

use std::fmt;
use std::sync::Arc;

use crate::catalog_provider::DuckDBTable;
use crate::operator::Operator;
use custom_deserializer::CustomDeserializer;

/// A single node in the logical plan tree.
///
/// Each node holds an [`Operator`] (the relational operation it performs) and
/// zero or more child `inputs` that feed rows into it.
///
/// The plan forms a **tree, not a DAG**: every operator outputs to exactly one
/// parent. An operator may *read from* multiple children (e.g. a JOIN has two
/// `inputs`), but its output is always consumed by a single parent operator.
///
/// For example, `SELECT name FROM users WHERE age > 30` produces:
///
/// ```text
/// Projection(#0)           // outputs to the query result (root)
///   Filter(#1 > 30)        // outputs to Projection
///     Input(users, [name, age])  // outputs to Filter
/// ```
///
/// Each node above feeds its rows to exactly one parent — `Input` cannot
/// simultaneously feed both `Filter` and some other operator.
#[derive(CustomDeserializer, Debug)]
pub struct PlanNode {
    pub name: String,
    pub inputs: Vec<PlanNode>,
    pub operator: Operator,
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indented(f, 0)
    }
}

impl PlanNode {
    fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let prefix = "  ".repeat(indent);
        writeln!(f, "{}{}", prefix, self.operator)?;
        for child in &self.inputs {
            child.fmt_indented(f, indent + 1)?;
        }
        Ok(())
    }

    /// Walk the plan tree and turn each `RawInput` into a resolved
    /// [`Input`](crate::operator::Input) by looking up the `table_id` in the
    /// `tables` vector.
    pub(crate) fn resolve_inputs(mut self, tables: &[Arc<dyn DuckDBTable>]) -> Self {
        self.inputs = self
            .inputs
            .into_iter()
            .map(|n| n.resolve_inputs(tables))
            .collect();

        let Operator::RawInput(raw) = self.operator else {
            return self;
        };

        PlanNode {
            name: self.name,
            inputs: self.inputs,
            operator: Operator::Input(crate::operator::Input {
                table: tables[raw.table_id].clone(),
                columns: raw.columns,
            }),
        }
    }
}
