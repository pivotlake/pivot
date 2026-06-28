//! The [`PlanNode`] tree that represents a DuckDB logical plan.

use std::fmt;

use crate::catalog_provider::DuckDBTable;
use crate::operator::Operator;

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
#[derive(Debug)]
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

    /// Walk the plan tree and upgrade each `RawInput`/`RawMaterialize` to its
    /// resolved form by attaching the `Box<dyn DuckDBTable>` at index `table_id`.
    ///
    /// A `table_id` can be referenced more than once — a late-materialized query
    /// shares one between its narrow scan and its `Materialize` — so each
    /// reference gets a [`clone_box`](DuckDBTable::clone_box) of the resolved
    /// table rather than the single bound instance.
    ///
    /// Dynamic-filter references are left as-is here — they carry only a
    /// `slot_id` and are bound to a shared slot later, at compile time.
    pub(crate) fn resolve_inputs(self, tables: Vec<Box<dyn DuckDBTable>>) -> Self {
        self.resolve_inputs_walker(&tables)
    }

    fn resolve_inputs_walker(mut self, tables: &[Box<dyn DuckDBTable>]) -> Self {
        self.inputs = self
            .inputs
            .into_iter()
            .map(|n| n.resolve_inputs_walker(tables))
            .collect();

        let operator = match self.operator {
            Operator::RawInput(raw) => Operator::Input(crate::operator::Input {
                table: tables[raw.table_id].clone_box(),
                columns: raw.columns,
                dynamic_filters: raw.dynamic_filters,
                emit_row_group_metadata: raw.emit_row_group_metadata,
            }),
            Operator::RawMaterialize(raw) => Operator::Materialize(crate::operator::Materialize {
                table: tables[raw.table_id].clone_box(),
                columns: raw.columns,
            }),
            other => other,
        };

        PlanNode {
            name: self.name,
            inputs: self.inputs,
            operator,
        }
    }
}
