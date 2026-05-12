//! The [`PlanNode`] tree that represents a deserialized DuckDB logical plan.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use crate::catalog_provider::DuckDBTable;
use crate::dynamic_filter::DynamicFilterSlot;
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

    /// Walk the plan tree, attaching external state to each operator:
    /// * each `RawInput` is upgraded to an [`Input`](crate::operator::Input)
    ///   by moving the `Box<dyn DuckDBTable>` at index `table_id` out of
    ///   `tables` (each table id is consumed exactly once).
    /// * every [`DynamicFilter`](crate::dynamic_filter::DynamicFilter) on an
    ///   Input or TopN is bound to a shared [`DynamicFilterSlot`] keyed by
    ///   `slot_id`. Slots are allocated lazily on first encounter, so the
    ///   same id seen on a producer and on N consumer scans all resolve to
    ///   the same `Arc`.
    pub(crate) fn resolve_inputs(self, tables: Vec<Box<dyn DuckDBTable>>) -> Self {
        let mut table_slots: Vec<Option<Box<dyn DuckDBTable>>> =
            tables.into_iter().map(Some).collect();
        let mut dynamic_filter_slots: HashMap<usize, Arc<DynamicFilterSlot>> = HashMap::new();
        self.resolve_inputs_walker(&mut table_slots, &mut dynamic_filter_slots)
    }

    fn resolve_inputs_walker(
        mut self,
        tables: &mut [Option<Box<dyn DuckDBTable>>],
        dynamic_filter_slots: &mut HashMap<usize, Arc<DynamicFilterSlot>>,
    ) -> Self {
        self.inputs = self
            .inputs
            .into_iter()
            .map(|n| n.resolve_inputs_walker(tables, dynamic_filter_slots))
            .collect();

        // Bind in-place dynamic-filter references on the operator before any
        // RawInput → Input upgrade so the same code path handles both sides.
        match &mut self.operator {
            Operator::TopN(t) => {
                if let Some(df) = &mut t.produces_dynamic_filter {
                    let slot = dynamic_filter_slots
                        .entry(df.slot_id)
                        .or_insert_with(|| Arc::new(RwLock::new(None)));
                    df.slot = Some(Arc::clone(slot));
                }
            }
            Operator::RawInput(raw) => {
                for df in &mut raw.dynamic_filters {
                    let slot = dynamic_filter_slots
                        .entry(df.slot_id)
                        .or_insert_with(|| Arc::new(RwLock::new(None)));
                    df.slot = Some(Arc::clone(slot));
                }
            }
            _ => {}
        }

        let Operator::RawInput(raw) = self.operator else {
            return self;
        };

        PlanNode {
            name: self.name,
            inputs: self.inputs,
            operator: Operator::Input(crate::operator::Input {
                table: tables[raw.table_id]
                    .take()
                    .expect("table id already consumed by another Input"),
                columns: raw.columns,
                dynamic_filters: raw.dynamic_filters,
            }),
        }
    }
}
