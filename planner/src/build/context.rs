//! State shared by every node built during one walk of a DuckDB plan.

use std::collections::HashMap;

use duckdb_planner::handle::DynamicFilterRef;

use crate::dynamic_filter::DynamicFilter;
use crate::expression::Error as ExpressionError;
use crate::operator::{CteScan, Error as OperatorError};
use crate::types::Type;

/// Synthetic CTEs live far above DuckDB's table-index space so their indices
/// cannot collide with indices carried by real CTEs in the DuckDB plan.
const SYNTHETIC_CTE_BASE: usize = usize::MAX / 2;

/// Per-walk state used to connect plan nodes that refer to the same logical
/// resource, such as a dynamic-filter slot or CTE definition.
#[derive(Default)]
pub(super) struct BuildCtx {
    /// Dense dynamic-filter slot IDs, keyed by the pointer identity of DuckDB's
    /// shared `DynamicFilterData` cell. Shares the ID space with the
    /// join-produced filter slots.
    dynamic_filter_slots: HashMap<usize, usize>,
    /// How many join-produced filter slots have been allocated. Those slots
    /// need no keyed lookup (each is wired to its producer and consumer at one
    /// site), only IDs disjoint from `dynamic_filter_slots`.
    join_filter_slot_count: usize,
    /// How many key-bitset slots have been allocated. Their own id space:
    /// key-bitset slots live in a separate compile-time registry.
    key_bitset_slot_count: usize,
    /// The output shape of every CTE definition walked so far, keyed by its CTE
    /// index. A CTE scan has no child from which to obtain this information.
    cte_outputs: HashMap<usize, (Vec<Type>, Vec<bool>)>,
    /// The number of scan sites found for each CTE.
    cte_sites: HashMap<usize, usize>,
    /// The synthetic CTE published by each enclosing delim join, innermost
    /// last. A `DELIM_GET` reads the last entry.
    delim_targets: Vec<usize>,
    /// Number of synthetic CTE indices allocated during this walk.
    synthetic_ctes: usize,
}

impl BuildCtx {
    /// Build a planner dynamic filter and assign the shared DuckDB cell a
    /// stable, dense slot ID.
    pub(super) fn dynamic_filter(
        &mut self,
        df: DynamicFilterRef,
    ) -> Result<DynamicFilter, ExpressionError> {
        let next_slot = self.dynamic_filter_slots.len() + self.join_filter_slot_count;
        let slot_id = *self
            .dynamic_filter_slots
            .entry(df.data_id)
            .or_insert(next_slot);
        Ok(DynamicFilter {
            slot_id,
            column_idx: df.column,
            compare_type: df.comparison.try_into()?,
        })
    }

    /// Allocate a fresh slot ID for one bound of a join-produced filter,
    /// disjoint from the [`dynamic_filter`](Self::dynamic_filter) IDs.
    pub(super) fn allocate_join_filter_slot(&mut self) -> usize {
        let slot_id = self.dynamic_filter_slots.len() + self.join_filter_slot_count;
        self.join_filter_slot_count += 1;
        slot_id
    }

    /// Allocate a fresh key-bitset slot ID, in that registry's own id space.
    pub(super) fn allocate_key_bitset_slot(&mut self) -> usize {
        let slot_id = self.key_bitset_slot_count;
        self.key_bitset_slot_count += 1;
        slot_id
    }

    /// Record the output shape of a real CTE definition before its body is
    /// walked.
    pub(super) fn register_cte_output(
        &mut self,
        cte_index: usize,
        types: Vec<Type>,
        nullability: Vec<bool>,
    ) {
        self.cte_outputs.insert(cte_index, (types, nullability));
    }

    /// Allocate a synthetic CTE index and record its complete output shape.
    ///
    /// Keeping allocation and registration together prevents callers from
    /// creating an index that a later `CteScan` cannot resolve.
    pub(super) fn create_synthetic_cte(
        &mut self,
        types: Vec<Type>,
        nullability: Vec<bool>,
    ) -> usize {
        let cte_index = SYNTHETIC_CTE_BASE + self.synthetic_ctes;
        self.synthetic_ctes += 1;
        self.register_cte_output(cte_index, types, nullability);
        cte_index
    }

    /// Create one scan of a registered CTE and count the new read site.
    pub(super) fn create_cte_scan(&mut self, cte_index: usize) -> Result<CteScan, OperatorError> {
        let Some((types, nullable)) = self.cte_outputs.get(&cte_index) else {
            return Err(OperatorError::Unsupported(format!(
                "CTE #{cte_index} is read outside the plan that defines it"
            )));
        };
        *self.cte_sites.entry(cte_index).or_default() += 1;
        Ok(CteScan {
            cte_index,
            types: types.clone(),
            nullable: nullable.clone(),
        })
    }

    /// Finish walking a CTE body and return how many scans of it were found.
    pub(super) fn take_cte_sites(&mut self, cte_index: usize) -> usize {
        self.cte_sites.remove(&cte_index).unwrap_or_default()
    }

    /// The distinct-key CTE published by the innermost enclosing delim join.
    pub(super) fn delim_target(&self) -> Option<usize> {
        self.delim_targets.last().copied()
    }

    /// Walk a delim join's subquery side with its distinct-key CTE in scope.
    /// The pop happens even when `walk` returns an error.
    pub(super) fn with_delim_target<T, E>(
        &mut self,
        dedup_keys_cte: usize,
        walk: impl FnOnce(&mut BuildCtx) -> Result<T, E>,
    ) -> Result<T, E> {
        self.delim_targets.push(dedup_keys_cte);
        let result = walk(self);
        self.delim_targets.pop();
        result
    }
}
