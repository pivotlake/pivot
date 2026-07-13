//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::manifest::PartitionEqFilter;
use crate::parquet::{
    ParquetTable, ScanEqualityPredicate, materialize, row_group_eliminated, row_group_filter_from,
    scan_order_from, table_input_with_filter_and_eq_predicates,
};
use crate::parquet_writing::{ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE, encode_spec};
use arrow_array::{Array, ArrayRef, RecordBatch, Scalar, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec, Sender};
use planner::catalog::{
    CatalogTransaction, Column, DynamicScanPredicate, Error as CatalogError,
    Result as CatalogResult, Table,
};
use planner::expression::{CompareType, Expression, TableFilter};

use super::{CatalogTable, ParquetTransaction};

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding. Recorded as-is; applied at [`compile`](Table::compile)
/// time after the row-group metadata exists — min/max stats prune row groups,
/// and equality additionally prunes by dictionary contents in the decoder. The
/// upstream `Filter` always runs, so this is a pure optimization.
#[derive(Clone, Debug)]
struct PushedPredicate {
    column_idx: usize,
    compare_type: CompareType,
    value: Scalar<ArrayRef>,
}

/// A catalog table resolved by name from one transaction: it carries the
/// table's name, schema, and this query's pushed-down predicates, but no
/// catalog state. The file set is read from the transaction handed to the
/// compile-time methods, so the binding is a plain value and the transaction
/// stays the single owner of the frozen view. Cloned per-binding so each query
/// accumulates its own predicates.
#[derive(Clone, Debug)]
pub struct TableBinding {
    /// The catalog name resolved, used to fetch this table's row groups from
    /// the transaction's snapshot at compile time.
    name: String,
    pub columns: Vec<Column>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `Table` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
}

impl TableBinding {
    /// A binding over `name`, with no predicates pushed yet.
    pub(super) fn new(name: String, columns: Vec<Column>) -> Self {
        Self {
            name,
            columns,
            predicates: Vec::new(),
        }
    }

    /// This table's row groups in `transaction`'s snapshot, partition-pruned by
    /// the pushed equality predicates. Pure in-memory; the snapshot is
    /// immutable, so a scan and its late materialize (same filters) build
    /// identical views addressing the same global row-group indices. The
    /// transaction is always our own [`ParquetTransaction`] (a `ParquetCatalog`
    /// only ever compiles its own bindings), so the downcast is an invariant;
    /// failing it is an error, never a silent empty scan.
    fn resolve_files(
        &self,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<Arc<ParquetTable>> {
        let transaction = transaction
            .as_any()
            .downcast_ref::<ParquetTransaction>()
            .ok_or_else(|| CatalogError::Other("transaction is not a ParquetTransaction".into()))?;
        transaction
            .snapshot
            .parquet(&self.name, self.partition_filter_candidates())
    }

    /// This binding's pushed equality predicates as partition-filter candidates:
    /// the column name and the typed scalar. The catalog intersects these with
    /// the table's partition columns, so yielding every equality predicate (not
    /// just ones on partition columns, which the binding can't tell apart) is fine
    /// — a non-partition column prunes no files.
    fn partition_filter_candidates(&self) -> impl Iterator<Item = PartitionEqFilter> + '_ {
        self.predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .filter_map(|p| {
                Some(PartitionEqFilter {
                    column: self.columns.get(p.column_idx)?.name.clone(),
                    value: p.value.clone(),
                })
            })
    }
}

impl Table for TableBinding {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The transaction's snapshot hands back the file set this query was
        // bound against, partition-pruned. All in-memory: the background
        // refresh already materialized every footer.
        let current = self.resolve_files(transaction)?;

        // Equality predicates additionally let the decoder skip row groups whose
        // dictionary for that column excludes the constant.
        let eq_predicates: Vec<ScanEqualityPredicate> = self
            .predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .map(|p| ScanEqualityPredicate {
                column_idx: p.column_idx,
                value: p.value.clone(),
            })
            .collect();

        // Prune the row groups by the pushed-down predicates' stats. No footer
        // re-read — the cache already holds the materialized row groups.
        let parquet = Arc::new(self.pruned_parquet(&current));
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Same transaction as the scan and its snapshot is immutable, so this
        // reads an identical view and their global row-group indices line up.
        // Late materialization re-reads rows by that global index, so it uses
        // the full table, not the pruned scan view.
        Ok(materialize(
            input,
            self.resolve_files(transaction)?,
            projection,
        ))
    }

    /// Build the transactional Parquet write pipeline for this table.
    ///
    /// Dispatch workers conform and encode batches, then fan completed files
    /// into [`TransactionWriter::submit`](super::transaction::TransactionWriter::submit).
    /// Submission is an in-memory channel send; a dedicated transaction thread
    /// performs object-store writes and retains their paths until commit adds
    /// them to Delta or rollback deletes them. The pipeline's only output is the
    /// inserted row count used for PostgreSQL's command-completion tag.
    fn insert(
        &self,
        source: RecordBatchOperatorSpec,
        transaction: &dyn CatalogTransaction,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        let transaction = transaction
            .as_any()
            .downcast_ref::<ParquetTransaction>()
            .ok_or_else(|| CatalogError::Other("transaction is not a ParquetTransaction".into()))?;
        let table = transaction
            .snapshot
            .tables
            .get(&self.name)
            .cloned()
            .ok_or_else(|| {
                CatalogError::Other(format!("table {:?} is not in this snapshot", self.name).into())
            })?;

        let rows_written = Arc::new(AtomicU64::new(0));
        let source = self.conform_batches(source, &table, rows_written.clone());
        let encoded = encode_spec(
            source,
            Arc::from(table.partition_by()),
            Arc::from(table.sort_by()),
            ROW_GROUP_ROWS,
            ROW_GROUPS_PER_FILE,
        );
        let writer = transaction.insert_writer();
        let table_name = self.name.clone();
        let staged = encoded.fan_in(
            (writer, table_name),
            |state, encoded, _sender: &mut dyn Sender<RecordBatch>| {
                state
                    .0
                    .submit(state.1.clone(), encoded)
                    .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))
            },
            move |_state, sender: &mut dyn Sender<RecordBatch>| {
                let count = UInt64Array::from(vec![rows_written.load(Ordering::Relaxed)]);
                let schema = Schema::new(vec![Field::new("count", DataType::UInt64, false)]);
                sender.send(RecordBatch::try_new(
                    Arc::new(schema),
                    vec![Arc::new(count)],
                )?)?;
                Ok(())
            },
        );
        Ok(RecordBatchOperatorSpec::from_spec(staged))
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        let TableFilter::Expression(expr) = filter else {
            return Ok(false);
        };
        let Expression::Compare(compare) = expr.as_ref() else {
            return Ok(false);
        };
        let (reference, constant) = match (compare.left.as_ref(), compare.right.as_ref()) {
            (Expression::Ref(r), Expression::Constant(k))
            | (Expression::Constant(k), Expression::Ref(r)) => (r, k),
            _ => return Ok(false),
        };

        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.predicates.push(PushedPredicate {
            column_idx: reference.column_idx,
            compare_type: compare.compare_type,
            value: constant.clone(),
        });

        Ok(false)
    }

    fn column_min_max(
        &self,
        column: usize,
        transaction: &dyn CatalogTransaction,
    ) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // Read the transaction's snapshot files, the same view a scan would
        // see; an unresolvable snapshot means "scan".
        let parquet = self.resolve_files(transaction).ok()?;
        let row_groups = parquet.row_groups();
        if row_groups.is_empty() {
            return None;
        }
        // Every row group must carry both bounds; fold them with the arrow
        // comparison kernels (stats come back typed as the physical column).
        let mut min: Option<Scalar<ArrayRef>> = None;
        let mut max: Option<Scalar<ArrayRef>> = None;
        for rg in row_groups {
            let stats = rg.column_statistics(column)?;
            let (rg_min, rg_max) = (stats.min.as_ref()?, stats.max.as_ref()?);
            min = Some(match min {
                Some(m) if scalar_lt(&m, rg_min) => m,
                _ => rg_min.clone(),
            });
            max = Some(match max {
                Some(m) if scalar_lt(rg_max, &m) => m,
                _ => rg_max.clone(),
            });
        }
        Some((min?, max?))
    }

    fn row_count(&self, transaction: &dyn CatalogTransaction) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // The parquet footer carries each row group's exact row count, so the
        // table's count is their sum, with no data pages read.
        let parquet = self.resolve_files(transaction).ok()?;
        Some(parquet.row_groups().iter().map(|rg| rg.num_rows).sum())
    }
}

/// `a < b` over two single-value scalars of the same physical type. A null,
/// type mismatch, or kernel error reads as `false`. In [`column_min_max`] every
/// comparison is between two same-typed integer/temporal stat bounds, where the
/// kernel never errors and the bounds are non-null. (Mirrors the stricter,
/// private `scalar_lt` in `parquet::reading::fetching`; worth consolidating.)
///
/// [`column_min_max`]: TableBinding::column_min_max
fn scalar_lt(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    arrow_ord::cmp::lt(a, b).is_ok_and(|r| r.len() == 1 && r.is_valid(0) && r.value(0))
}

impl TableBinding {
    /// Put source batches into the target table's physical Parquet shape.
    ///
    /// DuckDB has already bound each VALUES column to a compatible logical
    /// type, and [`Insert::compile`](planner::operator::Insert::compile) has
    /// reordered explicit column lists into table order. This projection casts
    /// those arrays to Pivot's exact physical Arrow types, replaces the source
    /// schema with the table schema, and counts rows for PostgreSQL's
    /// `INSERT 0 n` completion tag. It performs CPU-only Arrow work; Parquet
    /// encoding and object-store I/O happen in later stages.
    fn conform_batches(
        &self,
        source: RecordBatchOperatorSpec,
        table: &CatalogTable,
        rows_written: Arc<AtomicU64>,
    ) -> RecordBatchOperatorSpec {
        let schema = table.physical_arrow_schema();
        source.project(move || {
            let rows_written = rows_written.clone();
            let schema = schema.clone();
            move |batch: RecordBatch| {
                rows_written.fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
                let columns = batch
                    .columns()
                    .iter()
                    .zip(schema.fields())
                    .map(|(column, field)| {
                        arrow_cast::cast(column, field.data_type())
                            .expect("a bound INSERT value must cast to its table column")
                    })
                    .collect();
                RecordBatch::try_new(schema.clone(), columns)
                    .expect("conformed INSERT columns must match the table schema")
            }
        })
    }

    /// Clone `parquet`'s row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`Table::compile`] actually
    /// scans over the table's current files. A min/max stat that proves no row in
    /// a group can match drops it; a stats-comparison error means "can't prune"
    /// (kept) — never wrong, just unoptimized. No footer I/O. Exposed so pruning
    /// can be asserted directly.
    pub fn pruned_parquet(&self, parquet: &ParquetTable) -> ParquetTable {
        let mut parquet = parquet.clone();
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                row_group_eliminated(rg.as_ref(), p.column_idx, p.compare_type, &p.value)
                    .unwrap_or(false)
            })
        });
        parquet
    }
}
