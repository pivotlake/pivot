//! The probe side of the hash join: for each probe row whose key hash matches
//! a build row's, emit the probe row's columns followed by the matched build
//! row's columns (gathered from the concatenated build payload batch).
//!
//! Matches are decided on the full 64-bit combined key hash (see
//! [`hash_key_row`]); the planner layers an equality filter over the join's
//! output that re-compares the actual key columns, which screens out both
//! hash collisions and any residual join conditions in one place.
//!
//! Matches are collected as two parallel selection vectors — probe row index
//! and build payload row index — and materialized per output batch with the
//! arrow `take` kernel, so the hot loops touch only hashes, the directory, and
//! the key/row arenas.

use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::Unary;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::build::{filter_null_keys, hash_key_row, strict_key_columns};
use crate::operations::unary::join::directory::{Directory, PtrBuffer, prefetch_ptr_l2};
use crate::operations::unary::join::primitive_builder::JoinPrimitiveBuilder;
use crate::operations::unary::join::{JoinArena, JoinMode, JoinTable};
use ahash::RandomState;
use arrow::compute::take;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt32Type};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::cmp::min;
use std::mem;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::sync::atomic::Ordering;

const PREFETCH_LENGTH: usize = 63;

pub struct Probe {
    table: JoinTable,
    hash_state: RandomState,
    key_columns: Vec<usize>,
    /// The key columns' indices in the build payload, used by the semi/anti
    /// modes' exact match verification.
    build_key_columns: Vec<usize>,
    /// Aligned with `key_columns`: `true` for a null-safe key (see the build
    /// side's field of the same name).
    null_safe: Vec<bool>,
    mode: JoinMode,
    /// The probe side's column types as the planner declared them, used by
    /// the BuildOuter mode's padded emission (which may run on a worker that
    /// never saw a probe batch).
    probe_types: Vec<DataType>,
    use_probe_array: bool,
    allocator: SlabAllocator,
    /// Probe fields followed (for an inner join) by build payload fields;
    /// built on first batch.
    output_schema: Option<Arc<Schema>>,
    /// The LEFT mode's build payload with its all-null padding row appended.
    padded_build_rows: Option<RecordBatch>,
}

impl Probe {
    pub(crate) fn new(
        table: JoinTable,
        hash_state: RandomState,
        key_columns: Vec<usize>,
        build_key_columns: Vec<usize>,
        null_safe: Vec<bool>,
        mode: JoinMode,
        probe_types: Vec<DataType>,
        use_probe_array: bool,
    ) -> Self {
        Self {
            table,
            hash_state,
            key_columns,
            build_key_columns,
            null_safe,
            mode,
            probe_types,
            use_probe_array,
            allocator: SlabAllocator::new(false),
            output_schema: None,
            padded_build_rows: None,
        }
    }

    /// Semi/anti probe over one batch: for each probe row decide whether some
    /// build row's keys equal its keys — verified exactly against the build
    /// payload's key columns, since no downstream filter can re-check a match
    /// that emits probe columns only — and emit the row for a match (semi) or
    /// for no match (anti). A row with a null key matches nothing.
    #[allow(clippy::too_many_arguments)]
    fn run_existence<S: Sender<RecordBatch>, B>(
        &mut self,
        directory: &Directory<B>,
        keys: &JoinArena<u64>,
        rows: &JoinArena<u32>,
        build_rows: &RecordBatch,
        build_key_columns: &[usize],
        batch: &RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()>
    where
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    {
        let probe_keys: Vec<&Int64Array> = self
            .key_columns
            .iter()
            .map(|&c| batch.column(c).as_primitive::<Int64Type>())
            .collect();
        let build_keys: Vec<&Int64Array> = build_key_columns
            .iter()
            .map(|&c| build_rows.column(c).as_primitive::<Int64Type>())
            .collect();
        let emit_on_match = self.mode == JoinMode::Semi;
        let null_safe = self.null_safe.clone();

        let output_schema = self
            .output_schema
            .get_or_insert_with(|| batch.schema())
            .clone();

        let mut sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut output_idx = 0;
        for row in 0..batch.num_rows() {
            // A null in a strict key column matches nothing; a null-safe
            // column's null hashes and compares through its validity.
            let strict_valid = probe_keys
                .iter()
                .zip(&null_safe)
                .all(|(column, &null_safe)| null_safe || column.is_valid(row));
            let matched = strict_valid && {
                let hash = hash_key_row(&self.hash_state, &probe_keys, &null_safe, row);
                directory.matches_bloom(hash) && {
                    let slot = directory.slot_for(hash);
                    let start = directory.end_ptr(slot as isize);
                    let end = directory.end_ptr((slot + 1) as isize);
                    (start..end).any(|j| {
                        keys[j] == hash
                            && build_keys.iter().zip(&probe_keys).zip(&null_safe).all(
                                |((b, p), &null_safe)| {
                                    let build_row = rows[j] as usize;
                                    if null_safe {
                                        match (b.is_valid(build_row), p.is_valid(row)) {
                                            (true, true) => b.value(build_row) == p.value(row),
                                            (false, false) => true,
                                            _ => false,
                                        }
                                    } else {
                                        b.value(build_row) == p.value(row)
                                    }
                                },
                            )
                    })
                }
            };
            if matched == emit_on_match {
                sel.write(output_idx, row as u32);
                output_idx += 1;
                if output_idx == RECORD_BATCH_SIZE {
                    let sel_array = mem::replace(
                        &mut sel,
                        JoinPrimitiveBuilder::<UInt32Type>::new(
                            &mut self.allocator,
                            RECORD_BATCH_SIZE,
                        ),
                    )
                    .into_array(output_idx);
                    send_selected(batch, &sel_array, &output_schema, sender)?;
                    output_idx = 0;
                }
            }
        }
        if output_idx > 0 {
            let sel_array = sel.into_array(output_idx);
            send_selected(batch, &sel_array, &output_schema, sender)?;
        }
        Ok(())
    }
}

impl Probe {
    /// The build-emitting existence modes' probe pass over one batch: mark
    /// every build row some probe row's keys equal (verified exactly against
    /// the build payload). Emission happens once, in the last probe worker's
    /// finish.
    #[allow(clippy::too_many_arguments)]
    fn run_marking<B>(
        &mut self,
        directory: &Directory<B>,
        keys: &JoinArena<u64>,
        rows: &JoinArena<u32>,
        build_rows: &RecordBatch,
        build_key_columns: &[usize],
        batch: &RecordBatch,
    ) -> unary::Result<()>
    where
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    {
        let probe_keys: Vec<&Int64Array> = self
            .key_columns
            .iter()
            .map(|&c| batch.column(c).as_primitive::<Int64Type>())
            .collect();
        let build_keys: Vec<&Int64Array> = build_key_columns
            .iter()
            .map(|&c| build_rows.column(c).as_primitive::<Int64Type>())
            .collect();
        let matched = unsafe { &*self.table.matched.get() };
        let null_safe = &self.null_safe;

        for row in 0..batch.num_rows() {
            let strict_valid = probe_keys
                .iter()
                .zip(null_safe)
                .all(|(column, &null_safe)| null_safe || column.is_valid(row));
            if !strict_valid {
                continue;
            }
            let hash = hash_key_row(&self.hash_state, &probe_keys, null_safe, row);
            if !directory.matches_bloom(hash) {
                continue;
            }
            let slot = directory.slot_for(hash);
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            for j in start..end {
                if keys[j] != hash {
                    continue;
                }
                let build_row = rows[j] as usize;
                let key_equal = build_keys.iter().zip(&probe_keys).zip(null_safe).all(
                    |((b, p), &null_safe)| {
                        if null_safe {
                            match (b.is_valid(build_row), p.is_valid(row)) {
                                (true, true) => b.value(build_row) == p.value(row),
                                (false, false) => true,
                                _ => false,
                            }
                        } else {
                            b.value(build_row) == p.value(row)
                        }
                    },
                );
                if key_equal {
                    matched[build_row].store(true, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    /// LEFT OUTER probe over one batch: emit every verified match, and each
    /// probe row without one paired with the all-null sentinel row appended
    /// to the build payload.
    #[allow(clippy::too_many_arguments)]
    fn run_left<S: Sender<RecordBatch>, B>(
        &mut self,
        directory: &Directory<B>,
        keys: &JoinArena<u64>,
        rows: &JoinArena<u32>,
        build_rows: &RecordBatch,
        build_key_columns: &[usize],
        batch: &RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()>
    where
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    {
        let probe_keys: Vec<&Int64Array> = self
            .key_columns
            .iter()
            .map(|&c| batch.column(c).as_primitive::<Int64Type>())
            .collect();
        let build_keys: Vec<&Int64Array> = build_key_columns
            .iter()
            .map(|&c| build_rows.column(c).as_primitive::<Int64Type>())
            .collect();
        let null_safe = self.null_safe.clone();

        // The padded build payload: one all-null row appended, which
        // unmatched probe rows select. Build once per probe operator.
        let padded = self.padded_build_rows(build_rows)?;
        let sentinel = build_rows.num_rows() as u32;
        let output_schema = self
            .output_schema
            .get_or_insert_with(|| {
                let fields: Vec<Field> = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone())
                    .chain(
                        padded
                            .schema()
                            .fields()
                            .iter()
                            .map(|field| field.as_ref().clone().with_nullable(true)),
                    )
                    .collect();
                Arc::new(Schema::new(fields))
            })
            .clone();

        let mut probe_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut build_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut output_idx = 0;
        macro_rules! push_pair {
            ($row:expr, $build_row:expr) => {{
                probe_sel.write(output_idx, $row as u32);
                build_sel.write(output_idx, $build_row);
                output_idx += 1;
                if output_idx == RECORD_BATCH_SIZE {
                    let probe_array = mem::replace(
                        &mut probe_sel,
                        JoinPrimitiveBuilder::<UInt32Type>::new(
                            &mut self.allocator,
                            RECORD_BATCH_SIZE,
                        ),
                    )
                    .into_array(output_idx);
                    let build_array = mem::replace(
                        &mut build_sel,
                        JoinPrimitiveBuilder::<UInt32Type>::new(
                            &mut self.allocator,
                            RECORD_BATCH_SIZE,
                        ),
                    )
                    .into_array(output_idx);
                    send_joined(
                        batch,
                        &padded,
                        &probe_array,
                        &build_array,
                        &output_schema,
                        sender,
                    )?;
                    output_idx = 0;
                }
            }};
        }
        for row in 0..batch.num_rows() {
            let strict_valid = probe_keys
                .iter()
                .zip(&null_safe)
                .all(|(column, &null_safe)| null_safe || column.is_valid(row));
            let mut matched = false;
            if strict_valid {
                let hash = hash_key_row(&self.hash_state, &probe_keys, &null_safe, row);
                if directory.matches_bloom(hash) {
                    let slot = directory.slot_for(hash);
                    let start = directory.end_ptr(slot as isize);
                    let end = directory.end_ptr((slot + 1) as isize);
                    for j in start..end {
                        if keys[j] != hash {
                            continue;
                        }
                        let build_row = rows[j] as usize;
                        let key_equal = build_keys.iter().zip(&probe_keys).zip(&null_safe).all(
                            |((b, p), &null_safe)| {
                                if null_safe {
                                    match (b.is_valid(build_row), p.is_valid(row)) {
                                        (true, true) => b.value(build_row) == p.value(row),
                                        (false, false) => true,
                                        _ => false,
                                    }
                                } else {
                                    b.value(build_row) == p.value(row)
                                }
                            },
                        );
                        if key_equal {
                            matched = true;
                            push_pair!(row, build_row as u32);
                        }
                    }
                }
            }
            if !matched {
                push_pair!(row, sentinel);
            }
        }
        if output_idx > 0 {
            let probe_array = probe_sel.into_array(output_idx);
            let build_array = build_sel.into_array(output_idx);
            send_joined(
                batch,
                &padded,
                &probe_array,
                &build_array,
                &output_schema,
                sender,
            )?;
        }
        Ok(())
    }

    /// Build-side outer probe over one batch: emit every verified match (probe
    /// columns then build columns) and mark the matched build rows; the last
    /// probe worker later emits the unmarked build rows padded with null probe
    /// columns (see `emit_unmatched_padded`). Matches verify exactly against
    /// the build payload, since a downstream filter can't re-check padded rows.
    ///
    /// Like the inner probe, null strict keys are filtered up front (a probe
    /// row is not preserved by this mode, so dropping non-matching rows early
    /// is sound), hashes are precomputed per batch, and the directory and key
    /// arena are prefetched a fixed lookahead ahead of the scan.
    #[allow(clippy::too_many_arguments)]
    fn run_build_outer<S: Sender<RecordBatch>, B>(
        &mut self,
        directory: &Directory<B>,
        keys: &JoinArena<u64>,
        rows: &JoinArena<u32>,
        build_rows: &RecordBatch,
        build_key_columns: &[usize],
        batch: &RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()>
    where
        B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    {
        let batch = filter_null_keys(
            batch.clone(),
            &strict_key_columns(&self.key_columns, &self.null_safe),
        );
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let probe_keys: Vec<&Int64Array> = self
            .key_columns
            .iter()
            .map(|&c| batch.column(c).as_primitive::<Int64Type>())
            .collect();
        let build_keys: Vec<&Int64Array> = build_key_columns
            .iter()
            .map(|&c| build_rows.column(c).as_primitive::<Int64Type>())
            .collect();
        let hashes: Vec<u64> = (0..batch.num_rows())
            .map(|i| hash_key_row(&self.hash_state, &probe_keys, &self.null_safe, i))
            .collect();
        // The common single strict Int64 key compares raw value slices; the
        // general path (composite or null-safe keys) re-checks validity.
        let single_key = probe_keys.len() == 1 && !self.null_safe[0];
        let (probe_values, build_values) = (probe_keys[0].values(), build_keys[0].values());
        let matched = unsafe { &*self.table.matched.get() };
        let null_safe = self.null_safe.clone();
        let output_schema = self.build_outer_schema(build_rows);

        let mut probe_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut build_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut output_idx = 0;
        for row in 0..batch.num_rows() {
            const DIRECTORY_PREFETCH_DISTANCE: usize = 16;
            const ARENA_PREFETCH_DISTANCE: usize = 8;
            if row + DIRECTORY_PREFETCH_DISTANCE < hashes.len() {
                directory.prefetch_l2(hashes[row + DIRECTORY_PREFETCH_DISTANCE]);
            }
            if row + ARENA_PREFETCH_DISTANCE < hashes.len() {
                let future_hash = hashes[row + ARENA_PREFETCH_DISTANCE];
                let future_slot = directory.slot_for(future_hash);
                let future_start = directory.end_ptr(future_slot as isize);
                prefetch_ptr_l2(keys.ptr_at_index(future_start) as *const u8);
            }

            let hash = hashes[row];
            if !directory.matches_bloom(hash) {
                continue;
            }
            let slot = directory.slot_for(hash);
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            for j in start..end {
                if keys[j] != hash {
                    continue;
                }
                let build_row = rows[j] as usize;
                let key_equal = if single_key {
                    build_values[build_row] == probe_values[row]
                } else {
                    build_keys.iter().zip(&probe_keys).zip(&null_safe).all(
                        |((b, p), &null_safe)| {
                            if null_safe {
                                match (b.is_valid(build_row), p.is_valid(row)) {
                                    (true, true) => b.value(build_row) == p.value(row),
                                    (false, false) => true,
                                    _ => false,
                                }
                            } else {
                                b.value(build_row) == p.value(row)
                            }
                        },
                    )
                };
                if !key_equal {
                    continue;
                }
                matched[build_row].store(true, Ordering::Relaxed);
                probe_sel.write(output_idx, row as u32);
                build_sel.write(output_idx, build_row as u32);
                output_idx += 1;
                if output_idx == RECORD_BATCH_SIZE {
                    let probe_array = mem::replace(
                        &mut probe_sel,
                        JoinPrimitiveBuilder::<UInt32Type>::new(
                            &mut self.allocator,
                            RECORD_BATCH_SIZE,
                        ),
                    )
                    .into_array(output_idx);
                    let build_array = mem::replace(
                        &mut build_sel,
                        JoinPrimitiveBuilder::<UInt32Type>::new(
                            &mut self.allocator,
                            RECORD_BATCH_SIZE,
                        ),
                    )
                    .into_array(output_idx);
                    send_joined(
                        &batch,
                        build_rows,
                        &probe_array,
                        &build_array,
                        &output_schema,
                        sender,
                    )?;
                    output_idx = 0;
                }
            }
        }
        if output_idx > 0 {
            let probe_array = probe_sel.into_array(output_idx);
            let build_array = build_sel.into_array(output_idx);
            send_joined(
                &batch,
                build_rows,
                &probe_array,
                &build_array,
                &output_schema,
                sender,
            )?;
        }
        Ok(())
    }

    /// The BuildOuter output schema: the planner-declared probe columns
    /// (nullable, they hold the padding) followed by the build payload's.
    /// Derived from `probe_types` rather than a probe batch so the padded
    /// emission can build it on a worker that never consumed one.
    fn build_outer_schema(&mut self, build_rows: &RecordBatch) -> Arc<Schema> {
        self.output_schema
            .get_or_insert_with(|| {
                let fields: Vec<Field> = self
                    .probe_types
                    .iter()
                    .enumerate()
                    .map(|(i, data_type)| Field::new(format!("probe_{i}"), data_type.clone(), true))
                    .chain(
                        build_rows
                            .schema()
                            .fields()
                            .iter()
                            .map(|field| field.as_ref().clone()),
                    )
                    .collect();
                Arc::new(Schema::new(fields))
            })
            .clone()
    }

    /// Emit the build rows no probe row matched, their probe columns null.
    /// Runs once, in the last probe worker, after every mark is in.
    fn emit_unmatched_padded<S: Sender<RecordBatch>>(
        &mut self,
        sender: &mut S,
    ) -> unary::Result<()> {
        let build_rows = unsafe { &*self.table.build_rows.get() };
        let Some(build_rows) = build_rows else {
            return Ok(());
        };
        let build_rows = build_rows.clone();
        let matched = unsafe { &*self.table.matched.get() };
        let schema = self.build_outer_schema(&build_rows);

        let mut sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut output_idx = 0;
        macro_rules! flush_padded {
            ($sel_array:expr, $len:expr) => {{
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
                for data_type in &self.probe_types {
                    columns.push(arrow_array::new_null_array(data_type, $len));
                }
                for column in build_rows.columns() {
                    columns.push(take(column, &$sel_array, None)?);
                }
                sender.send(RecordBatch::try_new(schema.clone(), columns)?)?;
            }};
        }
        for (row, flag) in matched.iter().enumerate() {
            if flag.load(Ordering::Relaxed) {
                continue;
            }
            sel.write(output_idx, row as u32);
            output_idx += 1;
            if output_idx == RECORD_BATCH_SIZE {
                let sel_array = mem::replace(
                    &mut sel,
                    JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE),
                )
                .into_array(output_idx);
                flush_padded!(sel_array, RECORD_BATCH_SIZE);
                output_idx = 0;
            }
        }
        if output_idx > 0 {
            let sel_array = sel.into_array(output_idx);
            flush_padded!(sel_array, output_idx);
        }
        Ok(())
    }

    /// The build payload with one all-null row appended (the LEFT join's
    /// padding target), built lazily once.
    fn padded_build_rows(&mut self, build_rows: &RecordBatch) -> unary::Result<RecordBatch> {
        if let Some(padded) = &self.padded_build_rows {
            return Ok(padded.clone());
        }
        let fields: Vec<Field> = build_rows
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let null_row = RecordBatch::try_new(
            schema.clone(),
            build_rows
                .columns()
                .iter()
                .map(|column| arrow_array::new_null_array(column.data_type(), 1))
                .collect(),
        )?;
        let relaxed = RecordBatch::try_new(schema.clone(), build_rows.columns().to_vec())?;
        let padded = arrow::compute::concat_batches(&schema, [&relaxed, &null_row])?;
        self.padded_build_rows = Some(padded.clone());
        Ok(padded)
    }

    /// Emit the marked (semi) or unmarked (anti) build rows, in payload
    /// order, chunked to batch size. Runs once, in the last probe worker.
    fn emit_marked<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<()> {
        let build_rows = unsafe { &*self.table.build_rows.get() };
        let Some(build_rows) = build_rows else {
            return Ok(());
        };
        let matched = unsafe { &*self.table.matched.get() };
        let emit_on_match = self.mode == JoinMode::BuildSemi;
        let schema = build_rows.schema();

        let mut sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let mut output_idx = 0;
        for (row, flag) in matched.iter().enumerate() {
            if flag.load(Ordering::Relaxed) != emit_on_match {
                continue;
            }
            sel.write(output_idx, row as u32);
            output_idx += 1;
            if output_idx == RECORD_BATCH_SIZE {
                let sel_array = mem::replace(
                    &mut sel,
                    JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE),
                )
                .into_array(output_idx);
                send_selected(build_rows, &sel_array, &schema, sender)?;
                output_idx = 0;
            }
        }
        if output_idx > 0 {
            let sel_array = sel.into_array(output_idx);
            send_selected(build_rows, &sel_array, &schema, sender)?;
        }
        Ok(())
    }
}

/// Emit joined rows: probe columns taken by `probe_sel`, build columns taken
/// by `build_sel` (which may select the LEFT padding row).
fn send_joined<S: Sender<RecordBatch>>(
    probe: &RecordBatch,
    build: &RecordBatch,
    probe_sel: &ArrayRef,
    build_sel: &ArrayRef,
    schema: &Arc<Schema>,
    sender: &mut S,
) -> unary::Result<()> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for column in probe.columns() {
        columns.push(take(column, probe_sel, None)?);
    }
    for column in build.columns() {
        columns.push(take(column, build_sel, None)?);
    }
    sender.send(RecordBatch::try_new(schema.clone(), columns)?)?;
    Ok(())
}

/// Emit the probe rows selected by `sel` (semi/anti output: probe columns
/// only).
fn send_selected<S: Sender<RecordBatch>>(
    batch: &RecordBatch,
    sel: &ArrayRef,
    schema: &Arc<Schema>,
    sender: &mut S,
) -> unary::Result<()> {
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column, sel, None))
        .collect::<Result<Vec<_>, _>>()?;
    sender.send(RecordBatch::try_new(schema.clone(), columns)?)?;
    Ok(())
}

/// The per-batch probe state: scans the precomputed key `hashes` against the
/// directory, collecting matched (probe row, build payload row) pairs into the
/// two selection-vector builders and flushing full output batches through
/// `sender`.
struct BatchProbe<'a, 'b, S: Sender<RecordBatch>> {
    keys: &'a JoinArena<u64>,
    rows: &'a JoinArena<u32>,
    hashes: &'b [u64],
    probe_batch: &'b RecordBatch,
    build_rows: &'a RecordBatch,
    output_schema: &'a Arc<Schema>,

    probe_sel: JoinPrimitiveBuilder<UInt32Type>,
    build_sel: JoinPrimitiveBuilder<UInt32Type>,
    output_idx: usize,

    sender: &'a mut S,
    allocator: &'a mut SlabAllocator,
}

impl<'a, 'b, S: Sender<RecordBatch>> BatchProbe<'a, 'b, S> {
    #[inline(never)]
    fn flush(&mut self) -> unary::Result<()> {
        if self.output_idx == 0 {
            return Ok(());
        }
        let probe_sel = mem::replace(
            &mut self.probe_sel,
            JoinPrimitiveBuilder::<UInt32Type>::new(self.allocator, RECORD_BATCH_SIZE),
        )
        .into_array(self.output_idx);
        let build_sel = mem::replace(
            &mut self.build_sel,
            JoinPrimitiveBuilder::<UInt32Type>::new(self.allocator, RECORD_BATCH_SIZE),
        )
        .into_array(self.output_idx);

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.output_schema.fields().len());
        for column in self.probe_batch.columns() {
            columns.push(take(column, &probe_sel, None)?);
        }
        for column in self.build_rows.columns() {
            columns.push(take(column, &build_sel, None)?);
        }
        let batch = RecordBatch::try_new(self.output_schema.clone(), columns)?;
        self.sender.send(batch)?;
        self.output_idx = 0;
        Ok(())
    }

    /// Record the match candidate at arena index `j` for probe row
    /// `probe_row`: write both selection vectors at the current cursor and
    /// advance it only when the full hash matches (branchless on the match
    /// itself; the capacity flush branch is almost never taken).
    #[inline(always)]
    fn record_match(&mut self, j: usize, probe_row: usize) -> unary::Result<()> {
        let key = self.keys[j];
        self.probe_sel.write(self.output_idx, probe_row as u32);
        self.build_sel.write(self.output_idx, self.rows[j]);
        self.output_idx += (key == self.hashes[probe_row]) as usize;
        if self.output_idx == RECORD_BATCH_SIZE {
            self.flush()?;
        }
        Ok(())
    }

    /// Scalar reference probe: bloom-check and scan each probe row's
    /// directory slot in order, with light lookahead prefetching.
    fn run_scalar<B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer>(
        mut self,
        directory: &Directory<B>,
    ) -> unary::Result<()> {
        for row_idx in 0..self.hashes.len() {
            const DIRECTORY_PREFETCH_DISTANCE: usize = 16;
            const ARENA_PREFETCH_DISTANCE: usize = 8;

            if row_idx + DIRECTORY_PREFETCH_DISTANCE < self.hashes.len() {
                directory.prefetch_l2(self.hashes[row_idx + DIRECTORY_PREFETCH_DISTANCE]);
            }

            if row_idx + ARENA_PREFETCH_DISTANCE < self.hashes.len() {
                let future_hash = self.hashes[row_idx + ARENA_PREFETCH_DISTANCE];
                let future_slot = directory.slot_for(future_hash);
                let future_start = directory.end_ptr(future_slot as isize);
                prefetch_ptr_l2(self.keys.ptr_at_index(future_start) as *const u8);
            }

            let hash = self.hashes[row_idx];
            if !directory.matches_bloom(hash) {
                continue;
            }

            let slot = directory.slot_for(hash);
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);
            for j in start..end {
                self.record_match(j, row_idx)?;
            }
        }

        self.flush()
    }
}

/// The prefetch-pipelined probe: bloom-filters the precomputed hashes into a
/// ring of matched directory slots, prefetches their arena ranges, then drains
/// matches a window behind — keeping many independent loads in flight.
struct ProbeArray<
    'a,
    'b,
    B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    S: Sender<RecordBatch>,
> {
    row_idx: usize,
    directory: &'a Directory<B>,

    // If we made this two separate variables, e.g. matched_slots and next_matched_slots, swapping
    // became an issue. The reason for this is that because it is of constant size, it was being
    // unrolled, causing multiple pointer swaps to occur. This may seem like a small issue, and it's
    // certainly not a big one, but keeping instructions to a minimum is important for the ROB.
    matched_slots: [[(usize, usize); PREFETCH_LENGTH]; 2],
    matched_size: [usize; 2],
    matched_idx: usize,

    out: BatchProbe<'a, 'b, S>,
}

impl<'a, 'b, B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer, S: Sender<RecordBatch>>
    ProbeArray<'a, 'b, B, S>
{
    /// Bloom-check `length` rows, recording matched (slot, row) pairs. When
    /// `PREFETCH_AHEAD`, also prefetch the directory slot of the row a window
    /// ahead so its entry is resident by the time its own pass reads it.
    #[inline(always)]
    pub fn generate_matched_slots<const PREFETCH_AHEAD: bool>(&mut self, length: usize) {
        let next_matched_slots: &mut [(usize, usize); PREFETCH_LENGTH] =
            &mut self.matched_slots[self.matched_idx];
        let mut size = self.matched_size[self.matched_idx];
        let mut row_idx = self.row_idx;
        let shift = self.directory.shift;
        let hashes = self.out.hashes;
        for _ in 0..length {
            if PREFETCH_AHEAD {
                let ahead = hashes[row_idx + PREFETCH_LENGTH];
                prefetch_ptr_l2(self.directory.ptr_for_slot((ahead >> shift) as usize) as *const u8);
            }

            // bloom check, arena
            let hash = hashes[row_idx];

            let slot = (hash >> shift) as usize;
            let stored = unsafe { *self.directory.ptr_for_slot(slot) };
            let probe = Directory::<B>::compute_tag(hash) as u64;

            if (stored & probe) == probe {
                // Given that we matched, let's prefetch the slot after ours. Cache lines are 64 bytes,
                // so there's a one in eight chance this will be relevant
                prefetch_ptr_l2(self.directory.ptr_for_slot(slot + 1) as *const u8);
                next_matched_slots[size % PREFETCH_LENGTH] = (slot, row_idx);
                size += 1;
            }

            row_idx += 1;
        }
        self.matched_size[self.matched_idx] = size;
        self.row_idx = row_idx;
    }

    #[inline(always)]
    pub fn build_output<const PREFETCH_NEXT: bool>(&mut self) -> unary::Result<()> {
        // Field-level destructuring keeps the slot-ring borrows disjoint from
        // the &mut the match recorder needs.
        let Self {
            matched_slots,
            matched_size,
            matched_idx,
            directory,
            out,
            ..
        } = self;
        let cur = *matched_idx ^ 1;
        let nxt = *matched_idx;

        let current_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &matched_slots[cur];
        let next_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &matched_slots[nxt];
        for i in 0..matched_size[cur] {
            // Since all our memory should be in l2 (or on it's way) for the current slots being
            // built, we want to overlap future memory access. We therefore begin pulling the ptrs
            // from the next iterations matched slots
            if PREFETCH_NEXT {
                let (slot, _) = next_matched_slots[i];
                let start = directory.end_ptr(slot as isize);
                let end = directory.end_ptr((slot + 1) as isize);
                prefetch_ptr_l2(out.keys.ptr_at_index(start) as *const u8);
                prefetch_ptr_l2(out.keys.ptr_at_index(end) as *const u8);
            }

            let (slot, idx) = current_matched_slots[i];
            let start = directory.end_ptr(slot as isize);
            let end = directory.end_ptr((slot + 1) as isize);

            for j in start..end {
                out.record_match(j, idx)?;
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub fn only_prefetch(&mut self) {
        let nxt = self.matched_idx;

        let next_matched_slots: &[(usize, usize); PREFETCH_LENGTH] = &self.matched_slots[nxt];
        for i in 0..self.matched_size[nxt] {
            // Since all our memory should be in l2 (or on it's way) for the current slots being
            // built, we want to overlap future memory access. We therefore begin pulling the ptrs
            // from the next iterations matched slots
            let (slot, _) = next_matched_slots[i];
            let start = self.directory.end_ptr(slot as isize);
            let end = self.directory.end_ptr((slot + 1) as isize);
            prefetch_ptr_l2(self.out.keys.ptr_at_index(start) as *const u8);
            prefetch_ptr_l2(self.out.keys.ptr_at_index(end) as *const u8);
        }
    }

    #[inline(always)]
    fn swap_matched_slots(&mut self) {
        self.matched_idx ^= 1;
        self.matched_size[self.matched_idx] = 0;
    }

    #[inline(always)]
    pub fn run(mut self) -> unary::Result<()> {
        let len = self.out.hashes.len();

        // No row has a `row + PREFETCH_LENGTH`, so never prefetch ahead.
        if len <= PREFETCH_LENGTH {
            if len != 0 {
                self.generate_matched_slots::<false>(len);
                self.only_prefetch();
                self.swap_matched_slots();
                self.build_output::<false>()?;
            }

            return self.out.flush();
        }

        // From here on, len > PREFETCH_LENGTH, so this cannot underflow.
        let ahead_end = len - PREFETCH_LENGTH;

        // Prime the pipeline: generate first matched buffer and prefetch its arena ranges.
        let first = min(PREFETCH_LENGTH, ahead_end);
        self.generate_matched_slots::<true>(first);
        self.only_prefetch();
        self.swap_matched_slots();

        while self.row_idx < ahead_end {
            let n = min(PREFETCH_LENGTH, ahead_end - self.row_idx);

            self.generate_matched_slots::<true>(n);
            self.build_output::<true>()?;
            self.swap_matched_slots();
        }

        // Tail rows have no row a window ahead; do not prefetch ahead.
        self.generate_matched_slots::<false>(len - self.row_idx);
        self.build_output::<true>()?;
        self.swap_matched_slots();

        // Drain final generated buffer.
        self.build_output::<false>()?;

        self.out.flush()
    }
}

/// Run the pipelined probe over one batch: monomorphized per directory
/// backing so the inner loops carry no per-element dispatch.
fn run_probe_array<'a, 'b, B, S>(
    directory: &'a Directory<B>,
    out: BatchProbe<'a, 'b, S>,
) -> unary::Result<()>
where
    B: Index<usize, Output = u64> + IndexMut<usize> + PtrBuffer,
    S: Sender<RecordBatch>,
{
    ProbeArray {
        row_idx: 0,
        directory,
        matched_slots: [[(0, 0); PREFETCH_LENGTH]; 2],
        matched_size: [0; 2],
        matched_idx: 0,
        out,
    }
    .run()
}

impl Unary<RecordBatch, RecordBatch> for Probe {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> unary::Result<()> {
        // The build dataflow's collect can return while its final stolen
        // partition job is still running on another worker; the gate flips
        // when the join table is fully populated.
        while !self.table.gate.load(Ordering::Acquire) {
            std::thread::yield_now();
        }

        let build_rows = unsafe { &*self.table.build_rows.get() };
        let Some(build_rows) = build_rows else {
            // Empty build side: nothing matches, so an inner or semi join
            // emits nothing and an anti join passes every probe row through.
            // (A LEFT join should pad every probe row with nulls, but the
            // build side's schema is unknown with zero build batches; fail
            // loudly rather than emit the wrong shape.)
            match self.mode {
                JoinMode::Anti if batch.num_rows() > 0 => sender.send(batch)?,
                JoinMode::Left => {
                    return Err(crate::operations::unary::Error::Operator(
                        "LEFT join over an empty build side is not supported yet".into(),
                    ));
                }
                _ => {}
            }
            return Ok(());
        };

        if matches!(self.mode, JoinMode::Semi | JoinMode::Anti) {
            let directory = unsafe { &*self.table.directory.get() };
            let keys = unsafe { &*self.table.keys.get() };
            let rows = unsafe { &*self.table.rows.get() };
            let build_key_columns = self.build_key_columns.clone();
            return self.run_existence(
                directory,
                keys,
                rows,
                build_rows,
                &build_key_columns,
                &batch,
                sender,
            );
        }
        if matches!(self.mode, JoinMode::BuildSemi | JoinMode::BuildAnti) {
            let directory = unsafe { &*self.table.directory.get() };
            let keys = unsafe { &*self.table.keys.get() };
            let rows = unsafe { &*self.table.rows.get() };
            let build_key_columns = self.build_key_columns.clone();
            return self.run_marking(
                directory,
                keys,
                rows,
                build_rows,
                &build_key_columns,
                &batch,
            );
        }
        if self.mode == JoinMode::BuildOuter {
            let directory = unsafe { &*self.table.directory.get() };
            let keys = unsafe { &*self.table.keys.get() };
            let rows = unsafe { &*self.table.rows.get() };
            let build_key_columns = self.build_key_columns.clone();
            return self.run_build_outer(
                directory,
                keys,
                rows,
                build_rows,
                &build_key_columns,
                &batch,
                sender,
            );
        }
        if self.mode == JoinMode::Left {
            let directory = unsafe { &*self.table.directory.get() };
            let keys = unsafe { &*self.table.keys.get() };
            let rows = unsafe { &*self.table.rows.get() };
            let build_key_columns = self.build_key_columns.clone();
            return self.run_left(
                directory,
                keys,
                rows,
                build_rows,
                &build_key_columns,
                &batch,
                sender,
            );
        }

        let batch = filter_null_keys(
            batch,
            &strict_key_columns(&self.key_columns, &self.null_safe),
        );
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let key_columns: Vec<&Int64Array> = self
            .key_columns
            .iter()
            .map(|&c| batch.column(c).as_primitive::<Int64Type>())
            .collect();
        let hashes: Vec<u64> = (0..batch.num_rows())
            .map(|i| hash_key_row(&self.hash_state, &key_columns, &self.null_safe, i))
            .collect();

        let output_schema = self
            .output_schema
            .get_or_insert_with(|| {
                let fields: Vec<Field> = batch
                    .schema()
                    .fields()
                    .iter()
                    .chain(build_rows.schema().fields())
                    .map(|field| field.as_ref().clone())
                    .collect();
                Arc::new(Schema::new(fields))
            })
            .clone();

        let keys = unsafe { &*self.table.keys.get() };
        let rows = unsafe { &*self.table.rows.get() };
        let probe_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let build_sel =
            JoinPrimitiveBuilder::<UInt32Type>::new(&mut self.allocator, RECORD_BATCH_SIZE);
        let out = BatchProbe {
            keys,
            rows,
            hashes: &hashes,
            probe_batch: &batch,
            build_rows,
            output_schema: &output_schema,
            probe_sel,
            build_sel,
            output_idx: 0,
            sender,
            allocator: &mut self.allocator,
        };

        let directory = unsafe { &*self.table.directory.get() };
        if !self.use_probe_array {
            return out.run_scalar(directory);
        }
        run_probe_array(directory, out)
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        if matches!(
            self.mode,
            JoinMode::BuildSemi | JoinMode::BuildAnti | JoinMode::BuildOuter
        ) && self.table.probes_remaining.fetch_sub(1, Ordering::AcqRel) == 1
        {
            // Last probe worker: every mark is in, emit the build rows. An
            // all-empty probe input still lands here (finish always runs), so
            // the gate may not have been awaited yet.
            while !self.table.gate.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            if self.mode == JoinMode::BuildOuter {
                self.emit_unmatched_padded(sender)?;
            } else {
                self.emit_marked(sender)?;
            }
        }
        Ok(true)
    }
}
