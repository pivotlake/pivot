//! Buffers verified join matches and emits full output batches.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use arrow_array::{RecordBatch, new_null_array};
use arrow_schema::{Field, Schema, SchemaRef};

use crate::RECORD_BATCH_SIZE;
use crate::arrays::accumulator::BatchAccumulator;
use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::join::JoinCell;
use crate::operations::unary::join::build_rows::{self, BuildRows};
use crate::operations::unary::join::residual_filter::ResidualFilter;

/// Owns the buffered matches and accumulators that produce the join's output.
/// Probe and build columns accumulate separately because their rows come from
/// different sources; every emitted batch splices them under one schema.
pub(super) struct ProbeMatchOutputter {
    /// The probe columns listed in the output, then the build columns.
    output_schema: SchemaRef,
    probe: BatchAccumulator,
    build: BatchAccumulator,
    allocator: SlabAllocator,
    build_rows: Arc<JoinCell<BuildRows>>,
    /// Matched `(probe row, build row id)` pairs waiting to be drained. A semi
    /// join fills only the probe indices.
    pub(super) probe_indices: Vec<u32>,
    pub(super) build_indices: Vec<u32>,
    pub(super) matched: usize,
    /// The join's residual predicate, applied to the collected pairs at drain
    /// time, before any pair is flagged or emitted.
    pub(super) residual_filters: Option<ResidualFilter>,
    /// The probed batch's rows that matched nothing, collected by a
    /// probe-side outer join while the batch is probed and padded out once it
    /// has been. See [`begin_probe_outer_batch`](Self::begin_probe_outer_batch)
    /// for who records into it.
    missed_probe_rows: Vec<u32>,
    /// Whether the join carries a residual predicate, which is what forces
    /// the settled-flag supplement below. Cached off `residual_filters` so
    /// the hot recording sites read a plain bool.
    has_residual: bool,
    /// One flag per row of the batch being probed, set when the row's fate
    /// becomes known: a miss recorded inline, or a pair the residual let
    /// through at drain time. Rows still unsettled after the batch's last
    /// drain had candidates the residual rejected every one of, the one kind
    /// of miss no inline site can see. Maintained (and allocated) only when
    /// a residual exists.
    probe_row_settled: Vec<u8>,
    /// The probe rows nothing matched, accumulated by a probe-side outer join
    /// until a full batch of them can be emitted null-padded. `None` for every
    /// other kind: an accumulator eagerly takes a pooled slab per column, a
    /// real per-query cost no other kind may pay for a path it never runs.
    unmatched_probe: Option<BatchAccumulator>,
}

impl ProbeMatchOutputter {
    pub(super) fn new(
        probe_fields: &[Field],
        build_fields: &[Field],
        build_rows: Arc<JoinCell<BuildRows>>,
        residual_filters: Option<ResidualFilter>,
        outer_join_probe_side: bool,
    ) -> Self {
        let mut allocator = SlabAllocator::new(false);
        let fields: Vec<Field> = probe_fields.iter().chain(build_fields).cloned().collect();
        Self {
            output_schema: Arc::new(Schema::new(fields)),
            probe: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(probe_fields.to_vec())),
                &mut allocator,
            ),
            build: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(build_fields.to_vec())),
                &mut allocator,
            ),
            unmatched_probe: outer_join_probe_side.then(|| {
                BatchAccumulator::retaining_source_buffers(
                    Arc::new(Schema::new(probe_fields.to_vec())),
                    &mut allocator,
                )
            }),
            allocator,
            build_rows,
            probe_indices: vec![0; RECORD_BATCH_SIZE],
            build_indices: vec![0; RECORD_BATCH_SIZE],
            matched: 0,
            has_residual: residual_filters.is_some(),
            residual_filters,
            missed_probe_rows: Vec::new(),
            probe_row_settled: Vec::new(),
        }
    }

    pub(super) fn has_buffered_matches(&self) -> bool {
        !self.probe.is_empty()
    }

    /// Start miss tracking for the next probed batch of `rows` rows.
    ///
    /// Most misses are definitive the moment the probe passes the row: the
    /// bloom check rejected it (no candidates at all), or its arena range
    /// verified nothing (only false candidates). A residual cannot change
    /// either verdict, it only ever rejects pairs, so those two sites always
    /// record straight into `missed_probe_rows`. What a residual adds is one
    /// more kind of miss, a row whose verified pairs are all rejected at
    /// drain time; the settled flags exist to find those rows (see
    /// [`probe_row_settled`](Self::probe_row_settled)), and a join without a
    /// residual never allocates them.
    pub(super) fn begin_probe_outer_batch(&mut self, rows: usize) {
        self.missed_probe_rows.clear();
        self.missed_probe_rows.reserve(rows);
        if self.has_residual {
            self.probe_row_settled.clear();
            self.probe_row_settled.resize(rows, 0);
        }
    }

    /// Record one probed row as matching nothing. The caller guarantees the
    /// verdict is final (see
    /// [`begin_probe_outer_batch`](Self::begin_probe_outer_batch)).
    #[inline(always)]
    pub(super) fn record_missed_probe_row(&mut self, probe_row: usize) {
        // Never reallocates: `begin_probe_outer_batch` reserved one slot per
        // row of the batch.
        self.missed_probe_rows.push(probe_row as u32);
        if self.has_residual {
            self.probe_row_settled[probe_row] = 1;
        }
    }

    /// Emit one batch of probe rows that matched nothing: the accumulated
    /// probe columns, and an all-null column of the right shape for each
    /// build column.
    pub(super) fn emit_unmatched_probe_rows(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let unmatched_probe = self
            .unmatched_probe
            .as_mut()
            .expect("only a probe-side outer join pads probe rows");
        if unmatched_probe.is_empty() {
            return Ok(());
        }
        let probe_part = unmatched_probe.take_batch(&mut self.allocator)?;
        let rows = probe_part.num_rows();
        let columns = probe_part
            .columns()
            .iter()
            .cloned()
            .chain(
                self.build
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| new_null_array(field.data_type(), rows)),
            )
            .collect();
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(rows));
        sender.send(RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            columns,
            &options,
        )?)?;
        Ok(())
    }

    /// Finish unmatched-row processing once the probed batch is fully probed
    /// and drained. The inline sites recorded most misses; under a residual,
    /// the rows still unsettled, whose every pair the residual rejected, join
    /// them here.
    pub(super) fn finish_probe_outer_batch(
        &mut self,
        probe_source: &RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        if self.has_residual {
            for row in 0..self.probe_row_settled.len() {
                if self.probe_row_settled[row] == 0 {
                    self.missed_probe_rows.push(row as u32);
                }
            }
        }
        self.append_missed_probe_rows(probe_source, sender)
    }

    /// Append every row of `probe_source` as missed: for the probe rows a
    /// probe-side outer join never probes (null-keyed, or an empty build side).
    pub(super) fn append_probe_rows_padded(
        &mut self,
        probe_source: &RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        self.missed_probe_rows.clear();
        self.missed_probe_rows
            .extend(0..probe_source.num_rows() as u32);
        self.append_missed_probe_rows(probe_source, sender)
    }

    /// Append the collected `missed_probe_rows` of `probe_source` to the
    /// padding accumulator, emitting whenever a full batch gathers. Unlike
    /// the build-side pass, whose stored batches are pre-chunked to one
    /// output batch's worth, a probed batch can miss more rows than the
    /// accumulator has room for, so the append goes in capacity-sized chunks.
    fn append_missed_probe_rows(
        &mut self,
        probe_source: &RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let mut missed = std::mem::take(&mut self.missed_probe_rows);
        let mut remaining = missed.as_slice();
        while !remaining.is_empty() {
            let unmatched_probe = self
                .unmatched_probe
                .as_mut()
                .expect("only a probe-side outer join pads probe rows");
            let room = unmatched_probe.capacity() - unmatched_probe.len();
            let (chunk, rest) = remaining.split_at(room.min(remaining.len()));
            unmatched_probe.append_batch_by_indices(probe_source, chunk, &mut self.allocator);
            if unmatched_probe.has_full_batch() {
                self.emit_unmatched_probe_rows(sender)?;
            }
            remaining = rest;
        }
        missed.clear();
        self.missed_probe_rows = missed;
        Ok(())
    }

    /// Emit one combined batch from the two sides' accumulated rows.
    pub(super) fn emit(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<()> {
        let probe_part = self.probe.take_batch(&mut self.allocator)?;
        let build_part = self.build.take_batch(&mut self.allocator)?;
        let columns = probe_part
            .columns()
            .iter()
            .chain(build_part.columns())
            .cloned()
            .collect();
        let options =
            arrow_array::RecordBatchOptions::new().with_row_count(Some(probe_part.num_rows()));
        sender.send(RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            columns,
            &options,
        )?)?;
        Ok(())
    }

    /// Emit one batch of build rows that matched nothing: the accumulated build
    /// columns, and an all-null column of the right shape for each probe column.
    pub(super) fn emit_unmatched_build_rows(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        if self.build.is_empty() {
            return Ok(());
        }
        let build_part = self.build.take_batch(&mut self.allocator)?;
        let rows = build_part.num_rows();
        let columns = self
            .probe
            .schema()
            .fields()
            .iter()
            .map(|field| new_null_array(field.data_type(), rows))
            .chain(build_part.columns().iter().cloned())
            .collect();
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(rows));
        sender.send(RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            columns,
            &options,
        )?)?;
        Ok(())
    }

    pub(super) fn append_unmatched_build_rows(
        &mut self,
        batch: &RecordBatch,
        first_row_id: usize,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let build_rows = unsafe { &*self.build_rows.get() };
        let mut found = 0;
        for row in 0..batch.num_rows() {
            // Branchless, as in the match collector: write the row and keep it
            // only if its flag is still clear. A stored batch holds at most
            // one output batch's worth of rows, so a pass never overfills the
            // accumulator.
            self.build_indices[found] = row as u32;
            found += (build_rows.matched[first_row_id + row] == 0) as usize;
        }
        if found > 0 {
            self.build.append_batch_by_indices(
                batch,
                &self.build_indices[..found],
                &mut self.allocator,
            );
        }
        if self.build.has_full_batch() {
            self.emit_unmatched_build_rows(sender)?;
        }
        Ok(())
    }

    /// Append the collected pairs to the output accumulators, emitting if a
    /// full batch accumulated.
    #[inline(never)]
    pub(super) fn drain<
        const OUTER_JOIN_BUILD_SIDE: bool,
        const SEMI_PROBE_SIDE: bool,
        const OUTER_JOIN_PROBE_SIDE: bool,
    >(
        &mut self,
        probe_source: &RecordBatch,
        probe_batch: &RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        if self.matched == 0 {
            return Ok(());
        }
        let build_rows = unsafe { &*self.build_rows.get() };
        // The residual runs first: a pair it rejects is not a match, so it
        // must neither flag its build row nor reach the output.
        if let Some(residual_filters) = &mut self.residual_filters {
            self.matched = residual_filters.filter_combined_batch(
                probe_batch,
                &build_rows.batches,
                &mut self.probe_indices,
                &mut self.build_indices,
                self.matched,
            )?;
            if self.matched == 0 {
                return Ok(());
            }
        }
        if OUTER_JOIN_PROBE_SIDE && self.has_residual {
            // These indices are the pairs the residual just let through:
            // their rows matched, so settle them. A row none of whose pairs
            // ever survives a drain stays unsettled and pads as a miss.
            for &row in &self.probe_indices[..self.matched] {
                self.probe_row_settled[row as usize] = 1;
            }
        }
        if OUTER_JOIN_BUILD_SIDE {
            // These indices are the verified matches, so flagging them here
            // keeps the match loop itself untouched. Relaxed because the flags
            // are only read after a barrier that orders them, and atomic only
            // because peer workers write the same bytes concurrently.
            for &row in &self.build_indices[..self.matched] {
                let flag = build_rows.matched.ptr_at_index(row as usize);
                unsafe { AtomicU8::from_ptr(flag) }.store(1, Ordering::Relaxed);
            }
        }
        self.probe.append_batch_by_indices(
            probe_source,
            &self.probe_indices[..self.matched],
            &mut self.allocator,
        );
        // A semi join has no build columns and buffered no build rows: its
        // build side only ever contributes its (empty) column list on emit.
        if !SEMI_PROBE_SIDE {
            self.build.append_from_batches(
                &build_rows.output_columns,
                build_rows::row_id_shift(),
                &self.build_indices[..self.matched],
                &mut self.allocator,
            );
        }
        self.matched = 0;
        if self.probe.has_full_batch() {
            self.emit(sender)?;
        }
        Ok(())
    }
}
