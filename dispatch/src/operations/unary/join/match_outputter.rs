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
///
/// The accumulators start from the spec's declared fields but follow the
/// *actual* layout of what they accumulate, because a variant column's
/// physical struct type is chosen per file and only known from the data:
/// [`set_build_schema_from_rows`](Self::set_build_schema_from_rows) sets the
/// build side's physical types from the published rows once, and
/// [`switch_probe_schema`](Self::switch_probe_schema) switches the probe side
/// whenever the probe stream crosses into a differently-laid-out file. The
/// switch flushes rows buffered under the old schema first. Appending a batch
/// into an accumulator with different physical types would silently drop
/// fields for which the accumulator has no slot, so both sides must have the
/// source's types before any append.
pub(super) struct ProbeMatchOutputter {
    /// The probe accumulator's fields, then the build accumulator's, then a
    /// mark join's marker. Refreshed whenever either side's schema changes.
    output_schema: SchemaRef,
    /// Whether the output carries a mark join's trailing marker column.
    mark: bool,
    probe: BatchAccumulator,
    build: BatchAccumulator,
    allocator: SlabAllocator,
    build_rows: Arc<JoinCell<BuildRows>>,
    /// Matched `(probe row, build row id)` pairs waiting to be drained. A semi
    /// join fills only the probe indices.
    pub(super) probe_indices: Vec<u32>,
    pub(super) build_indices: Vec<u64>,
    pub(super) matched: usize,
    /// The rows of one stored build batch an unmatched-row scan keeps, as
    /// positions within that batch.
    kept_build_rows: Vec<u32>,
    /// The join's residual predicate, applied to the collected pairs at drain
    /// time, before any pair is flagged or emitted.
    pub(super) residual_filters: Option<ResidualFilter>,
    /// The probed batch's rows that matched nothing, collected by a
    /// probe-side outer or anti join while the batch is probed and emitted
    /// once it has been. See
    /// [`begin_unmatched_probe_tracking`](Self::begin_unmatched_probe_tracking)
    /// for who
    /// records into it.
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
    /// The probe rows nothing matched, accumulated by a probe-side outer or
    /// anti join until a full batch of them can be emitted (null-padded for
    /// outer; an anti join lists no build columns to pad). `None` for every
    /// other kind: an accumulator eagerly takes a pooled slab per column, a
    /// real per-query cost no other kind may pay for a path it never runs.
    unmatched_probe: Option<BatchAccumulator>,
    /// The matched-flag value the build-row scan keeps: 0 for a build-side
    /// outer or anti join, which emit the rows no pair ever flagged, and 1
    /// for a build-side semi join, which emits exactly the flagged ones.
    scan_kept_flag: u8,
}

impl ProbeMatchOutputter {
    pub(super) fn new(
        probe_fields: &[Field],
        build_fields: &[Field],
        build_rows: Arc<JoinCell<BuildRows>>,
        residual_filters: Option<ResidualFilter>,
        emits_unmatched_probe_rows: bool,
        mark: bool,
        scan_keeps_matched_build_rows: bool,
    ) -> Self {
        let mut allocator = SlabAllocator::new(false);
        let mut fields: Vec<Field> = probe_fields.iter().chain(build_fields).cloned().collect();
        if mark {
            fields.push(Field::new("mark", arrow_schema::DataType::Boolean, true));
        }
        Self {
            output_schema: Arc::new(Schema::new(fields)),
            mark,
            probe: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(probe_fields.to_vec())),
                &mut allocator,
            ),
            build: BatchAccumulator::retaining_source_buffers(
                Arc::new(Schema::new(build_fields.to_vec())),
                &mut allocator,
            ),
            unmatched_probe: emits_unmatched_probe_rows.then(|| {
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
            kept_build_rows: vec![0; RECORD_BATCH_SIZE],
            has_residual: residual_filters.is_some(),
            residual_filters,
            missed_probe_rows: Vec::new(),
            probe_row_settled: Vec::new(),
            scan_kept_flag: scan_keeps_matched_build_rows as u8,
        }
    }

    pub(super) fn has_buffered_matches(&self) -> bool {
        !self.probe.is_empty()
    }

    /// Set the build accumulator's physical types from the published build
    /// rows. A no-op once they match (or for an empty build side); called before
    /// probing touches the accumulator, which is therefore still empty when it
    /// is replaced.
    pub(super) fn set_build_schema_from_rows(&mut self) {
        let build_rows = unsafe { &*self.build_rows.get() };
        let Some(first) = build_rows.output_batches.first() else {
            return;
        };
        if schemas_have_same_types(self.build.schema(), first.schema_ref()) {
            return;
        }
        debug_assert!(
            self.build.is_empty(),
            "the build schema settles before any build row accumulates"
        );
        let schema = schema_with_types_from(self.build.schema(), first.schema_ref());
        self.build = BatchAccumulator::retaining_source_buffers(schema, &mut self.allocator);
        self.refresh_output_schema();
    }

    /// Switch the probe-side accumulators to `source_schema`, the projected
    /// schema of the batch about to be probed. A no-op while its physical types
    /// are unchanged, which is every batch until the probe stream crosses into
    /// a differently-laid-out file; on a change, rows buffered under the old
    /// schema are emitted first (matched pairs flush both sides together, since
    /// their rows are paired).
    pub(super) fn switch_probe_schema(
        &mut self,
        source_schema: &SchemaRef,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        if schemas_have_same_types(self.probe.schema(), source_schema) {
            return Ok(());
        }
        if !self.probe.is_empty() {
            self.emit(sender)?;
        }
        let schema = schema_with_types_from(self.probe.schema(), source_schema);
        self.probe =
            BatchAccumulator::retaining_source_buffers(schema.clone(), &mut self.allocator);
        if self.unmatched_probe.is_some() {
            self.emit_unmatched_probe_rows(sender)?;
            self.unmatched_probe = Some(BatchAccumulator::retaining_source_buffers(
                schema,
                &mut self.allocator,
            ));
        }
        self.refresh_output_schema();
        Ok(())
    }

    /// Rebuild [`output_schema`](Self::output_schema) from the accumulators'
    /// current layouts.
    fn refresh_output_schema(&mut self) {
        let mut fields: Vec<Field> = self
            .probe
            .schema()
            .fields()
            .iter()
            .chain(self.build.schema().fields())
            .map(|field| field.as_ref().clone())
            .collect();
        if self.mark {
            fields.push(Field::new("mark", arrow_schema::DataType::Boolean, true));
        }
        self.output_schema = Arc::new(Schema::new(fields));
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
    pub(super) fn begin_unmatched_probe_tracking(&mut self, rows: usize) {
        self.missed_probe_rows.clear();
        self.missed_probe_rows.reserve(rows);
        if self.has_residual {
            self.probe_row_settled.clear();
            self.probe_row_settled.resize(rows, 0);
        }
    }

    /// Record one probed row as matching nothing. The caller guarantees the
    /// verdict is final (see
    /// [`begin_unmatched_probe_tracking`](Self::begin_unmatched_probe_tracking)).
    #[inline(always)]
    pub(super) fn record_missed_probe_row(&mut self, probe_row: usize) {
        // Never reallocates: `begin_unmatched_probe_tracking` reserved one
        // slot per row of the batch.
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
            .expect("only a probe-side outer or anti join emits unmatched probe rows");
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

    /// Emit `probe_source` with a mark column built from the batch's missed
    /// rows: TRUE for every row the miss tracking never recorded, and for the
    /// recorded misses FALSE, or NULL when a null-keyed build row exists (the
    /// null could have matched, so the miss is three-valued unknown).
    pub(super) fn emit_marked_batch_from_misses(
        &mut self,
        probe_source: &RecordBatch,
        build_saw_null_key: bool,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let rows = probe_source.num_rows();
        let mut values = arrow_buffer::BooleanBufferBuilder::new(rows);
        values.append_n(rows, true);
        for &row in &self.missed_probe_rows {
            values.set_bit(row as usize, false);
        }
        let validity = build_saw_null_key.then(|| {
            let mut validity = arrow_buffer::BooleanBufferBuilder::new(rows);
            validity.append_n(rows, true);
            for &row in &self.missed_probe_rows {
                validity.set_bit(row as usize, false);
            }
            arrow_buffer::NullBuffer::new(validity.finish())
        });
        self.missed_probe_rows.clear();
        let marker = arrow_array::BooleanArray::new(values.finish(), validity);
        self.send_marked(probe_source, Arc::new(marker), sender)
    }

    /// Emit `probe_source` with one marker value for every row: FALSE against
    /// an empty build side, NULL for null-keyed probe rows.
    pub(super) fn emit_marked_batch_with_constant(
        &mut self,
        probe_source: &RecordBatch,
        mark: Option<bool>,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let marker = arrow_array::BooleanArray::from(vec![mark; probe_source.num_rows()]);
        self.send_marked(probe_source, Arc::new(marker), sender)
    }

    /// A probed batch can be larger than an output batch (scans deliver
    /// whatever size the source produced), and downstream operators size
    /// their accumulators to [`RECORD_BATCH_SIZE`], so the marked rows go
    /// out in chunks of at most that many.
    fn send_marked(
        &self,
        probe_source: &RecordBatch,
        marker: arrow_array::ArrayRef,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        let rows = probe_source.num_rows();
        let mut start = 0;
        while start < rows {
            let len = (rows - start).min(RECORD_BATCH_SIZE);
            let columns = probe_source
                .columns()
                .iter()
                .map(|column| column.slice(start, len))
                .chain(std::iter::once(marker.slice(start, len)))
                .collect();
            let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(len));
            sender.send(RecordBatch::try_new_with_options(
                self.output_schema.clone(),
                columns,
                &options,
            )?)?;
            start += len;
        }
        Ok(())
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
                .expect("only a probe-side outer or anti join emits unmatched probe rows");
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

    /// Emit one combined batch from the two sides' accumulated rows. The
    /// batch can reach one row short of twice [`RECORD_BATCH_SIZE`] (a
    /// drain's remainder plus the next full drain); consumers own their
    /// sizing, chunking oversized inputs themselves.
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

    /// Append the rows of one stored build batch whose flag holds the value
    /// the join's scan keeps (see [`scan_kept_flag`](Self::scan_kept_flag)),
    /// emitting if a full batch accumulated.
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
            // only if its flag holds the kept value. A stored batch holds at
            // most one output batch's worth of rows, so a pass never overfills
            // the accumulator.
            self.kept_build_rows[found] = row as u32;
            found += (build_rows.matched[first_row_id + row] == self.scan_kept_flag) as usize;
        }
        if found > 0 {
            self.build.append_batch_by_indices(
                batch,
                &self.kept_build_rows[..found],
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
        const STOP_AFTER_FIRST_MATCH: bool,
        const TRACK_UNMATCHED_PROBE_ROWS: bool,
        const DISCARD_MATCHED_PAIRS: bool,
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
        if TRACK_UNMATCHED_PROBE_ROWS && self.has_residual {
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
        if DISCARD_MATCHED_PAIRS {
            // An anti or build-side semi join's matched pairs are not output
            // rows; the settling and flagging above is all a match
            // contributes.
            self.matched = 0;
            return Ok(());
        }
        self.probe.append_batch_by_indices(
            probe_source,
            &self.probe_indices[..self.matched],
            &mut self.allocator,
        );
        // The first-match path records no build row ids, and every join kind
        // using it has an empty build-side output projection.
        if !STOP_AFTER_FIRST_MATCH {
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

/// Whether `current` and `source` have the same field data types. Names,
/// nullability, and metadata deliberately do not participate: the accumulator
/// retains those from the join's declared schema (an outer join declares
/// columns nullable for padding even when the stored data has no nulls).
fn schemas_have_same_types(current: &SchemaRef, source: &SchemaRef) -> bool {
    current
        .fields()
        .iter()
        .zip(source.fields())
        .all(|(current, source)| current.data_type() == source.data_type())
}

/// Returns `declared` with each field's data type replaced by `source`'s while
/// preserving the declared name, nullability, and metadata.
fn schema_with_types_from(declared: &SchemaRef, source: &SchemaRef) -> SchemaRef {
    let fields: Vec<Field> = declared
        .fields()
        .iter()
        .zip(source.fields())
        .map(|(declared, source)| {
            declared
                .as_ref()
                .clone()
                .with_data_type(source.data_type().clone())
        })
        .collect();
    Arc::new(Schema::new(fields))
}
