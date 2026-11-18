//! The operator that holds the input files' rows and plans their merge.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::RecordBatch;
use dispatch::{
    BuildContext, Built, BuiltUnary, ChannelInputSender, DataFlowDispatcher, KWayMergePlan,
    MapFactory, MergeRun, OperatorFactory, OperatorSpec, OrderBy, Projection, Sender, Unary,
    UnaryFactory, UnaryResult, WorkStatus, channel_input, count_rows_not_after, is_batch_sorted,
    stealable, stealable_fifo, to_single_worker_mpsc,
};

use super::super::RANGES_DECODED_AT_A_TIME_PER_FILE;
use super::super::input_batch::{DecodedBatch, InputBatch};
use super::super::memory_budget::MemoryBudget;
use super::decode_gate::{DecodeGate, DecodeReleases};
use super::input_file_rows::InputFileRows;
use super::{MergeJob, MergedRun, RowsToMerge};
use crate::reading::DECODE_RANGE_ROWS;
use crate::types::metadata::QueryRowGroupMetadata;
use crate::writing::error::{WriteError, WriteResult};
use crate::{DecodeRange, ParquetTable};

/// Holds the rows of the input files that are read and not merged yet, asks
/// for each file's next row groups ahead of the merge, releases their rows to
/// be decoded a range at a time as the merge is about to need them, and
/// plans the merge of the rows that are safe to merge.
///
/// Every batch read is routed to one worker, the compaction worker. Its
/// planner [takes the table's files](Self::load_input_files); the other
/// workers' planners have no files, receive nothing and have nothing to do.
pub(in crate::writing::compaction) struct MergePlanner {
    table: Arc<ParquetTable>,
    /// The sort columns, as columns of the table. Empty for a table without
    /// sort columns, whose row groups go out as they are, one after the
    /// other.
    key_columns: Arc<[usize]>,
    /// How sort columns projected out of a batch are ordered.
    order_by: Arc<[OrderBy]>,
    files: Vec<InputFileRows>,
    /// Which of `files` each of the table's row groups belongs to.
    file_of_row_group: Vec<usize>,
    /// Where the requests for row groups go.
    requests: Arc<ChannelInputSender<QueryRowGroupMetadata>>,
    /// Where the planner says which rows may be decoded.
    releases: Arc<DecodeReleases>,
    /// The planned jobs not sent yet, in output order.
    jobs: VecDeque<MergeJob>,
    next_sequence: usize,
    /// How many rows the jobs sent so far hold.
    rows_sent: usize,
    row_count: usize,
    /// What the memory of the decoded rows is reserved from.
    memory_budget: Arc<MemoryBudget>,
}

impl MergePlanner {
    /// The rows of `table` in output order, as runs of rows of the batches
    /// they were read in. The table's files are each sorted by
    /// `sort_column_names`; without sort columns the order is every row
    /// group in turn.
    pub(in crate::writing::compaction) fn build_merged_runs_spec(
        dispatcher: &DataFlowDispatcher,
        table: &Arc<ParquetTable>,
        sort_column_names: Arc<[String]>,
        memory_budget: &Arc<MemoryBudget>,
        compaction_worker: usize,
    ) -> OperatorSpec<MergedRun, impl OperatorFactory<MergedRun> + 'static + use<>> {
        let worker_count = dispatcher.worker_count();
        let key_columns: Arc<[usize]> = sort_column_names
            .iter()
            .map(|name| {
                table
                    .schema()
                    .index_of(name)
                    .expect("sort columns name declared columns")
            })
            .collect();

        // Sized for every row group, so a request never finds the queue full.
        let request_capacity = table.row_groups().len().max(1);
        let (request_sender, requests) =
            channel_input::<QueryRowGroupMetadata>(dispatcher, request_capacity, Box::new(|| {}));
        // The compaction worker's planner sends its requests for row groups
        // through `request_sender`, and their batches come back to it.
        let request_sender = Arc::new(request_sender);
        let releases = DecodeReleases::new(dispatcher, table.row_groups().len(), compaction_worker);
        let planners = (0..worker_count)
            .map(|worker| {
                let mut planner = MergePlanner {
                    table: table.clone(),
                    order_by: (0..key_columns.len())
                        .map(|column| OrderBy::new(column, false, true))
                        .collect(),
                    key_columns: key_columns.clone(),
                    files: Vec::new(),
                    file_of_row_group: Vec::new(),
                    requests: request_sender.clone(),
                    releases: releases.clone(),
                    jobs: VecDeque::new(),
                    next_sequence: 0,
                    rows_sent: 0,
                    row_count: 0,
                    memory_budget: memory_budget.clone(),
                };
                if worker == compaction_worker {
                    planner.load_input_files();
                }
                planner
            })
            .collect();
        let topology = dispatcher.topology();
        let projection = Projection::all(table.schema().fields().len());
        let fetched =
            crate::reading::fetch_row_group_buffers(requests, &projection, table.get_read_medium());
        let released = crate::reading::cut_decode_ranges(
            fetched,
            projection,
            Arc::new(Vec::new()),
            // Row groups are claimed by explicit request, not by an injector,
            // so this count is never consulted.
            Arc::new(AtomicUsize::new(0)),
        )
        .chain(
            to_single_worker_mpsc::<DecodeRange>(worker_count, compaction_worker)
                .into_iter()
                .collect(),
            DecodeGate::factories(&releases, worker_count),
        );
        // The decoder appends the columns that say which rows of which row
        // group a batch holds. A worker runs what follows its own work
        // first, so the worker that takes a batch here is normally the one
        // that decoded it.
        crate::reading::decode_ranges(released, dispatch::RECORD_BATCH_SIZE, true)
            .chain(
                stealable::<RecordBatch>(topology).into_iter().collect(),
                (0..worker_count)
                    .map(|worker| {
                        MapFactory(move |batch: RecordBatch| Ok(DecodedBatch::new(batch, worker)))
                    })
                    .collect(),
            )
            .chain(
                to_single_worker_mpsc::<DecodedBatch>(worker_count, compaction_worker)
                    .into_iter()
                    .collect(),
                planners,
            )
            .chain(
                stealable_fifo::<MergeJob>(topology).into_iter().collect(),
                (0..worker_count)
                    .map(|_| {
                        let mut allocator = None;
                        MapFactory(move |job: MergeJob| job.merge_into_run(&mut allocator))
                    })
                    .collect(),
            )
    }

    /// Take the table's files as the files to merge, ask for the first row
    /// groups of each, and release each file's first rows to be decoded.
    fn load_input_files(&mut self) {
        let mut previous_group_in_file = None;
        for (group, metadata) in self.table.row_groups().iter().enumerate() {
            // A file's row groups are listed together in file order, so a
            // position that does not grow starts the next file. Without
            // sort columns nothing is merged, and the row groups are read
            // as if they were one file's.
            let starts_file = self.files.is_empty()
                || (!self.key_columns.is_empty()
                    && previous_group_in_file
                        .is_none_or(|previous| metadata.file_row_group_idx <= previous));
            previous_group_in_file = Some(metadata.file_row_group_idx);
            if starts_file {
                self.files.push(InputFileRows::default());
            }
            self.file_of_row_group.push(self.files.len() - 1);
            // A row group without rows decodes to nothing, so no batch would
            // ever say it arrived.
            if metadata.num_rows > 0 {
                self.files.last_mut().unwrap().add_row_group(group);
                self.row_count += metadata.num_rows as usize;
            }
        }
        self.read_ahead();
    }

    /// Ask for each file's next row groups, release the next ranges of rows
    /// of the files the merge is waiting for, and end the requests once the
    /// last one is out.
    ///
    /// The merge goes on only with rows of every file, so it is waiting for
    /// the files that have no rows ready to merge. A file with rows ready
    /// gets no further range: its rows would only take memory until the
    /// merge gets to them. A file with none gets its next ranges, up to
    /// [`RANGES_DECODED_AT_A_TIME_PER_FILE`] being decoded at a time, while
    /// the memory budget is not full. A range's memory is reserved when its
    /// rows arrive, so the budget can be passed by the ranges being decoded
    /// when it fills.
    ///
    /// One range is released although the budget is full: that of a file
    /// with no rows ready or being decoded, while no output row group is
    /// being encoded or written. Then nothing is on its way to give memory
    /// back.
    fn read_ahead(&mut self) {
        for file in &mut self.files {
            file.request_row_groups(&self.table, &self.requests);
            while !file.has_ready_rows() && file.count_next_range_rows() > 0 {
                let rows_being_decoded = file.count_rows_being_decoded();
                if rows_being_decoded + file.count_next_range_rows()
                    > RANGES_DECODED_AT_A_TIME_PER_FILE * DECODE_RANGE_ROWS as usize
                {
                    break;
                }
                let merge_cannot_go_on =
                    rows_being_decoded == 0 && !self.memory_budget.has_row_group_in_flight();
                if self.memory_budget.is_full() && !merge_cannot_go_on {
                    break;
                }
                file.release_next_range(&self.releases);
                // Releasing rows makes room to ask for the next row group.
                file.request_row_groups(&self.table, &self.requests);
            }
        }
        if !self
            .files
            .iter()
            .any(InputFileRows::has_unrequested_row_groups)
        {
            self.requests.close();
        }
    }

    /// The sort columns of the last arrived row that sorts first among the
    /// files with rows still to come, as a one-row batch: no row not read
    /// yet sorts before it. `None` when every file has all arrived, so every
    /// ready row can be merged.
    fn find_last_safe_key(&self) -> WriteResult<Option<RecordBatch>> {
        let mut bound: Option<RecordBatch> = None;
        for file in self
            .files
            .iter()
            .filter(|file| !file.has_all_rows_arrived())
        {
            let last_key = file.last_key().expect("the file has ready rows").clone();
            bound = Some(match bound {
                None => last_key,
                Some(current) => {
                    let sorts_first = count_rows_not_after(
                        &self.order_by,
                        std::slice::from_ref(&last_key),
                        &current,
                        0,
                    )? == 1;
                    if sorts_first { last_key } else { current }
                }
            });
        }
        Ok(bound)
    }

    /// Plan the merge of every ready row that no row still to come can
    /// precede, as merge jobs. Returns whether there was anything to plan.
    ///
    /// The steps: check that the merge can run, take the rows that are safe
    /// to merge out of the files, have the merge of their sort columns
    /// planned, and queue one job per part of that plan. A table without
    /// sort columns has nothing to merge, and its rows are taken as they
    /// are.
    fn plan_merge_of_safe_rows(&mut self) -> WriteResult<bool> {
        if !self.has_rows_of_every_file() {
            return Ok(false);
        }
        let (sources, planned) = if self.key_columns.is_empty() {
            self.take_rows_in_file_order()?
        } else {
            let (sources, key_runs) = self.take_safe_rows()?;
            (sources, self.plan_rows_to_merge(key_runs)?)
        };
        self.queue_merge_jobs(sources, planned);
        self.read_ahead();
        Ok(true)
    }

    /// Whether the merge can run: there are rows ready, and no file that has
    /// rows still to come is without ready rows. The merge takes the next
    /// row from whichever file has the smallest, so it cannot go on while it
    /// does not know one file's next row.
    fn has_rows_of_every_file(&self) -> bool {
        let awaits_rows = self
            .files
            .iter()
            .any(|file| !file.has_all_rows_arrived() && !file.has_ready_rows());
        !awaits_rows && self.files.iter().any(InputFileRows::has_ready_rows)
    }

    /// For a table without sort columns: take every ready row, in the order
    /// the rows were read. Returns the batches taken and, for each, that its
    /// rows go out as they are.
    fn take_rows_in_file_order(&mut self) -> WriteResult<(Vec<InputBatch>, Vec<RowsToMerge>)> {
        let taken = self.files[0].take_rows_not_after(None, &self.order_by)?;
        let mut sources = Vec::with_capacity(taken.len());
        let mut planned = Vec::with_capacity(taken.len());
        for (rows, _) in taken {
            planned.push(RowsToMerge::Consecutive {
                source: sources.len() as u32,
                first_row: 0,
                row_count: rows.row_count(),
            });
            sources.push(rows);
        }
        Ok((sources, planned))
    }

    /// Take out of every file the rows that are safe to merge: those whose
    /// sort columns sort at or before the last safe key (see
    /// [`find_last_safe_key`](Self::find_last_safe_key)).
    ///
    /// Returns the batches taken, which are the merge's source batches, and
    /// the merge's input: one run per file, of the batches' sort columns.
    /// Each key is labelled with the number of its source batch and its row
    /// there, so that the merged keys say which row each output row is.
    fn take_safe_rows(&mut self) -> WriteResult<(Vec<InputBatch>, Vec<MergeRun>)> {
        let bound = self.find_last_safe_key()?;
        let mut sources: Vec<InputBatch> = Vec::new();
        let mut key_runs = Vec::new();
        for file in &mut self.files {
            let taken = file.take_rows_not_after(bound.as_ref(), &self.order_by)?;
            if taken.is_empty() {
                continue;
            }
            let mut run_keys = Vec::with_capacity(taken.len());
            for (rows, keys) in taken {
                run_keys.push(MergeJob::label_keys(keys, sources.len() as u32)?);
                sources.push(rows);
            }
            key_runs.push(MergeRun::new(run_keys, 0));
        }
        Ok((sources, key_runs))
    }

    /// Have the merge of `key_runs` planned, and say what each part of the
    /// plan leaves to do.
    ///
    /// When the files' rows interleave, the plan is slices of the merge that
    /// any worker can do, each on its own. When they do not, there is
    /// nothing to merge: the plan is the key batches in output order, and
    /// each is consecutive rows of one source batch. A plan of nothing means
    /// a file's rows are out of order: the file whose last ready key is the
    /// last safe key always has its own rows at or before it.
    fn plan_rows_to_merge(&self, key_runs: Vec<MergeRun>) -> WriteResult<Vec<RowsToMerge>> {
        Ok(
            match KWayMergePlan::try_new(self.order_by.clone(), key_runs, 1)? {
                KWayMergePlan::Empty => return Err(WriteError::UnsortedCompactionInput),
                KWayMergePlan::Identity(output) => output
                    .into_batches()
                    .into_iter()
                    .map(|batch| batch.into_batch())
                    .filter(|keys| keys.num_rows() > 0)
                    .map(|keys| MergeJob::read_consecutive_rows(&keys))
                    .collect(),
                KWayMergePlan::Parallel(tasks) => {
                    tasks.into_iter().map(RowsToMerge::Slice).collect()
                }
            },
        )
    }

    /// Queue one merge job per part of a plan, numbered in output order.
    /// The jobs of one plan share its source batches.
    fn queue_merge_jobs(&mut self, sources: Vec<InputBatch>, planned: Vec<RowsToMerge>) {
        let sources: Arc<[InputBatch]> = sources.into();
        for rows in planned {
            self.jobs.push_back(MergeJob {
                sequence: self.next_sequence,
                sources: sources.clone(),
                rows,
            });
            self.next_sequence += 1;
        }
    }

    /// Send out the planned jobs, planning the next round whenever they are
    /// all out. Returns whether any job went out.
    fn plan_and_send_merge_jobs(&mut self, sender: &mut dyn Sender<MergeJob>) -> UnaryResult<bool> {
        let mut sent = false;
        loop {
            while let Some(job) = self.jobs.front() {
                self.rows_sent += job.row_count();
                sender.send(self.jobs.pop_front().unwrap())?;
                sent = true;
            }
            if !self.plan_merge_of_safe_rows()? {
                return Ok(sent);
            }
        }
    }
}

impl UnaryFactory<DecodedBatch, MergeJob> for MergePlanner {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<DecodedBatch, MergeJob> for MergePlanner {
    fn consume(
        &mut self,
        decoded: DecodedBatch,
        sender: &mut dyn Sender<MergeJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if decoded.row_count() > 0 {
            // The rows are in memory from here on, at this size, until the
            // row groups they go to are encoded.
            let batch = decoded.into_input_batch(&self.memory_budget);
            if !self.key_columns.is_empty() {
                // A file whose rows are out of sort order cannot be merged.
                let keys = batch
                    .batch
                    .project(&self.key_columns)
                    .map_err(WriteError::from)?;
                if !is_batch_sorted(&self.order_by, &keys).map_err(WriteError::from)? {
                    return Err(WriteError::UnsortedCompactionInput.into());
                }
            }
            let file = self.file_of_row_group[batch.row_group as usize];
            self.files[file].add_batch(batch, &self.key_columns, &self.order_by)?;
            self.read_ahead();
        }
        self.plan_and_send_merge_jobs(sender)?;
        Ok(())
    }

    fn run(
        &mut self,
        sender: &mut dyn Sender<MergeJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        // Every worker has a planner, and only the compaction worker's has
        // files. The others must not read ahead: with no file left to ask
        // for, they would end the requests the one with files still sends.
        if self.files.is_empty() {
            return Ok(WorkStatus::Pending);
        }
        // Memory given back since the last time may have made room.
        self.read_ahead();
        Ok(if self.plan_and_send_merge_jobs(sender)? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// Rows not released for want of room in the memory budget are still to
    /// be merged.
    fn has_pending_work(&self) -> bool {
        self.rows_sent < self.row_count
    }
}
