//! Reading only some rows of a scan's files: a stage ahead of the fetchers
//! that asks, file by file, which rows are worth reading.
//!
//! ```text
//!   files of the scan ─▶ RowSelectionStage ─▶ fetch ─▶ decompress ─▶ cut ─▶ decode
//!                          │
//!                          │ per file: look it up once
//!                          │ per row group of the file:
//!                          ├ no row      dropped, nothing of it is read
//!                          ├ some rows   handed on, and only those rows are decoded
//!                          └ every row   handed on whole
//! ```
//!
//! What decides is the scan's [`RowSelector`], which answers for a whole file
//! at a time: whatever it consults (an index stored beside the data file,
//! say) covers the file, so one lookup serves every row group of it. The
//! lookup may need reads. The selector issues them through the
//! [`OperatorIO`] it is handed and says which read it waits on; the stage
//! keeps the file and what the selector was in the middle of, and hands both
//! back with the read's response.
//!
//! Rows the selector names are read as they are: nothing above the scan needs
//! to check them again, and rows the table format deleted must not be among
//! them.
//!
//! ## Which file next, and which row group
//!
//! The files wait in one queue every worker's stage takes from
//! ([`ScannedFiles`]), ordered so that a scan uses what is in memory before
//! its own reads can push it out: first the files whose lookup has most in
//! memory already, then the files with most of their own pages in the
//! decompressed cache, then the largest.
//!
//! A looked-up file's row groups are handed to the fetchers in the order a
//! scan without a selector hands them out (most cached pages first, then
//! largest), each to the NUMA node such a scan assigns it to
//! ([`SelectedRowGroupQueues`]). Which rows of a row group are read is worked
//! out only as it is handed over, so what is held for a file is the lookup's
//! answer, never the row numbers of all its row groups at once.
//!
//! ## Staying just ahead of the fetchers
//!
//! A fetcher asks for a row group's read ahead of its worker's CPU work, so
//! the stage works at the same two moments a fetcher does: whenever the
//! worker saturates the available IO, and on its CPU turn.
//!
//! - It hands over a row group only while none is waiting for a fetcher of
//!   that row group's node, so the row groups decided and not yet asked for
//!   are at most one per node.
//! - It looks up another file only while the worker's read-ahead budget for
//!   the medium the table is read from has room (see
//!   [`dispatch::io::ReadAheadBudget`]). A file counts its projected bytes
//!   against that budget while it is being looked up, which is what its row
//!   groups will count once fetchers ask for them: the files in lookup are
//!   about as many as it takes to fill the budget. A stage also keeps no more
//!   files open, in lookup or part handed over, than a worker keeps reads in
//!   flight for file metadata ([`http_readahead`](crate::http_readahead)).

use crate::RowGroupFilter;
use crate::reading::fetching::affinity_node;
use crate::types::metadata::{QueryRowGroupMetadata, RowGroupMetadata, RowSelection};
use crate::types::projection::Projection;
use crate::types::table::ParquetTable;
use crossbeam_deque::{Injector, Steal};
use dispatch::io::{
    OpenFile, OperatorIO, ReadAheadCharge, ReadMedium, ReadRequestId, ReadResponse,
};
use dispatch::{
    BuildContext, Built, BuiltUnary, Receiver, RootChannelFactory, Sender, Unary, UnaryFactory,
    UnaryResult, WorkStatus,
};
use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Decides which rows of a scan's files are read. See the
/// [module docs](self).
pub trait RowSelector {
    /// A looked-up file's answer: what the rows of each of its row groups
    /// are selected from.
    type FileRows;
    /// What a lookup waiting on a read keeps until the response is in.
    type Pending;

    /// How many bytes of what looking `file` up reads are in memory already.
    /// A scan looks up first the files with most.
    fn count_bytes_in_memory(&self, file: &OpenFile) -> usize;

    /// Look `file` up, as far as that goes without waiting.
    fn look_up_file(
        &mut self,
        file: &OpenFile,
        io: &mut OperatorIO,
    ) -> UnaryResult<FileLookupProgress<Self::FileRows, Self::Pending>>;

    /// Carry on a lookup with the `response` to the read `pending` was
    /// waiting on.
    fn resume_file_lookup(
        &mut self,
        pending: Self::Pending,
        response: ReadResponse,
        io: &mut OperatorIO,
    ) -> UnaryResult<FileLookupProgress<Self::FileRows, Self::Pending>>;

    /// The rows of `row_group` to read, given the answer for its file:
    /// [`RowSelection::All`] reads it whole, and an empty list of indices
    /// drops it.
    fn select_rows(
        &self,
        file_rows: &Self::FileRows,
        row_group: &RowGroupMetadata,
    ) -> UnaryResult<RowSelection>;
}

/// How far a selector has got with one file.
pub enum FileLookupProgress<FileRows, Pending> {
    /// The selector waits on `read`. Its response goes to
    /// [`RowSelector::resume_file_lookup`] together with `pending`.
    Reading {
        read: ReadRequestId,
        pending: Pending,
    },
    LookedUp(FileRows),
}

/// The row groups a scan reads of one data file, in the order they are handed
/// to the fetchers. Never empty.
pub(crate) struct ScannedFile {
    row_groups: VecDeque<QueryRowGroupMetadata>,
}

/// The files of a scan still to be looked up, in the order they are taken,
/// shared by every worker's [`RowSelectionStage`]. See the
/// [module docs](self) for the order.
#[derive(Clone)]
pub(crate) struct ScannedFiles {
    files: Arc<Injector<ScannedFile>>,
}

impl ScannedFiles {
    /// The files of `table`, of which a scan reads the columns of
    /// `projection` and `row_selector` decides the rows.
    pub(crate) fn of_table<Selector: RowSelector>(
        table: &ParquetTable,
        projection: &Projection,
        row_selector: &Selector,
    ) -> Self {
        // A row group's cached pages and projected size, each read once: the
        // cache counters move while the scan is planned.
        struct RowGroupWeight {
            index: usize,
            cached_pages: usize,
            projected_bytes: i64,
        }
        let mut weights_by_file: Vec<Vec<RowGroupWeight>> = Vec::new();
        let mut position_of_file: HashMap<&OpenFile, usize> = HashMap::new();
        for (index, row_group) in table.row_groups.iter().enumerate() {
            let position = *position_of_file
                .entry(&row_group.open_file)
                .or_insert_with(|| {
                    weights_by_file.push(Vec::new());
                    weights_by_file.len() - 1
                });
            weights_by_file[position].push(RowGroupWeight {
                index,
                cached_pages: row_group.live_decompressed_pages.load(Ordering::Relaxed),
                projected_bytes: projection
                    .indices()
                    .iter()
                    .map(|&column| row_group.columns[column].total_compressed_size)
                    .sum(),
            });
        }

        let mut files: Vec<_> = weights_by_file
            .into_iter()
            .map(|mut weights| {
                weights
                    .sort_by_key(|weight| Reverse((weight.cached_pages, weight.projected_bytes)));
                let file = &table.row_groups[weights[0].index].open_file;
                let order = Reverse((
                    row_selector.count_bytes_in_memory(file),
                    weights
                        .iter()
                        .map(|weight| weight.cached_pages)
                        .sum::<usize>(),
                    weights
                        .iter()
                        .map(|weight| weight.projected_bytes)
                        .sum::<i64>(),
                ));
                let row_groups = weights
                    .iter()
                    .map(|weight| {
                        QueryRowGroupMetadata::new(table, weight.index, RowSelection::All)
                    })
                    .collect();
                (order, ScannedFile { row_groups })
            })
            .collect();
        files.sort_by_key(|(order, _)| *order);

        let queue = Injector::new();
        for (_, file) in files {
            queue.push(file);
        }
        Self {
            files: Arc::new(queue),
        }
    }
}

/// The receiving side is the queue itself: what a stage's operator watches to
/// see whether files are left.
impl RootChannelFactory<ScannedFile> for ScannedFiles {
    type Receiver = ScannedFiles;

    fn build(self) -> Self::Receiver {
        self
    }
}

impl Receiver<ScannedFile> for ScannedFiles {
    fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn try_recv(&self) -> Option<ScannedFile> {
        loop {
            match self.files.steal() {
                Steal::Success(file) => return Some(file),
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
    }

    /// `try_recv` already reaches every file.
    fn steal(&self) -> Option<ScannedFile> {
        None
    }
}

/// The row groups a scan's stages have decided to read, waiting for a fetcher
/// to ask for them: one queue per NUMA node, each row group in the queue of
/// the node a scan without a selector assigns it to, so its cached bytes stay
/// on one node query after query.
///
/// A stage puts a row group in only while its node's queue is empty. A stage
/// that found it occupied is woken once a fetcher has emptied it.
#[derive(Clone)]
pub(crate) struct SelectedRowGroupQueues {
    shared: Arc<SharedSelectedRowGroups>,
}

struct SharedSelectedRowGroups {
    node_queues: Vec<Injector<QueryRowGroupMetadata>>,
    /// Set by a stage that had a row group to hand over and found its node's
    /// queue occupied; the fetcher that next empties a queue clears it and
    /// wakes the workers.
    is_a_stage_waiting: AtomicBool,
}

impl SelectedRowGroupQueues {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            shared: Arc::new(SharedSelectedRowGroups {
                node_queues: (0..node_count).map(|_| Injector::new()).collect(),
                is_a_stage_waiting: AtomicBool::new(false),
            }),
        }
    }

    fn node_of(&self, row_group: &RowGroupMetadata) -> usize {
        affinity_node(row_group, self.shared.node_queues.len())
    }

    /// Whether `row_group` can be queued now: no row group is waiting for a
    /// fetcher of its node. When one is, a fetcher emptying a queue wakes the
    /// workers, so the stage asking gets to ask again.
    fn has_room_for(&self, row_group: &RowGroupMetadata) -> bool {
        let queue = &self.shared.node_queues[self.node_of(row_group)];
        if queue.is_empty() {
            return true;
        }
        self.shared
            .is_a_stage_waiting
            .store(true, Ordering::Release);
        // A fetcher may have emptied the queue between the two looks, before
        // it could see that a stage is waiting.
        queue.is_empty()
    }

    fn push(&self, row_group: QueryRowGroupMetadata) {
        let node = self.node_of(row_group.get_metadata());
        self.shared.node_queues[node].push(row_group);
        dispatch::waker::waker_set().notify_one_near(node);
    }
}

/// Runs on the worker thread, so the receiver can snapshot which node's
/// queue is local to it.
impl RootChannelFactory<QueryRowGroupMetadata> for SelectedRowGroupQueues {
    type Receiver = SelectedRowGroupsReceiver;

    fn build(self) -> Self::Receiver {
        SelectedRowGroupsReceiver {
            queues: self,
            numa_node_idx: dispatch::worker::current_node(),
        }
    }
}

/// A fetcher's view of the [`SelectedRowGroupQueues`]: its own node's queue
/// first, the other nodes' only when its own is empty.
pub(crate) struct SelectedRowGroupsReceiver {
    queues: SelectedRowGroupQueues,
    numa_node_idx: usize,
}

impl SelectedRowGroupsReceiver {
    fn take_from_node(&self, node: usize) -> Option<QueryRowGroupMetadata> {
        let shared = &self.queues.shared;
        let queue = &shared.node_queues[node];
        loop {
            match queue.steal() {
                Steal::Success(row_group) => {
                    if queue.is_empty() && shared.is_a_stage_waiting.swap(false, Ordering::AcqRel) {
                        dispatch::waker::waker_set().notify_all();
                    }
                    return Some(row_group);
                }
                Steal::Retry => continue,
                Steal::Empty => return None,
            }
        }
    }
}

impl Receiver<QueryRowGroupMetadata> for SelectedRowGroupsReceiver {
    fn is_empty(&self) -> bool {
        self.queues
            .shared
            .node_queues
            .iter()
            .all(|queue| queue.is_empty())
    }

    fn try_recv(&self) -> Option<QueryRowGroupMetadata> {
        self.take_from_node(self.numa_node_idx)
    }

    /// A node that finished its share takes from the others' queues rather
    /// than idle.
    fn steal(&self) -> Option<QueryRowGroupMetadata> {
        let node_count = self.queues.shared.node_queues.len();
        (0..node_count)
            .find_map(|offset| self.take_from_node((self.numa_node_idx + offset) % node_count))
    }
}

/// Builds one worker's [`RowSelectionStage`].
pub(crate) struct RowSelectionStageFactory<Selector> {
    row_selector: Selector,
    files: ScannedFiles,
    filter: Option<RowGroupFilter>,
    projection: Arc<Projection>,
    read_medium: ReadMedium,
    selected_row_groups: Option<SelectedRowGroupQueues>,
}

impl<Selector> RowSelectionStageFactory<Selector> {
    /// A stage looking `files` up with `row_selector`, for a scan reading the
    /// columns of `projection` from `read_medium`. A row group `filter`
    /// rules out when it comes to be handed over is dropped.
    ///
    /// The row groups it keeps go to `selected_row_groups`, where the scan's
    /// fetchers ask for them. A scan that reads nothing of a row group has no
    /// fetchers, and they go down the stage's output channel instead.
    pub(crate) fn new(
        row_selector: Selector,
        files: ScannedFiles,
        filter: Option<RowGroupFilter>,
        projection: Arc<Projection>,
        read_medium: ReadMedium,
        selected_row_groups: Option<SelectedRowGroupQueues>,
    ) -> Self {
        Self {
            row_selector,
            files,
            filter,
            projection,
            read_medium,
            selected_row_groups,
        }
    }
}

impl<Selector> UnaryFactory<ScannedFile, QueryRowGroupMetadata>
    for RowSelectionStageFactory<Selector>
where
    Selector: RowSelector + Send + 'static,
{
    type Unary = RowSelectionStage<Selector>;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(RowSelectionStage {
            row_selector: self.row_selector,
            files: self.files,
            filter: self.filter,
            projection: self.projection,
            read_medium: self.read_medium,
            selected_row_groups: self.selected_row_groups,
            looking_up: HashMap::new(),
            looked_up: VecDeque::new(),
        })
    }
}

/// A file whose lookup waits on a read.
struct FileInLookup<Pending> {
    file: ScannedFile,
    /// What the selector was in the middle of.
    pending: Pending,
    /// The file's projected bytes, counted against the worker's read-ahead
    /// budget while it is looked up.
    read_ahead_charge: ReadAheadCharge,
}

/// A looked-up file, with the row groups of it still to be handed over.
struct LookedUpFile<FileRows> {
    file: ScannedFile,
    rows: FileRows,
}

/// Looks a scan's files up with its [`RowSelector`] and hands the fetchers
/// the row groups worth reading. See the [module docs](self).
///
/// The stage takes its files itself, when it has room for one, so it never
/// asks to be fed: its input channel is only how the dataflow sees that files
/// are left.
pub(crate) struct RowSelectionStage<Selector: RowSelector> {
    row_selector: Selector,
    files: ScannedFiles,
    filter: Option<RowGroupFilter>,
    /// The columns the scan reads from every row group.
    projection: Arc<Projection>,
    /// The medium the table is read from, whose read-ahead budget a file
    /// counts against while it is looked up.
    read_medium: ReadMedium,
    /// Where the scan's fetchers ask for the row groups this stage keeps;
    /// `None` in a scan that reads nothing of a row group.
    selected_row_groups: Option<SelectedRowGroupQueues>,
    /// The files whose lookup waits on a read, by that read.
    looking_up: HashMap<ReadRequestId, FileInLookup<Selector::Pending>>,
    /// The files looked up and not yet handed over whole, oldest first.
    looked_up: VecDeque<LookedUpFile<Selector::FileRows>>,
}

impl<S: RowSelector> RowSelectionStage<S> {
    /// Hand the fetchers the next row group of the oldest looked-up file, or
    /// drop it when no row of it is read. Waits while a row group is still
    /// waiting for a fetcher of the same node.
    fn hand_over_next_row_group(
        &mut self,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
    ) -> UnaryResult<WorkStatus> {
        let Some(looked_up) = self.looked_up.front_mut() else {
            return Ok(WorkStatus::Pending);
        };
        let next = looked_up
            .file
            .row_groups
            .front()
            .expect("a file handed over whole is no longer kept");
        if let Some(queues) = &self.selected_row_groups
            && !queues.has_room_for(next.get_metadata())
        {
            return Ok(WorkStatus::Pending);
        }

        let row_group = looked_up
            .file
            .row_groups
            .pop_front()
            .expect("the row group looked at is still there");
        let is_ruled_out = self
            .filter
            .as_ref()
            .is_some_and(|filter| !filter(row_group.get_metadata()));
        let selection = if is_ruled_out {
            RowSelection::Indices(Vec::new().into())
        } else {
            self.row_selector
                .select_rows(&looked_up.rows, row_group.get_metadata())?
        };
        if looked_up.file.row_groups.is_empty() {
            self.looked_up.pop_front();
        }

        let row_group = match selection {
            RowSelection::Indices(rows) if rows.is_empty() => return Ok(WorkStatus::Ran),
            RowSelection::All => row_group,
            selection => row_group.with_selection(selection),
        };
        match &self.selected_row_groups {
            Some(queues) => queues.push(row_group),
            None => sender.send(row_group)?,
        }
        Ok(WorkStatus::Ran)
    }

    /// Take the next file and look it up, while the stage has room for one
    /// (see the [module docs](self)).
    fn look_up_next_file(&mut self, io: &mut OperatorIO) -> UnaryResult<WorkStatus> {
        let open_files = self.looking_up.len() + self.looked_up.len();
        if open_files >= crate::http_readahead() || !io.has_read_ahead_room(self.read_medium) {
            return Ok(WorkStatus::Pending);
        }
        let Some(file) = self.files.try_recv() else {
            return Ok(WorkStatus::Pending);
        };
        // A file the filter rules out whole needs no lookup.
        let is_ruled_out = self.filter.as_ref().is_some_and(|filter| {
            file.row_groups
                .iter()
                .all(|row_group| !filter(row_group.get_metadata()))
        });
        if is_ruled_out {
            return Ok(WorkStatus::Ran);
        }

        let projected_bytes: i64 = file
            .row_groups
            .iter()
            .flat_map(|row_group| {
                self.projection.indices().iter().map(move |&column| {
                    row_group.row_group_metadata.columns[column].total_compressed_size
                })
            })
            .sum();
        let read_ahead_charge = io.charge_read_ahead(self.read_medium, projected_bytes as usize);
        let open_file = file.row_groups[0].get_metadata().open_file.clone();
        let progress = self.row_selector.look_up_file(&open_file, io)?;
        self.settle_lookup(file, read_ahead_charge, progress);
        Ok(WorkStatus::Ran)
    }

    /// Keep `file` for the read its lookup waits on, or, once it is looked
    /// up, for its row groups to be handed over. `read_ahead_charge` is what
    /// the file counts against the budget, given back once it is looked up:
    /// from then on its row groups are counted as fetchers ask for them.
    fn settle_lookup(
        &mut self,
        file: ScannedFile,
        read_ahead_charge: ReadAheadCharge,
        progress: FileLookupProgress<S::FileRows, S::Pending>,
    ) {
        match progress {
            FileLookupProgress::Reading { read, pending } => {
                self.looking_up.insert(
                    read,
                    FileInLookup {
                        file,
                        pending,
                        read_ahead_charge,
                    },
                );
            }
            FileLookupProgress::LookedUp(rows) => {
                self.looked_up.push_back(LookedUpFile { file, rows });
            }
        }
    }
}

impl<S: RowSelector> Unary<ScannedFile, QueryRowGroupMetadata> for RowSelectionStage<S> {
    fn consume(
        &mut self,
        _file: ScannedFile,
        _sender: &mut dyn Sender<QueryRowGroupMetadata>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        unreachable!("the stage takes its files itself and is never ready to be fed one")
    }

    fn ready_for_more_work(&mut self) -> bool {
        false
    }

    fn saturates_available_io(&self) -> bool {
        true
    }

    /// Ahead of the worker's CPU work: hand a fetcher its next row group, so
    /// it finds one when it asks for its next read, or else look up another
    /// file. In a scan with no fetchers the row groups are handed on only on
    /// the CPU turn.
    fn saturate_available_io(
        &mut self,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
        io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        if self.selected_row_groups.is_some()
            && let WorkStatus::Ran = self.hand_over_next_row_group(sender)?
        {
            return Ok(WorkStatus::Ran);
        }
        self.look_up_next_file(io)
    }

    /// A read a file's lookup was waiting on is back.
    fn process_read_response(
        &mut self,
        _sender: &mut dyn Sender<QueryRowGroupMetadata>,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> UnaryResult<()> {
        let FileInLookup {
            file,
            pending,
            read_ahead_charge,
        } = self
            .looking_up
            .remove(&response.id())
            .expect("response for a read this stage did not issue");
        let progress = self
            .row_selector
            .resume_file_lookup(pending, response, io)?;
        self.settle_lookup(file, read_ahead_charge, progress);
        Ok(())
    }

    /// The worker's CPU turn.
    fn run(
        &mut self,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
        io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        if let WorkStatus::Ran = self.hand_over_next_row_group(sender)? {
            return Ok(WorkStatus::Ran);
        }
        self.look_up_next_file(io)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<QueryRowGroupMetadata>) -> UnaryResult<bool> {
        Ok(self.looking_up.is_empty() && self.looked_up.is_empty())
    }

    /// The files still to be taken keep the stage from finishing, on every
    /// worker: any of them may be the one to take the next. So do the files
    /// this worker's stage has open.
    fn has_pending_work(&self) -> bool {
        !self.files.is_empty() || !self.looking_up.is_empty() || !self.looked_up.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::dummy_row_group;
    use dispatch::io::LocalFile;
    use std::convert::Infallible;
    use std::sync::atomic::AtomicUsize;

    /// A selector that only says how much of each file's lookup is in memory.
    struct LookupsInMemory {
        bytes_by_file: Vec<(OpenFile, usize)>,
    }

    impl RowSelector for LookupsInMemory {
        type FileRows = ();
        type Pending = Infallible;

        fn count_bytes_in_memory(&self, file: &OpenFile) -> usize {
            let (_, bytes) = self
                .bytes_by_file
                .iter()
                .find(|(known, _)| known == file)
                .unwrap();
            *bytes
        }

        fn look_up_file(
            &mut self,
            _file: &OpenFile,
            _io: &mut OperatorIO,
        ) -> UnaryResult<FileLookupProgress<(), Infallible>> {
            unreachable!("no file is looked up")
        }

        fn resume_file_lookup(
            &mut self,
            pending: Infallible,
            _response: ReadResponse,
            _io: &mut OperatorIO,
        ) -> UnaryResult<FileLookupProgress<(), Infallible>> {
            match pending {}
        }

        fn select_rows(
            &self,
            _file_rows: &(),
            _row_group: &RowGroupMetadata,
        ) -> UnaryResult<RowSelection> {
            unreachable!("no row group is handed over")
        }
    }

    fn open_file() -> OpenFile {
        OpenFile::Local(LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
    }

    /// The one row group of `file`, with `cached_pages` of its pages in the
    /// decompressed cache.
    fn row_group_of(file: &OpenFile, cached_pages: usize) -> Arc<RowGroupMetadata> {
        Arc::new(RowGroupMetadata {
            open_file: file.clone(),
            live_decompressed_pages: Arc::new(AtomicUsize::new(cached_pages)),
            ..(*dummy_row_group()).clone()
        })
    }

    #[test]
    fn files_are_taken_by_what_is_in_memory_of_their_lookup_and_then_of_their_pages() {
        let files = [open_file(), open_file(), open_file()];
        let table = ParquetTable::new(vec![
            row_group_of(&files[0], 5),
            row_group_of(&files[1], 0),
            row_group_of(&files[2], 9),
        ]);
        let row_selector = LookupsInMemory {
            bytes_by_file: vec![
                (files[0].clone(), 0),
                (files[1].clone(), 100),
                (files[2].clone(), 0),
            ],
        };

        let scanned = ScannedFiles::of_table(&table, &Projection::columns([]), &row_selector);

        let taken: Vec<usize> = std::iter::from_fn(|| scanned.try_recv())
            .map(|file| file.row_groups[0].index())
            .collect();
        assert_eq!(taken, vec![1, 2, 0]);
    }
}
