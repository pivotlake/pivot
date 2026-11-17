//! Searching a scan's files in their text indexes, ahead of the fetch.
//!
//! ```text
//!   the scan's files ──▶ TextIndexSearcher ──▶ row groups, each with ──▶ fetch ──▶ … ──▶ decode
//!   (most of their index   (one per worker)    the rows to read
//!    in memory first)
//! ```
//!
//! A scan that looks for tokens (`has_token`) hands its files to the
//! searchers instead of handing its row groups to the fetchers. A searcher
//! takes a file, finds the rows of every token in the file's index
//! ([`TokenLookup`]: a read for each piece of the index that is not in its
//! cache), and sends each of the file's row groups on with only the rows that
//! have all the tokens selected. A row group none of whose rows do is never
//! fetched. From there on the row groups go the way a late materialization's
//! do: fetched, cut and decoded for the selected rows only.
//!
//! A file without an index is not searched: its row groups are sent on whole.
//! So what comes out of the scan is every row that has the tokens, and maybe
//! rows that do not. Whoever asked for the tokens still tests the rows the
//! scan returns.
//!
//! The files are handed out in the order of how much of their sparse and
//! dict indexes is in memory, the most first: those are searched with the
//! fewest reads, and their rows are on their way to the decoders while the
//! other files' indexes are still being read.

use crate::ParquetTable;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crossbeam_deque::{Injector, Steal};
use dispatch::io::{OpenFile, OperatorIO, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use dispatch::text_index::{LetterCase, LookupProgress, RowsWithToken, TokenLookup};
use dispatch::{
    BuildContext, Built, BuiltUnary, Receiver, RootChannelFactory, Sender,
    SharedWorkQueueChannelFactory, SharedWorkQueueSender, Unary, UnaryFactory,
};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// How many files one worker's searcher has in hand at once. Each waits on
/// one read per token at a time, so this is how many index reads a worker
/// keeps on their way.
const MOST_FILES_SEARCHED_AT_ONCE: usize = 16;

/// The index file of one data file, opened for a scan to search.
#[derive(Clone)]
pub struct OpenTextIndexFile {
    pub open_file: OpenFile,
    /// How many bytes long the index file is.
    pub len: usize,
    /// What the index did to the letter case of the tokens it holds, which a
    /// token looked up in it has to have done to it too.
    pub letter_case: LetterCase,
}

/// One token a file's rows are searched for, with the index file that says
/// which of them have it.
pub struct TokenToFind {
    /// The token as the index holds it.
    token: Vec<u8>,
    index_file: OpenTextIndexFile,
}

impl TokenToFind {
    /// `token` as text has it, to be found in `index_file`. An index that
    /// lowers its tokens holds the token lowered, whichever case the text
    /// has it in, so it is looked up lowered: the rows found are those that
    /// have the token in any letter case.
    pub fn new(token: &[u8], index_file: OpenTextIndexFile) -> Self {
        let token = match index_file.letter_case {
            LetterCase::Kept => token.to_vec(),
            LetterCase::Lowered => token.to_ascii_lowercase(),
        };
        Self { token, index_file }
    }
}

/// One data file of a scan, and what its indexes are asked.
pub struct FileToSearch {
    /// The file's row groups the scan reads, by their positions in the
    /// scan's table, in file order.
    pub row_groups: Vec<usize>,
    /// The tokens a row has to have to be read, each with the index file to
    /// find it in. Empty for a file with no index, whose rows are all read.
    pub tokens: Vec<TokenToFind>,
}

impl FileToSearch {
    /// How many bytes of the file's sparse and dict indexes are in memory
    /// now. Asked on a worker, where the caches are reached.
    fn count_index_bytes_in_memory(&self) -> usize {
        let caches = memory_ctx().text_index_caches();
        self.tokens
            .iter()
            .map(|token| {
                let index_file = &token.index_file.open_file;
                caches.sparse_indexes.count_cached_bytes(index_file)
                    + caches.dict_indexes.count_cached_bytes(index_file)
            })
            .sum()
    }
}

/// The files of one scan, for the workers' searchers to take: every worker's
/// [`FilesToSearch`] takes from the same queue.
///
/// The queue is put in order by the first worker to build its receiver, not
/// when the scan is planned: how much of an index is in memory is asked of
/// the caches, which only a worker reaches.
#[derive(Clone)]
pub struct FilesToSearchFactory {
    /// The files as the scan listed them, until a worker orders them.
    listed_files: Arc<Mutex<Vec<FileToSearch>>>,
    ordered_files: Arc<OnceLock<Arc<OrderedFiles>>>,
}

/// The queue of a scan's files, the ones to search first at the front.
struct OrderedFiles {
    files: Injector<FileToSearch>,
    /// The files not taken yet. Whoever takes the last wakes every worker.
    files_left: AtomicUsize,
}

impl FilesToSearchFactory {
    pub fn new(files: Vec<FileToSearch>) -> Self {
        Self {
            listed_files: Arc::new(Mutex::new(files)),
            ordered_files: Arc::default(),
        }
    }
}

impl RootChannelFactory<FileToSearch> for FilesToSearchFactory {
    type Receiver = FilesToSearch;

    /// Runs on the worker thread. The first worker here orders the files:
    /// the most index bytes in memory first, and among files with as many,
    /// the order the scan listed them in.
    fn build(self) -> FilesToSearch {
        let ordered_files = self.ordered_files.get_or_init(|| {
            let mut files = std::mem::take(&mut *self.listed_files.lock().unwrap());
            // Each file's count is read once: the caches change under the
            // sort, and a count read again per comparison would not order.
            files.sort_by_cached_key(|file| Reverse(file.count_index_bytes_in_memory()));
            let files_left = AtomicUsize::new(files.len());
            let queue = Injector::new();
            for file in files {
                queue.push(file);
            }
            Arc::new(OrderedFiles {
                files: queue,
                files_left,
            })
        });
        FilesToSearch {
            ordered_files: Arc::clone(ordered_files),
        }
    }
}

/// One worker's end of a scan's queue of files.
pub struct FilesToSearch {
    ordered_files: Arc<OrderedFiles>,
}

impl FilesToSearch {
    /// Count a file taken. Taking the last one wakes every worker: a worker
    /// that saw the queue hold a file when it tried to finish has parked
    /// without finishing, and nothing else tells it the queue ran dry.
    fn note_file_taken(&self) {
        if self
            .ordered_files
            .files_left
            .fetch_sub(1, Ordering::Relaxed)
            == 1
        {
            dispatch::waker::waker_set().notify_all();
        }
    }
}

impl Receiver<FileToSearch> for FilesToSearch {
    fn is_empty(&self) -> bool {
        self.ordered_files.files.is_empty()
    }

    /// One attempt, so a worker with other work to do does not spin on a
    /// queue another worker is at.
    fn try_recv(&self) -> Option<FileToSearch> {
        if self.is_empty() {
            return None;
        }
        match self.ordered_files.files.steal() {
            Steal::Success(file) => {
                self.note_file_taken();
                Some(file)
            }
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<FileToSearch> {
        while !self.is_empty() {
            match self.ordered_files.files.steal() {
                Steal::Success(file) => {
                    self.note_file_taken();
                    return Some(file);
                }
                Steal::Empty => return None,
                Steal::Retry => continue,
            }
        }
        None
    }
}

/// Builds one worker's [`TextIndexSearcher`].
pub struct TextIndexSearcherFactory {
    table: Arc<ParquetTable>,
    io_row_groups_channel: SharedWorkQueueChannelFactory<QueryRowGroupMetadata>,
}

impl TextIndexSearcherFactory {
    /// A searcher of the files of `table`. `io_row_groups_channel` is the
    /// queue the row groups expected to need a read are sent to, which every
    /// worker's fetcher takes from.
    pub fn new(
        table: Arc<ParquetTable>,
        io_row_groups_channel: SharedWorkQueueChannelFactory<QueryRowGroupMetadata>,
    ) -> Self {
        Self {
            table,
            io_row_groups_channel,
        }
    }
}

impl UnaryFactory<FileToSearch, QueryRowGroupMetadata> for TextIndexSearcherFactory {
    type Unary = TextIndexSearcher;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<TextIndexSearcher> {
        Built::new(TextIndexSearcher {
            table: self.table,
            io_row_groups_sender: self.io_row_groups_channel.build_sender(),
            files_being_searched: HashMap::new(),
            awaited_reads: HashMap::new(),
            next_file_number: 0,
        })
    }
}

/// One token being looked up in one file's index.
struct TokenBeingFound {
    index_file: OpenFile,
    lookup: TokenLookup,
    /// Whether the lookup asked for a read that has not come back. It goes
    /// no further until it has been handed that read: a piece another
    /// token's lookup cached meanwhile would take it on to its next read,
    /// and the one on its way would come back as the wrong piece.
    awaits_read: bool,
    /// The token's rows, once the lookup reached them.
    rows: Option<RowsWithToken>,
}

/// A file a searcher has taken and not yet sent the row groups of.
struct FileBeingSearched {
    row_groups: Vec<usize>,
    tokens: Vec<TokenBeingFound>,
}

/// Which lookup a read was asked for: the file, by the number its searcher
/// gave it, and the token's place among the file's tokens.
struct AwaitedRead {
    file_number: usize,
    token_position: usize,
}

/// Takes a scan's files, finds in each file's index the rows that have the
/// tokens looked for, and sends the file's row groups on with those rows
/// selected.
///
/// A row group goes one of two ways, as in a table scan. One the
/// decompressed cache holds pages of is expected to be in memory, and is
/// sent down the channel for a fetcher to take when its worker has no other
/// work. Any other is expected to need a read, and is pushed to the queue
/// the fetchers take from ahead of their workers' CPU work.
pub struct TextIndexSearcher {
    table: Arc<ParquetTable>,
    /// Where the row groups expected to need a read are sent.
    io_row_groups_sender: SharedWorkQueueSender<QueryRowGroupMetadata>,
    /// The files in hand, each waiting on a read of its index.
    files_being_searched: HashMap<usize, FileBeingSearched>,
    awaited_reads: HashMap<ReadRequestId, AwaitedRead>,
    next_file_number: usize,
}

impl TextIndexSearcher {
    /// Take every lookup of file `file_number` that is not waiting for a
    /// read as far as what is in memory allows, asking for the read each one
    /// stops at. A file all of whose lookups reached their rows has its row
    /// groups sent on.
    fn advance_file(
        &mut self,
        file_number: usize,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let file = self
            .files_being_searched
            .get_mut(&file_number)
            .expect("a file advanced is a file in hand");
        for (token_position, token) in file.tokens.iter_mut().enumerate() {
            if token.rows.is_some() || token.awaits_read {
                continue;
            }
            match token.lookup.advance() {
                LookupProgress::Found(rows) => token.rows = Some(rows),
                LookupProgress::NeedsRead(range) => {
                    let read = io.read(
                        &memory_ctx().page_caches().compressed,
                        token.index_file.clone(),
                        [range],
                    )?;
                    token.awaits_read = true;
                    self.awaited_reads.insert(
                        read,
                        AwaitedRead {
                            file_number,
                            token_position,
                        },
                    );
                }
            }
        }
        if file.tokens.iter().any(|token| token.rows.is_none()) {
            return Ok(());
        }
        let file = self
            .files_being_searched
            .remove(&file_number)
            .expect("the file was just advanced");
        self.send_row_groups(file, sender)
    }

    /// Send on every row group of a searched file that has a row to read:
    /// one that every token appears in, and that the table format did not
    /// delete, since a selection of rows never names a deleted one.
    fn send_row_groups(
        &mut self,
        file: FileBeingSearched,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
    ) -> dispatch::UnaryResult<()> {
        // The token in the fewest rows is the one walked. The others are
        // only asked about the rows it is in.
        let mut rows_of_tokens: Vec<RowsWithToken> = file
            .tokens
            .into_iter()
            .map(|token| token.rows.expect("every lookup reached its rows"))
            .collect();
        rows_of_tokens.sort_by_key(RowsWithToken::count_rows);

        for row_group_index in file.row_groups {
            let selection = match rows_of_tokens.split_first() {
                None => RowSelection::All,
                Some((rarest_token, other_tokens)) => {
                    let row_group = &self.table.row_groups()[row_group_index];
                    // An index holds a row by its position in the file, and
                    // a selection by its position in the row group.
                    let first_row = row_group.file_row_offset as u32;
                    let rows_of_row_group = first_row..first_row + row_group.num_rows as u32;
                    let mut rows_to_read = Vec::new();
                    rarest_token.for_each_row_in(rows_of_row_group, |row| {
                        let is_deleted = row_group
                            .deleted_rows
                            .as_ref()
                            .is_some_and(|deleted| deleted.contains(row as u64));
                        if !is_deleted && other_tokens.iter().all(|token| token.contains_row(row)) {
                            rows_to_read.push(row - first_row);
                        }
                    });
                    if rows_to_read.is_empty() {
                        continue;
                    }
                    RowSelection::Indices(rows_to_read.into())
                }
            };
            let row_group = QueryRowGroupMetadata::new(&self.table, row_group_index, selection);
            let is_expected_in_memory = row_group
                .get_metadata()
                .live_decompressed_pages
                .load(Ordering::Relaxed)
                > 0;
            if is_expected_in_memory {
                sender.send(row_group)?;
            } else {
                self.io_row_groups_sender.send(row_group)?;
            }
        }
        Ok(())
    }
}

impl Unary<FileToSearch, QueryRowGroupMetadata> for TextIndexSearcher {
    /// Take a file: start a lookup for each of its tokens, and go as far as
    /// what is in memory allows.
    fn consume(
        &mut self,
        file: FileToSearch,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let tokens = file
            .tokens
            .into_iter()
            .map(|token| TokenBeingFound {
                lookup: TokenLookup::new(
                    token.index_file.open_file.clone(),
                    token.index_file.len,
                    &token.token,
                ),
                index_file: token.index_file.open_file,
                awaits_read: false,
                rows: None,
            })
            .collect();
        let file_number = self.next_file_number;
        self.next_file_number += 1;
        self.files_being_searched.insert(
            file_number,
            FileBeingSearched {
                row_groups: file.row_groups,
                tokens,
            },
        );
        self.advance_file(file_number, sender, io)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.files_being_searched.len() < MOST_FILES_SEARCHED_AT_ONCE
    }

    /// A piece of an index came back: hand it to the lookup that asked for
    /// it, which decodes it into its cache, and go on with the file.
    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let AwaitedRead {
            file_number,
            token_position,
        } = self
            .awaited_reads
            .remove(&response.id())
            .expect("a read that came back is a read a lookup asked for");
        let piece = response
            .into_locations()
            .into_iter()
            .next()
            .expect("a lookup asks for one range");
        let token = &mut self
            .files_being_searched
            .get_mut(&file_number)
            .expect("a file waiting on a read is a file in hand")
            .tokens[token_position];
        token
            .lookup
            .supply(piece.bytes.as_slice())
            .map_err(crate::op_err)?;
        token.awaits_read = false;
        self.advance_file(file_number, sender, io)
    }

    fn finish(
        &mut self,
        _sender: &mut dyn Sender<QueryRowGroupMetadata>,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.files_being_searched.is_empty())
    }

    /// The files in hand keep the searcher from finishing: their row groups
    /// are still to be sent on.
    fn has_pending_work(&self) -> bool {
        !self.files_being_searched.is_empty()
    }
}
