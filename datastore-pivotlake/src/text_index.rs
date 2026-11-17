//! Text indexes over a table's data files.
//!
//! An index covers one column of one data file and lives in its own object in
//! the table's location, named after the data file and the column:
//!
//! ```text
//!   pivot-1f0c….parquet
//!   pivot-1f0c….parquet.9c41d2aa.message.textidx     index of column `message`
//! ```
//!
//! Data files never change, so neither does an index: it is built once from
//! the file's rows and is valid for as long as the file exists. Creating an
//! index on a table therefore means building one for every data file that has
//! none ([`CatalogTable::create_text_index`]), and can be repeated to cover
//! files written since. The files are built a few at a time, since a file's
//! tokens are held in memory until its last row has been read.
//!
//! ```text
//!   files without an index
//!     │ scan the column, with row positions
//!     ▼
//!   TokenizeRows ──▶ IndexAssembler ──▶ WriteIndexFiles
//!   every worker     one worker per     writes each index
//!                    file               through dispatch IO
//! ```
//!
//! A scan with a token predicate on an indexed column looks each of its data
//! files up in the file's index, once, and reads of each row group only the
//! rows the index says hold the tokens ([`TextIndexRowSelector`]):
//!
//! ```text
//!   data file taken by the scan
//!     │ has an index: look the tokens up, once      has none
//!     ▼                                             ▼
//!   per row group of the file:                      every row group is read whole,
//!     no row    ─▶ nothing of it is read            and the filter above the scan
//!     some rows ─▶ only those rows are decoded      checks its rows
//! ```
//!
//! An index names exactly the rows holding the tokens. So when every data
//! file has one, the scan answers the predicate by itself and no filter is
//! planned above it; when some file has none, the filter stays, and checks
//! the rows of the files without an index.
//!
//! A scan feeding a Top-N is the exception: it is read in the order of the
//! Top-N's key and under its throttle, which go by row group, and takes no
//! part in this. It checks the tokens on the text of the rows it reads.
//!
//! Which files have an index is found by listing the table's location once
//! per column and remembered for the table's lifetime.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use dispatch::io::{
    FsWriteRequest, HttpUploadRequest, LocalFile, OpenFile, OperatorIO, ReadResponse, RemoteFile,
};
use dispatch::text_index::{
    AssembledIndex, FileTokenRun, IndexAssembler, IndexFile, LookupProgress, MatchingRows,
    TextIndexLookup, TextQuery, TokenRunBuilder,
};
use dispatch::{
    BuildContext, Built, BuiltUnary, DefaultUnaryFactory, OperatorSpec, Projection, Sender, Unary,
    UnaryFactory, UnaryResult, memory_ctx, return_to_worker_mpsc, stealable, values_input,
};
use object_storage::{DataFileLocation, ObjectPath, ObjectStore};
use parquet_engine::{FileLookupProgress, RowGroupMetadata, RowSelection, RowSelector};
use planner::expression::HasTokens;

use crate::{CatalogTable, Error, Result};

const INDEX_SUFFIX: &str = ".textidx";

/// The name of the index object for `column` of the data file at `data_file`.
/// The data file's own name and the column's, with the characters a file
/// name cannot safely hold replaced, keep it readable. A hash of the data
/// file's whole path and of the column's name as it is written keeps apart
/// two files of the same name in different directories, and two columns whose
/// names read the same once replaced (`user-agent` and `user_agent`).
fn index_object_name(data_file: &ObjectPath, column: &str) -> String {
    let readable_column: String = column
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!(
        "{}.{:08x}.{readable_column}{INDEX_SUFFIX}",
        data_file.name(),
        stable_hash(data_file.as_str().as_bytes(), column.as_bytes()) as u32,
    )
}

/// FNV-1a over `path` and `column`: a hash that is the same in every build, so
/// an index written by one process is found by another.
fn stable_hash(path: &[u8], column: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in path.iter().chain([&0u8]).chain(column) {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A column's index files, by the data file each one covers.
#[derive(Default)]
pub(crate) struct ColumnIndexFiles {
    by_data_file: HashMap<ObjectPath, IndexFile>,
}

/// A column's text indexes over the data files of one copy of a table.
pub(crate) struct ColumnIndexes {
    /// The index of each data file that has one, by the open data file.
    by_open_file: HashMap<OpenFile, FileIndex>,
    covers_every_file: bool,
}

/// One data file's index.
struct FileIndex {
    index: IndexFile,
    /// Rows in the data file, which the index has to cover exactly.
    file_row_count: u64,
}

impl ColumnIndexes {
    /// Whether no data file has an index.
    pub(crate) fn is_empty(&self) -> bool {
        self.by_open_file.is_empty()
    }

    /// Whether every data file holding rows has an index, so that a scan
    /// selecting its rows from these indexes emits exactly the rows that
    /// match.
    pub(crate) fn covers_every_file(&self) -> bool {
        self.covers_every_file
    }
}

/// A token predicate pushed into a scan.
#[derive(Clone)]
pub(crate) struct TokenPredicate {
    /// The call as the query wrote it.
    pub(crate) call: HasTokens,
    /// The table column the call reads.
    pub(crate) column_idx: usize,
    /// The tokens the call looks for.
    pub(crate) query: Arc<TextQuery>,
    /// The column's indexes to find them in.
    pub(crate) indexes: Arc<ColumnIndexes>,
    /// Whether the plan keeps no filter for the call, because every data
    /// file has an index: the scan then has to emit exactly the rows the
    /// call holds for.
    pub(crate) is_answered_by_the_scan: bool,
}

/// What the indexes say about one data file: for each of the scan's token
/// predicates the file has an index for, the rows of the file holding its
/// tokens.
pub(crate) struct FileMatches {
    per_predicate: Vec<MatchingRows>,
}

/// A file's lookup, waiting on a read of one of its indexes.
pub(crate) struct PendingFileLookup {
    file: OpenFile,
    /// The answers of the predicates before `predicate`.
    answered: Vec<MatchingRows>,
    /// Which of the scan's predicates `lookup` answers.
    predicate: usize,
    lookup: TextIndexLookup,
}

/// Decides which rows of a scan's files are read from the files' text
/// indexes: the rows holding the tokens of every one of the scan's token
/// predicates the file has an index for. A file with no index for any of them
/// is read whole.
#[derive(Clone)]
pub(crate) struct TextIndexRowSelector {
    predicates: Arc<[TokenPredicate]>,
}

impl TextIndexRowSelector {
    pub(crate) fn new(predicates: &[TokenPredicate]) -> Self {
        Self {
            predicates: predicates.into(),
        }
    }

    /// Look the predicates up in `file`'s indexes from `next_predicate` on,
    /// having `answered` from the ones before, up to the first lookup that
    /// has to wait for a read.
    fn look_up_predicates_from(
        &self,
        next_predicate: usize,
        mut answered: Vec<MatchingRows>,
        file: &OpenFile,
        io: &mut OperatorIO,
    ) -> UnaryResult<FileLookupProgress<FileMatches, PendingFileLookup>> {
        for (predicate, token_predicate) in self.predicates.iter().enumerate().skip(next_predicate)
        {
            // Once one predicate holds for no row of the file, no row holds
            // for all of them, whatever the others say.
            if answered.last().is_some_and(MatchingRows::is_empty) {
                break;
            }
            let Some(file_index) = token_predicate.indexes.by_open_file.get(file) else {
                continue;
            };
            let mut lookup =
                TextIndexLookup::new(file_index.index.clone(), token_predicate.query.clone());
            match lookup.advance(io)? {
                LookupProgress::Reading(read) => {
                    return Ok(FileLookupProgress::Reading {
                        read,
                        pending: PendingFileLookup {
                            file: file.clone(),
                            answered,
                            predicate,
                            lookup,
                        },
                    });
                }
                LookupProgress::Done(matching) => {
                    answered.push(file_index.check_covers_the_file(matching)?);
                }
            }
        }
        Ok(FileLookupProgress::LookedUp(FileMatches {
            per_predicate: answered,
        }))
    }
}

impl FileIndex {
    /// `matching`, the index's answer, once it is seen to speak of exactly
    /// the data file's rows. An index of another row count was built from a
    /// different file, and the rows it names are not this file's.
    fn check_covers_the_file(&self, matching: MatchingRows) -> UnaryResult<MatchingRows> {
        if matching.indexed_row_count() != self.file_row_count {
            return Err(parquet_engine::op_err(Error::TextIndex(format!(
                "a text index covers {} rows of a data file that holds {}",
                matching.indexed_row_count(),
                self.file_row_count
            ))));
        }
        Ok(matching)
    }
}

impl RowSelector for TextIndexRowSelector {
    type FileRows = FileMatches;
    type Pending = PendingFileLookup;

    fn count_bytes_in_memory(&self, file: &OpenFile) -> usize {
        let caches = memory_ctx().text_index_caches();
        self.predicates
            .iter()
            .filter_map(|token_predicate| token_predicate.indexes.by_open_file.get(file))
            .map(|file_index| caches.count_cached_bytes(&file_index.index.file))
            .sum()
    }

    fn look_up_file(
        &mut self,
        file: &OpenFile,
        io: &mut OperatorIO,
    ) -> UnaryResult<FileLookupProgress<FileMatches, PendingFileLookup>> {
        self.look_up_predicates_from(0, Vec::new(), file, io)
    }

    fn resume_file_lookup(
        &mut self,
        pending: PendingFileLookup,
        response: ReadResponse,
        io: &mut OperatorIO,
    ) -> UnaryResult<FileLookupProgress<FileMatches, PendingFileLookup>> {
        let PendingFileLookup {
            file,
            mut answered,
            predicate,
            mut lookup,
        } = pending;
        match lookup.process_read_response(response, io)? {
            LookupProgress::Reading(read) => {
                return Ok(FileLookupProgress::Reading {
                    read,
                    pending: PendingFileLookup {
                        file,
                        answered,
                        predicate,
                        lookup,
                    },
                });
            }
            LookupProgress::Done(matching) => {
                let file_index = self.predicates[predicate]
                    .indexes
                    .by_open_file
                    .get(&file)
                    .expect("the lookup was started from this file's index");
                answered.push(file_index.check_covers_the_file(matching)?);
            }
        }
        self.look_up_predicates_from(predicate + 1, answered, &file, io)
    }

    fn select_rows(
        &self,
        file_rows: &FileMatches,
        row_group: &RowGroupMetadata,
    ) -> UnaryResult<RowSelection> {
        let Some((first, rest)) = file_rows.per_predicate.split_first() else {
            // The file has no index for any of the predicates.
            return Ok(RowSelection::All);
        };
        if file_rows.per_predicate.iter().any(MatchingRows::is_empty) {
            return Ok(RowSelection::Indices(Vec::new().into()));
        }
        let first_row = row_group.file_row_offset as u32;
        let rows_in_file = first_row..first_row + row_group.num_rows as u32;
        let mut rows = first.select_rows(rows_in_file.clone())?;
        for matching in rest {
            if rows.selects_none() {
                break;
            }
            rows.intersect_with(&matching.select_rows(rows_in_file.clone())?);
        }
        // An index covers every row its file was written with, and names a
        // row whether or not the table has deleted it since.
        for deleted_row in row_group.deleted_rows_in(0..row_group.num_rows as u32) {
            rows.deselect(deleted_row as usize);
        }
        Ok(if rows.selects_all() {
            RowSelection::All
        } else {
            RowSelection::Indices(rows.selected_rows().collect())
        })
    }
}

/// The index files known for a table, per column. Shared by every copy of the
/// table: an index belongs to a data file, not to a version of the table.
#[derive(Default)]
pub(crate) struct TextIndexRegistry {
    columns: Mutex<HashMap<String, Arc<ColumnIndexFiles>>>,
}

/// One row group's place in its data file, for turning the scan's
/// (row group, row in group) positions into file rows.
struct RowGroupPlace {
    /// Position of the file among those being indexed.
    file: usize,
    first_row: u32,
    file_row_count: u32,
}

/// Turns scanned batches of one text column into token runs. Each batch holds
/// consecutive rows of row groups, named by the scan's trailing row-group and
/// row-index columns.
struct TokenizeRows {
    places: Arc<[RowGroupPlace]>,
    worker_count: usize,
    builder: TokenRunBuilder,
}

impl Unary<RecordBatch, FileTokenRun<usize>> for TokenizeRows {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<FileTokenRun<usize>>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let texts = batch.column(0).as_string_view();
        let row_groups = parquet_engine::global_row_group(&batch);
        let rows_in_group = parquet_engine::row_index(&batch);
        let group_of_run = row_groups
            .values()
            .as_primitive::<arrow_array::types::UInt32Type>();

        let mut run_start = 0usize;
        for (run, &run_end) in row_groups.run_ends().values().iter().enumerate() {
            let run_end = run_end as usize;
            let place = &self.places[group_of_run.value(run) as usize];
            let first_row = place.first_row + rows_in_group.value(run_start);
            for batch_row in run_start..run_end {
                let row = place.first_row + rows_in_group.value(batch_row);
                if row != first_row + (batch_row - run_start) as u32 {
                    return Err(parquet_engine::op_err(Error::TextIndex(
                        "a scanned batch does not hold consecutive rows".to_string(),
                    )));
                }
                if texts.is_valid(batch_row) {
                    self.builder.add_row(row, texts.value(batch_row).as_bytes());
                }
            }
            sender.send(FileTokenRun {
                file: place.file,
                file_row_count: place.file_row_count,
                assembly_worker: place.file % self.worker_count,
                run: self
                    .builder
                    .finish_run(first_row..first_row + (run_end - run_start) as u32),
            })?;
            run_start = run_end;
        }
        Ok(())
    }
}

struct TokenizeRowsFactory {
    places: Arc<[RowGroupPlace]>,
    worker_count: usize,
}

impl UnaryFactory<RecordBatch, FileTokenRun<usize>> for TokenizeRowsFactory {
    type Unary = TokenizeRows;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(TokenizeRows {
            places: self.places,
            worker_count: self.worker_count,
            builder: TokenRunBuilder::new(),
        })
    }
}

/// An index written to the store: which file it covers and how large it is.
struct WrittenIndex {
    file: usize,
    len: usize,
}

/// Writes each assembled index to its object and reports it once the write
/// has landed.
struct WriteIndexFiles {
    store: Arc<dyn ObjectStore>,
    /// The index object of each file being indexed, by the file's position.
    index_keys: Arc<[ObjectPath]>,
    /// Writes issued and not yet landed, by the address of their bytes, which
    /// the completion carries back.
    in_flight: HashMap<usize, WrittenIndex>,
}

impl WriteIndexFiles {
    fn complete(
        &mut self,
        bytes_address: usize,
        sender: &mut dyn Sender<WrittenIndex>,
    ) -> dispatch::UnaryResult<()> {
        let written = self
            .in_flight
            .remove(&bytes_address)
            .expect("completion for an index write not in flight");
        sender.send(written)?;
        Ok(())
    }
}

impl Unary<AssembledIndex<usize>, WrittenIndex> for WriteIndexFiles {
    fn consume(
        &mut self,
        index: AssembledIndex<usize>,
        _sender: &mut dyn Sender<WrittenIndex>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let data = Arc::new(index.bytes);
        let key = &self.index_keys[index.file];
        let open_file = match self.store.sink(key).map_err(parquet_engine::op_err)? {
            DataFileLocation::Local(path) => {
                let file = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(path)
                    .map_err(parquet_engine::op_err)?;
                OpenFile::Local(LocalFile::new(file).map_err(parquet_engine::op_err)?)
            }
            DataFileLocation::Remote { url, auth } => {
                let remote = RemoteFile::open(url, auth, data.len() as u64)
                    .map(Arc::new)
                    .map_err(parquet_engine::op_err)?;
                OpenFile::Remote(remote)
            }
        };
        io.write(open_file, data.clone())?;
        self.in_flight.insert(
            Arc::as_ptr(&data) as usize,
            WrittenIndex {
                file: index.file,
                len: data.len(),
            },
        );
        Ok(())
    }

    fn process_fs_write_response(
        &mut self,
        sender: &mut dyn Sender<WrittenIndex>,
        request: FsWriteRequest,
    ) -> dispatch::UnaryResult<()> {
        self.complete(Arc::as_ptr(&request.data) as usize, sender)
    }

    fn process_http_upload_response(
        &mut self,
        sender: &mut dyn Sender<WrittenIndex>,
        request: HttpUploadRequest,
    ) -> dispatch::UnaryResult<()> {
        self.complete(Arc::as_ptr(&request.data) as usize, sender)
    }

    fn has_pending_work(&self) -> bool {
        !self.in_flight.is_empty()
    }
}

struct WriteIndexFilesFactory {
    store: Arc<dyn ObjectStore>,
    index_keys: Arc<[ObjectPath]>,
}

impl UnaryFactory<AssembledIndex<usize>, WrittenIndex> for WriteIndexFilesFactory {
    type Unary = WriteIndexFiles;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(WriteIndexFiles {
            store: self.store,
            index_keys: self.index_keys,
            in_flight: HashMap::new(),
        })
    }
}

impl CatalogTable {
    /// Build the text index of `column` for every data file that has none,
    /// returning how many were built. Files that already have one are left
    /// alone, so calling this again covers the files written since.
    pub fn create_text_index(&self, column: &str) -> Result<usize> {
        let column_idx = self
            .columns()
            .iter()
            .position(|declared| declared.name == column)
            .ok_or_else(|| Error::TextIndex(format!("`{column}` is not a column of the table")))?;
        if self.columns()[column_idx].col_type != planner::types::Type::Utf8 {
            return Err(Error::TextIndex(format!(
                "`{column}` is not a text column, so it cannot have a text index"
            )));
        }

        let indexed = self.text_index_files(column)?;
        let unindexed: Vec<_> = self
            .files()
            .iter()
            .filter(|file| !indexed.by_data_file.contains_key(&file.file_ref().path))
            .filter(|file| !file.row_groups.is_empty())
            .collect();
        if unindexed.is_empty() {
            return Ok(0);
        }

        // A file's tokens stay in memory from its first row read to its last,
        // and a scan reads the row groups of all its files in no particular
        // order. One file per worker at a time bounds how many files are
        // partly read at once, and gives each its own worker to merge on.
        let dispatcher = self.dispatcher();
        let worker_count = dispatcher.worker_count();
        let mut built = 0;
        for files in unindexed.chunks(worker_count) {
            let mut places = Vec::new();
            let mut row_groups = Vec::new();
            let mut index_keys = Vec::new();
            for (file, table_file) in files.iter().enumerate() {
                let file_row_count: i64 = table_file.row_groups.iter().map(|rg| rg.num_rows).sum();
                let file_row_count = u32::try_from(file_row_count).map_err(|_| {
                    Error::TextIndex(format!(
                        "`{}` has more rows than a text index can address",
                        table_file.file_ref().path.as_str()
                    ))
                })?;
                for row_group in &table_file.row_groups {
                    places.push(RowGroupPlace {
                        file,
                        first_row: row_group.file_row_offset as u32,
                        file_row_count,
                    });
                    row_groups.push(row_group.clone());
                }
                index_keys.push(self.index_key(&table_file.file_ref().path, column));
            }
            let index_keys: Arc<[ObjectPath]> = index_keys.into();
            let places: Arc<[RowGroupPlace]> = places.into();

            let parquet = Arc::new(parquet_engine::ParquetTable::new(row_groups));
            let (_, scan) = parquet_engine::table_input(
                dispatcher,
                &parquet,
                Projection::columns([column_idx]),
                true,
            )
            .into_parts();
            let written = OperatorSpec::new(dispatcher.clone(), scan)
                .chain(
                    stealable(dispatcher.topology()).into_iter().collect(),
                    (0..worker_count)
                        .map(|_| TokenizeRowsFactory {
                            places: places.clone(),
                            worker_count,
                        })
                        .collect(),
                )
                .chain(
                    return_to_worker_mpsc(worker_count).into_iter().collect(),
                    DefaultUnaryFactory::<IndexAssembler<usize>>::create_for_workers(worker_count),
                )
                .chain(
                    stealable(dispatcher.topology()).into_iter().collect(),
                    (0..worker_count)
                        .map(|_| WriteIndexFilesFactory {
                            store: self.store(),
                            index_keys: index_keys.clone(),
                        })
                        .collect(),
                )
                .map_each(|written: WrittenIndex| Ok((written.file, written.len)))
                .collect()
                .map_err(Error::TextIndexDataflow)?;

            let new_files = written
                .into_iter()
                .map(|(file, len)| {
                    (
                        files[file].file_ref().path.clone(),
                        index_keys[file].clone(),
                        len,
                    )
                })
                .collect();
            let opened = self.open_index_files(new_files)?;
            built += opened.len();
            self.text_indexes().extend(column, opened);
        }
        Ok(built)
    }

    /// The text indexes of `column` over the data files of this copy of the
    /// table.
    pub(crate) fn column_text_indexes(&self, column: &str) -> Result<Arc<ColumnIndexes>> {
        let indexed = self.text_index_files(column)?;
        let mut by_open_file = HashMap::new();
        let mut covers_every_file = true;
        for table_file in self.files() {
            // A file with no row group holds no row to select.
            let Some(first) = table_file.row_groups.first() else {
                continue;
            };
            match indexed.by_data_file.get(&table_file.file_ref().path) {
                Some(index) => {
                    let file_row_count: i64 =
                        table_file.row_groups.iter().map(|rg| rg.num_rows).sum();
                    by_open_file.insert(
                        first.open_file.clone(),
                        FileIndex {
                            index: index.clone(),
                            file_row_count: file_row_count as u64,
                        },
                    );
                }
                None => covers_every_file = false,
            }
        }
        Ok(Arc::new(ColumnIndexes {
            by_open_file,
            covers_every_file,
        }))
    }

    fn index_key(&self, data_file: &ObjectPath, column: &str) -> ObjectPath {
        self.object_location()
            .join(&index_object_name(data_file, column))
    }

    /// The index files of `column`, found by listing the table's location the
    /// first time the column is asked about.
    fn text_index_files(&self, column: &str) -> Result<Arc<ColumnIndexFiles>> {
        if let Some(known) = self.text_indexes().get(column) {
            return Ok(known);
        }
        let listed: HashMap<String, u64> = self
            .store()
            .list(self.object_location())?
            .objects
            .into_iter()
            .filter(|object| object.file.path.as_str().ends_with(INDEX_SUFFIX))
            .map(|object| (object.file.path.name().to_string(), object.file.size))
            .collect();
        let found = self
            .files()
            .iter()
            .filter_map(|file| {
                let data_file = &file.file_ref().path;
                let size = listed.get(&index_object_name(data_file, column))?;
                Some((
                    data_file.clone(),
                    self.index_key(data_file, column),
                    *size as usize,
                ))
            })
            .collect();
        let opened = self.open_index_files(found)?;
        Ok(self.text_indexes().extend(column, opened))
    }

    /// Open index files for reading: `(data file, index object, size)` each.
    /// Opened on a worker, which is where a newly opened descriptor is
    /// registered with the caches. The open files are kept for the table's
    /// lifetime, so what the caches hold for them stays addressable.
    fn open_index_files(
        &self,
        files: Vec<(ObjectPath, ObjectPath, usize)>,
    ) -> Result<Vec<(ObjectPath, IndexFile)>> {
        let store = self.store();
        values_input(self.dispatcher(), files)
            .map_each(move |(data_file, index_key, len)| {
                let file = store
                    .source(&index_key)
                    .map_err(parquet_engine::op_err)?
                    .open_read(len as u64)
                    .map_err(parquet_engine::op_err)?;
                memory_ctx().caches().invalidate_reopened_file(&file);
                Ok((data_file, IndexFile { file, len }))
            })
            .collect()
            .map_err(Error::TextIndexDataflow)
    }
}

impl TextIndexRegistry {
    fn get(&self, column: &str) -> Option<Arc<ColumnIndexFiles>> {
        self.columns.lock().unwrap().get(column).cloned()
    }

    /// Add `files` to what is known for `column` and return the result.
    fn extend(&self, column: &str, files: Vec<(ObjectPath, IndexFile)>) -> Arc<ColumnIndexFiles> {
        let mut columns = self.columns.lock().unwrap();
        let mut by_data_file = columns
            .get(column)
            .map(|known| known.by_data_file.clone())
            .unwrap_or_default();
        by_data_file.extend(files);
        let known = Arc::new(ColumnIndexFiles { by_data_file });
        columns.insert(column.to_string(), known.clone());
        known
    }
}
