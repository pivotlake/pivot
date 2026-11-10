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
//! files written since.
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
//! A scan with a token predicate on an indexed column asks the indexes which
//! row groups can hold a match before it reads any of them
//! ([`CatalogTable::row_groups_without_match`]). Which files have an index is
//! found by listing the table's location once per column and remembered for
//! the table's lifetime.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use dispatch::io::{
    FsWriteRequest, HttpUploadRequest, LocalFile, OpenFile, OperatorIO, RemoteFile,
};
use dispatch::text_index::{
    AssembledIndex, FileTokenRun, IndexAssembler, IndexFile, IndexLookupsFactory, LookupRequest,
    TextQuery, TokenRunBuilder,
};
use dispatch::{
    BuildContext, Built, BuiltUnary, DefaultUnaryFactory, OperatorSpec, Projection, Sender, Unary,
    UnaryFactory, memory_ctx, return_to_worker_mpsc, stealable, values_input,
};
use object_storage::{DataFileLocation, ObjectPath, ObjectStore};
use parquet_engine::RowGroupMetadata;

use crate::{CatalogTable, Error, Result};

const INDEX_SUFFIX: &str = ".textidx";

/// The name of the index object for `column` of the data file at `data_file`.
/// The data file's own name keeps it readable; a hash of its whole path keeps
/// two files of the same name in different directories apart.
fn index_object_name(data_file: &ObjectPath, column: &str) -> String {
    let column: String = column
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!(
        "{}.{:08x}.{column}{INDEX_SUFFIX}",
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
            let mut builder = TokenRunBuilder::new();
            for batch_row in run_start..run_end {
                let row = place.first_row + rows_in_group.value(batch_row);
                if row != first_row + (batch_row - run_start) as u32 {
                    return Err(parquet_engine::op_err(Error::TextIndex(
                        "a scanned batch does not hold consecutive rows".to_string(),
                    )));
                }
                if texts.is_valid(batch_row) {
                    builder.add_row(row, texts.value(batch_row).as_bytes());
                }
            }
            sender.send(FileTokenRun {
                file: place.file,
                file_row_count: place.file_row_count,
                assembly_worker: place.file % self.worker_count,
                run: builder.finish(first_row..first_row + (run_end - run_start) as u32),
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

/// Identifies a row group across the views built over a table's files.
pub(crate) type RowGroupKey = (OpenFile, usize);

pub(crate) fn row_group_key(row_group: &RowGroupMetadata) -> RowGroupKey {
    (row_group.open_file.clone(), row_group.file_row_group_idx)
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

        let mut places = Vec::new();
        let mut row_groups = Vec::new();
        let mut index_keys = Vec::new();
        for (file, table_file) in unindexed.iter().enumerate() {
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

        let dispatcher = self.dispatcher();
        let worker_count = dispatcher.worker_count();
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
                    unindexed[file].file_ref().path.clone(),
                    index_keys[file].clone(),
                    len,
                )
            })
            .collect();
        let opened = self.open_index_files(new_files)?;
        let built = opened.len();
        self.text_indexes().extend(column, opened);
        Ok(built)
    }

    /// The row groups of this table's files that hold no row matching `query`
    /// on `column`, according to the column's indexes. Row groups of files
    /// without an index are never listed: nothing is known about them.
    pub(crate) fn row_groups_without_match(
        &self,
        column: &str,
        query: &Arc<TextQuery>,
        candidates: &[Arc<RowGroupMetadata>],
    ) -> Result<HashSet<RowGroupKey>> {
        let indexed = self.text_index_files(column)?;
        if indexed.by_data_file.is_empty() {
            return Ok(HashSet::new());
        }
        let scanned_files: HashSet<&OpenFile> = candidates.iter().map(|rg| &rg.open_file).collect();

        // One lookup per indexed file the scan reads, over all of its rows.
        let mut requests = Vec::new();
        let mut row_groups_of_request = Vec::new();
        for table_file in self.files() {
            let Some(index) = indexed.by_data_file.get(&table_file.file_ref().path) else {
                continue;
            };
            let Some(first) = table_file.row_groups.first() else {
                continue;
            };
            if !scanned_files.contains(&first.open_file) {
                continue;
            }
            let file_row_count: i64 = table_file.row_groups.iter().map(|rg| rg.num_rows).sum();
            requests.push(LookupRequest {
                index: index.clone(),
                rows: 0..file_row_count as u32,
                payload: row_groups_of_request.len(),
            });
            row_groups_of_request.push(&table_file.row_groups);
        }
        if requests.is_empty() {
            return Ok(HashSet::new());
        }

        let dispatcher = self.dispatcher();
        let answers = values_input(dispatcher, requests)
            .chain(
                stealable(dispatcher.topology()).into_iter().collect(),
                IndexLookupsFactory::create_for_workers(dispatcher.worker_count(), query.clone()),
            )
            .map_each(|result| Ok((result.payload, result.matching)))
            .collect()
            .map_err(Error::TextIndexDataflow)?;

        let mut without_match = HashSet::new();
        for (request, matching) in answers {
            for row_group in row_groups_of_request[request] {
                let first_row = row_group.file_row_offset as usize;
                let rows = first_row..first_row + row_group.num_rows as usize;
                if !matching.selects_any_in(rows) {
                    without_match.insert(row_group_key(row_group));
                }
            }
        }
        Ok(without_match)
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
                memory_ctx().caches().file_reopened(&file);
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
