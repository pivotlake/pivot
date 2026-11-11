//! The live data files of one partition of a table version, held once: every
//! file as one record, in an order the version fixes.
//!
//! # The order
//!
//! Files stand by where they start on the table's sort key, their lower
//! bounds on it compared as [`crate::sort_key`] orders them. Files that
//! start at the same place, which is every file of a table without a sort
//! key, stand in the order they were committed in, and files of one commit
//! by path, which only makes the order total. A walk over a partition
//! therefore meets the files of an unsorted table oldest first.
//!
//! # The records
//!
//! A format keeps one record per file: what the format itself reads of the
//! file, with the bounds its metadata recorded for the file's columns,
//! beside the facts every format knows ([`FileFacts`]). A record says what
//! it holds through [`IndexedRecord`]: its facts, and its bounds on the
//! table's sort key, read where the record holds them. Maintenance
//! therefore walks the files of any format alike, and compares the
//! stretches of the sort key two files cover with no bound copied out of a
//! record.
//!
//! # Persistence
//!
//! A [`FileIndex`] is persistent: cloning shares every record and tree
//! node, and an edit copies only what it touches. The next version's index
//! of a partition is the previous one's with the files a commit removed
//! taken out and those it added put in, so a version costs what changed,
//! and a query keeps the index it started on.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::ControlFlow;
use std::sync::Arc;

use imbl::{HashMap as SharedMap, OrdSet};
use object_storage::{FileRef, ObjectPath};

use parquet_engine::measure_allocation;

use crate::sort_key::{BoundSide, SortValue, compare_sort_bounds};

/// What every table format knows of one live data file.
#[derive(Clone, Debug)]
pub struct FileFacts {
    /// The file's path as its table names it, and its size.
    file: FileRef,
    /// How many rows the file holds, deleted ones included, when the
    /// table's metadata records it.
    row_count: Option<u64>,
    /// Where the file's commit stands among the table's commits: a later
    /// commit's is not below an earlier one's.
    commit_sequence: u64,
}

impl FileFacts {
    /// The facts of `file`, holding `row_count` rows that the commit
    /// standing at `commit_sequence` among its table's added.
    pub fn new(file: FileRef, row_count: Option<u64>, commit_sequence: u64) -> Self {
        Self {
            file,
            row_count,
            commit_sequence,
        }
    }

    /// The file's path as its table names it, and its size.
    pub fn file(&self) -> &FileRef {
        &self.file
    }

    /// How many rows the file holds, deleted ones included, when the
    /// table's metadata records it.
    pub fn row_count(&self) -> Option<u64> {
        self.row_count
    }

    /// Where the file's commit stands among the table's commits.
    pub fn commit_sequence(&self) -> u64 {
        self.commit_sequence
    }

    /// An estimate of the memory the facts hold outside themselves, with
    /// what an index's own maps spend on one file: the file's path, at the
    /// size an allocation of it takes, and the file's entry in the order
    /// and in the map by path, each of which holds the file's record and
    /// nothing of its own.
    pub fn estimate_heap_bytes(&self) -> u64 {
        /// What the order and the map by path spend on one file between
        /// them: each a pointer to the file's record, and its share of the
        /// nodes the pointers sit in.
        const INDEX_ENTRY_BYTES: usize = 2 * size_of::<usize>() + 96;

        let total = INDEX_ENTRY_BYTES + measure_allocation(self.file.path.as_str().len());
        total as u64
    }
}

/// One file as a format's index holds it: the facts every format knows,
/// beside whatever else the format keeps of the file, the file's bounds on
/// the table's sort key included.
pub trait IndexedRecord: Send + Sync + 'static {
    fn facts(&self) -> &FileFacts;

    /// How many columns the file's table is sorted by.
    fn count_sort_columns(&self) -> usize;

    /// The file's bound on `side` of the sort column at `position`, which
    /// is below [`count_sort_columns`](Self::count_sort_columns), read
    /// where the record holds it, as a value of the column's current type.
    /// The same value for as long as the record lives: the order of the
    /// index the record stands in depends on it.
    fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_>;

    /// An estimate of the memory the record holds, itself included, with
    /// what an index's own maps spend on it
    /// ([`FileFacts::estimate_heap_bytes`]).
    fn estimate_heap_bytes(&self) -> u64;
}

/// A record as the order of an index holds it: it stands by the lower
/// bound of the table's sort key in its file, then by when its file was
/// committed, then by its file's path, which only makes the order total.
struct RecordInOrder<Record>(Arc<Record>);

impl<Record> Clone for RecordInOrder<Record> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<Record: IndexedRecord> PartialEq for RecordInOrder<Record> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<Record: IndexedRecord> Eq for RecordInOrder<Record> {}

impl<Record: IndexedRecord> PartialOrd for RecordInOrder<Record> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Record: IndexedRecord> Ord for RecordInOrder<Record> {
    fn cmp(&self, other: &Self) -> Ordering {
        let (left, right) = (self.0.facts(), other.0.facts());
        compare_sort_bounds(&*self.0, BoundSide::Lower, &*other.0, BoundSide::Lower)
            .then_with(|| left.commit_sequence.cmp(&right.commit_sequence))
            .then_with(|| left.file.path.as_str().cmp(right.file.path.as_str()))
    }
}

/// A record as a map by path holds it: it is looked up by its file's path,
/// which it hashes and compares as.
struct RecordByPath<Record>(Arc<Record>);

impl<Record: IndexedRecord> RecordByPath<Record> {
    fn read_path(&self) -> &str {
        self.0.facts().file.path.as_str()
    }
}

impl<Record> Clone for RecordByPath<Record> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<Record: IndexedRecord> PartialEq for RecordByPath<Record> {
    fn eq(&self, other: &Self) -> bool {
        self.read_path() == other.read_path()
    }
}

impl<Record: IndexedRecord> Eq for RecordByPath<Record> {}

impl<Record: IndexedRecord> Hash for RecordByPath<Record> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.read_path().hash(state);
    }
}

impl<Record: IndexedRecord> Borrow<str> for RecordByPath<Record> {
    fn borrow(&self) -> &str {
        self.read_path()
    }
}

/// Files by their paths, in no order. Cloning shares everything; an edit
/// copies only what it touches. A file's path is held once, in its record,
/// which the map looks the file up by.
pub struct FilesByPath<Record> {
    files: SharedMap<RecordByPath<Record>, ()>,
}

impl<Record> Clone for FilesByPath<Record> {
    fn clone(&self) -> Self {
        Self {
            files: self.files.clone(),
        }
    }
}

impl<Record> Default for FilesByPath<Record> {
    fn default() -> Self {
        Self {
            files: SharedMap::new(),
        }
    }
}

impl<Record: IndexedRecord> FilesByPath<Record> {
    /// Add `record`, replacing a file already at its path.
    pub fn insert(&mut self, record: Arc<Record>) {
        self.remove(&record.facts().file.path);
        self.files.insert(RecordByPath(record), ());
    }

    /// Drop the file at `path`, if the map has one.
    pub fn remove(&mut self, path: &ObjectPath) -> Option<Arc<Record>> {
        let (RecordByPath(record), ()) = self.files.remove_with_key(path.as_str())?;
        Some(record)
    }

    /// The file at `path`.
    pub fn find(&self, path: &ObjectPath) -> Option<&Arc<Record>> {
        let (RecordByPath(record), ()) = self.files.get_key_value(path.as_str())?;
        Some(record)
    }

    /// How many files the map holds.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

/// The live data files of one partition of one table version. Cloning
/// shares everything; an edit copies only what it touches.
///
/// The order and the map by path each hold a file's record and read what
/// they order and look up by off it, so a file's path and bounds are held
/// once, in its record.
pub struct FileIndex<Record> {
    files: OrdSet<RecordInOrder<Record>>,
    by_path: FilesByPath<Record>,
    /// The sum of the files' [`IndexedRecord::estimate_heap_bytes`].
    heap_bytes: u64,
}

impl<Record> Clone for FileIndex<Record> {
    fn clone(&self) -> Self {
        Self {
            files: self.files.clone(),
            by_path: self.by_path.clone(),
            heap_bytes: self.heap_bytes,
        }
    }
}

impl<Record> Default for FileIndex<Record> {
    fn default() -> Self {
        Self {
            files: OrdSet::new(),
            by_path: FilesByPath::default(),
            heap_bytes: 0,
        }
    }
}

impl<Record: IndexedRecord> FileIndex<Record> {
    /// Add `record`, replacing a file already at its path.
    pub fn insert(&mut self, record: Arc<Record>) {
        self.remove(&record.facts().file.path);
        self.heap_bytes += record.estimate_heap_bytes();
        self.by_path.insert(Arc::clone(&record));
        self.files.insert(RecordInOrder(record));
    }

    /// Drop the file at `path`, if the index has one.
    pub fn remove(&mut self, path: &ObjectPath) -> Option<Arc<Record>> {
        let in_order = RecordInOrder(self.by_path.remove(path)?);
        self.files.remove(&in_order);
        self.heap_bytes -= in_order.0.estimate_heap_bytes();
        Some(in_order.0)
    }

    /// The file at `path`.
    pub fn find(&self, path: &ObjectPath) -> Option<&Arc<Record>> {
        self.by_path.find(path)
    }

    /// Every file, in key order.
    pub fn files(&self) -> impl Iterator<Item = &Arc<Record>> {
        self.files.iter().map(|RecordInOrder(record)| record)
    }

    /// How many files the index holds.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// An estimate of the memory the index holds: the sum of its files'.
    pub fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// What a walk does with each file of a partition: `Break` ends the walk.
pub type VisitFacts<'v> = dyn FnMut(&FileFacts) -> ControlFlow<()> + 'v;

/// What a walk does with the record of each file of a partition: `Break`
/// ends the walk.
pub type VisitRecord<'v> = dyn FnMut(&Arc<dyn IndexedRecord>) -> ControlFlow<()> + 'v;

/// The files of one partition as whatever walks them reads them, whichever
/// format's records the partition's index holds.
pub trait PartitionFiles: Send + Sync {
    /// Visit every file, in key order; `Break` when the visit ended the
    /// walk.
    fn visit_files(&self, visit: &mut VisitFacts<'_>) -> ControlFlow<()>;

    /// Visit the record of every file, in key order, for a walk that
    /// compares the files' bounds on the sort key and may keep a record
    /// past its visit; `Break` when the visit ended the walk.
    fn visit_records(&self, visit: &mut VisitRecord<'_>) -> ControlFlow<()>;

    /// The file at `path`.
    fn find_file(&self, path: &ObjectPath) -> Option<&FileFacts>;

    /// How many files the partition holds.
    fn count_files(&self) -> usize;
}

impl<Record: IndexedRecord> PartitionFiles for FileIndex<Record> {
    fn visit_files(&self, visit: &mut VisitFacts<'_>) -> ControlFlow<()> {
        for record in self.files() {
            visit(record.facts())?;
        }
        ControlFlow::Continue(())
    }

    fn visit_records(&self, visit: &mut VisitRecord<'_>) -> ControlFlow<()> {
        for record in self.files() {
            let record: Arc<dyn IndexedRecord> = Arc::clone(record) as _;
            visit(&record)?;
        }
        ControlFlow::Continue(())
    }

    fn find_file(&self, path: &ObjectPath) -> Option<&FileFacts> {
        self.find(path).map(|record| record.facts())
    }

    fn count_files(&self) -> usize {
        self.len()
    }
}
