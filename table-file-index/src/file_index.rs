//! The live data files of one partition of a table version, held once: every
//! file as one record, in an order the version fixes.
//!
//! # The order
//!
//! Files stand by where they start on the table's sort key. Files that start
//! at the same place, which is every file of a table without a sort key,
//! stand in the order they were committed in, and files of one commit by
//! path, which only makes the order total. A walk over a partition
//! therefore meets the files of an unsorted table oldest first.
//!
//! # The records
//!
//! What every format knows of a file is its [`FileFacts`]: maintenance
//! reads nothing else, so it walks the files of any format alike, and
//! compares the stretches of the sort key two files cover as bytes, with no
//! bound decoded. A format's record holds the facts beside what the format
//! itself reads of the file, and says so through [`IndexedRecord`].
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

use parquet_engine::{measure_allocation, measure_shared_allocation};

/// Where a file sorts among its partition's files: by the lower bound of
/// the table's sort key in it, then by when it was committed, then by its
/// path, which only makes the order total. A view of the file's
/// [`FileFacts`], which hold what it orders by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FileKey<'f> {
    sort_minimum: &'f [u8],
    commit_sequence: u64,
    path: &'f str,
}

/// What every table format knows of one live data file.
#[derive(Clone, Debug)]
pub struct FileFacts {
    /// The file's path as its table names it, and its size.
    file: FileRef,
    /// How many rows the file holds, deleted ones included, when the
    /// table's metadata records it.
    row_count: Option<u64>,
    /// The sort key's lower bound across the file as one row whose bytes
    /// order as the bounds do; a column without a bound orders first. Empty
    /// for a table without a sort key.
    sort_minimum: Arc<[u8]>,
    /// The sort key's upper bound across the file, in the byte form of the
    /// key's lower bound; `None` when the table has no sort key with an
    /// ordered byte form, or the file's metadata does not bound the column
    /// the key leads with on both sides.
    sort_maximum: Option<Arc<[u8]>>,
    /// Where the file's commit stands among the table's commits: a later
    /// commit's is not below an earlier one's.
    commit_sequence: u64,
}

impl FileFacts {
    /// The facts of `file`, holding `row_count` rows that the commit
    /// standing at `commit_sequence` among its table's added, whose rows
    /// start at `sort_minimum` on the table's sort key and end at
    /// `sort_maximum` (see [`SortKeyEncoder`](crate::SortKeyEncoder)).
    pub fn new(
        file: FileRef,
        row_count: Option<u64>,
        sort_minimum: Arc<[u8]>,
        sort_maximum: Option<Arc<[u8]>>,
        commit_sequence: u64,
    ) -> Self {
        Self {
            file,
            row_count,
            sort_minimum,
            sort_maximum,
            commit_sequence,
        }
    }

    /// The file's path as its table names it, and its size.
    pub fn file(&self) -> &FileRef {
        &self.file
    }

    /// Where the file sorts among its partition's files.
    pub(crate) fn key(&self) -> FileKey<'_> {
        FileKey {
            sort_minimum: &self.sort_minimum,
            commit_sequence: self.commit_sequence,
            path: self.file.path.as_str(),
        }
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

    /// The stretch of the table's sort key the file covers, its lowest and
    /// highest keys as bytes that order as the keys do. `None` for a table
    /// without a sort key, and for a file whose metadata does not bound the
    /// column the key leads with.
    pub fn sort_range(&self) -> Option<(&[u8], &[u8])> {
        let sort_maximum = self.sort_maximum.as_deref()?;
        Some((&self.sort_minimum, sort_maximum))
    }

    /// An estimate of the memory the facts hold outside themselves, with
    /// what an index's own maps spend on one file: the file's path and each
    /// bound of the sort key, at the size an allocation of it takes, and
    /// the file's entry in the order and in the map by path, each of which
    /// holds the file's record and nothing of its own.
    pub fn estimate_heap_bytes(&self) -> u64 {
        /// What the order and the map by path spend on one file between
        /// them: each a pointer to the file's record, and its share of the
        /// nodes the pointers sit in.
        const INDEX_ENTRY_BYTES: usize = 2 * size_of::<usize>() + 96;

        let bound_bytes = measure_shared_allocation(self.sort_minimum.len())
            + self
                .sort_maximum
                .as_ref()
                .map_or(0, |bound| measure_shared_allocation(bound.len()));
        let total =
            INDEX_ENTRY_BYTES + measure_allocation(self.file.path.as_str().len()) + bound_bytes;
        total as u64
    }
}

/// One file as a format's index holds it: the facts every format knows,
/// beside whatever else the format keeps of the file.
pub trait IndexedRecord: Send + Sync + 'static {
    fn facts(&self) -> &FileFacts;

    /// An estimate of the memory the record holds, itself included, with
    /// what an index's own maps spend on it
    /// ([`FileFacts::estimate_heap_bytes`]).
    fn estimate_heap_bytes(&self) -> u64;
}

/// A record as the order of an index holds it: it stands where its
/// [`FileKey`] does.
struct RecordInOrder<Record>(Arc<Record>);

impl<Record> Clone for RecordInOrder<Record> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<Record: IndexedRecord> PartialEq for RecordInOrder<Record> {
    fn eq(&self, other: &Self) -> bool {
        self.0.facts().key() == other.0.facts().key()
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
        self.0.facts().key().cmp(&other.0.facts().key())
    }
}

/// A record as the map by path of an index holds it: it is looked up by
/// its file's path, which it hashes and compares as.
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

/// The live data files of one partition of one table version. Cloning
/// shares everything; an edit copies only what it touches.
///
/// The order and the map by path each hold a file's record and read what
/// they order and look up by off its facts, so a file's path and bounds
/// are held once, in its record.
pub struct FileIndex<Record> {
    files: OrdSet<RecordInOrder<Record>>,
    by_path: SharedMap<RecordByPath<Record>, ()>,
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
            by_path: SharedMap::new(),
            heap_bytes: 0,
        }
    }
}

impl<Record: IndexedRecord> FileIndex<Record> {
    /// Add `record`, replacing a file already at its path.
    pub fn insert(&mut self, record: Arc<Record>) {
        self.remove(&record.facts().file.path);
        self.heap_bytes += record.estimate_heap_bytes();
        self.by_path.insert(RecordByPath(Arc::clone(&record)), ());
        self.files.insert(RecordInOrder(record));
    }

    /// Drop the file at `path`, if the index has one.
    pub fn remove(&mut self, path: &ObjectPath) -> Option<Arc<Record>> {
        let (RecordByPath(record), ()) = self.by_path.remove_with_key(path.as_str())?;
        let in_order = RecordInOrder(record);
        self.files.remove(&in_order);
        self.heap_bytes -= in_order.0.estimate_heap_bytes();
        Some(in_order.0)
    }

    /// The file at `path`.
    pub fn find(&self, path: &ObjectPath) -> Option<&Arc<Record>> {
        let (RecordByPath(record), ()) = self.by_path.get_key_value(path.as_str())?;
        Some(record)
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

/// The files of one partition as whatever walks them reads them, whichever
/// format's records the partition's index holds.
pub trait PartitionFiles: Send + Sync {
    /// Visit every file, in key order; `Break` when the visit ended the
    /// walk.
    fn visit_files(&self, visit: &mut VisitFacts<'_>) -> ControlFlow<()>;

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

    fn find_file(&self, path: &ObjectPath) -> Option<&FileFacts> {
        self.find(path).map(|record| record.facts())
    }

    fn count_files(&self) -> usize {
        self.len()
    }
}
