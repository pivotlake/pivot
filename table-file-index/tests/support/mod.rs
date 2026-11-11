//! What this crate's integration tests share: a file record whose bounds on
//! the sort key are the values a test gives it.

#![allow(dead_code)]

use std::sync::Arc;

use object_storage::{FileRef, ObjectPath};
use table_file_index::{BoundSide, FileFacts, FileIndex, IndexedRecord, SortValue};

pub struct Record {
    facts: FileFacts,
    lower: Vec<SortValue<'static>>,
    upper: Vec<SortValue<'static>>,
}

impl IndexedRecord for Record {
    fn facts(&self) -> &FileFacts {
        &self.facts
    }

    fn count_sort_columns(&self) -> usize {
        self.lower.len()
    }

    fn read_sort_bound(&self, position: usize, side: BoundSide) -> SortValue<'_> {
        match side {
            BoundSide::Lower => self.lower[position],
            BoundSide::Upper => self.upper[position],
        }
    }

    fn estimate_heap_bytes(&self) -> u64 {
        self.facts.estimate_heap_bytes()
    }
}

/// The record of the file at `path`, committed at `commit_sequence`, whose
/// rows run from `lower` to `upper` on the sort key, a value per column.
pub fn record_between(
    path: &str,
    lower: &[SortValue<'static>],
    upper: &[SortValue<'static>],
    commit_sequence: u64,
) -> Arc<Record> {
    let file = FileRef {
        path: ObjectPath::new(path),
        size: 10,
    };
    Arc::new(Record {
        facts: FileFacts::new(file, Some(1), commit_sequence),
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    })
}

/// The record of the file at `path`, committed at `commit_sequence`, whose
/// rows all hold `key` on the sort key, a value per column.
pub fn record(path: &str, key: &[SortValue<'static>], commit_sequence: u64) -> Arc<Record> {
    record_between(path, key, key, commit_sequence)
}

/// The index of `records`, inserted in the order given.
pub fn index_of(records: impl IntoIterator<Item = Arc<Record>>) -> FileIndex<Record> {
    let mut index = FileIndex::default();
    for record in records {
        index.insert(record);
    }
    index
}

pub fn list_paths(index: &FileIndex<Record>) -> Vec<&str> {
    index
        .files()
        .map(|record| record.facts().file().path.as_str())
        .collect()
}
