//! What tests take of a table that nothing else does. The tests of the
//! crates built on this one reach it through the `test-support` feature;
//! this crate's unit tests also share one worker pool and a table around
//! hand-made metadata.

use std::ops::ControlFlow;

use crate::Result;
use crate::files::data_files::DataFileEntry;
use crate::table::Table;

#[cfg(test)]
mod fixtures;
#[cfg(test)]
pub(crate) use fixtures::{build_table_at, test_pool};

impl Table {
    /// Every live data file of this version as a commit records it, by
    /// partition and then in the partition's order. Every data manifest is
    /// read.
    pub fn entries(&self) -> Result<Vec<DataFileEntry>> {
        let mut entries = Vec::new();
        self.visit_partitions(&mut |_, index| {
            for file in index.files() {
                entries.push(file.to_entry(self.metadata(), self.columns())?);
            }
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(entries)
    }
}
