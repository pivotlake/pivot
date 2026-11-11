//! What tests take of a table that nothing else does: its files listed
//! whole, a metadata store that keeps its versions in memory, and an object
//! store that fails on request. The tests of the crates built on this one
//! reach it through the `test-support` feature.

use std::ops::ControlFlow;

use crate::Result;
use crate::files::data_files::DataFileEntry;
use crate::table::Table;

mod faulty_store;
mod memory_metadata_store;

pub use faulty_store::{Faults, FaultyStore, HeldRead, Requests};
pub use memory_metadata_store::MemoryMetadataStore;

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
