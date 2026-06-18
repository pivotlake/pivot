//! Source stage: a work-stealing queue that hands out every input file exactly
//! once, to whatever worker steals it next.

use crate::store::DataFile;
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::sync::Arc;

/// Builds a [`FileInjector`] per worker, all sharing one queue of the files.
#[derive(Clone)]
pub(super) struct FileInjectorFactory {
    files: Arc<Injector<DataFile>>,
}

impl FileInjectorFactory {
    pub(super) fn new(files: &[DataFile]) -> Self {
        let injector = Injector::new();
        for file in files {
            injector.push(file.clone());
        }
        Self {
            files: Arc::new(injector),
        }
    }
}

impl RootChannelFactory<DataFile> for FileInjectorFactory {
    type Receiver = FileInjector;

    fn build(self) -> FileInjector {
        FileInjector { files: self.files }
    }
}

pub(super) struct FileInjector {
    files: Arc<Injector<DataFile>>,
}

impl Receiver<DataFile> for FileInjector {
    fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn try_recv(&self) -> Option<DataFile> {
        // Drive footer loading from the eager `run_cpu_work` path (which calls
        // `try_recv`), not only the worker's idle `steal` path — otherwise each
        // worker fetches one footer at a time and a large catch-up (many new
        // files committed since the last query) serialises into ~1s. A single
        // non-spinning attempt keeps the hot loop from spinning on `Steal::Retry`;
        // the next iteration retries. Mirrors the `RowGroupInjector` fix.
        match self.files.steal() {
            Steal::Success(file) => Some(file),
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<DataFile> {
        loop {
            match self.files.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                Steal::Success(file) => return Some(file),
            }
        }
    }
}
