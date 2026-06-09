//! Source stage: a work-stealing queue that hands out every input file (tagged
//! with its index) exactly once, to whatever worker steals it next.

use super::IndexedFile;
use crate::store::DataFileLocation;
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::sync::Arc;

/// Builds a [`FileInjector`] per worker, all sharing one queue of the files.
#[derive(Clone)]
pub(super) struct FileInjectorFactory {
    files: Arc<Injector<IndexedFile>>,
}

impl FileInjectorFactory {
    pub(super) fn new(files: &[DataFileLocation]) -> Self {
        let injector = Injector::new();
        for (idx, location) in files.iter().enumerate() {
            injector.push((idx, location.clone()));
        }
        Self {
            files: Arc::new(injector),
        }
    }
}

impl RootChannelFactory<IndexedFile> for FileInjectorFactory {
    type Receiver = FileInjector;

    fn build(self) -> FileInjector {
        FileInjector { files: self.files }
    }
}

pub(super) struct FileInjector {
    files: Arc<Injector<IndexedFile>>,
}

impl Receiver<IndexedFile> for FileInjector {
    fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn try_recv(&self) -> Option<IndexedFile> {
        None
    }

    fn steal(&self) -> Option<IndexedFile> {
        loop {
            match self.files.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                Steal::Success(file) => return Some(file),
            }
        }
    }
}
