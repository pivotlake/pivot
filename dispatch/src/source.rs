use crossbeam_deque::{Injector, Steal};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;
use tracing::debug;

/// A source of parquet files, built from a directory of parquets. It holds an Injector of Paths,
/// continuously being stolen from as work is done on the given parquet paths.
pub struct Source {
    parquets: Injector<PathBuf>,
}

impl Source {
    pub fn new(parquets: Injector<PathBuf>) -> Self {
        Self { parquets }
    }

    pub fn from_directory(path: &Path) -> Self {
        let injector = Injector::new();

        if let Ok(rd) = fs::read_dir(path) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_file() {
                    injector.push(p);
                }
            }
        }
        debug!("Injector start count: {:?}", injector.len());

        Self::new(injector)
    }

    pub fn is_empty(&self) -> bool {
        self.parquets.is_empty()
    }

    pub fn pop_parquet_path(&self) -> Option<PathBuf> {
        loop {
            return match self.parquets.steal() {
                Steal::Empty => None,
                Steal::Success(s) => Some(s),
                Steal::Retry => {
                    sleep(Duration::from_millis(1));
                    continue;
                }
            };
        }
    }
}
