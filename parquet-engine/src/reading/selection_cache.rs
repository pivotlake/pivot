//! Row selections remembered across queries: which rows of a row group
//! passed a scan predicate, keyed by the file, the row group within it, and
//! the predicate's fingerprint.
//!
//! A scan meeting a remembered predicate reads only the rows that passed it,
//! decoding and testing nothing else, and skips a row group none of whose
//! rows passed. A scan meeting a predicate the cache has nothing for reads
//! the row group whole and gathers which rows pass while its ranges decode,
//! on whichever workers, filing the selection once every row was tested.
//!
//! Reading by selection pays only when the selection is sparse: every run
//! of kept rows costs a decoder a stop and a restart, so a predicate most
//! rows pass would read slower by selection than whole. Such a predicate is
//! remembered as dense, and a scan meeting it again neither reads by it nor
//! gathers it again.
//!
//! Gathering costs the scan that does it, so a predicate is gathered only
//! from its second scan of a row group on: the first scan just notes having
//! met it, and a predicate run once never pays. `PIVOT_SELECTION_CACHE_BYTES`
//! bounds the cache (`0` turns it off); past the bound the oldest entries go.

use dispatch::io::OpenFile;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

const DEFAULT_BUDGET_BYTES: usize = 512 << 20;
const BUDGET_VARIABLE: &str = "PIVOT_SELECTION_CACHE_BYTES";
/// A selection is kept when at most this share of the row group's rows pass:
/// one row in 64.
const SPARSE_DIVISOR: usize = 64;

/// Which rows of a row group pass one predicate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SelectionKey {
    file: OpenFile,
    row_group_in_file: usize,
    predicate: u64,
}

impl SelectionKey {
    pub fn new(file: OpenFile, row_group_in_file: usize, predicate: u64) -> Self {
        Self {
            file,
            row_group_in_file,
            predicate,
        }
    }
}

/// What the cache remembers of a predicate on a row group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Remembered {
    /// Met once, and not yet worth gathering.
    Met,
    /// The rows passing it, sorted ascending: few enough to read by.
    Rows(Arc<[u32]>),
    /// Too many rows pass it for reading by selection to pay.
    Dense,
}

impl Remembered {
    fn bytes(&self) -> usize {
        match self {
            Remembered::Rows(rows) => rows.len() * size_of::<u32>(),
            Remembered::Met | Remembered::Dense => size_of::<Self>(),
        }
    }
}

struct Selections {
    by_key: HashMap<SelectionKey, Remembered>,
    /// Keys in insertion order, oldest first, for eviction.
    insertion_order: VecDeque<SelectionKey>,
    bytes: usize,
}

/// The process-wide store of remembered selections.
pub struct SelectionCache {
    selections: Mutex<Selections>,
    budget_bytes: usize,
}

impl SelectionCache {
    /// The cache every scan shares, sized by `PIVOT_SELECTION_CACHE_BYTES`.
    pub fn global() -> &'static SelectionCache {
        static CACHE: OnceLock<SelectionCache> = OnceLock::new();
        CACHE.get_or_init(|| SelectionCache::with_budget(budget_from_environment()))
    }

    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            selections: Mutex::new(Selections {
                by_key: HashMap::new(),
                insertion_order: VecDeque::new(),
                bytes: 0,
            }),
            budget_bytes,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.budget_bytes > 0
    }

    /// What is remembered of `key`'s predicate on its row group.
    pub fn get(&self, key: &SelectionKey) -> Option<Remembered> {
        self.selections.lock().unwrap().by_key.get(key).cloned()
    }

    /// Remembers `remembered` for `key`, letting the oldest selections go
    /// while the cache is over its budget.
    pub fn insert(&self, key: SelectionKey, remembered: Remembered) {
        let mut selections = self.selections.lock().unwrap();
        let bytes = remembered.bytes();
        if let Some(previous) = selections.by_key.insert(key.clone(), remembered) {
            selections.bytes -= previous.bytes();
        } else {
            selections.insertion_order.push_back(key);
        }
        selections.bytes += bytes;
        while selections.bytes > self.budget_bytes {
            let oldest = selections
                .insertion_order
                .pop_front()
                .expect("a cache over budget holds a selection");
            if let Some(evicted) = selections.by_key.remove(&oldest) {
                selections.bytes -= evicted.bytes();
            }
        }
    }

    pub fn len(&self) -> usize {
        self.selections.lock().unwrap().by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn budget_from_environment() -> usize {
    dispatch::env::get_env_var_with_default(BUDGET_VARIABLE, DEFAULT_BUDGET_BYTES)
}

/// Gathers the rows of one row group that pass one predicate while the row
/// group's ranges decode, on any worker, and files the selection once every
/// row was tested.
pub struct SelectionCollector {
    /// The predicate's position among the scan's predicates.
    predicate_index: usize,
    key: SelectionKey,
    /// The row group's row count: the selection is complete once this many
    /// rows were tested.
    rows: u32,
    gathered: Mutex<Gathered>,
}

struct Gathered {
    passing: Vec<u32>,
    tested: u32,
}

impl SelectionCollector {
    pub fn new(predicate_index: usize, key: SelectionKey, rows: u32) -> Self {
        Self {
            predicate_index,
            key,
            rows,
            gathered: Mutex::new(Gathered {
                passing: Vec::new(),
                tested: 0,
            }),
        }
    }

    pub fn predicate_index(&self) -> usize {
        self.predicate_index
    }

    /// Records `tested` more rows of the row group as tested, `passing` being
    /// the row-group rows among them that passed. Files the selection with
    /// the cache once every row of the row group was tested: the sorted
    /// passing rows when they are sparse, a dense verdict otherwise.
    pub fn record(&self, tested: u32, passing: impl Iterator<Item = u32>) {
        let mut gathered = self.gathered.lock().unwrap();
        gathered.passing.extend(passing);
        gathered.tested += tested;
        if gathered.tested < self.rows {
            return;
        }
        let mut rows = std::mem::take(&mut gathered.passing);
        drop(gathered);
        let remembered = if rows.len() * SPARSE_DIVISOR > self.rows as usize {
            Remembered::Dense
        } else {
            rows.sort_unstable();
            Remembered::Rows(rows.into())
        };
        SelectionCache::global().insert(self.key.clone(), remembered);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dispatch::io::LocalFile;

    fn key(predicate: u64) -> SelectionKey {
        let file = LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
        SelectionKey::new(OpenFile::Local(file), 0, predicate)
    }

    #[test]
    fn an_inserted_selection_is_found_until_the_budget_evicts_it() {
        let cache = SelectionCache::with_budget(12);
        let first = key(1);

        cache.insert(first.clone(), Remembered::Rows(Arc::from([1u32, 2])));
        let found = cache.get(&first);
        cache.insert(key(2), Remembered::Rows(Arc::from([5u32, 6])));
        let evicted = cache.get(&first);

        assert_eq!(found, Some(Remembered::Rows(Arc::from([1u32, 2]))));
        assert!(evicted.is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn a_collector_files_the_sorted_selection_once_every_row_was_tested() {
        let key = key(7);
        let collector = SelectionCollector::new(0, key.clone(), 640);

        collector.record(320, [400u32, 500].into_iter());
        let before_last_rows = SelectionCache::global().get(&key);
        collector.record(320, [1u32].into_iter());
        let after_last_rows = SelectionCache::global().get(&key);

        assert!(before_last_rows.is_none());
        assert_eq!(
            after_last_rows,
            Some(Remembered::Rows(Arc::from([1u32, 400, 500])))
        );
    }

    #[test]
    fn a_met_predicate_is_replaced_by_its_gathered_rows() {
        let key = key(9);
        let cache = SelectionCache::with_budget(1 << 10);

        cache.insert(key.clone(), Remembered::Met);
        let met = cache.get(&key);
        cache.insert(key.clone(), Remembered::Rows(Arc::from([2u32])));
        let gathered = cache.get(&key);

        assert_eq!(met, Some(Remembered::Met));
        assert_eq!(gathered, Some(Remembered::Rows(Arc::from([2u32]))));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn a_predicate_many_rows_pass_is_remembered_as_dense() {
        let key = key(8);
        let collector = SelectionCollector::new(0, key.clone(), 64);

        collector.record(64, [0u32, 1].into_iter());

        assert_eq!(SelectionCache::global().get(&key), Some(Remembered::Dense));
    }
}
