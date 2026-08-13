//! Dictionaries built once per column chunk and shared by every claim reading
//! it.
//!
//! A dictionary belongs to a whole column chunk, so whichever rows a reader
//! wants it needs all of it. Splitting a row group puts several readers on the
//! same chunk at once, and without sharing each of them builds the same
//! dictionary again — turning a k-way split into k copies of that work, which
//! is what a short query notices first.
//!
//! The store is scoped to one scan rather than cached on the row group. A built
//! dictionary lives in slab memory and keeps the 2MB write buffer it was carved
//! from alive, so holding one for the process's lifetime would steadily drain
//! the buffer pool. Entries are held weakly on top of that: the dictionary is
//! freed as soon as the last split reading it drops, rather than lingering until
//! the scan ends.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Identifies a column chunk within a scan: its row group, and the leaf within
/// that row group.
pub type ChunkKey = (usize, usize);

#[derive(Default)]
pub struct SharedDictionaries {
    built: Mutex<HashMap<ChunkKey, Weak<dyn Any + Send + Sync>>>,
}

impl SharedDictionaries {
    pub fn new() -> Self {
        Self::default()
    }

    /// The dictionary already built for `key`, if one is still alive.
    ///
    /// Answering `None` only costs a rebuild, never correctness, so a
    /// dictionary dropped between this call and its use is simply rebuilt.
    pub fn get<D: Any + Send + Sync>(&self, key: ChunkKey) -> Option<Arc<D>> {
        let built = self
            .built
            .lock()
            .expect("dictionary store is never poisoned");
        built.get(&key)?.upgrade()?.downcast::<D>().ok()
    }

    /// Publishes `dictionary` as this chunk's, returning whichever is current.
    ///
    /// Two splits can reach a chunk's dictionary page at once and both build
    /// one; the loser takes the winner's copy and drops its own, so every reader
    /// of a chunk ends up sharing a single dictionary.
    pub fn publish<D: Any + Send + Sync>(&self, key: ChunkKey, dictionary: Arc<D>) -> Arc<D> {
        let mut built = self
            .built
            .lock()
            .expect("dictionary store is never poisoned");
        if let Some(existing) = built
            .get(&key)
            .and_then(Weak::upgrade)
            .and_then(|held| held.downcast::<D>().ok())
        {
            return existing;
        }
        let erased: Arc<dyn Any + Send + Sync> = dictionary.clone();
        built.insert(key, Arc::downgrade(&erased));
        dictionary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Dictionary(u32);

    #[test]
    fn a_published_dictionary_is_handed_to_the_next_reader() {
        let store = SharedDictionaries::new();
        let held = store.publish((7, 2), Arc::new(Dictionary(42)));

        let found: Arc<Dictionary> = store.get((7, 2)).expect("published");

        assert_eq!(*found, Dictionary(42));
        assert!(
            Arc::ptr_eq(&held, &found),
            "the same dictionary, not a copy"
        );
    }

    /// Chunks are keyed by row group and leaf together, so the same leaf in
    /// another row group is a different dictionary.
    #[test]
    fn dictionaries_are_separate_per_chunk() {
        let store = SharedDictionaries::new();
        let _held = store.publish((7, 2), Arc::new(Dictionary(42)));

        assert!(store.get::<Dictionary>((8, 2)).is_none());
        assert!(store.get::<Dictionary>((7, 3)).is_none());
    }

    /// Two readers racing to build the same chunk's dictionary converge on one:
    /// the loser is handed the winner's.
    #[test]
    fn a_racing_publish_returns_the_dictionary_already_held() {
        let store = SharedDictionaries::new();
        let first = store.publish((0, 0), Arc::new(Dictionary(1)));

        let second = store.publish((0, 0), Arc::new(Dictionary(2)));

        assert_eq!(*second, Dictionary(1));
        assert!(Arc::ptr_eq(&first, &second));
    }

    /// Held weakly: once every reader has finished, the dictionary's slab
    /// memory goes back rather than being pinned until the scan ends.
    #[test]
    fn a_dictionary_is_released_once_its_readers_drop() {
        let store = SharedDictionaries::new();
        drop(store.publish((0, 0), Arc::new(Dictionary(1))));

        assert!(store.get::<Dictionary>((0, 0)).is_none());
    }
}
