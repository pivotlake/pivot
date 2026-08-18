//! A read-through memory cache for the immutable objects of a Delta log.
//!
//! Rebuilding a table's state reads the log end to end: the newest checkpoint,
//! then every commit written above it. Kernel re-reads those same objects on
//! every rebuild, and re-reads a checkpoint several times within one rebuild
//! (once per pass it makes over the file). A table that commits often therefore
//! pays a full log's worth of object-store GETs on every refresh sweep, growing
//! with each commit until the next checkpoint resets it.
//!
//! Those objects never change. The Delta protocol writes a commit, a checkpoint,
//! and a CRC once, under a name that encodes its version, and no writer rewrites
//! one afterwards. So a version's bytes can be held and served again for as long
//! as there's room. `_last_checkpoint` is the exception the protocol overwrites
//! in place, so it is never cached, and neither is anything outside a `_delta_log`
//! directory: the data files are far larger and are already served through the
//! reader's own disk cache.
//!
//! The cache wraps whatever backend a datastore opened, so it applies to every
//! store alike, and it caches whole objects: a rebuild reads all of a checkpoint
//! in the end, just a range at a time, so the first range read fetches the file
//! and every later one is served from memory. An object too large for
//! [`MAX_CACHED_OBJECT_BYTES`] is passed straight through instead.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::{
    Attributes, CopyOptions, DynObjectStore, GetOptions, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, Result,
};
use futures::stream::{BoxStream, StreamExt, TryStreamExt};

/// How many bytes of log objects one datastore holds at once. A commit is
/// kilobytes and a checkpoint a few megabytes, so this holds a busy table's
/// whole log several times over; past it, the least recently read object is
/// dropped.
const CAPACITY_BYTES: u64 = 128 * 1024 * 1024;

/// The largest object worth holding. A checkpoint above this belongs to a table
/// with millions of files, where caching it would cost more memory than the
/// re-reads it saves, so it is read straight from the store as asked.
const MAX_CACHED_OBJECT_BYTES: u64 = 32 * 1024 * 1024;

/// An object store that serves a Delta log's immutable objects from memory.
#[derive(Debug)]
pub(crate) struct CachingLogStore {
    inner: Arc<DynObjectStore>,
    cache: Arc<Mutex<LogCache>>,
}

impl CachingLogStore {
    pub(crate) fn new(inner: Arc<DynObjectStore>) -> Self {
        Self::with_capacity(inner, CAPACITY_BYTES)
    }

    fn with_capacity(inner: Arc<DynObjectStore>, capacity_bytes: u64) -> Self {
        Self {
            inner,
            cache: Arc::new(Mutex::new(LogCache::new(capacity_bytes))),
        }
    }

    /// The held copy of `location`, marked as just used.
    fn cached(&self, location: &Path) -> Option<CachedObject> {
        self.cache.lock().unwrap().get(location)
    }

    fn hold(&self, object: CachedObject) {
        self.cache.lock().unwrap().insert(object);
    }

    fn drop_held(&self, location: &Path) {
        self.cache.lock().unwrap().remove(location);
    }
}

/// Whether `location` names an object the Delta protocol writes once and never
/// rewrites, so a held copy can never go stale.
fn is_immutable_log_object(location: &Path) -> bool {
    if !location.parts().any(|part| part.as_ref() == "_delta_log") {
        return false;
    }
    let Some(name) = location.filename() else {
        return false;
    };
    // `_last_checkpoint` names the latest checkpoint and is overwritten as the
    // log advances; everything else in the directory carries its version in its
    // name.
    name.ends_with(".json") || name.ends_with(".parquet") || name.ends_with(".crc")
}

/// Whether `options` asks for exactly the object's bytes, so a held copy answers
/// it. A conditional or versioned request is about the object's identity at the
/// store rather than its content, and goes to the store to be answered.
fn is_plain_read(options: &GetOptions) -> bool {
    options.if_match.is_none()
        && options.if_none_match.is_none()
        && options.if_modified_since.is_none()
        && options.if_unmodified_since.is_none()
        && options.version.is_none()
        && !options.head
}

/// One held object: its bytes and the metadata the store reported for them.
#[derive(Clone, Debug)]
struct CachedObject {
    meta: ObjectMeta,
    bytes: Bytes,
    attributes: Attributes,
}

impl CachedObject {
    /// Answer a get for `range` (the whole object when `None`) out of the held
    /// bytes, shaped as the store would have returned it.
    fn to_get_result(&self, options: &GetOptions) -> Result<GetResult> {
        let size = self.bytes.len() as u64;
        let range = match &options.range {
            Some(requested) => requested.as_range(size).map_err(|source| {
                delta_kernel::object_store::Error::Generic {
                    store: "CachingLogStore",
                    source: Box::new(source),
                }
            })?,
            None => 0..size,
        };
        let bytes = self.bytes.slice(range.start as usize..range.end as usize);
        Ok(GetResult {
            payload: GetResultPayload::Stream(
                futures::stream::once(async move { Ok(bytes) }).boxed(),
            ),
            meta: self.meta.clone(),
            range,
            attributes: self.attributes.clone(),
        })
    }
}

/// The held objects and their byte budget. Eviction is least-recently-read,
/// tracked by a counter stamped on each read rather than a list, since the map
/// holds tens of entries per table and a scan over it costs less than
/// maintaining the ordering.
#[derive(Debug)]
struct LogCache {
    entries: HashMap<Path, Entry>,
    held_bytes: u64,
    capacity_bytes: u64,
    reads: u64,
}

#[derive(Debug)]
struct Entry {
    object: CachedObject,
    last_read: u64,
}

impl LogCache {
    fn new(capacity_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            held_bytes: 0,
            capacity_bytes,
            reads: 0,
        }
    }

    fn get(&mut self, location: &Path) -> Option<CachedObject> {
        self.reads += 1;
        let reads = self.reads;
        let entry = self.entries.get_mut(location)?;
        entry.last_read = reads;
        Some(entry.object.clone())
    }

    fn insert(&mut self, object: CachedObject) {
        let bytes = object.bytes.len() as u64;
        if bytes > self.capacity_bytes {
            return;
        }
        self.remove(&object.meta.location);
        while self.held_bytes + bytes > self.capacity_bytes {
            let Some(coldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_read)
                .map(|(location, _)| location.clone())
            else {
                break;
            };
            self.remove(&coldest);
        }
        self.reads += 1;
        self.held_bytes += bytes;
        self.entries.insert(
            object.meta.location.clone(),
            Entry {
                object,
                last_read: self.reads,
            },
        );
    }

    fn remove(&mut self, location: &Path) {
        if let Some(entry) = self.entries.remove(location) {
            self.held_bytes -= entry.object.bytes.len() as u64;
        }
    }
}

#[async_trait::async_trait]
impl ObjectStore for CachingLogStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> Result<PutResult> {
        // A write to a held name means the assumption behind holding it was
        // wrong; drop the copy rather than serve bytes the store no longer has.
        self.drop_held(location);
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.drop_held(location);
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        if !is_immutable_log_object(location) || !is_plain_read(&options) {
            return self.inner.get_opts(location, options).await;
        }
        if let Some(object) = self.cached(location) {
            return object.to_get_result(&options);
        }
        // Fetch the whole object, not the requested range: the caller is part
        // way through a log read that will ask for the rest of it, and one GET
        // for the file beats one per range.
        let whole = self.inner.get_opts(location, GetOptions::default()).await?;
        if whole.meta.size > MAX_CACHED_OBJECT_BYTES {
            drop(whole);
            return self.inner.get_opts(location, options).await;
        }
        let meta = whole.meta.clone();
        let attributes = whole.attributes.clone();
        let bytes = whole.bytes().await?;
        let object = CachedObject {
            meta,
            bytes,
            attributes,
        };
        let result = object.to_get_result(&options);
        self.hold(object);
        result
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        let cache = Arc::clone(&self.cache);
        self.inner
            .delete_stream(locations)
            .inspect_ok(move |deleted| cache.lock().unwrap().remove(deleted))
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.drop_held(to);
        self.inner.copy_opts(from, to, options).await
    }
}

impl std::fmt::Display for CachingLogStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CachingLogStore({})", self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::object_store::memory::InMemory;
    use delta_kernel::object_store::{ObjectStoreExt, PutPayload};

    /// Drive one store call to completion. The crate's tokio has no `macros`
    /// feature, so the tests build their own runtime rather than annotate.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(future)
    }

    /// Put `bytes` at `raw` in `store`.
    fn put(store: &dyn ObjectStore, raw: &str, bytes: &'static [u8]) -> Path {
        let location = Path::from(raw);
        block_on(store.put(&location, PutPayload::from_static(bytes))).unwrap();
        location
    }

    /// Read `location` whole, as the store's bytes.
    fn read(store: &dyn ObjectStore, location: &Path) -> Result<Bytes> {
        block_on(async { store.get(location).await?.bytes().await })
    }

    /// Read `range` of `location`.
    fn read_range(store: &dyn ObjectStore, location: &Path, range: std::ops::Range<u64>) -> Bytes {
        block_on(store.get_range(location, range)).unwrap()
    }

    /// Take `location` out from under the cache, so a later read that still
    /// answers proves the bytes came from memory rather than the store.
    fn delete_behind_the_cache(inner: &Arc<InMemory>, location: &Path) {
        block_on(inner.delete(location)).unwrap();
    }

    #[test]
    fn a_commit_is_served_from_memory_once_it_has_been_read() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::new(Arc::clone(&inner) as Arc<DynObjectStore>);
        let commit = put(
            &store,
            "db/t/_delta_log/00000000000000000001.json",
            b"commit",
        );

        read(&store, &commit).unwrap();
        delete_behind_the_cache(&inner, &commit);

        assert_eq!(read(&store, &commit).unwrap().as_ref(), b"commit");
    }

    #[test]
    fn one_range_read_of_a_checkpoint_holds_the_whole_file() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::new(Arc::clone(&inner) as Arc<DynObjectStore>);
        let checkpoint = put(
            &store,
            "db/t/_delta_log/00000000000000000200.checkpoint.parquet",
            b"0123456789",
        );

        read_range(&store, &checkpoint, 0..2);
        delete_behind_the_cache(&inner, &checkpoint);

        assert_eq!(
            read_range(&store, &checkpoint, 6..10).as_ref(),
            b"6789",
            "a later pass over the file reads no further than memory"
        );
    }

    #[test]
    fn the_checkpoint_pointer_is_read_from_the_store_every_time() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::new(Arc::clone(&inner) as Arc<DynObjectStore>);
        let pointer = put(&store, "db/t/_delta_log/_last_checkpoint", b"{}");

        read(&store, &pointer).unwrap();
        delete_behind_the_cache(&inner, &pointer);

        assert!(
            read(&store, &pointer).is_err(),
            "the pointer is rewritten in place, so it is never held"
        );
    }

    #[test]
    fn a_data_file_is_read_from_the_store_every_time() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::new(Arc::clone(&inner) as Arc<DynObjectStore>);
        let data = put(&store, "db/t/part-0001.parquet", b"rows");

        read(&store, &data).unwrap();
        delete_behind_the_cache(&inner, &data);

        assert!(read(&store, &data).is_err());
    }

    #[test]
    fn a_full_cache_drops_the_least_recently_read_object() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::with_capacity(Arc::clone(&inner) as Arc<DynObjectStore>, 8);
        let first = put(&store, "db/t/_delta_log/00000000000000000001.json", b"1111");
        let second = put(&store, "db/t/_delta_log/00000000000000000002.json", b"2222");
        let third = put(&store, "db/t/_delta_log/00000000000000000003.json", b"3333");
        for commit in [&first, &second] {
            read(&store, commit).unwrap();
        }

        read(&store, &first).unwrap();
        read(&store, &third).unwrap();
        for commit in [&first, &second, &third] {
            delete_behind_the_cache(&inner, commit);
        }

        assert_eq!(read(&store, &first).unwrap().as_ref(), b"1111");
        assert_eq!(read(&store, &third).unwrap().as_ref(), b"3333");
        assert!(
            read(&store, &second).is_err(),
            "the commit no read touched is the one dropped"
        );
    }

    #[test]
    fn a_rewritten_object_is_dropped_rather_than_served_stale() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::new(Arc::clone(&inner) as Arc<DynObjectStore>);
        let commit = put(
            &store,
            "db/t/_delta_log/00000000000000000001.json",
            b"first",
        );
        read(&store, &commit).unwrap();

        put(
            &store,
            "db/t/_delta_log/00000000000000000001.json",
            b"second",
        );

        assert_eq!(read(&store, &commit).unwrap().as_ref(), b"second");
    }

    #[test]
    fn a_deleted_object_is_dropped_rather_than_served_stale() {
        let inner = Arc::new(InMemory::new());
        let store = CachingLogStore::new(Arc::clone(&inner) as Arc<DynObjectStore>);
        let commit = put(
            &store,
            "db/t/_delta_log/00000000000000000001.json",
            b"commit",
        );
        read(&store, &commit).unwrap();

        block_on(store.delete(&commit)).unwrap();

        assert!(read(&store, &commit).is_err());
    }
}
