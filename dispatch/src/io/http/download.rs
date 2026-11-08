//! The body of a whole-object download: a `GET` of a remote object with no
//! `Range` (see [`GetBody::Whole`](super::GetBody::Whole)), and where it lands.
//!
//! The object's size is unknown until the response head carries it, so the
//! body's destination is allocated then, by the thread the head reached. On a
//! worker the body lands straight in the compressed cache: the whole object is
//! looked up as one range, every extent that lookup hands this download the
//! fill of receives the bytes it covers in place, and landing commits them, so
//! the next read of the object is a sized read of cached bytes. A fragment the
//! lookup finds cached already, or being filled by another worker's download
//! of the same object, is received into scratch memory and served from there.
//! On a thread with no memory context (the blocking transport pool of a
//! non-Linux host) the body lands on the heap, and the worker that lands it
//! copies it into the cache.

use crate::io::{OpenFile, RemoteFile};
use crate::memory::compressed_cache::MissingExtent;
use crate::memory::{WriteBuffer, has_memory_context, memory_ctx};
use bytes::Bytes;
use std::sync::Arc;

/// Where a download's body lands: runs of memory allocated once the response
/// head says how long the body is, in body order.
pub(crate) struct DownloadBody {
    runs: Vec<BodyRun>,
    len: usize,
}

enum BodyRun {
    /// A compressed-cache extent this download fills: the body bytes at its
    /// file offset land in the slot the cache placed it in, and landing
    /// commits them. `data` is the view of those bytes the response is
    /// answered with.
    Extent { fill: MissingExtent, data: Bytes },
    /// A free-pool slot holding `len` body bytes the cache already has, or
    /// that another worker's download is filling: received, since the
    /// response carries them, and served from here.
    Scratch { buffer: WriteBuffer, len: usize },
    /// The whole body, on a thread with no ring. Copied into the cache by the
    /// worker that lands it.
    Heap(Vec<u8>),
}

impl BodyRun {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            // SAFETY: `dest` is the start of the extent's blocks in a slot its
            // pin keeps alive, only this download writes them while they are
            // invalid, and `data` views exactly the bytes of the body they hold.
            Self::Extent { fill, data } => unsafe {
                std::slice::from_raw_parts_mut(fill.dest(), data.len())
            },
            Self::Scratch { buffer, len } => &mut buffer.as_mut_slice()[..*len],
            Self::Heap(bytes) => bytes,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Extent { data, .. } => data.len(),
            Self::Scratch { len, .. } => *len,
            Self::Heap(bytes) => bytes.len(),
        }
    }
}

impl DownloadBody {
    /// Room for the `len`-byte body of a download of `remote`, on the thread
    /// the response head reached. On a worker, that is the compressed-cache
    /// extents of the whole object, which this download then owns the fill
    /// of, with scratch memory for any fragment the cache already holds or
    /// another worker is filling, and the cache keeps the object's size with
    /// them. On a thread with no memory context, the heap.
    pub(crate) fn allocate(remote: &Arc<RemoteFile>, len: u64) -> Self {
        let len = usize::try_from(len).expect("a response body longer than the address space");
        if !has_memory_context() {
            return Self {
                runs: vec![BodyRun::Heap(vec![0u8; len])],
                len,
            };
        }
        let open_file = OpenFile::Remote(remote.clone());
        let cache = memory_ctx().compressed_cache();
        let mut runs = Vec::new();
        for lookup in cache.get(&open_file, 0, len) {
            let owned_fill = lookup.missing().filter(|fill| fill.fill_owner()).cloned();
            let data = lookup.into_data();
            runs.push(match owned_fill {
                Some(fill) => BodyRun::Extent { fill, data },
                None => BodyRun::Scratch {
                    buffer: memory_ctx().get_write_buffer(false),
                    len: data.len(),
                },
            });
        }
        cache.remember_size(&open_file, len as u64);
        Self { runs, len }
    }

    /// Total body length.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The rest of the run that body offset `offset` falls in: where the next
    /// bytes of a body `offset` bytes of which have landed go. Empty once the
    /// whole body has.
    pub(crate) fn region_at(&mut self, offset: usize) -> &mut [u8] {
        assert!(
            offset <= self.len,
            "offset {offset} past a {} byte body",
            self.len
        );
        let mut start = 0;
        for run in &mut self.runs {
            let len = run.len();
            if offset < start + len {
                return &mut run.as_mut_slice()[offset - start..];
            }
            start += len;
        }
        &mut []
    }

    /// Land the fully received body of a download of `remote` on the worker
    /// that issued it: commit the cache extents it filled, or copy a body
    /// received on the heap into the cache, and return the body as its runs,
    /// in order. Bytes in the cache stay pinned there for as long as they are
    /// held; scratch slots are freed with them.
    pub(crate) fn land(self, remote: &Arc<RemoteFile>) -> Vec<Bytes> {
        self.runs
            .into_iter()
            .map(|run| match run {
                BodyRun::Extent { fill, data } => {
                    fill.commit_prefix(data.len());
                    data
                }
                BodyRun::Scratch { buffer, len } => Bytes::from_owner(buffer).slice(..len),
                BodyRun::Heap(bytes) => {
                    copy_into_cache(&OpenFile::Remote(remote.clone()), &bytes);
                    Bytes::from(bytes)
                }
            })
            .collect()
    }
}

/// Enter the whole `bytes` of `open_file` into the compressed cache: fill and
/// commit every extent of it this worker is handed the fill of, leaving what
/// is cached or being filled elsewhere alone, and make the size known.
fn copy_into_cache(open_file: &OpenFile, bytes: &[u8]) {
    let cache = memory_ctx().compressed_cache();
    let mut offset = 0;
    for lookup in cache.get(open_file, 0, bytes.len()) {
        let owned_fill = lookup.missing().filter(|fill| fill.fill_owner()).cloned();
        let fragment = &bytes[offset..offset + lookup.into_data().len()];
        if let Some(fill) = owned_fill {
            // SAFETY: `dest` is the start of the extent's blocks in a slot its
            // pin keeps alive, only this fill writes them while they are
            // invalid, and the fragment is the body bytes the extent covers.
            unsafe {
                std::ptr::copy_nonoverlapping(fragment.as_ptr(), fill.dest(), fragment.len())
            };
            fill.commit_prefix(fragment.len());
        }
        offset += fragment.len();
    }
    cache.remember_size(open_file, bytes.len() as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{BUFFER_SIZE, CacheLookup, init_test_free_pool};
    use url::Url;

    /// An object of its own at `path`, so each test keys its own cache entry.
    fn remote(path: &str) -> Arc<RemoteFile> {
        let url = Url::parse(&format!("http://127.0.0.1:1/{path}")).unwrap();
        Arc::new(RemoteFile::open_unsized(url, None).unwrap())
    }

    fn pattern(offset: usize) -> u8 {
        (offset % 251) as u8
    }

    /// Fill the body the way a transport does: region by region, in order.
    fn receive(body: &mut DownloadBody, bytes: &[u8]) {
        let mut written = 0;
        while written < bytes.len() {
            let region = body.region_at(written);
            let n = region.len();
            region.copy_from_slice(&bytes[written..written + n]);
            written += n;
        }
    }

    fn concat(runs: impl IntoIterator<Item = Bytes>) -> Vec<u8> {
        runs.into_iter().flatten().collect()
    }

    /// The whole object as the cache now serves it: every byte a hit.
    fn cached_object(remote: &Arc<RemoteFile>, len: usize) -> Vec<u8> {
        let hits = memory_ctx()
            .compressed_cache()
            .get(&OpenFile::Remote(remote.clone()), 0, len);
        assert!(hits.iter().all(|hit| hit.missing().is_none()));
        concat(hits.into_iter().map(CacheLookup::into_data))
    }

    /// A body received on a worker lands in cache extents: it reads back as
    /// what was received, and the whole object is then a sized cache hit.
    #[test]
    fn a_body_received_on_a_worker_lands_in_the_cache() {
        init_test_free_pool(4);
        let remote = remote("on-worker");
        let object: Vec<u8> = (0..BUFFER_SIZE + 3).map(pattern).collect();
        let mut body = DownloadBody::allocate(&remote, object.len() as u64);

        receive(&mut body, &object);
        let landed = body.land(&remote);

        assert_eq!(concat(landed), object);
        assert_eq!(cached_object(&remote, object.len()), object);
        let known = memory_ctx()
            .compressed_cache()
            .known_size(&OpenFile::Remote(remote));
        assert_eq!(known, Some(object.len() as u64));
    }

    /// A body dropped before it lands, as a failed download's is, gives its
    /// extents up: the object misses again with its fill free to take, and
    /// with nothing of it cached its size is forgotten too, so the next
    /// download of it is again one request.
    #[test]
    fn a_body_dropped_before_landing_releases_its_extents_and_size() {
        init_test_free_pool(4);
        let remote = remote("dropped");
        let body = DownloadBody::allocate(&remote, 100);

        drop(body);

        let open_file = OpenFile::Remote(remote);
        let lookups = memory_ctx().compressed_cache().get(&open_file, 0, 100);
        assert!(
            lookups
                .iter()
                .all(|lookup| lookup.missing().is_some_and(MissingExtent::fill_owner))
        );
        assert_eq!(memory_ctx().compressed_cache().known_size(&open_file), None);
    }

    /// A fragment another lookup already owns the fill of is received into
    /// scratch and served from there; that owner's extent is left for it to
    /// commit.
    #[test]
    fn a_body_of_an_object_another_fill_owns_lands_in_scratch() {
        init_test_free_pool(4);
        let remote = remote("followed");
        let owner = memory_ctx()
            .compressed_cache()
            .get(&OpenFile::Remote(remote.clone()), 0, 100);
        let object: Vec<u8> = (0..100).map(pattern).collect();
        let mut body = DownloadBody::allocate(&remote, 100);

        receive(&mut body, &object);
        let landed = body.land(&remote);

        assert_eq!(concat(landed), object);
        assert!(!owner[0].missing().unwrap().is_committed());
    }

    /// A body received on a thread with no memory context lands on the heap,
    /// and enters the cache with the worker that lands it.
    #[test]
    fn a_body_received_off_worker_enters_the_cache_when_landed() {
        init_test_free_pool(4);
        let remote = remote("off-worker");
        let object: Vec<u8> = (0..BUFFER_SIZE + 3).map(pattern).collect();
        let mut body = std::thread::scope(|scope| {
            scope
                .spawn(|| DownloadBody::allocate(&remote, object.len() as u64))
                .join()
                .unwrap()
        });

        receive(&mut body, &object);
        let landed = body.land(&remote);

        assert_eq!(concat(landed), object);
        assert_eq!(cached_object(&remote, object.len()), object);
    }

    /// An empty body has no extent to land in, so the cache holds nothing of
    /// the object and keeps nothing about it: each download of it is one
    /// trivial request.
    #[test]
    fn an_empty_body_lands_as_no_bytes_and_leaves_nothing_cached() {
        init_test_free_pool(1);
        let remote = remote("empty");
        let body = DownloadBody::allocate(&remote, 0);

        let landed = body.land(&remote);

        assert!(landed.is_empty());
        let known = memory_ctx()
            .compressed_cache()
            .known_size(&OpenFile::Remote(remote));
        assert_eq!(known, None);
    }
}
