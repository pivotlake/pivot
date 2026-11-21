//! Reclaiming a slot of the decompressed cache whose only tenants are stale:
//! pages carved in it that were never published there.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use bytes::Bytes;
use dispatch::io::{LocalFile, OpenFile};
use dispatch::memory::{Cache, CacheKey, DecompressedPage, DecompressedPageMeta, RingCache};
use dispatch::{BUFFER_SIZE, Dispatch, memory_ctx};

/// The key of the page at `offset` of `open_file`.
fn page_key(open_file: &OpenFile, offset: usize) -> CacheKey {
    CacheKey::new(open_file.clone(), offset, 1)
}

/// Carve `len` decompressed bytes for `key`, fill them with `pattern`, and
/// publish the page.
fn insert_page(cache: &Cache<DecompressedPage>, key: CacheKey, len: usize, pattern: u8) {
    cache
        .insert(key, |fill| {
            for mut slab in fill.get_bytes(len) {
                slab.as_mut_slice().fill(pattern);
            }
            let header = vec![Bytes::from(vec![pattern])];
            let live_pages = Arc::new(AtomicUsize::new(0));
            Ok::<_, Infallible>(DecompressedPageMeta::new(header, live_pages))
        })
        .unwrap();
}

/// Carve `len` decompressed bytes for `key` in a build that then fails, so
/// the page is never published and its tenant is stale.
fn abandon_page(cache: &Cache<DecompressedPage>, key: CacheKey, len: usize) {
    let abandoned = cache.insert(key, |fill| {
        fill.get_bytes(len);
        Err::<DecompressedPageMeta, _>("the build was abandoned")
    });
    assert!(abandoned.is_err(), "an abandoned build publishes nothing");
}

/// The one slot of this worker's ring the decompressed cache owns.
fn find_decompressed_slot() -> usize {
    let decompressed = memory_ctx().page_caches().decompressed.id();
    let mut owned = (0..memory_ctx().ring().len())
        .filter(|slot| memory_ctx().clock().owner(*slot) == Some(decompressed));
    let slot = owned.next().expect("the cache owns a slot");
    assert!(owned.next().is_none(), "the cache owns one slot only");
    slot
}

#[test]
fn reclaiming_a_stale_orphan_spares_the_page_published_elsewhere() {
    let dispatch = Dispatch::spin_up(1, 16, None);

    let (orphan_slot, reclaimed_slot, orphan_owner, is_page_held) = dispatch
        .dispatcher()
        .run_on_worker(|| {
            let cache = &memory_ctx().page_caches().decompressed;
            let open_file = OpenFile::Local(LocalFile::new(tempfile::tempfile().unwrap()).unwrap());
            let key = page_key(&open_file, 0);
            abandon_page(cache, key.clone(), 64); // never published: its tenant is stale
            let orphan_slot = find_decompressed_slot();
            // Fill the slot, so the insert goes into a fresh one.
            abandon_page(cache, page_key(&open_file, 1000), BUFFER_SIZE - 64);
            insert_page(cache, key.clone(), 64, 0xBB);

            let reclaimed = cache.reclaim(orphan_slot).unwrap();

            (
                orphan_slot,
                reclaimed.buffer.slot_idx,
                memory_ctx().clock().owner(orphan_slot),
                cache.get(&key).is_some(),
            )
        })
        .unwrap();

    assert_eq!(reclaimed_slot, orphan_slot);
    assert_eq!(
        orphan_owner, None,
        "the orphan must be fully released, not returned while still owned"
    );
    assert!(
        is_page_held,
        "the page published at the same key must survive"
    );
    dispatch.exit();
}
