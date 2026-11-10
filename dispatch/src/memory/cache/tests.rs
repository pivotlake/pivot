//! The generic cache and the caches together: packing, lookups, reclaim,
//! reinforcement, weakening, and eviction across families. Every test
//! installs its own single-worker memory context with the standard setup.

use super::*;
use crate::io::{LocalFile, OpenFile};
use crate::memory::BUFFER_SIZE;
use crate::memory::context::{init_test_free_pool, memory_ctx};
use bytes::Bytes;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// One shared local file, for tests that key by offset alone.
#[allow(non_snake_case)]
fn FD() -> OpenFile {
    use std::sync::OnceLock;
    static FILE: OnceLock<LocalFile> = OnceLock::new();
    OpenFile::Local(
        FILE.get_or_init(|| LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
            .clone(),
    )
}

/// A fresh, independent local file (distinct key bucket).
fn new_file() -> OpenFile {
    OpenFile::Local(LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap())
}

fn key(offset: usize) -> CacheKey {
    CacheKey::new(FD(), offset, 1)
}

fn key_in(file: &OpenFile, offset: usize, span: usize) -> CacheKey {
    CacheKey::new(file.clone(), offset, span)
}

fn pages() -> &'static PageCaches {
    memory_ctx().page_caches()
}

fn decompressed() -> &'static Cache<DecompressedPage> {
    &pages().decompressed
}

fn compressed() -> &'static CompressedCache {
    &pages().compressed
}

fn clock() -> &'static crate::memory::Clock {
    memory_ctx().clock()
}

fn header(tag: u8) -> Vec<Bytes> {
    vec![Bytes::from(vec![tag])]
}

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

/// What a reinforced slot of `cache` gains per touch.
fn lives_at_reinforce(cache: &dyn RingCache) -> u8 {
    memory_ctx().caches().policy(cache.id()).lives_at_reinforce
}

/// Build a `len`-byte page filled with `pattern` straight into `cache`.
fn insert_page(
    cache: &Cache<DecompressedPage>,
    key: CacheKey,
    len: usize,
    pattern: u8,
) -> DecompressedPage {
    insert_page_counting(cache, key, len, pattern, counter())
}

fn insert_page_counting(
    cache: &Cache<DecompressedPage>,
    key: CacheKey,
    len: usize,
    pattern: u8,
    live_pages: Arc<AtomicUsize>,
) -> DecompressedPage {
    cache
        .insert(key, |fill| {
            for mut slab in fill.get_bytes(len) {
                slab.as_mut_slice().fill(pattern);
            }
            Ok::<_, Infallible>(DecompressedPageMeta::new(header(pattern), live_pages))
        })
        .unwrap()
}

fn all_bytes_are(page: &DecompressedPage, expected: u8) -> bool {
    page.data
        .iter()
        .flat_map(|b| b.iter())
        .all(|&b| b == expected)
}

fn total_len(page: &DecompressedPage) -> usize {
    page.data.iter().map(Bytes::len).sum()
}

/// The slot this worker's cursor for `cache` points at (where small entries pack).
fn cursor_slot(cache: &dyn RingCache) -> usize {
    memory_ctx().fill_cursor(cache.id()).slot()
}

fn release_cursor(cache: &dyn RingCache) {
    memory_ctx().fill_cursor(cache.id()).release();
}

/// Cache one whole slot of `FD()`'s compressed bytes at slot-aligned offset
/// `slot_number * BUFFER_SIZE`, touched a few times since, so it holds lives
/// like any extent an earlier query used. Returns its slot.
fn insert_compressed_slot(slot_number: usize) -> usize {
    let lookups = compressed().get(&FD(), slot_number * BUFFER_SIZE, BUFFER_SIZE);
    let missing = lookups[0].missing_extent().unwrap();
    missing.commit();
    let slot = memory_ctx().fill_cursor(compressed().id()).slot();
    for _ in 0..3 {
        clock().touch(slot);
    }
    slot
}

/// Cache `[offset, offset + len)` of `FD()` compressed and committed. Returns
/// the slot it was packed into.
fn insert_compressed(offset: usize, len: usize) -> usize {
    let lookups = compressed().get(&FD(), offset, len);
    let missing = lookups[0].missing_extent().unwrap();
    missing.commit();
    memory_ctx().fill_cursor(compressed().id()).slot()
}

/// Insert one decompressed page over `[offset, offset + len)` of `FD()` that
/// takes a whole slot by itself, then release the cursor so it is evictable.
fn insert_decompressed_slot(offset: usize, len: usize) -> usize {
    insert_page(decompressed(), key_in(&FD(), offset, len), BUFFER_SIZE, 0);
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());
    slot
}

fn compressed_has_misses(offset: usize, len: usize) -> bool {
    compressed()
        .get(&FD(), offset, len)
        .iter()
        .any(|l| l.missing_extent().is_some())
}

/// Take every slot still in the free pool as working memory, so the next
/// request has to evict.
fn take_every_free_slot() -> Vec<crate::memory::WriteBuffer> {
    std::iter::from_fn(|| memory_ctx().pop_free_idx(true))
        .map(|idx| memory_ctx().ring().try_write(idx).unwrap())
        .collect()
}

// -- Cache<V> on its own --

#[test]
fn get_misses_for_an_absent_key() {
    init_test_free_pool(1);

    assert!(decompressed().get(&key(0)).is_none());
}

#[test]
fn two_pages_pack_into_one_slot() {
    init_test_free_pool(4);

    insert_page(decompressed(), key(0), 100, 0xAA);
    insert_page(decompressed(), key(100), 100, 0xBB);

    let a = decompressed().get(&key(0)).unwrap();
    let b = decompressed().get(&key(100)).unwrap();
    assert_eq!(clock().owned(decompressed().id()), 1);
    assert!(all_bytes_are(&a, 0xAA));
    assert!(all_bytes_are(&b, 0xBB));
}

#[test]
fn a_straddling_page_reads_back_its_bytes() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), BUFFER_SIZE - 7, 0x11); // leave a 7-byte tail

    let built = insert_page(decompressed(), key(1000), 100, 0x22); // 7 + 93 across two slots

    let hit = decompressed().get(&key(1000)).unwrap();
    assert_eq!(hit.data.len(), 2, "two slabs across the slot boundary");
    assert_eq!(total_len(&hit), 100);
    assert!(all_bytes_are(&hit, 0x22));
    assert!(all_bytes_are(&built, 0x22));
}

#[test]
fn a_hit_is_ring_backed_and_copy_free() {
    init_test_free_pool(2);
    let built = insert_page(decompressed(), key(0), 8, 0);

    let hit = decompressed().get(&key(0)).unwrap();

    assert_eq!(hit.data[0].as_ptr(), built.data[0].as_ptr());
}

#[test]
fn a_hit_carries_the_header_stored_beside_it() {
    init_test_free_pool(2);
    insert_page(decompressed(), key(0), 8, 9);

    let hit = decompressed().get(&key(0)).unwrap();

    assert_eq!(hit.header, header(9));
}

#[test]
fn a_different_span_at_the_same_offset_misses() {
    init_test_free_pool(2);
    insert_page(decompressed(), key_in(&FD(), 0, 100), 8, 0);

    assert!(decompressed().get(&key_in(&FD(), 0, 200)).is_none());
}

#[test]
fn a_lost_dup_race_returns_readable_bytes_and_keeps_the_incumbent() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 50, 0xAA);

    let dup = insert_page(decompressed(), key(0), 50, 0xBB);

    assert!(all_bytes_are(&dup, 0xBB));
    assert!(all_bytes_are(&decompressed().get(&key(0)).unwrap(), 0xAA));
}

#[test]
fn a_failed_build_caches_nothing_and_leaves_the_slot_recyclable() {
    init_test_free_pool(4);
    let failed = decompressed().insert(key(0), |fill| {
        fill.get_bytes(64);
        Err::<DecompressedPageMeta, &str>("decompress failed")
    });
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(slot);

    assert!(matches!(failed, Err("decompress failed")));
    assert!(reclaimed.is_some(), "only a stale tenant, freely recycled");
    assert!(decompressed().is_empty());
}

#[test]
fn the_live_page_count_tracks_residency() {
    init_test_free_pool(4);
    let live = counter();
    insert_page_counting(decompressed(), key(0), 8, 0, live.clone());
    insert_page_counting(decompressed(), key(0), 8, 0, live.clone()); // loses the race
    let while_cached = live.load(std::sync::atomic::Ordering::Relaxed);

    decompressed().invalidate(&FD());

    assert_eq!(while_cached, 1);
    assert_eq!(live.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn reclaim_evicts_every_tenant_of_the_slot() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 100, 1);
    insert_page(decompressed(), key(100), 100, 2);
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(slot).unwrap();

    assert_eq!(reclaimed.buffer.slot_idx, slot);
    assert_eq!(reclaimed.dropped.len(), 2);
    assert!(decompressed().get(&key(0)).is_none());
    assert!(decompressed().get(&key(100)).is_none());
    assert!(decompressed().is_empty());
}

#[test]
fn evicting_one_slot_of_a_straddler_kills_the_entry_whole() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), BUFFER_SIZE - 7, 0x11);
    insert_page(decompressed(), key(1000), 100, 0x22); // 7 in slot A, 93 in slot B
    let second_slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(second_slot);

    assert!(reclaimed.is_some());
    assert!(
        decompressed().get(&key(1000)).is_none(),
        "straddler died whole"
    );
    assert!(decompressed().get(&key(0)).is_some());
}

#[test]
fn a_stale_partner_slot_recycles_cleanly() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), BUFFER_SIZE - 7, 0x11);
    let first_slot = cursor_slot(decompressed());
    insert_page(decompressed(), key(1000), 100, 0x22);
    let second_slot = cursor_slot(decompressed());
    release_cursor(decompressed());
    decompressed().reclaim(second_slot); // strands the straddler's bytes in first_slot

    let reclaimed = decompressed().reclaim(first_slot);

    assert!(reclaimed.is_some(), "stale tenant filtered, slot recycled");
    assert!(decompressed().is_empty());
}

#[test]
fn reclaiming_a_stale_orphan_spares_the_reinserted_entry() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 64, 0xAA);
    let orphan_slot = cursor_slot(decompressed());
    decompressed().invalidate(&FD()); // the entry is gone; its tenant is now stale
    release_cursor(decompressed()); // force the re-insert into a fresh slot
    insert_page(decompressed(), key(0), 64, 0xBB);
    release_cursor(decompressed());

    let reclaimed = decompressed().reclaim(orphan_slot).unwrap();

    assert_eq!(reclaimed.buffer.slot_idx, orphan_slot);
    assert_eq!(clock().owner(orphan_slot), None);
    assert!(decompressed().get(&key(0)).is_some());
}

#[test]
fn evicting_a_files_last_entry_releases_its_file_handle() {
    init_test_free_pool(2);
    let open_file = new_file();
    let OpenFile::Local(file) = &open_file else {
        unreachable!()
    };
    insert_page(decompressed(), key_in(&open_file, 0, 1), 8, 0);
    let slot = cursor_slot(decompressed());
    release_cursor(decompressed());

    decompressed().reclaim(slot);

    assert_eq!(file.strong_count(), 1);
}

#[test]
fn reclaim_is_none_for_an_unowned_slot() {
    init_test_free_pool(1);

    assert!(decompressed().reclaim(0).is_none());
}

#[test]
fn invalidate_drops_only_the_matching_file() {
    init_test_free_pool(4);
    let a = new_file();
    let b = new_file();
    insert_page(decompressed(), key_in(&a, 0, 1), 8, 0);
    insert_page(decompressed(), key_in(&b, 0, 1), 8, 0);

    decompressed().invalidate(&a);

    assert!(decompressed().get(&key_in(&a, 0, 1)).is_none());
    assert!(decompressed().get(&key_in(&b, 0, 1)).is_some());
}

#[test]
fn clear_drops_everything() {
    init_test_free_pool(4);
    insert_page(decompressed(), key(0), 8, 0);
    insert_page(decompressed(), key(100), 8, 0);

    let dropped = decompressed().clear();

    assert_eq!(dropped, 2);
    assert!(decompressed().is_empty());
    assert_eq!(clock().owned(decompressed().id()), 0);
}

#[test]
fn a_disabled_cache_builds_into_working_memory_and_never_hits() {
    init_test_free_pool(2);
    let cache: Cache<DecompressedPage> = Cache::with_enabled(CacheId::new(0), 128, false);

    let built = insert_page(&cache, key(0), 8, 7);

    assert!(all_bytes_are(&built, 7));
    assert!(cache.get(&key(0)).is_none());
    assert!(cache.is_empty());
}

#[test]
fn a_disabled_cache_answers_a_range_lookup_with_one_gap() {
    init_test_free_pool(2);
    let cache: Cache<DecompressedPage> = Cache::with_enabled(CacheId::new(0), 128, false);
    insert_page(&cache, key_in(&FD(), 0, 10), 8, 9);

    let segments = cache.get_range(&FD(), 0, 10);

    assert_eq!(shapes(&segments), vec![Shape::Gap(0, 10)]);
}

#[test]
fn is_empty_tracks_the_entry_count() {
    init_test_free_pool(2);
    assert!(decompressed().is_empty());

    insert_page(decompressed(), key(0), 8, 0);
    let while_cached = decompressed().is_empty();
    decompressed().invalidate(&FD());

    assert!(!while_cached);
    assert!(decompressed().is_empty());
}

#[test]
fn an_aligned_slab_never_straddles_a_slot() {
    init_test_free_pool(4);
    let sparse = &memory_ctx().sparse_index_caches().decoded;
    insert_page(decompressed(), key(0), 8, 0); // unrelated; the sparse cursor is its own
    let filler = sparse.insert(key_in(&FD(), 0, 1), |fill| {
        use crate::memory::slab::SlabSource;
        fill.get_aligned_slab(BUFFER_SIZE - 100, 64, false);
        Ok::<_, Infallible>(SparseIndexMeta)
    });
    let first_slot = cursor_slot(sparse.as_ref());

    let index = sparse
        .insert(key_in(&FD(), 1, 1), |fill| {
            use crate::memory::slab::SlabSource;
            let slab = fill.get_aligned_slab(1024, 64, true);
            assert_eq!(slab.as_slice().as_ptr() as usize % 64, 0);
            assert!(slab.as_slice().iter().all(|&b| b == 0));
            Ok::<_, Infallible>(SparseIndexMeta)
        })
        .unwrap();

    assert_eq!(filler.unwrap().slabs().len(), 1);
    assert_eq!(index.slabs().len(), 1);
    assert_ne!(
        cursor_slot(sparse.as_ref()),
        first_slot,
        "rotated rather than split"
    );
}

// -- get_range --

#[derive(Debug, PartialEq, Eq)]
enum Shape {
    Gap(usize, usize),
    Cached(usize, usize, u8),
}

fn shapes(segments: &[Segment<DecompressedPage>]) -> Vec<Shape> {
    segments
        .iter()
        .map(|s| match s {
            Segment::Gap { offset, len } => Shape::Gap(*offset, *len),
            Segment::Cached {
                offset,
                span,
                value,
            } => Shape::Cached(*offset, *span, value.header[0][0]),
        })
        .collect()
}

#[test]
fn get_range_over_an_empty_cache_is_one_gap() {
    init_test_free_pool(1);

    let segments = decompressed().get_range(&FD(), 100, 50);

    assert_eq!(shapes(&segments), vec![Shape::Gap(100, 50)]);
}

#[test]
fn get_range_finds_a_hit_surrounded_by_gaps() {
    init_test_free_pool(2);
    insert_page(decompressed(), key_in(&FD(), 10, 5), 8, 7);
    release_cursor(decompressed());

    let segments = decompressed().get_range(&FD(), 0, 20);

    assert_eq!(
        shapes(&segments),
        vec![
            Shape::Gap(0, 10),
            Shape::Cached(10, 5, 7),
            Shape::Gap(15, 5)
        ]
    );
}

#[test]
fn get_range_has_no_gap_between_adjacent_hits() {
    init_test_free_pool(2);
    insert_page(decompressed(), key_in(&FD(), 0, 5), 8, 1);
    insert_page(decompressed(), key_in(&FD(), 5, 5), 8, 2);
    release_cursor(decompressed());

    let segments = decompressed().get_range(&FD(), 0, 10);

    assert_eq!(
        shapes(&segments),
        vec![Shape::Cached(0, 5, 1), Shape::Cached(5, 5, 2)]
    );
}

// -- the caches together --

#[test]
fn evict_takes_a_page_read_once_before_the_compressed_bytes_under_it() {
    init_test_free_pool(8);
    insert_compressed(0, 4096);
    release_cursor(compressed());
    insert_decompressed_slot(0, 4096);

    memory_ctx().evict();

    assert_eq!(clock().owned(decompressed().id()), 0);
    assert!(!compressed_has_misses(0, 4096), "the source outlasted it");
}

#[test]
fn evict_takes_the_compressed_bytes_under_a_page_that_keeps_being_read() {
    init_test_free_pool(8);
    let source_slot = insert_compressed(0, 4096);
    release_cursor(compressed());
    insert_decompressed_slot(0, 4096);
    for _ in 0..clock().lives(source_slot) {
        decompressed().get(&key_in(&FD(), 0, 4096)).unwrap();
    }

    memory_ctx().evict();

    assert_eq!(clock().owned(compressed().id()), 0);
    assert!(decompressed().get(&key_in(&FD(), 0, 4096)).is_some());
}

/// A slot out of lives that a reader has pinned cannot be taken: the hand
/// passes over it and keeps ageing the rest until one of them gives.
#[test]
fn evict_passes_over_a_pinned_slot_and_takes_one_that_still_had_lives() {
    init_test_free_pool(8);
    let pinned_slot = insert_compressed(0, BUFFER_SIZE);
    release_cursor(compressed());
    let _reader = memory_ctx().ring().try_read(pinned_slot).unwrap();
    let ageing_slot = insert_compressed_slot(1); // holds more lives
    release_cursor(compressed());

    let evicted = memory_ctx().evict();

    assert_eq!(evicted.slot_idx, ageing_slot);
    assert_eq!(clock().owner(pinned_slot), Some(compressed().id()));
}

/// Every cached slot but one is pinned by a reader, however few lives it
/// holds: working memory comes from the one that is not.
#[test]
fn working_memory_takes_the_only_slot_no_reader_holds() {
    init_test_free_pool(8);
    let _pinned: Vec<DecompressedPage> = (0..3)
        .map(|i| {
            insert_decompressed_slot((100 + i) * BUFFER_SIZE, 4096);
            decompressed()
                .get(&key_in(&FD(), (100 + i) * BUFFER_SIZE, 4096))
                .unwrap()
        })
        .collect();
    let unpinned_slot = insert_compressed_slot(0);
    release_cursor(compressed());
    let _working_memory = take_every_free_slot();

    let buffer = memory_ctx().get_write_buffer(true);

    assert_eq!(buffer.slot_idx, unpinned_slot);
    assert_eq!(clock().owned(decompressed().id()), 3);
}

#[test]
fn evicting_a_compressed_extent_reinforces_its_decompressed_twin() {
    init_test_free_pool(8);
    let source_slot = insert_compressed(0, 4096);
    release_cursor(compressed());
    let twin_slot = insert_decompressed_slot(0, 4096); // same file range
    let lives_before = clock().lives(twin_slot);

    memory_ctx()
        .caches()
        .reclaim(compressed().id(), source_slot)
        .unwrap();

    assert_eq!(
        clock().lives(twin_slot),
        lives_before + lives_at_reinforce(decompressed())
    );
}

#[test]
fn evicting_a_decompressed_page_reinforces_its_compressed_source() {
    init_test_free_pool(8);
    let source_slot = insert_compressed(0, 4096);
    release_cursor(compressed());
    let page_slot = insert_decompressed_slot(0, 4096);
    let lives_before = clock().lives(source_slot);

    memory_ctx()
        .caches()
        .reclaim(decompressed().id(), page_slot)
        .unwrap();

    assert_eq!(
        clock().lives(source_slot),
        lives_before + lives_at_reinforce(compressed())
    );
}

#[test]
fn reading_a_decompressed_page_again_weakens_its_compressed_source() {
    init_test_free_pool(8);
    let source_slot = insert_compressed(0, 4096);
    release_cursor(compressed());
    insert_decompressed_slot(0, 4096);
    let lives_before = clock().lives(source_slot);

    decompressed().get(&key_in(&FD(), 0, 4096)).unwrap();

    assert_eq!(clock().lives(source_slot), lives_before - 1);
}

#[test]
fn a_run_of_decompressed_hits_weakens_the_compressed_slot_under_it_once() {
    init_test_free_pool(8);
    let source_slot = insert_compressed(0, 8192);
    release_cursor(compressed());
    insert_page(decompressed(), key_in(&FD(), 0, 4096), 64, 1);
    insert_page(decompressed(), key_in(&FD(), 4096, 4096), 64, 2);
    let lives_before = clock().lives(source_slot);

    let segments = decompressed().get_range(&FD(), 0, 8192);

    assert_eq!(segments.len(), 2);
    assert_eq!(clock().lives(source_slot), lives_before - 1);
}

#[test]
fn reading_compressed_bytes_leaves_the_decompressed_page_its_lives() {
    init_test_free_pool(8);
    insert_compressed(0, 4096);
    release_cursor(compressed());
    let page_slot = insert_decompressed_slot(0, 4096);
    let lives_before = clock().lives(page_slot);

    compressed().get(&FD(), 0, 4096);

    assert_eq!(clock().lives(page_slot), lives_before);
}

#[test]
fn a_heavier_cache_outlives_a_lighter_one_at_equal_recency() {
    init_test_free_pool(8);
    let sparse = &memory_ctx().sparse_index_caches().decoded;
    sparse
        .insert(key_in(&FD(), 0, 1), |fill| {
            fill.get_bytes(8);
            Ok::<_, Infallible>(SparseIndexMeta)
        })
        .unwrap();
    let sparse_slot = cursor_slot(sparse.as_ref());
    release_cursor(sparse.as_ref());
    let page_slot = insert_decompressed_slot(0, 4096);

    let evicted = memory_ctx().evict();

    assert_eq!(evicted.slot_idx, page_slot);
    assert_eq!(clock().owner(sparse_slot), Some(sparse.id()));
}

#[test]
fn reading_a_page_leaves_another_familys_cache_its_lives() {
    init_test_free_pool(8);
    let dict = &memory_ctx().dict_index_caches().compressed;
    let lookups = dict.get(&FD(), 0, 4096);
    lookups[0].missing_extent().unwrap().commit();
    let dict_slot = cursor_slot(dict.as_ref());
    release_cursor(dict.as_ref());
    insert_decompressed_slot(0, 4096); // the same file range, in the pages family
    let lives_before = clock().lives(dict_slot);

    decompressed().get(&key_in(&FD(), 0, 4096)).unwrap();

    assert_eq!(clock().lives(dict_slot), lives_before);
}
