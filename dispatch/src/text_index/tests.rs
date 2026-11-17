//! The three pieces of a text index, gathered in working memory and in their
//! caches. Every test installs its own single-worker memory context.

use super::*;
use crate::io::{FileRange, LocalFile, OpenFile};
use crate::memory::{CacheKey, RingCache, SlabAllocator, init_test_free_pool, memory_ctx};
use std::convert::Infallible;

/// The `index`-th of a run of ascending tokens.
fn token(index: usize) -> Vec<u8> {
    format!("token{index:08}").into_bytes()
}

/// A range that tells which token it was stored for.
fn range(index: usize) -> FileRange {
    FileRange::new(index * 100, 100)
}

/// A key for the piece at `offset` of some index file.
fn key(offset: usize) -> CacheKey {
    let index_file = LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
    CacheKey::new(OpenFile::Local(index_file), offset, 1)
}

fn build_sparse_index(smallest_tokens: &[&str]) -> SparseIndex {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = SparseIndexBuilder::new();
    for (index, smallest_token) in smallest_tokens.iter().enumerate() {
        builder.push(&mut allocator, smallest_token.as_bytes(), range(index));
    }
    builder.finish()
}

fn build_dict_index(token_count: usize) -> DictIndex {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = DictIndexBuilder::new();
    for index in 0..token_count {
        builder.push(&mut allocator, &token(index), range(index));
    }
    builder.finish()
}

/// Postings of a token appearing in every third row, `row_count` times.
fn build_postings(row_count: u32) -> Postings {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = PostingsBuilder::new();
    for index in 0..row_count {
        builder.push(&mut allocator, index * 3);
    }
    builder.finish()
}

#[test]
fn a_token_is_in_the_last_dict_index_whose_smallest_token_is_not_above_it() {
    init_test_free_pool(4);
    let sparse_index = build_sparse_index(&["apple", "date", "melon"]);

    let found = ["apple", "kiwi", "melon", "zebra"]
        .map(|token| sparse_index.find_dict_index(token.as_bytes()));

    assert_eq!(found, [0, 1, 2, 2].map(|index| Some(range(index))));
}

#[test]
fn a_token_below_every_token_of_the_file_is_in_no_dict_index() {
    init_test_free_pool(4);
    let sparse_index = build_sparse_index(&["apple", "date"]);

    let found = sparse_index.find_dict_index(b"aardvark");

    assert_eq!(found, None);
}

#[test]
fn an_empty_sparse_index_names_no_dict_index() {
    init_test_free_pool(4);
    let sparse_index = build_sparse_index(&[]);

    let found = sparse_index.find_dict_index(b"apple");

    assert_eq!(found, None);
    assert_eq!(sparse_index.count_dict_indexes(), 0);
}

#[test]
fn a_sparse_index_lying_in_many_chunks_finds_the_dict_index_of_every_token() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = SparseIndexBuilder::new();
    for index in 0..20_000 {
        builder.push(&mut allocator, &token(index), range(index));
    }
    let sparse_index = builder.finish();

    let every_token_found = (0..20_000).all(|index| {
        let just_above = [token(index), b"x".to_vec()].concat();
        sparse_index.find_dict_index(&token(index)) == Some(range(index))
            && sparse_index.find_dict_index(&just_above) == Some(range(index))
    });

    assert!(every_token_found);
    assert_eq!(sparse_index.count_dict_indexes(), 20_000);
}

#[test]
fn a_dict_index_finds_the_postings_of_each_token_it_holds() {
    init_test_free_pool(4);
    let dict_index = build_dict_index(3);

    let found = [0, 1, 2].map(|index| dict_index.find_postings(&token(index)));

    assert_eq!(found, [0, 1, 2].map(|index| Some(range(index))));
}

#[test]
fn a_dict_index_has_no_postings_for_a_token_it_does_not_hold() {
    init_test_free_pool(4);
    let dict_index = build_dict_index(3);

    let found = [&b"a"[..], b"token00000001x", b"token0000000", b"zebra"]
        .map(|token| dict_index.find_postings(token));

    assert_eq!(found, [None; 4]);
}

#[test]
fn a_dict_index_lying_in_many_chunks_finds_every_token() {
    init_test_free_pool(8);
    let dict_index = build_dict_index(20_000);

    let every_token_found =
        (0..20_000).all(|index| dict_index.find_postings(&token(index)) == Some(range(index)));

    assert!(every_token_found);
    assert_eq!(dict_index.count_tokens(), 20_000);
}

#[test]
#[should_panic(expected = "ascending order")]
fn a_token_added_out_of_order_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = DictIndexBuilder::new();
    builder.push(&mut allocator, b"melon", range(0));

    builder.push(&mut allocator, b"apple", range(1));
}

#[test]
fn postings_lying_in_many_chunks_hold_exactly_the_rows_added() {
    init_test_free_pool(8);
    let postings = build_postings(30_000);

    let wrongly_answered = (0..90_000u32)
        .filter(|row| postings.contains_row(*row) != row.is_multiple_of(3))
        .count();

    assert_eq!(wrongly_answered, 0);
    assert!(!postings.contains_row(90_000));
    assert_eq!(postings.count_rows(), 30_000);
}

#[test]
fn postings_visit_their_rows_in_ascending_order() {
    init_test_free_pool(8);
    let postings = build_postings(30_000);

    let mut rows = Vec::new();
    postings.for_each_row(|row| rows.push(row));

    assert_eq!(rows, (0..30_000).map(|index| index * 3).collect::<Vec<_>>());
}

#[test]
#[should_panic(expected = "ascending order")]
fn a_row_added_out_of_order_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = PostingsBuilder::new();
    builder.push(&mut allocator, 7);

    builder.push(&mut allocator, 7);
}

#[test]
fn a_sparse_index_built_into_its_cache_is_looked_up_from_there() {
    init_test_free_pool(8);
    let sparse_indexes = &memory_ctx().text_index_caches().sparse_indexes;
    let key = key(0);
    sparse_indexes
        .insert(key.clone(), |fill| {
            let mut builder = SparseIndexBuilder::new();
            for index in 0..1_000 {
                builder.push(fill, &token(index), range(index));
            }
            Ok::<_, Infallible>(builder.into_meta())
        })
        .unwrap();

    let sparse_index = sparse_indexes.get(&key).unwrap();

    assert!(
        (0..1_000).all(|index| sparse_index.find_dict_index(&token(index)) == Some(range(index)))
    );
}

#[test]
fn a_cached_sparse_index_larger_than_a_ring_slot_is_looked_up_across_slots() {
    init_test_free_pool(8);
    let sparse_indexes = &memory_ctx().text_index_caches().sparse_indexes;
    let key = key(0);
    sparse_indexes
        .insert(key.clone(), |fill| {
            let mut builder = SparseIndexBuilder::new();
            for index in 0..100_000 {
                builder.push(fill, &token(index), range(index));
            }
            Ok::<_, Infallible>(builder.into_meta())
        })
        .unwrap();

    let sparse_index = sparse_indexes.get(&key).unwrap();

    assert!(memory_ctx().clock().count_owned(sparse_indexes.id()) > 1);
    assert!(
        (0..100_000).all(|index| sparse_index.find_dict_index(&token(index)) == Some(range(index)))
    );
}

#[test]
fn a_dict_index_built_into_its_cache_is_looked_up_from_there() {
    init_test_free_pool(8);
    let dict_indexes = &memory_ctx().text_index_caches().dict_indexes;
    let key = key(0);
    dict_indexes
        .insert(key.clone(), |fill| {
            let mut builder = DictIndexBuilder::new();
            for index in 0..1_000 {
                builder.push(fill, &token(index), range(index));
            }
            Ok::<_, Infallible>(builder.into_meta())
        })
        .unwrap();

    let dict_index = dict_indexes.get(&key).unwrap();

    assert!((0..1_000).all(|index| dict_index.find_postings(&token(index)) == Some(range(index))));
    assert_eq!(dict_index.find_postings(b"token"), None);
}

#[test]
fn postings_built_into_their_cache_are_looked_up_from_there() {
    init_test_free_pool(8);
    let postings_cache = &memory_ctx().text_index_caches().postings;
    let key = key(0);
    postings_cache
        .insert(key.clone(), |fill| {
            let mut builder = PostingsBuilder::new();
            for index in 0..30_000 {
                builder.push(fill, index * 3);
            }
            Ok::<_, Infallible>(builder.into_meta())
        })
        .unwrap();

    let postings = postings_cache.get(&key).unwrap();

    assert_eq!(postings.count_rows(), 30_000);
    assert!(postings.contains_row(89_997));
    assert!(!postings.contains_row(89_998));
}
