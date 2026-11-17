use super::{count_bytes_cached, encoded_from, key, range, token};
use crate::memory::{RingCache, SlabAllocator, init_test_free_pool, memory_ctx};
use crate::text_index::{Error, SparseIndex, SparseIndexBuilder};
use std::convert::Infallible;

fn build_sparse_index(smallest_tokens: &[&str]) -> SparseIndex {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = SparseIndexBuilder::new();
    for (index, smallest_token) in smallest_tokens.iter().enumerate() {
        builder.push(&mut allocator, smallest_token.as_bytes(), range(index));
    }
    builder.finish()
}

/// A sparse index of `dict_index_count` dict indexes, the `index`-th
/// starting at `token(index)`.
fn build_long_sparse_index(dict_index_count: usize) -> SparseIndex {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = SparseIndexBuilder::new();
    for index in 0..dict_index_count {
        builder.push(&mut allocator, &token(index), range(index));
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
    let sparse_index = build_long_sparse_index(20_000);

    let every_token_found = (0..20_000).all(|index| {
        let just_above = [token(index), b"x".to_vec()].concat();
        sparse_index.find_dict_index(&token(index)) == Some(range(index))
            && sparse_index.find_dict_index(&just_above) == Some(range(index))
    });

    assert!(every_token_found);
    assert_eq!(sparse_index.count_dict_indexes(), 20_000);
}

#[test]
#[should_panic(expected = "ascending order")]
fn a_dict_index_added_out_of_order_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = SparseIndexBuilder::new();
    builder.push(&mut allocator, b"melon", range(0));

    builder.push(&mut allocator, b"apple", range(1));
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
            for index in 0..150_000 {
                builder.push(fill, &token(index), range(index));
            }
            Ok::<_, Infallible>(builder.into_meta())
        })
        .unwrap();

    let sparse_index = sparse_indexes.get(&key).unwrap();

    assert!(memory_ctx().clock().count_owned(sparse_indexes.id()) > 1);
    assert!(
        (0..150_000).all(|index| sparse_index.find_dict_index(&token(index)) == Some(range(index)))
    );
}

#[test]
fn a_sparse_index_decodes_to_the_dict_indexes_encoded() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let sparse_index = build_long_sparse_index(20_000);

    let encoded = sparse_index.encode(&mut allocator);
    let decoded = SparseIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish();

    assert_eq!(decoded.count_dict_indexes(), 20_000);
    assert!((0..20_000).all(|index| decoded.find_dict_index(&token(index)) == Some(range(index))));
}

#[test]
fn dict_indexes_at_any_place_in_the_file_survive_encoding() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let ranges = [(9_000, 10), (0, 5), (70, 0)];
    let mut builder = SparseIndexBuilder::new();
    for (index, (offset, len)) in ranges.into_iter().enumerate() {
        builder.push(
            &mut allocator,
            &token(index),
            crate::io::FileRange::new(offset, len),
        );
    }
    let sparse_index = builder.finish();

    let encoded = sparse_index.encode(&mut allocator);
    let decoded = SparseIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish();

    let found = [0, 1, 2].map(|index| {
        let range = decoded.find_dict_index(&token(index)).unwrap();
        (range.offset, range.len)
    });
    assert_eq!(found, ranges);
}

#[test]
fn a_sparse_index_decoded_into_its_cache_takes_the_bytes_of_its_entries_and_tokens() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let encoded = build_sparse_index(&["apple", "date", "melon"]).encode(&mut allocator);
    let sparse_indexes = &memory_ctx().text_index_caches().sparse_indexes;

    sparse_indexes
        .insert(key(0), |fill| {
            Ok::<_, Error>(SparseIndex::decode(&encoded, fill)?.into_meta())
        })
        .unwrap();

    // Three dict indexes at 16 bytes each and 14 bytes of tokens, with a few
    // bytes skipped to align the vecs.
    assert!(count_bytes_cached(sparse_indexes.id()) <= 3 * 16 + 14 + 8);
}

#[test]
fn a_token_sharing_more_bytes_than_the_token_before_it_has_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let a_then_a_token_sharing_five_bytes = encoded_from(&[2, 2, 0, 1, b'a', 0, 1, 5, 1, b'b']);

    let decoded = SparseIndex::decode(&a_then_a_token_sharing_five_bytes, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn smallest_tokens_encoded_out_of_order_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let b_then_a = encoded_from(&[2, 2, 0, 1, b'b', 0, 1, 0, 1, b'a', 0, 1]);

    let decoded = SparseIndex::decode(&b_then_a, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn more_token_bytes_than_the_sparse_index_says_it_has_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let one_byte_declared_two_present = encoded_from(&[1, 1, 0, 2, b'a', b'b', 0, 1]);

    let decoded = SparseIndex::decode(&one_byte_declared_two_present, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_sparse_index_cut_short_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let two_dict_indexes_with_one_record = encoded_from(&[2, 2, 0, 1, b'a', 0, 1]);

    let decoded = SparseIndex::decode(&two_dict_indexes_with_one_record, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn bytes_after_the_end_of_a_sparse_index_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let no_dict_indexes_then_a_byte = encoded_from(&[0, 0, 7]);

    let decoded = SparseIndex::decode(&no_dict_indexes_then_a_byte, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}
