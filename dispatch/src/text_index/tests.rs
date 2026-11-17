//! The three pieces of a text index, gathered in working memory and in their
//! caches. Every test installs its own single-worker memory context.

use super::*;
use crate::io::{FileRange, LocalFile, OpenFile};
use crate::memory::{CacheKey, RingCache, Slab, SlabAllocator, init_test_free_pool, memory_ctx};
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

// -- the encoded forms --

/// Postings of `row_count` rows, `step` rows apart.
fn build_postings_every(step: u32, row_count: u32) -> Postings {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = PostingsBuilder::new();
    for index in 0..row_count {
        builder.push(&mut allocator, index * step);
    }
    builder.finish()
}

fn rows_of(postings: &Postings) -> Vec<u32> {
    let mut rows = Vec::new();
    postings.for_each_row(|row| rows.push(row));
    rows
}

fn encoded_len(encoded: &[Slab]) -> usize {
    encoded.iter().map(Slab::size).sum()
}

/// `bytes` as encoded input.
fn encoded_from(bytes: &[u8]) -> Vec<Slab> {
    let mut slab = SlabAllocator::new(false).get_slab_of_size(bytes.len(), false);
    slab.as_mut_slice().copy_from_slice(bytes);
    vec![slab]
}

#[test]
fn postings_of_a_rare_token_decode_to_the_rows_encoded() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let postings = build_postings_every(1_000, 5_000);

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator).unwrap().finish();

    assert_eq!(rows_of(&decoded), rows_of(&postings));
    assert!(encoded_len(&encoded) <= 2 * 5_000 + 16);
}

#[test]
fn postings_of_a_token_in_most_rows_decode_to_the_rows_encoded() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let postings = build_postings_every(2, 100_000);

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator).unwrap().finish();

    assert_eq!(rows_of(&decoded), rows_of(&postings));
    assert!(encoded_len(&encoded) <= 200_000 / 8 + 16);
}

#[test]
fn postings_of_no_rows_decode_to_no_rows() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let postings = build_postings_every(1, 0);

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator).unwrap().finish();

    assert_eq!(decoded.count_rows(), 0);
}

#[test]
fn the_last_row_32_bits_can_name_survives_encoding() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = PostingsBuilder::new();
    builder.push(&mut allocator, 7);
    builder.push(&mut allocator, u32::MAX);
    let postings = builder.finish();

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator).unwrap().finish();

    assert_eq!(rows_of(&decoded), [7, u32::MAX]);
}

#[test]
fn a_dict_index_decodes_to_the_tokens_encoded() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let dict_index = build_dict_index(20_000);

    let encoded = dict_index.encode(&mut allocator);
    let decoded = DictIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish();

    assert_eq!(decoded.count_tokens(), 20_000);
    assert!((0..20_000).all(|index| decoded.find_postings(&token(index)) == Some(range(index))));
}

#[test]
fn an_encoded_dict_index_is_smaller_than_its_tokens_and_lies_in_several_slabs() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let dict_index = build_dict_index(20_000);
    let token_bytes_len = 20_000 * token(0).len();

    let encoded = dict_index.encode(&mut allocator);

    assert!(encoded_len(&encoded) < token_bytes_len);
    assert!(encoded.len() > 1);
}

#[test]
fn a_sparse_index_decodes_to_the_dict_indexes_encoded() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let sparse_index = build_sparse_index(&["apple", "date", "melon"]);

    let encoded = sparse_index.encode(&mut allocator);
    let decoded = SparseIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish();

    let found = ["apple", "kiwi", "zebra"].map(|token| decoded.find_dict_index(token.as_bytes()));
    assert_eq!(found, [0, 1, 2].map(|index| Some(range(index))));
    assert_eq!(decoded.find_dict_index(b"aardvark"), None);
}

#[test]
fn ranges_in_any_order_survive_encoding() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let ranges = [
        FileRange::new(9_000, 10),
        FileRange::new(0, 5),
        FileRange::new(70, 0),
    ];
    let mut builder = DictIndexBuilder::new();
    for (index, range) in ranges.into_iter().enumerate() {
        builder.push(&mut allocator, &token(index), range);
    }
    let dict_index = builder.finish();

    let encoded = dict_index.encode(&mut allocator);
    let decoded = DictIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish();

    let found = [0, 1, 2].map(|index| decoded.find_postings(&token(index)));
    assert_eq!(found, ranges.map(Some));
}

#[test]
fn a_dict_index_is_decoded_straight_into_its_cache() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let encoded = build_dict_index(1_000).encode(&mut allocator);
    let dict_indexes = &memory_ctx().text_index_caches().dict_indexes;
    let key = key(0);

    dict_indexes
        .insert(key.clone(), |fill| {
            Ok::<_, Error>(DictIndex::decode(&encoded, fill)?.into_meta())
        })
        .unwrap();
    let dict_index = dict_indexes.get(&key).unwrap();

    assert!((0..1_000).all(|index| dict_index.find_postings(&token(index)) == Some(range(index))));
}

#[test]
fn postings_are_decoded_straight_into_their_cache() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let rows = rows_of(&build_postings_every(2, 100_000));
    let encoded = build_postings_every(2, 100_000).encode(&mut allocator);
    let postings_cache = &memory_ctx().text_index_caches().postings;
    let key = key(0);

    postings_cache
        .insert(key.clone(), |fill| {
            Ok::<_, Error>(Postings::decode(&encoded, fill)?.into_meta())
        })
        .unwrap();
    let postings = postings_cache.get(&key).unwrap();

    assert_eq!(rows_of(&postings), rows);
}

#[test]
fn postings_cut_short_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let three_rows_with_one_gap = encoded_from(&[3, 0, 5]);

    let decoded = Postings::decode(&three_rows_with_one_gap, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn bytes_after_the_end_of_postings_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let no_rows_then_a_byte = encoded_from(&[0, 7]);

    let decoded = Postings::decode(&no_rows_then_a_byte, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn postings_in_an_unknown_form_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let one_row_in_form_nine = encoded_from(&[1, 9, 0]);

    let decoded = Postings::decode(&one_row_in_form_nine, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_bitmap_holding_another_number_of_rows_than_it_says_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let two_rows_but_one_bit_set = encoded_from(&[2, 1, 0, 8, 0b0000_0001]);

    let decoded = Postings::decode(&two_rows_but_one_bit_set, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_row_past_32_bits_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let one_row_at_two_to_the_32 = encoded_from(&[1, 0, 0x80, 0x80, 0x80, 0x80, 0x10]);

    let decoded = Postings::decode(&one_row_at_two_to_the_32, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn tokens_encoded_out_of_order_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let b_then_a = encoded_from(&[2, 0, 1, b'b', 0, 1, 0, 1, b'a', 0, 1]);

    let decoded = DictIndex::decode(&b_then_a, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_token_sharing_more_bytes_than_the_token_before_it_has_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let a_then_a_token_sharing_five_bytes = encoded_from(&[2, 0, 1, b'a', 0, 1, 5, 1, b'b', 0, 1]);

    let decoded = SparseIndex::decode(&a_then_a_token_sharing_five_bytes, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_dict_index_cut_short_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let two_tokens_with_one_record = encoded_from(&[2, 0, 1, b'a', 0, 1]);

    let decoded = DictIndex::decode(&two_tokens_with_one_record, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}
