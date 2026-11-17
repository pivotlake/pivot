use super::{count_bytes_cached, encoded_from, encoded_len, key, range, token};
use crate::io::FileRange;
use crate::memory::{RingCache, SlabAllocator, init_test_free_pool, memory_ctx};
use crate::text_index::{DictIndex, DictIndexBuilder, Error, FewRows, TokenRows};
use std::convert::Infallible;

/// What the `index`-th token of a test dict index is given: every other
/// token keeps a few rows in the dict index, the rest have postings.
fn rows_of_token(index: usize) -> TokenRows {
    if index.is_multiple_of(2) {
        let first_row = index as u32;
        return TokenRows::KeptInDict(FewRows::new(&[first_row, first_row + 1, first_row + 9]));
    }
    TokenRows::InPostings {
        postings: range(index),
        row_count: 1_000 + index,
    }
}

fn build_dict_index(token_count: usize) -> DictIndex {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = DictIndexBuilder::new();
    for index in 0..token_count {
        builder.push(&mut allocator, &token(index), rows_of_token(index));
    }
    builder.finish()
}

#[test]
fn a_token_in_few_rows_is_found_with_its_rows() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = DictIndexBuilder::new();
    builder.push(
        &mut allocator,
        b"kiwi",
        TokenRows::KeptInDict(FewRows::new(&[3, 17, 240])),
    );
    let dict_index = builder.finish();

    let found = dict_index.find_rows(b"kiwi");

    let Some(TokenRows::KeptInDict(few_rows)) = found else {
        panic!("the rows are kept in the dict index, found {found:?}");
    };
    assert_eq!(few_rows.as_slice(), [3, 17, 240]);
}

#[test]
fn a_token_in_many_rows_is_found_with_where_its_postings_are() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let in_postings = TokenRows::InPostings {
        postings: FileRange::new(52_040, 8_960),
        row_count: 12_000,
    };
    let mut builder = DictIndexBuilder::new();
    builder.push(&mut allocator, b"fig", in_postings);
    let dict_index = builder.finish();

    let found = dict_index.find_rows(b"fig");

    assert_eq!(found, Some(in_postings));
}

#[test]
fn a_dict_index_has_no_rows_for_a_token_it_does_not_hold() {
    init_test_free_pool(4);
    let dict_index = build_dict_index(3);

    let found = [&b"a"[..], b"token00000001x", b"token0000000", b"zebra"]
        .map(|token| dict_index.find_rows(token));

    assert_eq!(found, [None; 4]);
}

#[test]
fn a_dict_index_lying_in_many_chunks_finds_every_token() {
    init_test_free_pool(8);
    let dict_index = build_dict_index(20_000);

    let every_token_found =
        (0..20_000).all(|index| dict_index.find_rows(&token(index)) == Some(rows_of_token(index)));

    assert!(every_token_found);
    assert_eq!(dict_index.count_tokens(), 20_000);
}

#[test]
#[should_panic(expected = "ascending order")]
fn a_token_added_out_of_order_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = DictIndexBuilder::new();
    builder.push(&mut allocator, b"melon", rows_of_token(0));

    builder.push(&mut allocator, b"apple", rows_of_token(0));
}

#[test]
#[should_panic(expected = "keeps them in its dict index")]
fn postings_for_a_token_in_few_rows_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = DictIndexBuilder::new();

    builder.push(
        &mut allocator,
        b"kiwi",
        TokenRows::InPostings {
            postings: range(0),
            row_count: 8,
        },
    );
}

#[test]
#[should_panic(expected = "are not few")]
fn nine_rows_are_too_many_to_keep_in_a_dict_index() {
    FewRows::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
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
                builder.push(fill, &token(index), rows_of_token(index));
            }
            Ok::<_, Infallible>(builder.into_meta())
        })
        .unwrap();

    let dict_index = dict_indexes.get(&key).unwrap();

    assert!(
        (0..1_000).all(|index| dict_index.find_rows(&token(index)) == Some(rows_of_token(index)))
    );
    assert_eq!(dict_index.find_rows(b"token"), None);
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
    assert!(
        (0..20_000).all(|index| decoded.find_rows(&token(index)) == Some(rows_of_token(index)))
    );
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
fn the_last_row_32_bits_can_name_is_kept_through_encoding() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let rows = TokenRows::KeptInDict(FewRows::new(&[0, u32::MAX]));
    let mut builder = DictIndexBuilder::new();
    builder.push(&mut allocator, b"kiwi", rows);
    let dict_index = builder.finish();

    let encoded = dict_index.encode(&mut allocator);
    let decoded = DictIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish();

    assert_eq!(decoded.find_rows(b"kiwi"), Some(rows));
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

    assert!(
        (0..1_000).all(|index| dict_index.find_rows(&token(index)) == Some(rows_of_token(index)))
    );
}

#[test]
fn a_dict_index_decoded_into_its_cache_takes_twenty_bytes_a_token_beside_tokens_and_kept_rows() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let encoded = build_dict_index(1_000).encode(&mut allocator);
    let dict_indexes = &memory_ctx().text_index_caches().dict_indexes;
    let token_bytes_len = 1_000 * token(0).len();
    let kept_rows_len = 500 * 3 * size_of::<u32>();

    dict_indexes
        .insert(key(0), |fill| {
            Ok::<_, Error>(DictIndex::decode(&encoded, fill)?.into_meta())
        })
        .unwrap();

    // A few bytes are skipped wherever a chunk has to start aligned.
    let alignment_allowance = 64;
    assert!(
        count_bytes_cached(dict_indexes.id())
            <= 1_000 * 20 + token_bytes_len + kept_rows_len + alignment_allowance
    );
}

#[test]
fn tokens_encoded_out_of_order_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let b_then_a = encoded_from(&[2, 2, 2, 0, 1, b'b', 1, 5, 0, 1, b'a', 1, 6]);

    let decoded = DictIndex::decode(&b_then_a, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_token_in_no_rows_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let one_token_in_zero_rows = encoded_from(&[1, 1, 0, 0, 1, b'a', 0]);

    let decoded = DictIndex::decode(&one_token_in_zero_rows, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn more_kept_rows_than_the_dict_index_says_it_has_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let one_kept_row_declared_two_present = encoded_from(&[1, 1, 1, 0, 1, b'a', 2, 5, 0]);

    let decoded = DictIndex::decode(&one_kept_row_declared_two_present, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn fewer_kept_rows_than_the_dict_index_says_it_has_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let three_kept_rows_declared_one_present = encoded_from(&[1, 1, 3, 0, 1, b'a', 1, 5]);

    let decoded = DictIndex::decode(&three_kept_rows_declared_one_present, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn a_dict_index_cut_short_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let two_tokens_with_one_record = encoded_from(&[2, 2, 1, 0, 1, b'a', 1, 5]);

    let decoded = DictIndex::decode(&two_tokens_with_one_record, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}
