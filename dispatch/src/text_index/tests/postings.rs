use super::{count_bytes_cached, encoded_from, encoded_len, key};
use crate::memory::{RingCache, SlabAllocator, init_test_free_pool, memory_ctx};
use crate::text_index::{Error, Postings, PostingsBuilder};
use std::convert::Infallible;
use std::ops::Range;

fn build_postings(rows: impl IntoIterator<Item = u32>) -> Postings {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = PostingsBuilder::new();
    for row in rows {
        builder.push(&mut allocator, row);
    }
    builder.finish(&mut allocator)
}

/// The rows of a token that is rare in the file's first 200,000 rows (every
/// 1,000th) and common in the next 200,000 (every other row): some segments
/// listed, some as bits, and one that changes over halfway.
fn rows_rare_then_common() -> Vec<u32> {
    let rare = (0..200_000).step_by(1_000);
    let common = (200_000..400_000).step_by(2);
    rare.chain(common).collect()
}

fn rows_of(postings: &Postings) -> Vec<u32> {
    let mut rows = Vec::new();
    postings.for_each_row(|row| rows.push(row));
    rows
}

/// The rows `selected` selects, when its first bit is row `first_row`.
fn selected_rows(selected: &[u64], first_row: u32) -> Vec<u32> {
    let bits = selected.len() as u32 * 64;
    (0..bits)
        .filter(|bit| selected[*bit as usize / 64] & (1 << (bit % 64)) != 0)
        .map(|bit| first_row + bit)
        .collect()
}

/// Words enough for a bit for each row of `rows`, with every row selected
/// or none.
fn selection_over(rows: &Range<u32>, selected: bool) -> Vec<u64> {
    let mut words = vec![0u64; rows.len().div_ceil(64)];
    if selected {
        for bit in 0..rows.len() {
            words[bit / 64] |= 1 << (bit % 64);
        }
    }
    words
}

#[test]
fn postings_visit_the_rows_added_in_ascending_order() {
    init_test_free_pool(8);
    let rows = rows_rare_then_common();

    let postings = build_postings(rows.iter().copied());

    assert_eq!(rows_of(&postings), rows);
    assert_eq!(postings.count_rows(), rows.len());
}

#[test]
fn postings_hold_exactly_the_rows_added() {
    init_test_free_pool(8);
    let postings = build_postings(rows_rare_then_common());

    let wrongly_answered = (0..400_000u32)
        .filter(|row| {
            let was_added = if *row < 200_000 {
                row.is_multiple_of(1_000)
            } else {
                row.is_multiple_of(2)
            };
            postings.contains_row(*row) != was_added
        })
        .count();

    assert_eq!(wrongly_answered, 0);
    assert!(!postings.contains_row(400_000));
    assert!(!postings.contains_row(u32::MAX));
}

#[test]
fn postings_of_no_rows_hold_no_row() {
    init_test_free_pool(4);

    let postings = build_postings([]);

    assert_eq!(postings.count_rows(), 0);
    assert!(!postings.contains_row(0));
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
fn a_rare_token_takes_two_bytes_a_row_and_a_common_one_a_bit_a_row() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let rare = build_postings((0..65_536).step_by(100));
    let common = build_postings(0..65_536);

    let rare_len = encoded_len(&rare.encode(&mut allocator));
    let common_len = encoded_len(&common.encode(&mut allocator));

    assert_eq!(rare_len, 1 + 4 + 2 * 656);
    assert_eq!(common_len, 1 + 4 + 65_536 / 8);
}

#[test]
fn selecting_a_range_selects_the_token_rows_inside_it() {
    init_test_free_pool(8);
    let rows = rows_rare_then_common();
    let postings = build_postings(rows.iter().copied());
    let range = 130_990..262_200;
    let mut selected = selection_over(&range, false);

    postings.select_rows(range.clone(), &mut selected);

    let rows_in_range: Vec<u32> = rows.into_iter().filter(|row| range.contains(row)).collect();
    assert_eq!(selected_rows(&selected, range.start), rows_in_range);
}

#[test]
fn selecting_leaves_rows_already_selected_as_they_are() {
    init_test_free_pool(4);
    let postings = build_postings([10, 20]);
    let mut selected = vec![1u64 << 5];

    postings.select_rows(0..64, &mut selected);

    assert_eq!(selected_rows(&selected, 0), [5, 10, 20]);
}

#[test]
fn keeping_a_range_leaves_selected_only_the_token_rows_that_were() {
    init_test_free_pool(8);
    let rows = rows_rare_then_common();
    let postings = build_postings(rows.iter().copied());
    let range = 130_990..262_200;
    let mut selected = selection_over(&range, true);

    postings.keep_selected_rows(range.clone(), &mut selected);

    let rows_in_range: Vec<u32> = rows.into_iter().filter(|row| range.contains(row)).collect();
    assert_eq!(selected_rows(&selected, range.start), rows_in_range);
}

#[test]
fn keeping_selects_no_row_that_was_not_selected() {
    init_test_free_pool(4);
    let postings = build_postings([10, 20, 30]);
    let mut selected = vec![(1u64 << 20) | (1 << 21)];

    postings.keep_selected_rows(0..64, &mut selected);

    assert_eq!(selected_rows(&selected, 0), [20]);
}

#[test]
fn rows_all_of_two_tokens_appear_in_are_what_selecting_one_and_keeping_the_other_leaves() {
    init_test_free_pool(8);
    let every_third = build_postings((0..300_000).step_by(3));
    let every_fifth = build_postings((0..300_000).step_by(5));
    let range = 60_000..140_000;
    let mut selected = selection_over(&range, false);

    every_third.select_rows(range.clone(), &mut selected);
    every_fifth.keep_selected_rows(range.clone(), &mut selected);

    let every_fifteenth: Vec<u32> = range.clone().filter(|row| row.is_multiple_of(15)).collect();
    assert_eq!(selected_rows(&selected, range.start), every_fifteenth);
}

#[test]
fn a_range_the_token_never_appears_in_selects_nothing_and_keeps_nothing() {
    init_test_free_pool(4);
    let postings = build_postings([5, 1_000_000]);
    let range = 100..900_000;
    let mut selected_by_select = selection_over(&range, false);
    let mut selected_by_keep = selection_over(&range, true);

    postings.select_rows(range.clone(), &mut selected_by_select);
    postings.keep_selected_rows(range.clone(), &mut selected_by_keep);

    assert!(selected_by_select.iter().all(|word| *word == 0));
    assert!(selected_rows(&selected_by_keep, range.start).is_empty());
}

#[test]
fn postings_decode_to_the_rows_encoded() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let rows = rows_rare_then_common();
    let postings = build_postings(rows.iter().copied());

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator)
        .unwrap()
        .finish(&mut allocator);

    assert_eq!(rows_of(&decoded), rows);
    assert_eq!(decoded.count_rows(), rows.len());
    assert!(encoded.len() > 1);
}

#[test]
fn postings_of_no_rows_decode_to_no_rows() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let postings = build_postings([]);

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator)
        .unwrap()
        .finish(&mut allocator);

    assert_eq!(decoded.count_rows(), 0);
}

#[test]
fn the_last_row_32_bits_can_name_survives_encoding() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let postings = build_postings([7, u32::MAX]);

    let encoded = postings.encode(&mut allocator);
    let decoded = Postings::decode(&encoded, &mut allocator)
        .unwrap()
        .finish(&mut allocator);

    assert_eq!(rows_of(&decoded), [7, u32::MAX]);
    assert!(decoded.contains_row(u32::MAX));
}

#[test]
fn postings_built_into_their_cache_are_looked_up_from_there() {
    init_test_free_pool(8);
    let postings_cache = &memory_ctx().text_index_caches().postings;
    let rows = rows_rare_then_common();
    let key = key(0);
    postings_cache
        .insert(key.clone(), |fill| {
            let mut builder = PostingsBuilder::new();
            for &row in &rows {
                builder.push(fill, row);
            }
            Ok::<_, Infallible>(builder.into_meta(fill))
        })
        .unwrap();

    let postings = postings_cache.get(&key).unwrap();

    assert_eq!(rows_of(&postings), rows);
}

#[test]
fn postings_are_decoded_straight_into_their_cache() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let rows = rows_rare_then_common();
    let encoded = build_postings(rows.iter().copied()).encode(&mut allocator);
    let postings_cache = &memory_ctx().text_index_caches().postings;
    let key = key(0);

    postings_cache
        .insert(key.clone(), |fill| {
            Ok::<_, Error>(Postings::decode(&encoded, fill)?.into_meta(fill))
        })
        .unwrap();
    let postings = postings_cache.get(&key).unwrap();

    assert_eq!(rows_of(&postings), rows);
}

#[test]
fn postings_decoded_into_their_cache_take_the_bytes_of_their_segments_and_positions() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let encoded = build_postings([3, 17, 240, 70_000]).encode(&mut allocator);
    let postings_cache = &memory_ctx().text_index_caches().postings;

    postings_cache
        .insert(key(0), |fill| {
            Ok::<_, Error>(Postings::decode(&encoded, fill)?.into_meta(fill))
        })
        .unwrap();

    // Two segments at 8 bytes each and four positions at 2 bytes each.
    assert_eq!(count_bytes_cached(postings_cache.id()), 2 * 8 + 4 * 2);
}

#[test]
fn postings_cut_short_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let one_segment_of_two_rows_with_one_position = encoded_from(&[1, 0, 0, 1, 0, 5, 0]);

    let decoded = Postings::decode(&one_segment_of_two_rows_with_one_position, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn bytes_after_the_end_of_postings_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let no_segments_then_a_byte = encoded_from(&[0, 7]);

    let decoded = Postings::decode(&no_segments_then_a_byte, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn segments_encoded_out_of_order_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let segment_two_then_segment_one = encoded_from(&[2, 2, 0, 0, 0, 1, 0, 0, 0, 5, 0, 6, 0]);

    let decoded = Postings::decode(&segment_two_then_segment_one, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn positions_encoded_out_of_order_are_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let positions_nine_then_five = encoded_from(&[1, 0, 0, 1, 0, 9, 0, 5, 0]);

    let decoded = Postings::decode(&positions_nine_then_five, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}

#[test]
fn bits_holding_another_number_of_rows_than_their_segment_says_are_refused() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let mut encoded_bytes = vec![1, 0, 0, 0xff, 0xff]; // one segment, said to hold 65,536 rows
    encoded_bytes.extend(std::iter::repeat_n(0xff, 8_190)); // bits for 65,520 rows
    encoded_bytes.extend([0, 0]);
    let all_rows_said_but_sixteen_missing = encoded_from(&encoded_bytes);

    let decoded = Postings::decode(&all_rows_said_but_sixteen_missing, &mut allocator);

    assert!(matches!(decoded, Err(Error::Corrupt(_))));
}
