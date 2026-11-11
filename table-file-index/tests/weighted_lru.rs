//! What a map within a budget of bytes holds, and what it lets go of past
//! the budget.

use table_file_index::WeightedLru;

/// A map of `capacity_bytes` holding each of `keys`, at ten bytes each,
/// inserted in the order given.
fn fill_lru(capacity_bytes: u64, keys: &[&str]) -> WeightedLru<String, u32> {
    let mut lru = WeightedLru::new(capacity_bytes);
    for (value, key) in keys.iter().enumerate() {
        lru.insert(key.to_string(), value as u32, 10);
    }
    lru
}

#[test]
fn an_inserted_value_is_found_by_a_borrowed_key() {
    let mut lru = fill_lru(u64::MAX, &["a"]);

    let found = lru.get("a").copied();

    assert_eq!(found, Some(0));
}

#[test]
fn past_the_budget_the_entry_used_longest_ago_is_handed_back() {
    let mut lru = fill_lru(25, &["old", "kept"]);
    lru.get("old");

    let evicted = lru.insert("new".to_string(), 2, 10);

    assert_eq!(evicted, [("kept".to_string(), 1)]);
    assert!(lru.peek("old").is_some() && lru.peek("new").is_some());
}

#[test]
fn a_peeked_entry_is_not_counted_as_used() {
    let mut lru = fill_lru(25, &["peeked", "kept"]);
    lru.peek("peeked");

    let evicted = lru.insert("new".to_string(), 2, 10);

    assert_eq!(evicted, [("peeked".to_string(), 0)]);
}

#[test]
fn an_entry_marked_used_outlives_the_ones_used_before_it() {
    let mut lru = fill_lru(25, &["marked", "left"]);
    lru.mark_used("marked");

    let evicted = lru.insert("new".to_string(), 2, 10);

    assert_eq!(evicted, [("left".to_string(), 1)]);
}

#[test]
fn the_entry_just_inserted_stays_whatever_it_weighs() {
    let mut lru = fill_lru(25, &["a", "b"]);

    let evicted = lru.insert("heavy".to_string(), 2, 100);

    assert_eq!(evicted, [("a".to_string(), 0), ("b".to_string(), 1)]);
    assert_eq!(lru.peek("heavy"), Some(&2));
}

#[test]
fn inserting_under_a_held_key_frees_the_weight_of_the_value_it_drops() {
    let mut lru = fill_lru(25, &["a", "b"]);

    let evicted = lru.insert("a".to_string(), 7, 15);

    assert!(evicted.is_empty());
    assert_eq!((lru.peek("a"), lru.peek("b")), (Some(&7), Some(&1)));
}

#[test]
fn a_removed_entry_is_handed_back_and_leaves_its_room() {
    let mut lru = fill_lru(25, &["removed", "kept"]);

    let removed = lru.remove("removed");
    let evicted = lru.insert("new".to_string(), 2, 10);

    assert_eq!(removed, Some(("removed".to_string(), 0)));
    assert!(evicted.is_empty());
    assert!(lru.peek("removed").is_none() && lru.peek("kept").is_some());
}

/// A map holding `file-0` up to `file-{entry_count - 1}`, each under its
/// number at ten bytes, inserted in that order, which its budget just fits.
fn fill_numbered_lru(entry_count: u32) -> WeightedLru<String, u32> {
    let mut lru = WeightedLru::new(u64::from(entry_count) * 10);
    for number in 0..entry_count {
        lru.insert(format!("file-{number}"), number, 10);
    }
    lru
}

/// Remove `file-{number}` from `lru` for each of `numbers`.
fn remove_numbered(lru: &mut WeightedLru<String, u32>, numbers: std::ops::Range<u32>) {
    for number in numbers {
        lru.remove(format!("file-{number}").as_str());
    }
}

#[test]
fn slots_are_given_back_once_under_a_quarter_hold_an_entry() {
    let mut lru = fill_numbered_lru(5000);
    let slots_when_full = lru.count_allocated_slots();

    remove_numbered(&mut lru, 0..4000);

    assert!(slots_when_full >= 5000);
    assert!(lru.count_allocated_slots() < 1250);
}

#[test]
fn slots_are_kept_while_a_quarter_of_them_hold_an_entry() {
    let mut lru = fill_numbered_lru(5000);
    let slots_when_full = lru.count_allocated_slots();

    remove_numbered(&mut lru, 0..3750);

    assert_eq!(lru.count_allocated_slots(), slots_when_full);
}

#[test]
fn a_map_of_few_slots_keeps_them_all() {
    let mut lru = fill_numbered_lru(1000);
    let slots_when_full = lru.count_allocated_slots();

    remove_numbered(&mut lru, 0..999);

    assert_eq!(lru.count_allocated_slots(), slots_when_full);
    assert_eq!(lru.peek("file-999"), Some(&999));
}

#[test]
fn entries_keep_their_order_of_last_use_when_slots_are_given_back() {
    let mut lru = fill_numbered_lru(5000);
    lru.get("file-10");
    remove_numbered(&mut lru, 0..10);
    remove_numbered(&mut lru, 11..4997);

    let evicted = lru.insert("heavy".to_string(), 0, 50_000);

    let evicted_keys: Vec<&str> = evicted.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(
        evicted_keys,
        ["file-4997", "file-4998", "file-4999", "file-10"]
    );
}

#[test]
fn a_key_removed_before_slots_are_given_back_is_found_when_inserted_again() {
    let mut lru = fill_numbered_lru(5000);
    remove_numbered(&mut lru, 0..4990);

    lru.insert("file-7".to_string(), 70, 10);

    assert_eq!(lru.get("file-7"), Some(&70));
    assert_eq!(lru.get("file-4995"), Some(&4995));
}

#[test]
fn slots_given_back_are_taken_again_as_the_map_refills() {
    let mut lru = fill_numbered_lru(5000);
    remove_numbered(&mut lru, 0..5000);

    for number in 0..5000 {
        lru.insert(format!("refilled-{number}"), number, 10);
    }

    assert!(lru.count_allocated_slots() >= 5000);
    assert_eq!(lru.peek("refilled-0"), Some(&0));
    assert_eq!(lru.peek("refilled-4999"), Some(&4999));
}
