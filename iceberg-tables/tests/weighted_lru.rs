//! What a map of groups within a budget of bytes holds, and what it lets go
//! of past the budget.

use iceberg_tables::weighted_lru::WeightedLru;

/// The group the tests of one group hold their entries in.
const GROUP: &str = "group";

/// A map of `capacity_bytes` holding each of `keys` in [`GROUP`], at ten
/// bytes each, inserted in the order given.
fn fill_lru(capacity_bytes: u64, keys: &[&str]) -> WeightedLru<&'static str, String, u32> {
    let mut lru = WeightedLru::new(capacity_bytes);
    for (value, key) in keys.iter().enumerate() {
        lru.insert(GROUP, key.to_string(), value as u32, 10);
    }
    lru
}

#[test]
fn an_inserted_value_is_found_by_a_borrowed_key() {
    let mut lru = fill_lru(u64::MAX, &["a"]);

    let found = lru.get(&GROUP, "a").copied();

    assert_eq!(found, Some(0));
}

#[test]
fn past_the_budget_the_entry_used_longest_ago_is_handed_back() {
    let mut lru = fill_lru(25, &["old", "kept"]);
    lru.get(&GROUP, "old");

    let evicted = lru.insert(GROUP, "new".to_string(), 2, 10);

    assert_eq!(evicted, [("kept".to_string(), 1)]);
    assert!(lru.peek(&GROUP, "old").is_some() && lru.peek(&GROUP, "new").is_some());
}

#[test]
fn a_peeked_entry_is_not_counted_as_used() {
    let mut lru = fill_lru(25, &["peeked", "kept"]);
    lru.peek(&GROUP, "peeked");

    let evicted = lru.insert(GROUP, "new".to_string(), 2, 10);

    assert_eq!(evicted, [("peeked".to_string(), 0)]);
}

#[test]
fn an_entry_marked_used_outlives_the_ones_used_before_it() {
    let mut lru = fill_lru(25, &["marked", "left"]);
    lru.mark_used(&GROUP, "marked");

    let evicted = lru.insert(GROUP, "new".to_string(), 2, 10);

    assert_eq!(evicted, [("left".to_string(), 1)]);
}

#[test]
fn the_entry_just_inserted_stays_whatever_it_weighs() {
    let mut lru = fill_lru(25, &["a", "b"]);

    let evicted = lru.insert(GROUP, "heavy".to_string(), 2, 100);

    assert_eq!(evicted, [("a".to_string(), 0), ("b".to_string(), 1)]);
    assert_eq!(lru.peek(&GROUP, "heavy"), Some(&2));
}

#[test]
fn inserting_under_a_held_key_frees_the_weight_of_the_value_it_drops() {
    let mut lru = fill_lru(25, &["a", "b"]);

    let evicted = lru.insert(GROUP, "a".to_string(), 7, 15);

    assert!(evicted.is_empty());
    assert_eq!(
        (lru.peek(&GROUP, "a"), lru.peek(&GROUP, "b")),
        (Some(&7), Some(&1))
    );
}

#[test]
fn a_removed_entry_is_handed_back_and_leaves_its_room() {
    let mut lru = fill_lru(25, &["removed", "kept"]);

    let removed = lru.remove(&GROUP, "removed");
    let evicted = lru.insert(GROUP, "new".to_string(), 2, 10);

    assert_eq!(removed, Some(("removed".to_string(), 0)));
    assert!(evicted.is_empty());
    assert!(lru.peek(&GROUP, "removed").is_none() && lru.peek(&GROUP, "kept").is_some());
}

#[test]
fn groups_hold_entries_under_one_key_apart() {
    let mut lru: WeightedLru<&str, String, u32> = WeightedLru::new(u64::MAX);
    lru.insert("first", "a".to_string(), 1, 10);

    lru.insert("second", "a".to_string(), 2, 10);

    assert_eq!(
        (lru.peek(&"first", "a"), lru.peek(&"second", "a")),
        (Some(&1), Some(&2))
    );
}

#[test]
fn the_entry_used_longest_ago_goes_whichever_group_holds_it() {
    let mut lru: WeightedLru<&str, String, u32> = WeightedLru::new(25);
    lru.insert("first", "old".to_string(), 1, 10);
    lru.insert("second", "kept".to_string(), 2, 10);

    let evicted = lru.insert("second", "new".to_string(), 3, 10);

    assert_eq!(evicted, [("old".to_string(), 1)]);
}

#[test]
fn a_group_whose_last_entry_is_evicted_is_no_longer_listed() {
    let mut lru: WeightedLru<&str, String, u32> = WeightedLru::new(15);
    lru.insert("emptied", "a".to_string(), 1, 10);

    lru.insert("kept", "b".to_string(), 2, 10);

    assert_eq!(lru.list_groups().collect::<Vec<_>>(), [&"kept"]);
}

#[test]
fn a_removed_group_leaves_the_entries_of_the_others() {
    let mut lru: WeightedLru<&str, String, u32> = WeightedLru::new(35);
    lru.insert("removed", "a".to_string(), 1, 10);
    lru.insert("removed", "b".to_string(), 2, 10);
    lru.insert("kept", "a".to_string(), 3, 10);

    lru.remove_group(&"removed");
    let evicted = lru.insert("kept", "b".to_string(), 4, 10);

    assert!(evicted.is_empty());
    assert_eq!(lru.list_groups().collect::<Vec<_>>(), [&"kept"]);
    assert_eq!(lru.peek(&"kept", "a"), Some(&3));
}

/// A map holding `file-0` up to `file-{entry_count - 1}` in [`GROUP`], each
/// under its number at ten bytes, inserted in that order, which its budget
/// just fits.
fn fill_numbered_lru(entry_count: u32) -> WeightedLru<&'static str, String, u32> {
    let mut lru = WeightedLru::new(u64::from(entry_count) * 10);
    for number in 0..entry_count {
        lru.insert(GROUP, format!("file-{number}"), number, 10);
    }
    lru
}

/// Remove `file-{number}` from `lru` for each of `numbers`.
fn remove_numbered(
    lru: &mut WeightedLru<&'static str, String, u32>,
    numbers: std::ops::Range<u32>,
) {
    for number in numbers {
        lru.remove(&GROUP, format!("file-{number}").as_str());
    }
}

#[test]
fn entries_keep_their_order_of_last_use_when_slots_are_given_back() {
    let mut lru = fill_numbered_lru(5000);
    lru.get(&GROUP, "file-10");
    remove_numbered(&mut lru, 0..10);
    remove_numbered(&mut lru, 11..4997);

    let evicted = lru.insert(GROUP, "heavy".to_string(), 0, 50_000);

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

    lru.insert(GROUP, "file-7".to_string(), 70, 10);

    assert_eq!(lru.get(&GROUP, "file-7"), Some(&70));
    assert_eq!(lru.get(&GROUP, "file-4995"), Some(&4995));
}

#[test]
fn a_map_emptied_and_refilled_finds_what_it_holds() {
    let mut lru = fill_numbered_lru(5000);
    remove_numbered(&mut lru, 0..5000);

    for number in 0..5000 {
        lru.insert(GROUP, format!("refilled-{number}"), number, 10);
    }

    assert_eq!(lru.peek(&GROUP, "refilled-0"), Some(&0));
    assert_eq!(lru.peek(&GROUP, "refilled-4999"), Some(&4999));
}

#[test]
fn a_value_inserted_as_the_oldest_is_the_first_to_go() {
    let mut lru = fill_lru(35, &["used"]);
    lru.insert_as_oldest(GROUP, "unasked".to_string(), 7, 10);

    let evicted = lru.insert(GROUP, "new".to_string(), 2, 20);

    assert_eq!(evicted, [("unasked".to_string(), 7)]);
    assert!(lru.peek(&GROUP, "used").is_some());
}

#[test]
fn a_value_inserted_as_the_oldest_is_held_only_while_the_budget_has_room() {
    let mut lru = fill_lru(25, &["a", "b"]);

    let is_held = lru.insert_as_oldest(GROUP, "unasked".to_string(), 7, 10);

    assert!(!is_held);
    assert!(lru.peek(&GROUP, "unasked").is_none());
    assert!(lru.peek(&GROUP, "a").is_some() && lru.peek(&GROUP, "b").is_some());
}

#[test]
fn a_value_inserted_as_the_oldest_leaves_the_one_held_under_its_key_as_it_was() {
    let mut lru = fill_lru(25, &["old", "kept"]);

    let is_held = lru.insert_as_oldest(GROUP, "kept".to_string(), 7, 10);

    let evicted = lru.insert(GROUP, "new".to_string(), 2, 10);
    assert!(is_held);
    assert_eq!(evicted, [("old".to_string(), 0)]);
    assert_eq!(lru.peek(&GROUP, "kept"), Some(&1));
}

#[test]
fn a_value_inserted_as_the_oldest_and_then_used_outlives_the_ones_used_before_it() {
    let mut lru = fill_lru(35, &["left"]);
    lru.insert_as_oldest(GROUP, "unasked".to_string(), 7, 10);
    lru.get(&GROUP, "unasked");

    let evicted = lru.insert(GROUP, "new".to_string(), 2, 20);

    assert_eq!(evicted, [("left".to_string(), 0)]);
}

#[test]
fn the_budget_left_free_is_what_the_values_held_do_not_weigh() {
    let lru = fill_lru(25, &["a", "b"]);

    let free_bytes = lru.free_bytes();

    assert_eq!(free_bytes, 5);
}
