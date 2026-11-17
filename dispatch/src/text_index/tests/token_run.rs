use super::{bytes_of, find_rows, read_sparse_index};
use crate::memory::{SlabAllocator, init_test_free_pool};
use crate::text_index::{LetterCase, PlacedTokenRun, TokenRun, TokenRunBuilder};

/// The token run of `texts`, one row each.
fn build_run(letter_case: LetterCase, texts: &[&str]) -> TokenRun {
    let mut allocator = SlabAllocator::new(false);
    let mut builder = TokenRunBuilder::new(letter_case);
    for (row, text) in texts.iter().enumerate() {
        builder.push_text(row as u32, text.as_bytes());
    }
    builder.finish(&mut allocator)
}

/// The bytes of the index file of `runs`, each placed at its first row.
fn write_index_file(runs: Vec<(u32, TokenRun)>) -> Vec<u8> {
    let mut allocator = SlabAllocator::new(false);
    let placed_runs: Vec<PlacedTokenRun> = runs
        .into_iter()
        .map(|(first_row, run)| PlacedTokenRun { first_row, run })
        .collect();
    bytes_of(&TokenRun::write_index_file(&placed_runs, &mut allocator))
}

#[test]
fn the_tokens_of_a_run_are_indexed_with_the_rows_they_appear_in() {
    init_test_free_pool(8);
    let run = build_run(
        LetterCase::Kept,
        &[
            "GET /users timeout",
            "POST /users ok",
            "GET /orders timeout",
        ],
    );

    let file = write_index_file(vec![(0, run)]);

    let found = ["GET", "POST", "users", "orders", "timeout", "ok"]
        .map(|token| find_rows(&file, token.as_bytes()).unwrap());
    assert_eq!(
        found,
        [
            vec![0, 2],
            vec![1],
            vec![0, 1],
            vec![2],
            vec![0, 2],
            vec![1]
        ]
    );
}

#[test]
fn a_token_repeated_in_a_row_is_indexed_once_for_that_row() {
    init_test_free_pool(8);
    let run = build_run(LetterCase::Kept, &["retry retry retry", "retry"]);

    let file = write_index_file(vec![(0, run)]);

    assert_eq!(find_rows(&file, b"retry"), Some(vec![0, 1]));
}

#[test]
fn a_run_is_indexed_at_the_rows_of_the_file_it_is_placed_at() {
    init_test_free_pool(8);
    let first_run = build_run(LetterCase::Kept, &["disk full", "disk ok"]);
    let second_run = build_run(LetterCase::Kept, &["net ok", "disk full"]);

    let file = write_index_file(vec![(0, first_run), (1_000, second_run)]);

    let found = ["disk", "full", "net", "ok"].map(|token| find_rows(&file, token.as_bytes()));
    assert_eq!(
        found,
        [
            Some(vec![0, 1, 1_001]),
            Some(vec![0, 1_001]),
            Some(vec![1_000]),
            Some(vec![1, 1_000])
        ]
    );
}

#[test]
fn kept_letter_case_indexes_a_word_under_each_spelling() {
    init_test_free_pool(8);
    let run = build_run(LetterCase::Kept, &["Error", "error"]);

    let file = write_index_file(vec![(0, run)]);

    assert_eq!(find_rows(&file, b"Error"), Some(vec![0]));
    assert_eq!(find_rows(&file, b"error"), Some(vec![1]));
}

#[test]
fn lowered_letter_case_indexes_a_word_once_whatever_its_spelling() {
    init_test_free_pool(8);
    let run = build_run(LetterCase::Lowered, &["Error", "error", "ERROR"]);

    let file = write_index_file(vec![(0, run)]);

    assert_eq!(find_rows(&file, b"error"), Some(vec![0, 1, 2]));
    assert_eq!(find_rows(&file, b"Error"), None);
}

#[test]
fn a_builder_makes_one_run_after_another() {
    init_test_free_pool(8);
    let mut allocator = SlabAllocator::new(false);
    let mut builder = TokenRunBuilder::new(LetterCase::Kept);
    builder.push_text(0, b"apple melon");
    let first_run = builder.finish(&mut allocator);

    builder.push_text(0, b"melon");
    let second_run = builder.finish(&mut allocator);

    assert_eq!(first_run.count_tokens(), 2);
    assert_eq!(second_run.count_tokens(), 1);
}

#[test]
fn many_rows_in_several_runs_are_all_found() {
    init_test_free_pool(32);
    let texts: Vec<String> = (0..20_000)
        .map(|row| format!("common word{} unique{row}", row % 100))
        .collect();
    let runs: Vec<(u32, TokenRun)> = texts
        .chunks(5_000)
        .enumerate()
        .map(|(run_index, texts)| {
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            (
                run_index as u32 * 5_000,
                build_run(LetterCase::Kept, &texts),
            )
        })
        .collect();

    let file = write_index_file(runs);

    assert_eq!(
        find_rows(&file, b"common"),
        Some((0..20_000).collect::<Vec<u32>>())
    );
    assert_eq!(
        find_rows(&file, b"word7"),
        Some((7..20_000).step_by(100).collect::<Vec<u32>>())
    );
    assert_eq!(find_rows(&file, b"unique12345"), Some(vec![12_345]));
}

#[test]
fn rows_without_text_leave_an_index_file_of_no_tokens() {
    init_test_free_pool(8);
    let run = build_run(LetterCase::Kept, &["", " -- "]);

    let file = write_index_file(vec![(0, run)]);

    assert_eq!(read_sparse_index(&file).count_dict_indexes(), 0);
}
