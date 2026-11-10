//! [`RowMask`]: which rows of a row range an index lookup selected.

/// One bit per row of a row range: set when the row is selected.
///
/// A lookup answers for a contiguous range of a file's rows (a row group,
/// typically), and bit `i` stands for the range's `i`-th row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMask {
    words: Vec<u64>,
    len: usize,
}

impl RowMask {
    /// A mask over `len` rows selecting none of them.
    pub fn none(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// A mask over `len` rows selecting every one.
    pub fn all(len: usize) -> Self {
        let mut mask = Self {
            words: vec![u64::MAX; len.div_ceil(64)],
            len,
        };
        mask.clear_bits_past_the_end();
        mask
    }

    /// Rows the mask covers, selected or not.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn select(&mut self, row: usize) {
        assert!(row < self.len, "row {row} outside a {} row mask", self.len);
        self.words[row / 64] |= 1 << (row % 64);
    }

    /// Select every row whose bit is set in `word`, where the word's lowest
    /// bit stands for row `64 * word_index`.
    pub(crate) fn select_word(&mut self, word_index: usize, word: u64) {
        self.words[word_index] |= word;
        if word_index + 1 == self.words.len() {
            self.clear_bits_past_the_end();
        }
    }

    pub fn is_selected(&self, row: usize) -> bool {
        row < self.len && self.words[row / 64] & (1 << (row % 64)) != 0
    }

    pub fn selects_none(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    /// Whether any row of `rows` is selected.
    pub fn selects_any_in(&self, rows: std::ops::Range<usize>) -> bool {
        assert!(rows.end <= self.len, "rows outside a {} row mask", self.len);
        if rows.is_empty() {
            return false;
        }
        let (first_word, last_word) = (rows.start / 64, (rows.end - 1) / 64);
        (first_word..=last_word).any(|word_index| {
            let mut word = self.words[word_index];
            if word_index == first_word {
                word &= u64::MAX << (rows.start % 64);
            }
            if word_index == last_word {
                word &= u64::MAX >> (63 - (rows.end - 1) % 64);
            }
            word != 0
        })
    }

    pub fn selects_all(&self) -> bool {
        self.count_selected() == self.len
    }

    pub fn count_selected(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Keep only the rows `other` also selects.
    pub fn intersect_with(&mut self, other: &RowMask) {
        assert_eq!(self.len, other.len, "masks cover different row ranges");
        for (word, other_word) in self.words.iter_mut().zip(&other.words) {
            *word &= other_word;
        }
    }

    /// Select the rows `other` selects as well.
    pub fn union_with(&mut self, other: &RowMask) {
        assert_eq!(self.len, other.len, "masks cover different row ranges");
        for (word, other_word) in self.words.iter_mut().zip(&other.words) {
            *word |= other_word;
        }
    }

    /// The selected rows, ascending.
    pub fn selected_rows(&self) -> impl Iterator<Item = u32> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let mut remaining = word;
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        return None;
                    }
                    let bit = remaining.trailing_zeros();
                    remaining &= remaining - 1;
                    Some(word_index as u32 * 64 + bit)
                })
            })
    }

    fn clear_bits_past_the_end(&mut self) {
        let used_in_last = self.len % 64;
        if used_in_last != 0 {
            *self.words.last_mut().unwrap() &= (1u64 << used_in_last) - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_mask_selects_exactly_its_rows() {
        let mask = RowMask::all(70);

        assert_eq!(mask.count_selected(), 70);
        assert!(mask.selects_all());
        assert!(!mask.is_selected(70));
    }

    #[test]
    fn selected_rows_come_back_ascending() {
        let mut mask = RowMask::none(200);
        mask.select(130);
        mask.select(3);
        mask.select(64);

        let rows: Vec<u32> = mask.selected_rows().collect();

        assert_eq!(rows, vec![3, 64, 130]);
    }

    #[test]
    fn intersecting_keeps_the_rows_both_select() {
        let mut mask = RowMask::none(100);
        mask.select(1);
        mask.select(80);
        let mut other = RowMask::none(100);
        other.select(80);
        other.select(99);

        mask.intersect_with(&other);

        assert_eq!(mask.selected_rows().collect::<Vec<_>>(), vec![80]);
    }

    #[test]
    fn uniting_keeps_the_rows_either_selects() {
        let mut mask = RowMask::none(100);
        mask.select(1);
        let mut other = RowMask::none(100);
        other.select(99);

        mask.union_with(&other);

        assert_eq!(mask.selected_rows().collect::<Vec<_>>(), vec![1, 99]);
    }

    #[test]
    fn a_range_holds_a_selected_row_only_if_one_lies_inside_it() {
        let mut mask = RowMask::none(300);
        mask.select(70);
        mask.select(200);

        assert!(mask.selects_any_in(70..71));
        assert!(mask.selects_any_in(0..300));
        assert!(mask.selects_any_in(129..201));
        assert!(!mask.selects_any_in(71..200));
        assert!(!mask.selects_any_in(0..70));
        assert!(!mask.selects_any_in(201..300));
    }

    #[test]
    fn a_whole_word_never_selects_rows_past_the_end() {
        let mut mask = RowMask::none(70);

        mask.select_word(1, u64::MAX);

        assert_eq!(mask.count_selected(), 6);
    }
}
