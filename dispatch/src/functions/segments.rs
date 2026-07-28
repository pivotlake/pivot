//! [`SegmentMatcher`] — match strings against an ordered list of literals, the
//! shape a SQL `LIKE` pattern takes once it is split on its wildcards.

use crate::env::MAX_INLINE_STRING_VIEW;
use crate::functions::needle::NeedleSearcher;
use arrow::array::ByteView;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::BooleanBufferBuilder;

/// Matches every string that starts with `prefix`, ends with `suffix`, and
/// contains each entry of `segments` in order and without overlap in between.
///
/// An empty `prefix` or `suffix` leaves that end unanchored, since every string
/// starts and ends with the empty string. The parts may not overlap each other,
/// so a string matches only if it is at least as long as all of them
/// concatenated. That is exactly what a `LIKE` pattern means once split on `%`,
/// where the parts are separated by at least one character position.
///
/// Every segment is searched once per underlying data buffer (see
/// [`NeedleSearcher`]), so a batch costs one vectorized pass per segment per
/// buffer, and then a binary search per segment per row.
pub struct SegmentMatcher {
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    segments: Vec<NeedleSearcher>,
    /// The shortest string that can match: every part laid end to end.
    min_len: usize,
}

impl SegmentMatcher {
    pub fn new(prefix: &[u8], segments: &[&[u8]], suffix: &[u8]) -> Self {
        let min_len = prefix.len()
            + suffix.len()
            + segments.iter().map(|segment| segment.len()).sum::<usize>();
        Self {
            prefix: prefix.to_vec(),
            suffix: suffix.to_vec(),
            segments: segments.iter().map(NeedleSearcher::new).collect(),
            min_len,
        }
    }

    /// Test every string in `col`, returning a [`BooleanArray`] mask. `true` at
    /// position *i* means the string at *i* matches.
    ///
    /// The matcher is stateful: buffer scan results are cached across calls, so
    /// repeated invocations on arrays sharing backing buffers (common with
    /// dictionary-encoded or sliced data) avoid redundant work.
    pub fn run(&mut self, col: &StringViewArray) -> BooleanArray {
        for segment in &mut self.segments {
            segment.scan_buffers(col);
        }
        // The allocation behind each data buffer (with the offset of the array's
        // own window into it) for the anchor comparisons, and every segment's
        // offsets within it for the ordered search.
        let allocations: Vec<(&[u8], usize)> = col
            .data_buffers()
            .iter()
            .map(|buffer| (&buffer.bytes()[..], buffer.ptr_offset()))
            .collect();
        let scanned: Vec<_> = self
            .segments
            .iter()
            .map(|segment| segment.scanned_buffers(col))
            .collect();

        let mut bitmap = BooleanBufferBuilder::new(col.len());
        for &view in col.views().iter() {
            let len = view as u32 as usize;

            let matched = if len < self.min_len {
                false
            } else if len > MAX_INLINE_STRING_VIEW {
                let bv = ByteView::from(view);
                let buffer_index = bv.buffer_index as usize;
                let (allocation, base_offset) = allocations[buffer_index];
                let start = base_offset + bv.offset as usize;
                let value = &allocation[start..start + len];
                self.matches_anchors(value)
                    && self.matches_segments(|index, from| {
                        scanned[index][buffer_index]
                            .find_at_or_after(
                                start + from,
                                start + len - self.suffix.len(),
                                self.segments[index].needle_len(),
                            )
                            .map(|at| at - start)
                    })
            } else {
                let inline = view.to_le_bytes();
                let value = &inline[4..4 + len];
                self.matches_anchors(value)
                    && self.matches_segments(|index, from| {
                        let limit = len - self.suffix.len();
                        self.segments[index]
                            .finder()
                            .find(&value[from..limit])
                            .map(|at| at + from)
                    })
            };

            bitmap.append(matched);
        }

        bitmap.finish().into()
    }

    /// Whether one row's bytes satisfy the two anchors. Length is checked by the
    /// caller, which knows it before it has the bytes.
    #[inline(always)]
    fn matches_anchors(&self, value: &[u8]) -> bool {
        value.starts_with(&self.prefix) && value.ends_with(&self.suffix)
    }

    /// Walk the segments through one row, each starting where the previous one
    /// ended. `find(index, from)` returns the start of segment `index`'s first
    /// occurrence at or after `from`, both relative to the row, and is
    /// responsible for keeping the match clear of the suffix.
    #[inline(always)]
    fn matches_segments(&self, mut find: impl FnMut(usize, usize) -> Option<usize>) -> bool {
        // The anchors have already matched, so the segments only have to fit in
        // the region they leave behind.
        let mut cursor = self.prefix.len();

        for (index, segment) in self.segments.iter().enumerate() {
            let Some(start) = find(index, cursor) else {
                return false;
            };
            cursor = start + segment.needle_len();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::SegmentMatcher;
    use arrow_array::StringViewArray;

    fn run(prefix: &str, segments: &[&str], suffix: &str, values: &[&str]) -> Vec<bool> {
        let array = StringViewArray::from_iter_values(values.iter().copied());
        let segments: Vec<&[u8]> = segments.iter().map(|s| s.as_bytes()).collect();

        let result =
            SegmentMatcher::new(prefix.as_bytes(), &segments, suffix.as_bytes()).run(&array);

        (0..result.len()).map(|i| result.value(i)).collect()
    }

    #[test]
    fn matches_two_segments_in_order() {
        let values = [
            "a special package of requests",
            "requests before special ones",
            "special requests",
            "special",
        ];

        let result = run("", &["special", "requests"], "", &values);

        assert_eq!(result, vec![true, false, true, false]);
    }

    #[test]
    fn segments_may_not_overlap() {
        let values = ["abcd", "abcbcd"];

        let result = run("", &["abc", "bcd"], "", &values);

        assert_eq!(result, vec![false, true]);
    }

    #[test]
    fn anchors_bound_the_segments() {
        let values = [
            "start middle end",
            "no middle end",
            "start middle no",
            "start end middle",
        ];

        let result = run("start", &["middle"], "end", &values);

        assert_eq!(result, vec![true, false, false, false]);
    }

    #[test]
    fn matches_several_segments_between_anchors() {
        let values = [
            "abcde",
            "axxbyyczzdwwe",
            // Out of order: the c comes before the b.
            "acbde",
            "abcd",
            "bcde",
        ];

        let result = run("a", &["b", "c", "d"], "e", &values);

        assert_eq!(result, vec![true, true, false, false, false]);
    }

    #[test]
    fn anchors_may_not_overlap_each_other() {
        let values = ["ab", "aXb", "a"];

        let result = run("a", &[], "b", &values);

        assert_eq!(result, vec![true, true, false]);
    }

    #[test]
    fn matches_inline_and_buffered_values_alike() {
        let values = [
            "axb",
            "a string long enough to live in a data buffer, with x in it, and b",
        ];

        let result = run("a", &["x"], "b", &values);

        assert_eq!(result, vec![true, true]);
    }

    #[test]
    fn empty_pattern_matches_everything() {
        let values = ["", "anything at all", "x"];

        let result = run("", &[], "", &values);

        assert_eq!(result, vec![true, true, true]);
    }

    #[test]
    fn reuses_buffer_offsets_across_calls() {
        let array = StringViewArray::from_iter_values([
            "a long string with special and then requests in it",
            "a long string with requests and then special in it",
        ]);
        let mut matcher = SegmentMatcher::new(b"", &[b"special".as_slice(), b"requests"], b"");

        let first = matcher.run(&array);
        let second = matcher.run(&array);

        assert!(first.value(0));
        assert!(!first.value(1));
        assert!(second.value(0));
        assert!(!second.value(1));
    }
}
