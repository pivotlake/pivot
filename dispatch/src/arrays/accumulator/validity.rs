//! The validity bits of one accumulated column.

use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::column::SourceSelection;

/// Validity bits for the accumulated rows of one column, one bit per row.
/// All-ones until a null actually arrives, so null-free streams never touch it
/// beyond one flag check per append.
pub(super) struct ValidityMask {
    words: Vec<u64>,
    any_null: bool,
}

impl ValidityMask {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            words: vec![u64::MAX; capacity.div_ceil(64)],
            any_null: false,
        }
    }

    /// Record the validity of the appended rows: the `rows` of a column whose
    /// null buffer is `nulls`, landing at accumulated position `at`.
    pub(super) fn append(
        &mut self,
        nulls: Option<&NullBuffer>,
        selection: SourceSelection<'_>,
        at: usize,
    ) {
        let Some(nulls) = nulls else {
            return;
        };
        if nulls.null_count() == 0 {
            return;
        }
        for (offset, row) in selection.iter_positions().enumerate() {
            if !nulls.is_valid(row) {
                let position = at + offset;
                self.words[position / 64] &= !(1 << (position % 64));
                self.any_null = true;
            }
        }
    }

    /// The accumulated rows' null buffer (`None` when every row is valid),
    /// resetting for the next batch.
    pub(super) fn take(&mut self, len: usize) -> Option<NullBuffer> {
        if !self.any_null {
            return None;
        }
        let bits = BooleanBuffer::new(Buffer::from_slice_ref(&self.words), 0, len);
        self.words.fill(u64::MAX);
        self.any_null = false;
        Some(NullBuffer::new(bits))
    }
}
