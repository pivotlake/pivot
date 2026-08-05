//! The validity bits of one accumulated column.

use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

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

    /// Record validity for indexed rows of one batch.
    pub(super) fn append_indices(
        &mut self,
        nulls: Option<&NullBuffer>,
        indices: &[u32],
        at: usize,
    ) {
        let Some(nulls) = nulls else {
            return;
        };
        if nulls.null_count() == 0 {
            return;
        }
        for (offset, &row) in indices.iter().enumerate() {
            let row = row as usize;
            if !nulls.is_valid(row) {
                let position = at + offset;
                self.words[position / 64] &= !(1 << (position % 64));
                self.any_null = true;
            }
        }
    }

    /// Record validity for a contiguous range of one batch.
    pub(super) fn append_range(
        &mut self,
        nulls: Option<&NullBuffer>,
        start: usize,
        len: usize,
        at: usize,
    ) {
        let Some(nulls) = nulls else {
            return;
        };
        if nulls.null_count() == 0 {
            return;
        }
        for (offset, row) in (start..start + len).enumerate() {
            if !nulls.is_valid(row) {
                let position = at + offset;
                self.words[position / 64] &= !(1 << (position % 64));
                self.any_null = true;
            }
        }
    }

    /// Record the validity of rows gathered by encoded id: id
    /// `batch << shift | row` reads the validity `batch_nulls(batch)` returns,
    /// landing at accumulated position `at` onward in id order.
    pub(super) fn append_by_ids<'n>(
        &mut self,
        ids: &[u32],
        shift: u32,
        at: usize,
        batch_nulls: impl Fn(usize) -> Option<&'n NullBuffer>,
    ) {
        let mask = (1u32 << shift) - 1;
        for (offset, &id) in ids.iter().enumerate() {
            let Some(nulls) = batch_nulls((id >> shift) as usize) else {
                continue;
            };
            if !nulls.is_valid((id & mask) as usize) {
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
