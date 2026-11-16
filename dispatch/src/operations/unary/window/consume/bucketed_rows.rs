//! The rows a worker has received and not yet sorted, divided by hash bucket.
//!
//! [`BucketedRows`] does three things to each batch it is given, none of
//! which copies a row:
//!
//! 1. Hash the partition keys of every row to a bucket number. Rows with
//!    equal keys must get the same bucket, on every worker.
//! 2. Group the batch's row numbers by bucket.
//! 3. Keep the batch as it arrived, next to that grouping.
//!
//! When enough rows are pending, the worker takes them all out with
//! [`BucketedRows::take_batches_by_bucket`] and sorts each bucket's rows into
//! a run, which is the first time rows are copied.
//!
//! A window without partition keys has one bucket. Its batches are kept
//! as they are, with no hashing and no grouping.

use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::{ArrowError, DataType};

use crate::memory::{SlabAllocator, SlabBuffer};

/// Seeds of the hash applied to string keys. Every worker must send equal
/// partition keys to the same bucket, so the seeds are fixed.
const BYTES_HASH_SEEDS: [u64; 4] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
    0x2545_f491_4f6c_dd1d,
];

/// What every row's hash starts from.
const INITIAL_ROW_HASH: u64 = 0xcbf2_9ce4_8422_2325;

/// The number a NULL key contributes, whatever the column's type. NULL keys
/// form one partition in SQL, so they must all hash alike.
const NULL_KEY_NUMBER: u64 = 0x7fb5_d329_728e_a185;

/// The odd multiplier that spreads each folded-in key over the hash's bits.
/// The bucket is taken from the top bits, which a multiplication makes
/// depend on every bit below them.
const FOLD_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

/// Which rows of one batch belong to each bucket. The row numbers are held
/// on slab memory, since there is one for every pending row.
pub enum BucketGrouping {
    /// A batch of a window with one bucket: all its rows belong to it.
    SingleBucket { row_count: u32 },
    Grouped {
        /// Every row of the batch, ordered by bucket.
        rows_by_bucket: SlabBuffer<u32>,
        /// Where each bucket's rows start in `rows_by_bucket`, with the
        /// batch's row count as a final entry. Bucket `b` is rows
        /// `bucket_starts[b]..bucket_starts[b + 1]` of `rows_by_bucket`.
        bucket_starts: SlabBuffer<u32>,
    },
}

impl BucketGrouping {
    /// Appends the position of each of the batch's rows of `bucket` to
    /// `positions`, as `(batch_index, row)` pairs.
    pub fn append_bucket_positions(
        &self,
        bucket: usize,
        batch_index: u32,
        positions: &mut Vec<(u32, u32)>,
    ) {
        match self {
            BucketGrouping::SingleBucket { row_count } => {
                positions.extend((0..*row_count).map(|row| (batch_index, row)));
            }
            BucketGrouping::Grouped {
                rows_by_bucket,
                bucket_starts,
            } => {
                let start = bucket_starts[bucket] as usize;
                let end = bucket_starts[bucket + 1] as usize;
                // SAFETY: `group_rows_by_bucket` wrote every row number up to
                // the batch's row count, which is the last bucket's end.
                let bucket_rows = unsafe {
                    std::slice::from_raw_parts(rows_by_bucket.ptr_at_index(start), end - start)
                };
                positions.extend(bucket_rows.iter().map(|&row| (batch_index, row)));
            }
        }
    }
}

/// Batches of rows a worker has received and not yet sorted into runs, with
/// each batch's rows divided by hash bucket (see the module docs).
pub struct BucketedRows {
    partition_by: Vec<usize>,
    /// A power of two, and 1 for a window without partition keys.
    bucket_count: usize,
    batches: Vec<RecordBatch>,
    /// Per batch, which of its rows belong to each bucket.
    groupings: Vec<BucketGrouping>,
    row_count: usize,
    bytes_hash: RandomState,
    /// Per partition key, the row encoder for a column of a type that is
    /// hashed through its row encoding (a boolean, say), with the encoded
    /// rows of the latest batch. Created from the first batch, and `None`
    /// for the columns that are hashed directly.
    row_encoders: Vec<Option<(RowConverter, Rows)>>,
    /// Scratch space reused from batch to batch: each row's running hash, the
    /// number the column being folded in contributes for each row, each
    /// row's bucket, and where the next row of each bucket goes while a
    /// grouping is filled in.
    row_hashes: Vec<u64>,
    key_numbers: Vec<u64>,
    bucket_of_row: Vec<u16>,
    next_position_of_bucket: Vec<u32>,
    /// Ring memory that holds only the groupings. Taken on first use.
    ///
    /// The groupings are all dropped each time the pending rows are taken
    /// out. An allocator reuses a buffer once every slab cut from it is
    /// gone, so with an allocator of their own the groupings keep landing in
    /// the same few buffers. Sharing the allocator the sorted runs are
    /// written with would put groupings and runs in the same buffers, and
    /// the groupings' space would stay unusable for as long as the runs
    /// live.
    grouping_allocator: Option<SlabAllocator>,
}

impl BucketedRows {
    pub fn new(partition_by: Vec<usize>, bucket_count: usize) -> Self {
        assert!(bucket_count.is_power_of_two());
        let [seed_a, seed_b, seed_c, seed_d] = BYTES_HASH_SEEDS;
        Self {
            row_encoders: partition_by.iter().map(|_| None).collect(),
            partition_by,
            bucket_count,
            batches: Vec::new(),
            groupings: Vec::new(),
            row_count: 0,
            bytes_hash: RandomState::with_seeds(seed_a, seed_b, seed_c, seed_d),
            row_hashes: Vec::new(),
            key_numbers: Vec::new(),
            bucket_of_row: Vec::new(),
            next_position_of_bucket: Vec::new(),
            grouping_allocator: None,
        }
    }

    /// Adds a batch, working out which of its rows belong to each bucket.
    pub fn push(&mut self, batch: RecordBatch) -> Result<(), ArrowError> {
        let grouping = if self.bucket_count == 1 {
            BucketGrouping::SingleBucket {
                row_count: batch.num_rows() as u32,
            }
        } else {
            self.assign_buckets(&batch)?;
            self.group_rows_by_bucket()
        };
        self.row_count += batch.num_rows();
        self.batches.push(batch);
        self.groupings.push(grouping);
        Ok(())
    }

    /// Writes the bucket of every row of `batch` into `self.bucket_of_row`.
    ///
    /// The hash is computed a column at a time over the whole batch: each
    /// partition column is turned into one 64-bit number per row, and those
    /// numbers are folded into the row's running hash. The bucket is the
    /// hash's top bits.
    fn assign_buckets(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        let row_count = batch.num_rows();
        self.row_hashes.clear();
        self.row_hashes.resize(row_count, INITIAL_ROW_HASH);
        self.key_numbers.clear();
        self.key_numbers.resize(row_count, 0);

        for (key_index, &column) in self.partition_by.iter().enumerate() {
            let column = batch.column(column);
            Self::write_key_numbers(
                column,
                &self.bytes_hash,
                &mut self.row_encoders[key_index],
                &mut self.key_numbers,
            )?;
            if let Some(validity) = column.nulls().filter(|_| column.null_count() > 0) {
                for (key_number, is_valid) in self.key_numbers.iter_mut().zip(validity.iter()) {
                    if !is_valid {
                        *key_number = NULL_KEY_NUMBER;
                    }
                }
            }
            for (row_hash, &key_number) in self.row_hashes.iter_mut().zip(&self.key_numbers) {
                *row_hash = (row_hash.rotate_left(5) ^ key_number).wrapping_mul(FOLD_MULTIPLIER);
            }
        }

        let unused_hash_bits = u64::BITS - self.bucket_count.trailing_zeros();
        self.bucket_of_row.clear();
        self.bucket_of_row.extend(
            self.row_hashes
                .iter()
                .map(|row_hash| (row_hash >> unused_hash_bits) as u16),
        );
        Ok(())
    }

    /// Writes one number per row of `column` into `key_numbers`, equal for
    /// equal values. The number of a NULL row is unspecified; the caller
    /// overwrites it.
    ///
    /// It takes the fields it works on instead of `&mut self` because its
    /// caller is iterating over `self.partition_by`.
    fn write_key_numbers(
        column: &ArrayRef,
        bytes_hash: &RandomState,
        row_encoder: &mut Option<(RowConverter, Rows)>,
        key_numbers: &mut [u64],
    ) -> Result<(), ArrowError> {
        match column.data_type() {
            DataType::Utf8View => {
                let strings = column.as_string_view();
                for (row, key_number) in key_numbers.iter_mut().enumerate() {
                    *key_number = bytes_hash.hash_one(strings.value(row).as_bytes());
                }
            }
            DataType::BinaryView => {
                let values = column.as_binary_view();
                for (row, key_number) in key_numbers.iter_mut().enumerate() {
                    *key_number = bytes_hash.hash_one(values.value(row));
                }
            }
            data_type => match data_type.primitive_width() {
                Some(1) => Self::write_fixed_width_key_numbers::<1>(column, key_numbers),
                Some(2) => Self::write_fixed_width_key_numbers::<2>(column, key_numbers),
                Some(4) => Self::write_fixed_width_key_numbers::<4>(column, key_numbers),
                Some(8) => Self::write_fixed_width_key_numbers::<8>(column, key_numbers),
                Some(16) => Self::write_fixed_width_key_numbers::<16>(column, key_numbers),
                _ => {
                    let (converter, rows) = match row_encoder {
                        Some(row_encoder) => row_encoder,
                        None => {
                            let converter =
                                RowConverter::new(vec![SortField::new(data_type.clone())])?;
                            let rows = converter.empty_rows(0, 0);
                            row_encoder.insert((converter, rows))
                        }
                    };
                    rows.clear();
                    converter.append(rows, std::slice::from_ref(column))?;
                    for (key_number, encoded_value) in key_numbers.iter_mut().zip(rows.iter()) {
                        *key_number = bytes_hash.hash_one(encoded_value.data());
                    }
                }
            },
        }
        Ok(())
    }

    /// Writes the raw bits of each value of a column of `WIDTH`-byte values
    /// as its key number. Equal values have equal bits, whatever the column's
    /// type. A value wider than 8 bytes mixes its two halves, so that
    /// swapping them changes the number.
    fn write_fixed_width_key_numbers<const WIDTH: usize>(
        column: &ArrayRef,
        key_numbers: &mut [u64],
    ) {
        let values = column.to_data();
        // SAFETY: buffer 0 of a fixed-width array holds `offset + len` values
        // of `WIDTH` bytes each, and `key_numbers` is as long as the array.
        let first_value =
            unsafe { (values.buffers()[0].as_ptr() as *const [u8; WIDTH]).add(values.offset()) };
        let low_width = WIDTH.min(8);
        for (row, key_number) in key_numbers.iter_mut().enumerate() {
            let value: [u8; WIDTH] = unsafe { first_value.add(row).read_unaligned() };
            let mut low_half = [0u8; 8];
            let mut high_half = [0u8; 8];
            low_half[..low_width].copy_from_slice(&value[..low_width]);
            high_half[..WIDTH - low_width].copy_from_slice(&value[low_width..]);
            *key_number = u64::from_le_bytes(low_half)
                ^ u64::from_le_bytes(high_half).wrapping_mul(FOLD_MULTIPLIER);
        }
    }

    /// Groups the rows of the batch whose buckets are in `self.bucket_of_row`.
    ///
    /// The grouping is a counting sort: one pass counts the rows of each
    /// bucket, which gives every bucket's first position, and a second pass
    /// places each row at its bucket's next free position.
    fn group_rows_by_bucket(&mut self) -> BucketGrouping {
        let allocator = self
            .grouping_allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let next_position_of_bucket = &mut self.next_position_of_bucket;
        next_position_of_bucket.clear();
        next_position_of_bucket.resize(self.bucket_count + 1, 0);
        for &bucket in &self.bucket_of_row {
            next_position_of_bucket[bucket as usize + 1] += 1;
        }
        for bucket in 0..self.bucket_count {
            next_position_of_bucket[bucket + 1] += next_position_of_bucket[bucket];
        }
        // At this point entry `b` is where bucket `b` starts.
        let mut bucket_starts: SlabBuffer<u32> =
            allocator.create_slab_buffer(self.bucket_count + 1, false);
        for (bucket, &start) in next_position_of_bucket.iter().enumerate() {
            bucket_starts[bucket] = start;
        }

        let mut rows_by_bucket: SlabBuffer<u32> =
            allocator.create_slab_buffer(self.bucket_of_row.len(), false);
        for (row, &bucket) in self.bucket_of_row.iter().enumerate() {
            let position = &mut next_position_of_bucket[bucket as usize];
            rows_by_bucket[*position as usize] = row as u32;
            *position += 1;
        }
        BucketGrouping::Grouped {
            rows_by_bucket,
            bucket_starts,
        }
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Takes everything out, leaving nothing pending. Returns the batches as
    /// they arrived and, for each of them, which of its rows belong to each
    /// bucket.
    ///
    /// The rows are handed out as row numbers instead of per-bucket arrays
    /// because building an array of a bucket's rows would copy them, and the
    /// caller is about to copy them once anyway, in sorted order.
    pub fn take_batches_by_bucket(&mut self) -> (Vec<RecordBatch>, Vec<BucketGrouping>) {
        self.row_count = 0;
        (
            std::mem::take(&mut self.batches),
            std::mem::take(&mut self.groupings),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_array::{BooleanArray, Int64Array, StringViewArray};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    /// A batch of the given columns, all of which are partition keys.
    fn key_batch(columns: Vec<ArrayRef>) -> RecordBatch {
        let fields: Vec<_> = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                Field::new(format!("key_{index}"), column.data_type().clone(), true)
            })
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    /// Takes everything out of `pending` and lists the `(batch, row)`
    /// positions in each of `bucket_count` buckets.
    fn take_positions_by_bucket(
        pending: &mut BucketedRows,
        bucket_count: usize,
    ) -> Vec<Vec<(u32, u32)>> {
        let (_, groupings) = pending.take_batches_by_bucket();
        (0..bucket_count)
            .map(|bucket| {
                let mut positions = Vec::new();
                for (batch_index, grouping) in groupings.iter().enumerate() {
                    grouping.append_bucket_positions(bucket, batch_index as u32, &mut positions);
                }
                positions
            })
            .collect()
    }

    #[test]
    fn rows_equal_on_every_partition_key_share_a_bucket() {
        let names = StringViewArray::from(vec![Some("a"), None, Some("a"), None]);
        let numbers = Int64Array::from(vec![Some(7), None, Some(7), None]);
        let flags = BooleanArray::from(vec![Some(true), None, Some(true), None]);
        init_test_free_pool(4);
        let mut pending = BucketedRows::new(vec![0, 1, 2], 256);

        let batch = key_batch(vec![Arc::new(names), Arc::new(numbers), Arc::new(flags)]);
        pending.push(batch).unwrap();

        let buckets = take_positions_by_bucket(&mut pending, 256);
        let bucket_of = |row: u32| {
            buckets
                .iter()
                .position(|bucket| bucket.contains(&(0, row)))
                .unwrap()
        };
        assert_eq!(bucket_of(0), bucket_of(2));
        assert_eq!(bucket_of(1), bucket_of(3));
    }

    #[test]
    fn distinct_keys_spread_over_every_bucket() {
        init_test_free_pool(4);
        let mut pending = BucketedRows::new(vec![0], 256);

        let batch = key_batch(vec![Arc::new(Int64Array::from_iter_values(0..10_000))]);
        pending.push(batch).unwrap();

        let buckets = take_positions_by_bucket(&mut pending, 256);
        assert!(buckets.iter().all(|bucket| !bucket.is_empty()));
    }

    #[test]
    fn every_row_of_every_batch_is_pending_in_exactly_one_bucket() {
        init_test_free_pool(4);
        let mut pending = BucketedRows::new(vec![0], 16);

        for values in [0..100, 100..150] {
            let batch = key_batch(vec![Arc::new(Int64Array::from_iter_values(values))]);
            pending.push(batch).unwrap();
        }

        let buckets = take_positions_by_bucket(&mut pending, 16);
        let mut positions: Vec<_> = buckets.into_iter().flatten().collect();
        positions.sort_unstable();
        let every_row: Vec<(u32, u32)> = (0..100)
            .map(|row| (0, row))
            .chain((0..50).map(|row| (1, row)))
            .collect();
        assert_eq!(positions, every_row);
    }

    #[test]
    fn a_single_bucket_lists_every_row_in_arrival_order() {
        init_test_free_pool(4);
        let mut pending = BucketedRows::new(Vec::new(), 1);

        for values in [vec![10, 11], vec![20]] {
            let batch = key_batch(vec![Arc::new(Int64Array::from(values))]);
            pending.push(batch).unwrap();
        }

        let buckets = take_positions_by_bucket(&mut pending, 1);
        assert_eq!(buckets, vec![vec![(0, 0), (0, 1), (1, 0)]]);
    }
}
