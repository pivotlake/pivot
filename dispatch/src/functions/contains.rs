use arrow::array::ByteView;
use arrow_array::{Array, BooleanArray, StringViewArray};
use arrow_buffer::{BooleanBufferBuilder, Bytes};
use memchr::memmem::Finder;
use parquet::data_type::AsBytes;
use std::sync::{Arc, Weak};

/// The maximum number of bytes that can be stored inline in a byte view.
pub const MAX_INLINE_VIEW_LEN: u32 = 12;

#[derive(Debug, Default)]
struct BufferFindOffsets {
    base_ptr: usize,
    buffer: Weak<Bytes>,
    offsets: Vec<usize>,
}

impl BufferFindOffsets {
    fn is_valid(&self) -> bool {
        self.buffer.strong_count() > 0
    }
}

/// The `Contains` struct contains a specialized implementation for running contains with a needle
/// (e.g., WHERE LIKE '%google%') on `StringViewArray`s. The idea in general is to only do one pass
/// finding occurrences of the needle per underlying buffer.
///
/// In contrast to StringArray, StringViewArray has a vector of views (u128) which are either
/// pointers to underlying buffers or inlined strings. These underlying buffers can be re-used
/// across RecordBatches, and the same string may be pointed to many types from different places
/// (for example if it's in a Dict). They are also usually very large - they may be the original
/// allocations of the pages decompressed from the RowGroup.
/// The implementation here aims to only run the  memchr::memchr::Finder *once* per entire
/// underlying physical buffer.
///
/// This is advantageous for two reasons:
///     1. We're never running twice on one buffer
///     2. We can take advantage of vectorized capabilities by running on longer buffers
pub struct Contains {
    buffers_with_offsets: Vec<BufferFindOffsets>,
    finder: Finder<'static>,
}

impl Contains {
    pub fn new<B: ?Sized + AsRef<[u8]>>(needle: &B) -> Self {
        Self {
            buffers_with_offsets: vec![],
            finder: Finder::new(needle).into_owned(),
        }
    }

    pub fn run(&mut self, col: &StringViewArray) -> BooleanArray {
        self.register_underlying_buffers(col);
        self.create_bitmask(col)
    }

    /// Run find on underlying buffers *if this is the first time we've seen them*.
    fn register_underlying_buffers(&mut self, col: &StringViewArray) {
        self.buffers_with_offsets.retain_mut(|f| f.is_valid());

        for buffer in col.data_buffers() {
            // Get the underlying allocation (not the slice view)
            let base_ptr = buffer.bytes().as_ptr() as usize;

            if self
                .buffers_with_offsets
                .iter_mut()
                .any(|b| b.base_ptr == base_ptr)
            {
                continue;
            }

            self.buffers_with_offsets.push(BufferFindOffsets {
                base_ptr,
                buffer: Arc::downgrade(buffer.bytes()),
                offsets: self.finder.find_iter(buffer.bytes().as_bytes()).collect(),
            })
        }
    }

    /// Create the bitmask for a given array. This function takes it as given that all buffers have
    /// already been registered.
    ///
    /// The essential idea is to run through the columns `views`, which are either a pointer to the
    /// or a pointer to an underlying buffer or an inlined string.
    ///
    /// If it's a pointer, we run through the offsets for its corresponding buffer and see if any
    /// of them fall between its start_offset, and start_offset + length.
    /// If it's an inlined string - we just run `find` and turn on the corresponding bit if found.
    fn create_bitmask(&self, array: &StringViewArray) -> BooleanArray {
        let default_buffer_find_offsets = BufferFindOffsets::default();

        // A vector of base offset per buffer, together with its find offsets
        // The base offset is used to compute the "actual" offset of the string within the buffer,
        // as the offset given in the view is only relative to the ptr_offset within the buffer
        let offsets_with_buffer_find_offsets: Vec<_> = array
            .data_buffers()
            .iter()
            .map(|b| {
                let buffer_opt = self
                    .buffers_with_offsets
                    .iter()
                    .find(|f| b.bytes().as_ptr() as usize == f.base_ptr);
                (
                    b.ptr_offset(),
                    buffer_opt.unwrap_or(&default_buffer_find_offsets),
                )
            })
            .collect();

        let row_count = array.len();
        let mut bitmap = BooleanBufferBuilder::new(row_count);
        bitmap.append_n(row_count, false);

        for (i, &view) in array.views().iter().enumerate() {
            let len = view as u32;

            if len > MAX_INLINE_VIEW_LEN {
                let bytes = ByteView::from(view);
                let (buffer_offset, buffer) =
                    offsets_with_buffer_find_offsets[bytes.buffer_index as usize];
                let start_offset = bytes.offset as usize + buffer_offset;

                for &found_offset in &buffer.offsets {
                    if start_offset <= found_offset {
                        let within_range = start_offset + len as usize
                            >= found_offset + self.finder.needle().len();
                        bitmap.set_bit(i, within_range);
                        break;
                    }
                }
            } else {
                let view_bytes = view.to_le_bytes();
                let inline_data = &view_bytes[size_of::<u32>()..size_of::<u32>() + len as usize];

                // Early exit - no need to run find if len is smaller than needle,
                // it can't possibly exist
                if len < self.finder.needle().len() as u32 {
                    continue;
                }

                bitmap.set_bit(i, self.finder.find(inline_data).is_some());
            }
        }

        bitmap.finish().into()
    }
}
