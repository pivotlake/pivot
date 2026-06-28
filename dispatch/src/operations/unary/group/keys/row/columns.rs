//! The decode side: rebuild typed output columns from the persisted key blobs.

use super::schema::RowKeySchema;
use crate::arrays::{ArrayBuilder, PrimitiveBuilder, SlabColumn};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::keys::{ArenaKey, KeyColumns};
use arrow_array::builder::make_view;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{ArrayRef, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// Pops a little-endian integer off the front of a key blob, advancing it.
macro_rules! pop_le {
    ($blob:expr, $t:ty) => {{
        let (head, rest) = $blob.split_at(std::mem::size_of::<$t>());
        *$blob = rest;
        <$t>::from_le_bytes(head.try_into().unwrap())
    }};
}

/// Generates [`FieldBuilder`] (the decode-side column builder) and its three
/// methods from the shared `int_key_types!` list, so its integer arms can't
/// drift from the encode side. The `Str` arm is spelled out in each method
/// because it is genuinely different (length handling, zero-copy arena views).
macro_rules! define_field_builder {
    ( $( ($variant:ident, $dt:ident, $arrow:ty, $native:ty) ),+ $(,)? ) => {
        /// One output key column under construction. Integers decode into a typed
        /// primitive builder; strings into view headers pointing back into the arena.
        enum FieldBuilder {
            $( $variant(PrimitiveBuilder<$arrow>), )+
            Str(SlabColumn<u128>),
        }

        impl FieldBuilder {
            fn new(dt: &DataType, allocator: &mut SlabAllocator, rows: usize) -> Self {
                match dt {
                    $( DataType::$dt =>
                        FieldBuilder::$variant(PrimitiveBuilder::with_capacity(allocator, rows)), )+
                    DataType::Utf8View => FieldBuilder::Str(SlabColumn::with_capacity(allocator, rows)),
                    dt => unreachable!("row key column type not supported: {dt}"),
                }
            }

            /// Decode this field off the front of `blob`, advancing it, and push it into
            /// the column. `key`/`full_len` (the whole blob's length) place a non-inline
            /// string view into its arena buffer; `trailing` marks the prefix-less
            /// trailing string, whose bytes are the rest of `blob`.
            #[inline]
            fn decode(&mut self, blob: &mut &[u8], full_len: usize, key: &ArenaKey, trailing: bool) {
                match self {
                    $( FieldBuilder::$variant(b) => b.col.push(pop_le!(blob, $native)), )+
                    FieldBuilder::Str(views) => {
                        let s = if trailing {
                            std::mem::take(blob)
                        } else {
                            let len = pop_le!(blob, u32) as usize;
                            let (s, rest) = blob.split_at(len);
                            *blob = rest;
                            s
                        };
                        // Byte offset of `s` within the key's arena buffer. Strings ≤ 12
                        // bytes inline into the view header, so the buffer/offset args only
                        // matter for longer ones, and those live in a blob longer than the
                        // string, hence a non-inline blob with valid arena coordinates.
                        let pos_in_blob = (full_len - blob.len() - s.len()) as u32;
                        views.push(make_view(s, key.buffer_index(), key.offset() + pos_in_blob));
                    }
                }
            }

            /// Finish this builder into its arrow column and the matching schema field.
            /// String columns emit zero-copy views into the shared output buffers.
            #[inline]
            fn into_field(
                self,
                name: String,
                output_buffers: &Arc<[arrow_buffer::Buffer]>,
            ) -> (Field, ArrayRef) {
                let (dt, array): (DataType, ArrayRef) = match self {
                    $( FieldBuilder::$variant(b) => (DataType::$dt, b.into_array(None)), )+
                    FieldBuilder::Str(views) => {
                        let len = views.len();
                        let views = ScalarBuffer::<u128>::new(views.into_buffer(), 0, len);
                        // Safety: views built from valid blob slices; the shared buffers
                        // (Arc-holding the arena) keep the ring memory alive as long as the
                        // array exists. Every batch shares the one `Arc<[Buffer]>` built for
                        // this output phase, so the downstream concat/take reuse it instead
                        // of rebuilding the buffer list per batch.
                        let array: ArrayRef = Arc::new(unsafe {
                            StringViewArray::new_unchecked(views, output_buffers.clone(), None)
                        });
                        (DataType::Utf8View, array)
                    }
                };
                (Field::new(name, dt, false), array)
            }
        }
    };
}

int_key_types!(define_field_builder);

/// Emits the decoded key columns of a row-key GROUP BY result.
///
/// Persisted keys are buffered raw (decoding needs the arena, which only
/// [`finish`](KeyColumns::finish) receives). At `finish` each blob is walked
/// once: fixed-width fields scatter into primitive builders, string fields are
/// emitted as views into the blob's own arena bytes — zero-copy.
pub struct RowKeyColumns {
    schema: RowKeySchema,
    keys: SlabColumn<u128>,
}

impl KeyColumns for RowKeyColumns {
    type Key = ArenaKey;
    type Config = RowKeySchema;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, config: &RowKeySchema) -> Self {
        Self {
            schema: config.clone(),
            keys: SlabColumn::with_capacity(allocator, rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &ArenaKey) {
        self.keys.push(key.as_u128());
    }

    fn finish(
        self,
        arena: &Arc<SharedArena>,
        output_buffers: &Arc<[arrow_buffer::Buffer]>,
        allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let rows = self.keys.len();
        let mut builders: Vec<FieldBuilder> = self
            .schema
            .types()
            .iter()
            .map(|t| FieldBuilder::new(t, allocator, rows))
            .collect();

        let raw_keys = ScalarBuffer::<u128>::new(self.keys.into_buffer(), 0, rows);
        let last = builders.len() - 1;
        let trailing_str = matches!(self.schema.types().last(), Some(DataType::Utf8View));
        // Keys are walked here in (hash) slot order, but their arena blobs were
        // written in insertion order, so every non-inline `resolve` is a scattered
        // cache miss. Prefetch the blob a few keys ahead to hide that latency; for
        // an all-inline schema the `is_inline` guard skips it (no arena access).
        const DECODE_PREFETCH: usize = 16;
        for i in 0..rows {
            if i + DECODE_PREFETCH < rows {
                let ahead = ArenaKey::from_raw(raw_keys[i + DECODE_PREFETCH]);
                if !ahead.is_inline() {
                    arena.prefetch(ahead.buffer_index(), ahead.offset());
                }
            }
            let key = ArenaKey::from_raw(raw_keys[i]);
            let full = key.resolve(arena);
            let full_len = full.len();
            let mut blob = full;
            for (j, builder) in builders.iter_mut().enumerate() {
                builder.decode(&mut blob, full_len, &key, trailing_str && j == last);
            }
        }

        let mut fields = Vec::with_capacity(builders.len());
        let mut columns = Vec::with_capacity(builders.len());
        for (i, builder) in builders.into_iter().enumerate() {
            let (field, array) = builder.into_field(format!("k{i}"), output_buffers);
            fields.push(field);
            columns.push(array);
        }
        (fields, columns)
    }
}
