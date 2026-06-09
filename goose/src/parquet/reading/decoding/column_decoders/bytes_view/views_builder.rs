// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ViewsBuilder`] — an [`ArrayBuilder`] that produces Arrow
//! `StringViewArray`s.
//!
//! Each string value is stored as a 128-bit *view*. Short strings (≤ 12 bytes)
//! are inlined; longer strings reference a `(block_id, offset)` into the
//! `buffers` list. The builder accumulates views in a slab-allocated buffer
//! and converts to a `StringViewArray` with zero copies on
//! [`into_array`](ArrayBuilder::into_array).

use crate::parquet::reading::decoding::column_decoders::ArrayBuilder;
use arrow_array::{ArrayRef, StringViewArray, builder::make_view};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use dispatch::memory::SlabAllocator;
use dispatch::memory::SlabBuffer;
use std::ptr::NonNull;
use std::sync::Arc;

/// Accumulates 128-bit string views and their backing data blocks, then
/// finalises into a `StringViewArray`.
///
/// Does not reuse `GenericByteViewBuilder` because it needs slab-allocated
/// storage and direct control over the view buffer.
pub struct ViewsBuilder {
    /// Slab-allocated buffer of 128-bit views (one per value).
    pub views: SlabBuffer<u128>,
    /// Number of views pushed so far.
    len: usize,
    /// Data blocks referenced by non-inline views.
    pub buffers: Vec<Buffer>,
}

impl ViewsBuilder {
    /// Registers a data block and returns its block ID (used in non-inline
    /// views to reference string data).
    pub fn append_block(&mut self, block: Buffer) -> u32 {
        let block_id = self.buffers.len() as u32;
        self.buffers.push(block);
        block_id
    }

    /// # Safety
    /// This method is only safe when:
    /// - `block` is a valid index, i.e., the return value of `append_block`
    /// - `offset` and `offset + len` are valid indices into the buffer
    /// - The `(offset, offset + len)` is valid value for the native type.
    pub unsafe fn append_view_unchecked(&mut self, block: u32, offset: u32, len: u32) {
        let b = unsafe { self.buffers.get_unchecked(block as usize) };
        let end = offset.saturating_add(len);
        let b = unsafe { b.get_unchecked(offset as usize..end as usize) };

        let view = make_view(b, block, offset);

        self.views[self.len] = view;
        self.len += 1;
    }

    /// Directly append a view to the view array.
    /// This is used when we create a StringViewArray from a dictionary whose values are StringViewArray.
    ///
    /// # Safety
    /// The `view` must be a valid view as per the ByteView spec.
    pub unsafe fn append_raw_view_unchecked(&mut self, view: &u128) {
        self.views[self.len] = *view;
        self.len += 1;
    }
}

impl ArrayBuilder for ViewsBuilder {
    type Element = u128;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            views: allocator.create_slab_buffer(capacity, false),
            len: 0,
            buffers: vec![],
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, element: &Self::Element, amount: usize) {
        for _ in 0..amount {
            self.views[self.len] = *element;
            self.len += 1;
        }
    }

    #[inline]
    fn spare_mut(&mut self, count: usize) -> &mut [u128] {
        let start = self.len;
        self.len += count;
        unsafe { std::slice::from_raw_parts_mut(self.views.ptr_at_index(start), count) }
    }

    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef {
        let len = self.len;
        let ptr = NonNull::new(self.views.ptr_at_index(0) as *mut u8).unwrap();
        let buffer = unsafe {
            Buffer::from_custom_allocation(ptr, len * size_of::<u128>(), Arc::new(self.views))
        };
        let scalar_buffer = ScalarBuffer::new(buffer, 0, len);
        let nulls = null_buffer
            .map(|b| NullBuffer::new(BooleanBuffer::new(b, 0, len)))
            .filter(|n| n.null_count() != 0);
        // Safety: views were created correctly, and checked that the data is utf8 when building the buffer
        unsafe {
            Arc::new(StringViewArray::new_unchecked(
                scalar_buffer,
                self.buffers,
                nulls,
            ))
        }
    }
}
