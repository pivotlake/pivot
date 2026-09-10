//! Zero-copy keys for scattered string rows.
//!
//! A `StringViewArray` stores each value as a view: inline bytes when short,
//! otherwise a (buffer index, offset, length) triple into the array's data
//! buffers. An [`ArenaKey`] has the same layout, indexed into the arena's
//! buffer table instead. Registering the array's data buffers with the arena
//! therefore turns a view into a key by remapping its buffer index, with no
//! byte copied and the input buffer kept alive by the arena.

use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::keys::string::ArenaKey;
use arrow_array::StringViewArray;
use arrow_buffer::Buffer;

/// Per-worker cache of the arena slots registered for the buffer list a key
/// column last came with.
///
/// Lives in the extractor's scratch, so it outlives the batch: consecutive
/// batches often share one buffer list (an operator that emits views into
/// its own arena attaches the same list to every batch), and registering it
/// once instead of once per batch keeps the arena's slot table and the shared
/// registration lock off the consume path.
#[derive(Default)]
pub struct InputBufferSlots {
    /// Identity of the registered list: its address and length.
    registered: Option<(*const Buffer, usize)>,
    slots: Vec<u32>,
}

// SAFETY: the pointer is only compared for identity, never dereferenced.
unsafe impl Send for InputBufferSlots {}

impl InputBufferSlots {
    /// The key for view `idx` of `array`, pointing into the array's own
    /// buffer. Registers the array's buffers with the arena unless the
    /// previous call already registered this same list.
    #[inline(always)]
    pub(crate) fn key_at(
        &mut self,
        array: &StringViewArray,
        idx: usize,
        arena: &SharedArena,
    ) -> ArenaKey {
        let view = array.views()[idx];
        if ArenaKey::from_raw(view).is_inline() {
            return ArenaKey::from_raw(view);
        }
        let buffers = array.data_buffers();
        if self.registered != Some((buffers.as_ptr(), buffers.len())) {
            self.register(buffers, arena);
        }
        let buffer_index = (view >> 64) as u32;
        let remapped = (view & !(u128::from(u32::MAX) << 64))
            | (u128::from(self.slots[buffer_index as usize]) << 64);
        ArenaKey::from_raw(remapped)
    }

    #[cold]
    fn register(&mut self, buffers: &[Buffer], arena: &SharedArena) {
        self.slots.clear();
        self.slots.extend(
            buffers
                .iter()
                .map(|buffer| arena.register_input_buffer(buffer)),
        );
        self.registered = Some((buffers.as_ptr(), buffers.len()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    #[test]
    fn long_keys_point_into_the_input_buffer() {
        init_test_free_pool(4);
        let arena = SharedArena::new(4);
        let long = "a string well over twelve bytes";
        let array = StringViewArray::from(vec!["short", long]);
        let mut slots = InputBufferSlots::default();

        let short_key = slots.key_at(&array, 0, &arena);
        let long_key = slots.key_at(&array, 1, &arena);

        assert!(short_key.is_inline());
        assert_eq!(short_key.resolve(&arena), b"short");
        assert!(!long_key.is_inline());
        assert_eq!(long_key.resolve(&arena), long.as_bytes());
        assert_eq!(
            long_key.resolve(&arena).as_ptr(),
            array.value(1).as_ptr(),
            "the key reads the input bytes in place"
        );
    }

    #[test]
    fn a_shared_buffer_list_is_registered_once() {
        init_test_free_pool(4);
        let arena = SharedArena::new(4);
        let long = "a string well over twelve bytes";
        let array = StringViewArray::from(vec![long, long]);
        let first_batch = array.slice(0, 1);
        let second_batch = array.slice(1, 1);
        let mut slots = InputBufferSlots::default();

        let first_key = slots.key_at(&first_batch, 0, &arena);
        let second_key = slots.key_at(&second_batch, 0, &arena);

        assert_eq!(first_key.buffer_index(), second_key.buffer_index());
        assert_eq!(arena.to_arrow_buffers().len(), 1);
    }
}
