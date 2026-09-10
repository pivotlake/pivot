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

/// Arena slot per data buffer of one input array, registered on first use.
#[derive(Default)]
pub(crate) struct InputBufferSlots {
    slots: Option<Box<[u32]>>,
}

impl InputBufferSlots {
    /// The key for view `idx` of `array`, pointing into the array's own
    /// buffer. Registers the array's buffers with the arena on the first call.
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
        let slots = self.slots.get_or_insert_with(|| register(array, arena));
        let buffer_index = (view >> 64) as u32;
        let remapped = (view & !(u128::from(u32::MAX) << 64))
            | (u128::from(slots[buffer_index as usize]) << 64);
        ArenaKey::from_raw(remapped)
    }
}

#[cold]
fn register(array: &StringViewArray, arena: &SharedArena) -> Box<[u32]> {
    array
        .data_buffers()
        .iter()
        .map(|buffer| arena.register_input_buffer(buffer))
        .collect()
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
}
