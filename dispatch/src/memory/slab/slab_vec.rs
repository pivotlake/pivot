use std::marker::PhantomData;
use crate::memory::slab::Slab;
use crate::memory::{SlabAllocator, BUFFER_SIZE};

/// Number of elements of type `T` that fit in one `BUFFER_SIZE` slab.
#[inline(always)]
const fn elements_per_slab<T>() -> usize {
    BUFFER_SIZE / size_of::<T>()
}

pub struct SlabVec<T> {
    slabs: Vec<Slab>,
    ptr: *mut T,
    last_ptr: *mut T,
    len: usize,
    slab_allocator: SlabAllocator,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for SlabVec<T> {}

impl<T> SlabVec<T> {
    pub fn new(mut slab_allocator: SlabAllocator) -> Self {
        let slab = slab_allocator.get_slab_of_size(BUFFER_SIZE, false);
        let ptr = slab.ptr as *mut T;
        let last_ptr = unsafe { ptr.add(elements_per_slab::<T>()) };
        Self {
            ptr,
            last_ptr,
            slabs: vec![slab],
            len: 0,
            slab_allocator,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        unsafe {
            self.ptr.write(value);
            self.ptr = self.ptr.add(1);
            self.len += 1;
            if self.ptr >= self.last_ptr {
                let slab = self.slab_allocator.get_slab_of_size(BUFFER_SIZE, false);
                self.ptr = slab.ptr as *mut T;
                self.last_ptr = (slab.ptr as *mut T).add(elements_per_slab::<T>());
                self.slabs.push(slab);
            }
        }
    }
}

impl<T> IntoIterator for SlabVec<T> {
    type Item = T;
    type IntoIter = SlabVecIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabVecIterator::new(self.slabs, self.len)
    }
}

pub struct SlabVecIterator<T> {
    slabs: Vec<Slab>,
    slab_index: usize,
    ptr: *const T,
    slab_end: *const T,
    remaining: usize,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for SlabVecIterator<T> {}

impl<T> SlabVecIterator<T> {
    fn new(slabs: Vec<Slab>, len: usize) -> Self {
        if len == 0 {
            return Self {
                slabs,
                slab_index: 0,
                ptr: std::ptr::null(),
                slab_end: std::ptr::null(),
                remaining: 0,
                _phantom: PhantomData,
            };
        }
        let ptr = slabs[0].ptr as *const T;
        let slab_end = unsafe { (slabs[0].ptr as *const T).add(elements_per_slab::<T>()) };
        Self {
            slabs,
            slab_index: 0,
            ptr,
            slab_end,
            remaining: len,
            _phantom: PhantomData,
        }
    }
}

impl<T> Iterator for SlabVecIterator<T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        unsafe {
            let value = self.ptr.read();
            self.remaining -= 1;
            self.ptr = self.ptr.add(1);
            if self.ptr >= self.slab_end && self.remaining > 0 {
                self.slab_index += 1;
                let slab = &self.slabs[self.slab_index];
                self.ptr = slab.ptr as *const T;
                self.slab_end = (slab.ptr as *const T).add(elements_per_slab::<T>());
            }
            Some(value)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for SlabVecIterator<T> {}
