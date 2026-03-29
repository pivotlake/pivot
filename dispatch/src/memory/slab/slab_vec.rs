use std::marker::PhantomData;
use std::mem;
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
    next_size: usize,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for SlabVec<T> {}

impl<T> Default for SlabVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SlabVec<T> {
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            last_ptr: std::ptr::null_mut(),
            slabs: Vec::new(),
            len: 0,
            next_size: elements_per_slab::<T>(),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn push(&mut self, value: T, allocator: &mut SlabAllocator) {
        if self.ptr >= self.last_ptr {
            let slab = allocator.get_slab_of_size(self.next_size * size_of::<T>(), false);
            self.ptr = slab.ptr as *mut T;
            self.last_ptr = unsafe { (slab.ptr as *mut T).add(self.next_size) };
            self.slabs.push(slab);
            if self.next_size * 2 * size_of::<T>()  <= BUFFER_SIZE {
                self.next_size *= 2;
            }
        }
        unsafe {
            self.ptr.write(value);
            self.ptr = self.ptr.add(1);
            self.len += 1;
        }
    }

    pub fn slabs(&self) -> &[Slab] {
        &self.slabs
    }
}

// ---------------------------------------------------------------------------
// Consuming iterator (SlabVecIterator)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Borrowing iterator (SlabVecIter) — pointer-walking, no indexing math
// ---------------------------------------------------------------------------

impl<'a, T> IntoIterator for &'a SlabVec<T> {
    type Item = &'a T;
    type IntoIter = SlabVecIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SlabVecIter::new(&self.slabs, self.len)
    }
}

pub struct SlabVecIter<'a, T> {
    slabs: &'a [Slab],
    slab_index: usize,
    ptr: *const T,
    slab_end: *const T,
    remaining: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> SlabVecIter<'a, T> {
    fn new(slabs: &'a [Slab], len: usize) -> Self {
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

impl<'a, T> Iterator for SlabVecIter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        unsafe {
            let value = &*self.ptr;
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

impl<T> ExactSizeIterator for SlabVecIter<'_, T> {}
