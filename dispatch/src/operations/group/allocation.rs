/// `Allocation` is a trait that allows giving different types of backing allocations for a
/// HashTable. Currently, only `Vec` is a valid allocation, but this gives the option for trying
/// other methodologies for backing allocations such as mmaps, etc.
///
/// Note that `resize`, `len`, `capacity` all refer to sizes as multiples of `T`, *not* bytes
pub trait Allocation<T>: std::ops::Index<usize, Output = T> + std::ops::IndexMut<usize> {
    /// Resize backing allocation to a given size
    fn resize(&mut self, size: usize);
    /// Current length (backing allocation may have more ready)
    fn len(&self) -> usize;
    /// Current capacity (how many overall elements can the backing allocation hold)
    fn capacity(&self) -> usize;
}

impl<T: Default> Allocation<T> for Vec<T> {
    fn resize(&mut self, size: usize) {
        Vec::resize_with(self, size, Default::default)
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }
}
