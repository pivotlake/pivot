use crate::operations::unary::group::hashtables::Value;

/// A simple counting aggregation, that can be inline in a `HashTable` and holds the current count
/// for a key
#[derive(Default, Copy, Clone)]
pub struct Count {
    pub value: usize,
}

impl Count {
    /// Create a count with a specific initial value.
    pub fn new(size: usize) -> Self {
        Self { value: size }
    }
}

impl Value for Count {
    #[inline]
    fn single() -> Self {
        Self { value: 1 }
    }

    fn merge(mut self, v: Self) -> Self {
        self.value += v.value;
        self
    }
}
