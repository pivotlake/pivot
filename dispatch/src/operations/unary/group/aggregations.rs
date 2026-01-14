use crate::operations::unary::group::Value;

/// A simple counting aggregation, that can be inline in a `HashTable` and holds the current count
/// for a key
#[derive(Default, Copy, Clone)]
pub struct Count {
    pub value: usize,
}

impl Count {
    #[inline]
    pub fn single() -> Self {
        Self { value: 1 }
    }
}

impl Value for Count {
    fn merge(mut self, v: Self) -> Self {
        self.value += v.value;
        self
    }
}
