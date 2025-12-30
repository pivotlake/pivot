use std::ops::{Deref, DerefMut};

pub type Identifier = usize;

pub struct Identified<T> {
    id: Identifier,
    data: T,
}

impl<T> Identified<T> {
    pub fn new(id: Identifier, data: T) -> Self {
        Self { id, data }
    }

    pub fn id(&self) -> Identifier {
        self.id
    }

    pub fn take(self) -> T {
        self.data
    }
}

impl<T> Deref for Identified<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Identified<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
