use crate::env::{get_env_var_with_default, get_total_memory};
use crate::io::IOLocation;
use bytes::Bytes;
use moka::sync::Cache as MokaCache;
use std::sync::LazyLock;

pub static CACHE: LazyLock<Cache> = LazyLock::new(|| Cache {
    inner: MokaCache::builder()
        .max_capacity(get_env_var_with_default("CACHE_BYTES", get_total_memory() / 2) as u64)
        .eviction_listener(|_e, _b, _c| {})
        // .support_invalidation_closures()
        .build(),
});

pub struct Cache {
    inner: MokaCache<IOLocation, Bytes>,
}

impl Cache {
    pub fn get(&self, location: &IOLocation) -> Option<Bytes> {
        self.inner.get(location)
    }

    pub fn contains(&self, location: &IOLocation) -> bool {
        self.inner.contains_key(location)
    }

    pub fn insert(&self, location: IOLocation, bytes: Bytes) {
        self.inner.insert(location, bytes);
    }
}
