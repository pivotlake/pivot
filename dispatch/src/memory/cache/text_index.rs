//! The text index family: one cache for each kind of piece a text index file
//! is made of (see [`text_index`](crate::text_index)).
//!
//! The members do not hold the same bytes in different forms, as the pages
//! family's do. They hold the steps of one lookup, and a step is only reached
//! through the one before it: a dict index through the file's sparse index,
//! postings through a dict index. What the family is for is the order its
//! members give slots in, which [`setup`](super::setup) declares.

use crate::memory::cache::derived_cache::Cache;
use crate::text_index::{DictIndex, Postings, SparseIndex};
use std::sync::Arc;

/// The text index family's caches.
#[derive(Clone)]
pub struct TextIndexCaches {
    pub sparse_indexes: Arc<Cache<SparseIndex>>,
    pub dict_indexes: Arc<Cache<DictIndex>>,
    pub postings: Arc<Cache<Postings>>,
}
