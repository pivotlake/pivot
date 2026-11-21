//! Which part of which manifest an entry of the cache is
//! ([`CachedPartKey`]), and the part as the cache holds it
//! ([`ManifestPartValue`]).

use std::sync::Arc;

use crate::files::manifest_parts::{CachedColumnBounds, CachedPartitionFiles, ManifestPartitions};
use crate::files::partition::Partition;

/// Which part of one data manifest. The parts order with a manifest's list
/// of partitions first and then partition by partition, in the order of the
/// partitions: a partition's files, then what the manifest records of
/// each column in them, in the order of the fields' ids.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ManifestPart {
    /// The partitions the manifest holds a live file of.
    PartitionList,
    /// A part of what the manifest holds of the partition
    /// `partition`.
    ForPartition {
        partition: Arc<Partition>,
        part: PartitionPart,
    },
}

/// Which part of one partition in one data manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PartitionPart {
    /// The partition's live files in the manifest.
    Files,
    /// What the manifest records of the field `field_id` in those files.
    ColumnBounds { field_id: i32 },
}

/// A part of a manifest as whoever read it remembers it: the manifest by
/// the path its manifest list gives it, and which part of it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CachedPartKey {
    pub manifest: Arc<str>,
    pub part: ManifestPart,
}

impl CachedPartKey {
    /// The partitions the manifest at `manifest` holds a live file of.
    pub fn of_partitions(manifest: &Arc<str>) -> Self {
        Self {
            manifest: Arc::clone(manifest),
            part: ManifestPart::PartitionList,
        }
    }

    /// The files of the partition `partition` in the manifest at
    /// `manifest`.
    pub fn of_files(manifest: &Arc<str>, partition: &Arc<Partition>) -> Self {
        Self {
            manifest: Arc::clone(manifest),
            part: ManifestPart::ForPartition {
                partition: Arc::clone(partition),
                part: PartitionPart::Files,
            },
        }
    }

    /// What the manifest at `manifest` records of the field `field_id` in
    /// the files of the partition `partition`.
    pub fn of_bounds(manifest: &Arc<str>, partition: &Arc<Partition>, field_id: i32) -> Self {
        Self {
            manifest: Arc::clone(manifest),
            part: ManifestPart::ForPartition {
                partition: Arc::clone(partition),
                part: PartitionPart::ColumnBounds { field_id },
            },
        }
    }
}

/// A part of a manifest as the cache holds it, shared with whoever reads
/// it.
#[derive(Clone)]
pub(crate) enum ManifestPartValue {
    PartitionList(Arc<ManifestPartitions>),
    Files(Arc<CachedPartitionFiles>),
    ColumnBounds(Arc<CachedColumnBounds>),
}

impl ManifestPartValue {
    /// An estimate of the memory the part holds, itself included.
    pub(super) fn estimate_heap_bytes(&self) -> u64 {
        match self {
            Self::PartitionList(partitions) => partitions.estimate_heap_bytes(),
            Self::Files(files) => files.estimate_heap_bytes(),
            Self::ColumnBounds(bounds) => bounds.estimate_heap_bytes(),
        }
    }
}
