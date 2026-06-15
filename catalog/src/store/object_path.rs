//! [`ObjectPath`]: a path/key in the object store.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A `/`-separated key in the object store — e.g. `events/a.parquet` or
/// `_pivot_tables/t/00000000000000000001.json`.
///
/// A **leading `/` marks an absolute key**: addressed from the store's own root
/// (the filesystem root, or the bucket root), ignoring any database prefix. Any
/// other key is relative to the database root. The backends interpret that
/// distinction; everything above the store speaks `ObjectPath` so a store key is
/// never confused with a local filesystem path or an arbitrary string.
///
/// Serializes transparently as its string, so it round-trips through the JSON
/// manifests as a plain path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObjectPath(String);

impl ObjectPath {
    /// Wrap a raw key string.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// The key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether this is an absolute key (taken from the store's root, bypassing
    /// the database prefix).
    pub fn is_absolute(&self) -> bool {
        self.0.starts_with('/')
    }

    /// Whether the key is empty (the database root / no location).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The final path segment — the object's name within its directory.
    pub fn name(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or(&self.0)
    }

    /// This path with `segment` appended under it, preserving absoluteness
    /// (`events`.join(`a.parquet`) → `events/a.parquet`; a leading `/` is kept).
    pub fn join(&self, segment: &str) -> ObjectPath {
        let base = self.0.trim_matches('/');
        let segment = segment.trim_matches('/');
        let joined = if base.is_empty() {
            segment.to_string()
        } else {
            format!("{base}/{segment}")
        };
        ObjectPath(if self.is_absolute() {
            format!("/{joined}")
        } else {
            joined
        })
    }

    /// The directory this path sits in — everything before the final segment,
    /// keeping the leading `/` of an absolute key. `None` when there is no
    /// separator (a bare name at the root).
    pub fn parent(&self) -> Option<ObjectPath> {
        let trimmed = self.0.trim_end_matches('/');
        let idx = trimmed.rfind('/')?;
        Some(ObjectPath(trimmed[..idx].to_string()))
    }

    /// Resolve `child` against this path as a base directory: an **absolute**
    /// `child` is taken as-is (it escapes the base — e.g. a file in a shared
    /// directory); any other `child` is relative to this base and joined under
    /// it. How a table reads a [`FileRef`](crate::FileRef) whose path may be
    /// relative to the table's location or absolute.
    pub fn resolve(&self, child: &ObjectPath) -> ObjectPath {
        if child.is_absolute() {
            child.clone()
        } else {
            self.join(child.as_str())
        }
    }
}

impl fmt::Display for ObjectPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ObjectPath {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ObjectPath {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_preserves_absoluteness() {
        // A relative location yields a relative key (the store prepends its prefix).
        assert_eq!(ObjectPath::new("events").join("a.parquet").as_str(), "events/a.parquet");
        // An absolute location keeps its leading slash so the store reads it from
        // the bucket root.
        assert_eq!(
            ObjectPath::new("/shared/events").join("a.parquet").as_str(),
            "/shared/events/a.parquet"
        );
        // Joining onto the empty root is just the segment.
        assert_eq!(ObjectPath::default().join("a.parquet").as_str(), "a.parquet");
    }

    #[test]
    fn name_and_parent() {
        let p = ObjectPath::new("events/a.parquet");
        assert_eq!(p.name(), "a.parquet");
        assert_eq!(p.parent(), Some(ObjectPath::new("events")));

        let abs = ObjectPath::new("/shared/events/a.parquet");
        assert_eq!(abs.parent(), Some(ObjectPath::new("/shared/events")));

        // A bare name has no parent.
        assert_eq!(ObjectPath::new("a.parquet").parent(), None);
    }

    #[test]
    fn resolve_relative_under_base_absolute_escapes() {
        let location = ObjectPath::new("events");
        // A relative child is read under the table's location.
        assert_eq!(location.resolve(&ObjectPath::new("a.parquet")).as_str(), "events/a.parquet");
        // An absolute child escapes the location to the store root.
        assert_eq!(
            location.resolve(&ObjectPath::new("/shared/a.parquet")).as_str(),
            "/shared/a.parquet"
        );
        // An absolute location keeps its leading slash through resolution.
        assert_eq!(
            ObjectPath::new("/data/events").resolve(&ObjectPath::new("a.parquet")).as_str(),
            "/data/events/a.parquet"
        );
    }
}
