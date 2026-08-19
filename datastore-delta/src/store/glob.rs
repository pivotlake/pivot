//! Turning a location a query writes — `s3://bucket/events/*.parquet`,
//! `/data/hits.parquet` — into the set of objects it covers.
//!
//! A location splits into the **directory** holding its files and the
//! **name pattern** they must match within it. `*` stands for any run of
//! characters and `?` for exactly one; a location carrying neither names a
//! single object and matches it exactly.
//!
//! Only the final segment may carry a pattern. The directory is therefore
//! always literal, so it opens as a store the same way a datastore's location
//! does and one listing under it answers the match: a pattern higher up would
//! instead mean walking every directory below it, which the one-level
//! [`ObjectStore::list`] does not do, and is refused rather than quietly
//! matching nothing.

use super::{FileRef, ObjectPath, ObjectStore, Result, StoreError};

/// A location split into the directory its objects live in and the pattern
/// their names must match. Built by [`parse`](LocationPattern::parse), which is
/// where a location that names no object, or patterns a directory, is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocationPattern {
    /// The literal directory holding the objects, in the same spelling a
    /// datastore's location uses (`s3://bucket/events`, `/data`), so a store
    /// opens at it directly.
    pub directory_uri: String,
    /// The pattern the names directly under that directory must match.
    pub name_pattern: String,
}

impl LocationPattern {
    /// Split `location` into its directory and name pattern.
    pub fn parse(location: &str) -> Result<Self> {
        let (scheme, below_scheme) = match location.split_once("://") {
            Some((scheme, below_scheme)) => (Some(scheme), below_scheme),
            None => (None, location),
        };
        let Some((directory, name_pattern)) = below_scheme.rsplit_once('/') else {
            // A location with nothing below its bucket (`s3://bucket`), or a
            // bare local name with no directory part, which is read under the
            // working directory.
            if scheme.is_some() {
                return Err(StoreError::LocationWithoutName {
                    location: location.to_string(),
                });
            }
            return Self::build(".".to_string(), below_scheme, location);
        };
        if directory.contains(WILDCARDS) {
            return Err(StoreError::PatternInDirectory {
                location: location.to_string(),
            });
        }
        let directory_uri = match scheme {
            Some(scheme) => format!("{scheme}://{directory}"),
            // A local location whose directory part is empty is at the
            // filesystem root, which is that separator on its own.
            None if directory.is_empty() => "/".to_string(),
            None => directory.to_string(),
        };
        Self::build(directory_uri, name_pattern, location)
    }

    fn build(directory_uri: String, name_pattern: &str, location: &str) -> Result<Self> {
        if name_pattern.is_empty() {
            return Err(StoreError::LocationWithoutName {
                location: location.to_string(),
            });
        }
        Ok(Self {
            directory_uri,
            name_pattern: name_pattern.to_string(),
        })
    }
}

/// The wildcards a name pattern may carry.
const WILDCARDS: [char; 2] = ['*', '?'];

/// The objects directly under `store`'s root whose name matches `pattern`, by
/// name so a query reads them in a stable order however the backend listed
/// them. `store` is opened at the pattern's
/// [`directory_uri`](LocationPattern::directory_uri), so the listing is one
/// level and each name is matched whole.
pub fn list_matching(store: &dyn ObjectStore, pattern: &str) -> Result<Vec<FileRef>> {
    let mut matched: Vec<FileRef> = store
        .list(&ObjectPath::default())?
        .into_iter()
        .map(|object| object.file)
        .filter(|file| name_matches(pattern, file.path.name()))
        .collect();
    matched.sort_by(|a, b| a.path.as_str().cmp(b.path.as_str()));
    Ok(matched)
}

/// Whether `name` matches `pattern`, where `*` stands for any run of characters
/// (including none) and `?` for exactly one. Everything else matches itself.
///
/// Walks both in step, remembering the last `*` and how much of the name it had
/// consumed: a later mismatch resumes from there with the `*` swallowing one
/// more character, which is what lets a pattern with several `*`s find the one
/// split that matches.
pub fn name_matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    let (mut pattern_idx, mut name_idx) = (0, 0);
    // The open `*` to fall back to, and how much of the name it has consumed.
    let mut star: Option<usize> = None;
    let mut star_consumed = 0;

    while name_idx < name.len() {
        match pattern.get(pattern_idx) {
            Some('?') => {
                pattern_idx += 1;
                name_idx += 1;
            }
            Some('*') => {
                star = Some(pattern_idx);
                star_consumed = name_idx;
                pattern_idx += 1;
            }
            Some(c) if *c == name[name_idx] => {
                pattern_idx += 1;
                name_idx += 1;
            }
            // Mismatch: hand one more character to the last `*` and retry
            // everything after it. With no `*` behind us the name cannot match.
            _ => match star {
                Some(star_idx) => {
                    pattern_idx = star_idx + 1;
                    star_consumed += 1;
                    name_idx = star_consumed;
                }
                None => return false,
            },
        }
    }

    // The name is exhausted; trailing `*`s match the empty remainder.
    pattern[pattern_idx..].iter().all(|c| *c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_remote_location_splits_into_its_directory_and_pattern() {
        assert_eq!(
            LocationPattern::parse("s3://bucket/events/*.parquet").unwrap(),
            LocationPattern {
                directory_uri: "s3://bucket/events".to_string(),
                name_pattern: "*.parquet".to_string(),
            }
        );
        // Directly in the bucket, with no prefix between.
        assert_eq!(
            LocationPattern::parse("gs://bucket/a.parquet").unwrap(),
            LocationPattern {
                directory_uri: "gs://bucket".to_string(),
                name_pattern: "a.parquet".to_string(),
            }
        );
    }

    #[test]
    fn a_local_location_splits_the_same_way() {
        assert_eq!(
            LocationPattern::parse("/data/hits/*.parquet").unwrap(),
            LocationPattern {
                directory_uri: "/data/hits".to_string(),
                name_pattern: "*.parquet".to_string(),
            }
        );
        // A bare name is read under the working directory.
        assert_eq!(
            LocationPattern::parse("hits.parquet")
                .unwrap()
                .directory_uri,
            "."
        );
        // A name at the filesystem root keeps the root as its directory.
        assert_eq!(
            LocationPattern::parse("/hits.parquet")
                .unwrap()
                .directory_uri,
            "/"
        );
    }

    #[test]
    fn a_pattern_above_the_final_segment_is_refused() {
        let error = LocationPattern::parse("s3://bucket/*/a.parquet").unwrap_err();

        assert!(matches!(
            error,
            StoreError::PatternInDirectory { location } if location == "s3://bucket/*/a.parquet"
        ));
    }

    #[test]
    fn a_location_naming_no_object_is_refused() {
        for location in ["s3://bucket", "s3://bucket/events/"] {
            let error = LocationPattern::parse(location).unwrap_err();
            assert!(matches!(error, StoreError::LocationWithoutName { .. }));
        }
    }

    #[test]
    fn wildcards_match_runs_and_single_characters() {
        assert!(name_matches("*.parquet", "a.parquet"));
        assert!(name_matches("*.parquet", ".parquet"));
        assert!(!name_matches("*.parquet", "a.parquet.tmp"));
        assert!(name_matches("part-?.parquet", "part-3.parquet"));
        assert!(!name_matches("part-?.parquet", "part-42.parquet"));
        // Several wildcards need the one split that matches.
        assert!(name_matches("*-2026-*.parquet", "hits-2026-08.parquet"));
        assert!(!name_matches("*-2026-*.parquet", "hits-2025-08.parquet"));
        // A pattern with no wildcard names exactly one object.
        assert!(name_matches("a.parquet", "a.parquet"));
        assert!(!name_matches("a.parquet", "b.parquet"));
    }
}
