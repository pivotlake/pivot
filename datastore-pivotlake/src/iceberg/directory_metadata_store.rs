//! A table whose current version is kept in its own directory: numbered
//! metadata files under `<directory>/metadata/`, each version committed with a
//! conditional write that admits exactly one writer, so writers in several
//! processes share a table with no catalog service. An Iceberg reader reads
//! a version from the path of its metadata file; the directory holds no
//! `version-hint.text` to tell a reader that is given only the directory
//! which version is current.
//!
//! The current version is the highest numbered file, found by listing the
//! directory. A store's first listing of a table starts from the version
//! the table's hint names ([`VERSION_HINT_FILE`]), one at or below the
//! current version by up to [`VERSION_HINT_INTERVAL`], the number of
//! versions between two writes of it, and runs on to the newest file, so
//! the hint only shortens the listing and is never taken for the current
//! version.
//!
//! The metadata log of a version names every earlier metadata file except
//! the ones an expiry's commit was told had expired
//! ([`Table::expired_metadata_files`](iceberg_tables::Table::expired_metadata_files)):
//! `write.metadata.previous-versions-max` does not bound it, so a file leaves
//! the log, and is deleted, only once the file after it is older than the
//! retention window.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use iceberg_tables::commit::COMMIT_RETRIES;
use iceberg_tables::iceberg::spec::TableMetadata;
use iceberg_tables::iceberg::table::Table as SpecTable;
use iceberg_tables::iceberg::{self, ErrorKind, Runtime, TableCommit};
use iceberg_tables::maintenance::retention::snapshot_retention;
use iceberg_tables::{Error, MetadataFile, MetadataStore, Result, TableLocation};
use object_storage::StoreError;
use planner::catalog::SchemaQualifiedTableName;

/// The file under the metadata directory naming a version at or below the
/// current one, by up to [`VERSION_HINT_INTERVAL`] versions, for a store's
/// first listing of the table to start from. Written, best effort, by the
/// commit of each version that is a multiple of the interval: a write that
/// failed, or two writers' that landed out of order, leaves it further
/// behind. A store lists from the version it names to the newest, never
/// trusting it. The name is Pivot's own, not the `version-hint.text` other
/// engines open a table's directory by: they read the version that file
/// names as the current one, which this is not.
pub(crate) const VERSION_HINT_FILE: &str = "pivot-version-hint.text";
/// How many versions apart the hint is written, which bounds how many
/// metadata files a store's first listing of a table reads past the one the
/// hint names.
const VERSION_HINT_INTERVAL: u64 = 100;
/// Metadata versions are zero-padded to this many digits, so that the
/// store's byte-ordered listing is the version order: a listing started after
/// one version returns only newer ones, and the last key is the current one.
const VERSION_DIGITS: usize = 20;
const METADATA_FILE_PREFIX: &str = "v";
const METADATA_FILE_SUFFIX: &str = ".metadata.json";
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(10);
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(1);

/// `v{version}.metadata.json`, the version zero-padded to twenty digits.
fn metadata_file_name(version: u64) -> String {
    format!("{METADATA_FILE_PREFIX}{version:0VERSION_DIGITS$}{METADATA_FILE_SUFFIX}")
}

/// The version a metadata file name carries, or `None` for a name that is not
/// one (a sidecar, a file another tool dropped in): exactly `v`, the padded
/// digits, and the suffix.
pub(crate) fn parse_metadata_file_version(name: &str) -> Option<u64> {
    let digits = name
        .strip_prefix(METADATA_FILE_PREFIX)?
        .strip_suffix(METADATA_FILE_SUFFIX)?;
    if digits.len() != VERSION_DIGITS || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The hint's content for `version`: the padded digits the version's
/// metadata file name carries.
fn hint_content(version: u64) -> String {
    format!("{version:0VERSION_DIGITS$}")
}

/// The version a hint file holds. Anything but the padded digits (with
/// surrounding whitespace tolerated) is an error, never a silently ignored
/// hint: a corrupt hint is worth knowing about.
pub(crate) fn parse_hint(bytes: &[u8]) -> Result<u64> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| Error::MalformedVersionHint {
            content: String::from_utf8_lossy(bytes).into_owned(),
        })?
        .trim();
    if text.len() != VERSION_DIGITS || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Error::MalformedVersionHint {
            content: text.to_string(),
        });
    }
    text.parse().map_err(|_| Error::MalformedVersionHint {
        content: text.to_string(),
    })
}

/// The numbered metadata files of the table at one location.
#[derive(Clone)]
pub(crate) struct DirectoryMetadataStore {
    location: TableLocation,
    /// The latest version this store read or wrote, shared by its clones:
    /// where the listing for the table's latest version starts, and the
    /// metadata that answers when the listing ends there. Only ever replaced
    /// by a later version.
    latest_seen: Arc<Mutex<Option<SeenVersion>>>,
}

/// A metadata version as the store read or wrote it. A metadata file never
/// changes, so this stays true of its version.
#[derive(Clone)]
struct SeenVersion {
    version: u64,
    metadata_file: MetadataFile,
}

impl DirectoryMetadataStore {
    pub(crate) fn new(location: TableLocation) -> Self {
        Self {
            location,
            latest_seen: Arc::default(),
        }
    }

    /// The version the table's hint names, or `None` when the table has no
    /// hint, as one that has not reached version [`VERSION_HINT_INTERVAL`]
    /// has none. A hint that does not parse is an error.
    fn read_version_hint(&self) -> Result<Option<u64>> {
        let location = &self.location;
        location
            .store()
            .get(&location.metadata_key(VERSION_HINT_FILE))?
            .map(|bytes| parse_hint(&bytes))
            .transpose()
    }

    /// Write the version hint naming `version`. Best effort by the caller:
    /// the hint only shortens a store's first listing of the table, so a
    /// failed write costs nothing but a warning.
    fn write_version_hint(&self, version: u64) -> Result<()> {
        let location = &self.location;
        location
            .store()
            .put(
                &location.metadata_key(VERSION_HINT_FILE),
                hint_content(version).as_bytes(),
            )
            .map_err(Error::from)
    }

    /// The table's current version, its highest metadata file's, or `None`
    /// when it has no metadata file at all. The files are listed from the
    /// latest version this store has seen, or else from the version the
    /// hint names, which is at or below the current one; without a hint,
    /// or when the table has no file at or after the start, every metadata
    /// file is listed.
    fn find_current_version(&self) -> Result<Option<u64>> {
        // Copied out, so the lock is not held while the store is asked.
        let seen_version = self
            .latest_seen
            .lock()
            .unwrap()
            .as_ref()
            .map(|seen| seen.version);
        let start = match seen_version {
            Some(version) => Some(version),
            None => self.read_version_hint()?,
        };
        match (self.list_highest_version(start)?, start) {
            (None, Some(start)) => {
                tracing::warn!(
                    table = %self.location.table_uri(),
                    start,
                    "no metadata version at or after the one known exists; listing from the first version"
                );
                self.list_highest_version(None)
            }
            (found, _) => Ok(found),
        }
    }

    /// The highest metadata version listed from the version `start` up, or
    /// among every metadata file without one; `None` when the listing shows
    /// none. The start is a place in the order of the names, not a file
    /// that has to exist: when its metadata file was deleted, the versions
    /// after it are listed all the same.
    fn list_highest_version(&self, start: Option<u64>) -> Result<Option<u64>> {
        let location = &self.location;
        // The padded names sort in version order, so the name of the version
        // before the start is a cursor that admits the start itself and
        // everything newer.
        let after = start
            .and_then(|version| version.checked_sub(1))
            .map(metadata_file_name);
        let listed = location.store().list_filtered(
            &location.metadata_dir(),
            METADATA_FILE_PREFIX,
            after.as_deref(),
        )?;
        let mut versions: Vec<u64> = listed
            .objects
            .iter()
            .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
            .collect();
        versions.sort_unstable();
        Ok(versions.last().copied())
    }

    /// Read and parse metadata `version`.
    fn read_metadata(&self, version: u64) -> Result<MetadataFile> {
        let location = &self.location;
        let name = metadata_file_name(version);
        let bytes = location
            .store()
            .get(&location.metadata_key(&name))?
            .ok_or_else(|| Error::MetadataVersionMissing {
                table: location.table_uri(),
                version,
            })?;
        let metadata: TableMetadata = serde_json::from_slice(&bytes)?;
        Ok(MetadataFile {
            location: location.build_metadata_uri(&name),
            metadata: Arc::new(metadata),
        })
    }

    /// Remember `metadata` as version `version`, unless a later version is
    /// already remembered.
    fn remember_version(&self, version: u64, metadata_file: &MetadataFile) {
        let mut latest_seen = self.latest_seen.lock().unwrap();
        if latest_seen
            .as_ref()
            .is_none_or(|seen| seen.version < version)
        {
            *latest_seen = Some(SeenVersion {
                version,
                metadata_file: metadata_file.clone(),
            });
        }
    }

    /// Read metadata `version` and remember it.
    fn read_and_remember_metadata(&self, version: u64) -> Result<MetadataFile> {
        let metadata = self.read_metadata(version)?;
        self.remember_version(version, &metadata);
        Ok(metadata)
    }

    /// Whether `base` was committed at least its retention window ago: the
    /// age from which the metadata file after it may have been deleted,
    /// which takes the file after that one to be as old as the window.
    fn is_stale_base(base: &MetadataFile) -> bool {
        let retention_ms = snapshot_retention(&base.metadata)
            .map(|window| i64::try_from(window.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        base.metadata.last_updated_ms().saturating_add(retention_ms) <= now_unix_ms()
    }

    /// Make `metadata` the table's current version, as the successor of
    /// `base` (`None` for a table's first version): create
    /// `v{N+1}.metadata.json` only if nothing has that name yet.
    /// Every writer that loaded N derives the same name, so exactly one
    /// succeeds, and [`Error::NextVersionAlreadyPublished`] tells the others
    /// they lost. A version that is a multiple of [`VERSION_HINT_INTERVAL`]
    /// is then written to the hint, best effort.
    ///
    /// The name is free as well once the version the table had under it
    /// expired and was deleted, which takes a base committed at least its
    /// retention window ago ([`is_stale_base`](Self::is_stale_base)). The
    /// table's current file is never deleted, so a later version is then
    /// listed. A commit on such a base therefore lists the versions after
    /// its own before it writes, and when there is one it has lost, the
    /// same error, with nothing written. A listing that fails fails the commit as the
    /// error it is, since nothing was written. A write that succeeds is the
    /// table's version N+1 and stands, whatever another writer commits on
    /// top of it. The listing and the write are two requests, and the name
    /// can come free by expiry between them: other writers commit version
    /// N+1 and its successor there, and N+1 expires and is deleted there
    /// too. The write then succeeds under a name no reader looks at, which
    /// takes a retention window shorter than the time between the two
    /// requests.
    ///
    /// A write that fails without saying whether the store applied it is
    /// retried, and the commit stands when a later attempt succeeds. When
    /// every attempt fails, or a later one finds the name taken, which the
    /// failed attempt may itself have done, nothing tells whether the
    /// version under the name is this commit's: the commit is
    /// [`Error::CommitOutcomeUnknown`], never a loss, with the failure as
    /// its source, and is logged with the table and the version.
    pub(crate) fn commit_metadata(
        &self,
        base: Option<&MetadataFile>,
        metadata: Arc<TableMetadata>,
    ) -> Result<MetadataFile> {
        let location = &self.location;
        let store = location.store();
        let version = match base {
            Some(base) => metadata_location_version(&base.location)? + 1,
            None => 1,
        };
        let metadata_bytes = serde_json::to_vec(metadata.as_ref())?;
        let metadata_name = metadata_file_name(version);
        let metadata_key = location.metadata_key(&metadata_name);
        // Decided before anything is written: a version listed after the
        // write may be a successor of the write itself, which says nothing
        // of whether the name was free by expiry.
        if base.is_some_and(Self::is_stale_base) && self.has_version_after(&metadata_name)? {
            return Err(Error::NextVersionAlreadyPublished);
        }
        let put = retry_io(
            || match store.put_if_absent(&metadata_key, &metadata_bytes) {
                Ok(()) => Ok(true),
                Err(StoreError::VersionConflict { .. }) => Ok(false),
                Err(error) => Err(error),
            },
        );
        match put {
            Ok(RetriedAnswer { answer: true, .. }) => {}
            Ok(RetriedAnswer {
                answer: false,
                retried_failure: None,
            }) => return Err(Error::NextVersionAlreadyPublished),
            // The name is taken after an attempt of this commit failed
            // without saying whether the store applied it: the version
            // under the name may be this commit's or another writer's.
            Ok(RetriedAnswer {
                answer: false,
                retried_failure: Some(failure),
            }) => return Err(self.report_unknown_outcome(version, failure.into())),
            // No attempt said whether the store applied it.
            Err(error) => return Err(self.report_unknown_outcome(version, error)),
        }
        if version.is_multiple_of(VERSION_HINT_INTERVAL)
            && let Err(error) = self.write_version_hint(version)
        {
            tracing::warn!(
                table = %location.table_uri(),
                version,
                error = %error,
                "writing the version hint failed; a store's first listing of the table starts further back"
            );
        }
        let committed = MetadataFile {
            location: location.build_metadata_uri(&metadata_name),
            metadata,
        };
        self.remember_version(version, &committed);
        Ok(committed)
    }

    /// Whether the table has a numbered metadata file after the one named
    /// `metadata_name`, by a listing that starts after that name.
    fn has_version_after(&self, metadata_name: &str) -> Result<bool> {
        let location = &self.location;
        let listed = retry_io(|| {
            location.store().list_filtered(
                &location.metadata_dir(),
                METADATA_FILE_PREFIX,
                Some(metadata_name),
            )
        })?
        .answer;
        // A file another tool left here may start with `v` too and sort
        // after every version's name, a `version-hint.text` for one, so
        // only a numbered metadata file is a later version.
        Ok(listed
            .objects
            .iter()
            .any(|object| parse_metadata_file_version(object.file.path.as_str()).is_some()))
    }

    /// The error of a commit of `version` whose write ended with `failure`
    /// and left open whether the store applied it, logged here, where that
    /// is found out, with what tells an operator which version to look for.
    fn report_unknown_outcome(&self, version: u64, failure: Error) -> Error {
        tracing::warn!(
            table = %self.location.table_uri(),
            version,
            error = %failure,
            "the write of a metadata version failed without saying whether the store applied it; the commit's outcome is unknown and the files it wrote are kept"
        );
        Error::CommitOutcomeUnknown {
            source: Box::new(failure),
        }
    }
}

#[async_trait]
impl MetadataStore for DirectoryMetadataStore {
    /// The metadata of the table's current version: the one this store
    /// remembers when that is still the current version, which costs one
    /// short listing and no read, else read and remembered. The store is
    /// one table's directory, so the table's name tells it nothing.
    fn load_latest(&self, _name: &SchemaQualifiedTableName) -> Result<MetadataFile> {
        let version = self
            .find_current_version()?
            .ok_or_else(|| Error::TableMetadataMissing {
                table: self.location.table_uri(),
            })?;
        // Copied out, so the lock is not held while the store is asked.
        let latest_seen = self.latest_seen.lock().unwrap().clone();
        match latest_seen {
            Some(seen) if seen.version == version => Ok(seen.metadata_file),
            _ => self.read_and_remember_metadata(version),
        }
    }

    /// The successor's metadata log loses `expired_metadata_files` and
    /// nothing else: it keeps every other entry of the base's log, however
    /// many it holds, and adds the base's file.
    async fn commit(
        &self,
        base: &SpecTable,
        commit: TableCommit,
        expired_metadata_files: &[String],
    ) -> iceberg::Result<SpecTable> {
        self.commit_dropping_log_entries(base, commit, expired_metadata_files)
            .await
    }
}

impl DirectoryMetadataStore {
    /// Apply the commit to the base's metadata and publish the result as the
    /// base's successor with the one conditional write that settles whether
    /// the base is still current, or as the table's first version when the
    /// base was never published (a table being created). The successor's
    /// metadata log loses `expired_metadata_files` and nothing else, however
    /// many entries it holds: it keeps every other entry of the base's log
    /// and adds the base's file. A commit conflict means another writer
    /// published that version first.
    async fn commit_dropping_log_entries(
        &self,
        base: &SpecTable,
        mut commit: TableCommit,
        expired_metadata_files: &[String],
    ) -> iceberg::Result<SpecTable> {
        let base_metadata = base.metadata_ref();
        let published_base = base.metadata_location().map(|location| MetadataFile {
            location: location.to_string(),
            metadata: Arc::clone(&base_metadata),
        });
        for requirement in commit.take_requirements() {
            requirement.check(Some(&base_metadata))?;
        }
        let mut builder = base_metadata.as_ref().clone().into_builder(
            published_base
                .as_ref()
                .map(|published| published.location.clone()),
        );
        for update in commit.take_updates() {
            builder = update.apply(builder)?;
        }
        let metadata = Arc::new(
            builder
                .retain_metadata_log_except(expired_metadata_files.iter().cloned())
                .build()?
                .metadata,
        );
        let metadata_store = self.clone();
        // The store's calls block on the network, and the runtime's workers
        // must not.
        let published = tokio::task::spawn_blocking(move || {
            metadata_store.commit_metadata(published_base.as_ref(), metadata)
        })
        .await
        // A commit that panicked may have written the version.
        .unwrap_or_else(|panicked| {
            Err(Error::CommitOutcomeUnknown {
                source: Box::new(Error::spec(
                    iceberg::Error::new(ErrorKind::Unexpected, "publishing a version panicked")
                        .with_source(panicked),
                )),
            })
        })
        .map_err(|error| match error {
            // The kind is how the table learns it lost and builds its
            // commit again on the version that won.
            conflict @ Error::NextVersionAlreadyPublished => {
                iceberg::Error::new(ErrorKind::CatalogCommitConflicts, conflict.to_string())
            }
            // The error rides along as the source, which is how the
            // table tells an unknown outcome from a failure.
            error => iceberg::Error::new(
                ErrorKind::Unexpected,
                format!("publishing a version: {error}"),
            )
            .with_source(error),
        })?;
        SpecTable::builder()
            .metadata(published.metadata)
            .metadata_location(published.location)
            .identifier(base.identifier().clone())
            .file_io(base.file_io().clone())
            .runtime(Runtime::try_current()?)
            .build()
    }
}

/// What a store call answered, with the last failure that was retried before
/// it answered, when an attempt failed.
struct RetriedAnswer<T> {
    answer: T,
    retried_failure: Option<StoreError>,
}

/// Run `write` until it succeeds, retrying an IO or HTTP failure up to
/// [`COMMIT_RETRIES`] times with a jittered backoff.
fn retry_io<T>(mut write: impl FnMut() -> object_storage::Result<T>) -> Result<RetriedAnswer<T>> {
    let mut last_error = None;
    for attempt in 0..COMMIT_RETRIES {
        match write() {
            Ok(answer) => {
                return Ok(RetriedAnswer {
                    answer,
                    retried_failure: last_error,
                });
            }
            Err(error @ (StoreError::Io { .. } | StoreError::Http(_))) => {
                tracing::warn!(attempt, error = %error, "a commit write failed; retrying");
                last_error = Some(error);
            }
            Err(error) => return Err(error.into()),
        }
        let ceiling = RETRY_BACKOFF_BASE
            .saturating_mul(1 << attempt.min(16))
            .min(RETRY_BACKOFF_CAP);
        std::thread::sleep(jittered(ceiling));
    }
    Err(Error::CommitRetriesExhausted {
        attempts: COMMIT_RETRIES,
        source: last_error.expect("a retry only follows a recorded failure"),
    })
}

/// A sleep drawn from `[0, ceiling)`, seeded from the clock's subsecond
/// nanoseconds: enough to keep two writers from retrying in lockstep.
fn jittered(ceiling: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .subsec_nanos() as u64;
    Duration::from_nanos(nanos % (ceiling.as_nanos() as u64).max(1))
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

/// The version number a directory table's metadata location carries.
pub(crate) fn metadata_location_version(metadata_location: &str) -> Result<u64> {
    let name = metadata_location
        .rsplit('/')
        .next()
        .unwrap_or(metadata_location);
    parse_metadata_file_version(name).ok_or_else(|| Error::UnnumberedMetadataLocation {
        location: metadata_location.to_string(),
    })
}
