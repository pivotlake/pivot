//! The metadata document a table version was loaded from: the base every
//! commit from it is written on top of.

use std::sync::Arc;

use iceberg::spec::TableMetadata;

/// A metadata document the table was loaded from: where it is, as the next
/// version's metadata log records it, and what it says.
#[derive(Clone)]
pub struct CurrentMetadata {
    pub metadata_location: String,
    pub metadata: Arc<TableMetadata>,
}
