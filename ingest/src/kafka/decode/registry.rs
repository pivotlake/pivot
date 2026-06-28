//! A minimal Confluent Schema Registry client: fetch a writer schema by id and
//! cache it. Blocking HTTPS via `ureq` (the same client catalog's object stores
//! use); a fetch only happens on a cache miss (once per schema id), so the
//! Avro/Protobuf decoders amortise it to near-zero per message.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;

use super::DecodeError;

/// Caches the raw schema text (Avro JSON or `.proto`) for each schema id seen on
/// a topic. Shared behind an `Arc` by a source's consumer tasks.
pub(crate) struct SchemaRegistry {
    base_url: String,
    agent: ureq::Agent,
    cache: Mutex<HashMap<u32, Arc<str>>>,
}

/// The registry's `GET /schemas/ids/{id}` response (only the field we need).
#[derive(Deserialize)]
struct SchemaResponse {
    schema: String,
}

impl SchemaRegistry {
    pub(crate) fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        // Bounded timeouts so a slow/unreachable registry can't block a consumer
        // task (and thus `Ingestor::shutdown`) indefinitely on a cache miss.
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .build();
        Self {
            base_url,
            agent,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The raw schema text for `id`, fetched once and cached. The text is the
    /// Avro schema JSON or the Protobuf `.proto` source, depending on the topic's
    /// format.
    pub(crate) fn schema_text(&self, id: u32) -> Result<Arc<str>, DecodeError> {
        if let Some(text) = self.cache.lock().unwrap().get(&id) {
            return Ok(text.clone());
        }
        let url = format!("{}/schemas/ids/{}", self.base_url, id);
        let response: SchemaResponse = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| DecodeError::Registry(format!("GET {url}: {e}")))?
            .into_json()
            .map_err(|e| DecodeError::Registry(format!("parsing {url} response: {e}")))?;
        let text: Arc<str> = Arc::from(response.schema);
        self.cache.lock().unwrap().insert(id, text.clone());
        Ok(text)
    }
}
