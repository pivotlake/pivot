//! Avro decoder: strip the Confluent frame, fetch + cache the writer schema by
//! id, decode the datum, and render it as a JSON object so it joins the single
//! `arrow-json` construction path.
//!
//! Limitation: optional fields modelled as Avro unions (`["null","string"]`)
//! are rendered by their inner value; deeply nested unions are a known edge to
//! revisit if a schema needs them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use apache_avro::Schema as AvroSchema;
use serde_json::Value;

use super::{DecodeError, MessageDecoder, SchemaRegistry, confluent};

pub(super) struct AvroDecoder {
    registry: Arc<SchemaRegistry>,
    /// Parsed writer schemas, cached by id alongside the registry's text cache so
    /// each message after the first reuses the compiled schema.
    schemas: Mutex<HashMap<u32, Arc<AvroSchema>>>,
}

impl AvroDecoder {
    pub(super) fn new(registry: Arc<SchemaRegistry>) -> Self {
        Self {
            registry,
            schemas: Mutex::new(HashMap::new()),
        }
    }

    fn schema(&self, id: u32) -> Result<Arc<AvroSchema>, DecodeError> {
        if let Some(schema) = self.schemas.lock().unwrap().get(&id) {
            return Ok(schema.clone());
        }
        let text = self.registry.schema_text(id)?;
        let parsed = AvroSchema::parse_str(&text)
            .map_err(|e| DecodeError::Avro(format!("parsing avro schema {id}: {e}")))?;
        let schema = Arc::new(parsed);
        self.schemas.lock().unwrap().insert(id, schema.clone());
        Ok(schema)
    }
}

impl MessageDecoder for AvroDecoder {
    fn decode(&self, payload: &[u8]) -> Result<Value, DecodeError> {
        let frame = confluent::parse(payload)?;
        let schema = self.schema(frame.schema_id)?;
        let mut reader = frame.body;
        let avro_value = apache_avro::from_avro_datum(&schema, &mut reader, None)
            .map_err(|e| DecodeError::Avro(format!("decoding datum: {e}")))?;
        let json = Value::try_from(avro_value)
            .map_err(|e| DecodeError::Avro(format!("avro to json: {e}")))?;
        if json.is_object() {
            Ok(json)
        } else {
            Err(DecodeError::NotAnObject)
        }
    }
}
