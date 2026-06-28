//! JSON decoder: the payload is one JSON object, parsed straight to a
//! [`Value`]. No schema registry; the destination table's schema drives the
//! mapping at Arrow-construction time (see [`KafkaBatch`](super::KafkaBatch)).

use serde_json::Value;

use super::{DecodeError, MessageDecoder};

pub(super) struct JsonDecoder;

impl MessageDecoder for JsonDecoder {
    fn decode(&self, payload: &[u8]) -> Result<Value, DecodeError> {
        let value: Value = serde_json::from_slice(payload)?;
        if value.is_object() {
            Ok(value)
        } else {
            Err(DecodeError::NotAnObject)
        }
    }
}
