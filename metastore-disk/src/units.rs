//! How sizes and time spans are written in the config file.
//!
//! Both of the file's sections share this vocabulary, so `32g` means the same
//! thing whether it budgets the buffer pool or a compaction target.

use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use std::time::Duration;

use serde::de::{Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer, de};

/// A byte count written the way an operator thinks of one: `32g`, `512m`, or a
/// plain number of bytes.
///
/// Suffixes are base-1024 (`k`, `m`, `g`, `t`, each also accepting a trailing
/// `b`) and case-insensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSize(u64);

impl ByteSize {
    pub const fn from_bytes(bytes: u64) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(self) -> u64 {
        self.0
    }
}

impl FromStr for ByteSize {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (value, suffix) = split_number_and_unit(input)?;
        let multiplier: u64 = match suffix.as_str() {
            "" | "b" => 1,
            "k" | "kb" => 1024,
            "m" | "mb" => 1024 * 1024,
            "g" | "gb" => 1024 * 1024 * 1024,
            "t" | "tb" => 1024 * 1024 * 1024 * 1024,
            other => return Err(format!("`{other}` is not a known size suffix (k/m/g/t)")),
        };
        value
            .checked_mul(multiplier)
            .map(Self)
            .ok_or_else(|| format!("`{input}` overflows a byte count"))
    }
}

/// A span of time written with its unit: `500ms`, `30s`, `5m`, `1h`.
///
/// The unit is mandatory, so a bare number can never be read as the wrong one.
/// Zero is not a span: every interval configures a poll cadence, and a cadence
/// of zero is a busy loop rather than a fast one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval(Duration);

impl Interval {
    pub const fn from_duration(duration: Duration) -> Self {
        Self(duration)
    }

    pub const fn as_duration(self) -> Duration {
        self.0
    }
}

impl FromStr for Interval {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (value, unit) = split_number_and_unit(input)?;
        let millis_per_unit: u64 = match unit.as_str() {
            "ms" => 1,
            "s" => 1000,
            "m" => 60 * 1000,
            "h" => 60 * 60 * 1000,
            "" => return Err(format!("`{input}` needs a unit (ms/s/m/h)")),
            other => return Err(format!("`{other}` is not a known time unit (ms/s/m/h)")),
        };
        let millis = value
            .checked_mul(millis_per_unit)
            .ok_or_else(|| format!("`{input}` overflows a time span"))?;
        if millis == 0 {
            return Err(format!("`{input}` must be longer than zero"));
        }
        Ok(Self(Duration::from_millis(millis)))
    }
}

/// Split a value such as `128m` into its number and its lowercased unit.
fn split_number_and_unit(input: &str) -> Result<(u64, String), String> {
    let trimmed = input.trim();
    let digits_end = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (number, unit) = trimmed.split_at(digits_end);
    if unit.starts_with('.') {
        return Err(fractional_message(input));
    }
    let value: u64 = number
        .parse()
        .map_err(|_| format!("`{input}` does not start with a number"))?;
    Ok((value, unit.trim().to_ascii_lowercase()))
}

/// Both scalars count whole units, so a fraction is answered with the whole
/// number that means the same thing rather than a complaint about the `.`.
fn fractional_message(input: &str) -> String {
    format!(
        "`{input}` is fractional; write a whole number of a smaller unit instead (`512m`, `1500ms`)"
    )
}

/// Written back the way an operator would: the largest base-1024 suffix that
/// divides the count exactly, or a plain number of bytes. Parses back to the
/// same count.
impl Serialize for ByteSize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = self.0;
        for (suffix, multiplier) in [
            ("t", 1u64 << 40),
            ("g", 1 << 30),
            ("m", 1 << 20),
            ("k", 1 << 10),
        ] {
            if bytes != 0 && bytes.is_multiple_of(multiplier) {
                return serializer.serialize_str(&format!("{}{suffix}", bytes / multiplier));
            }
        }
        serializer.serialize_u64(bytes)
    }
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ScalarVisitor {
            expecting: "a size such as `32g`, `512m`, or a number of bytes",
            parsed: PhantomData,
        })
    }
}

impl<'de> Deserialize<'de> for Interval {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ScalarVisitor {
            expecting: "an interval such as `30s` or `500ms`",
            parsed: PhantomData,
        })
    }
}

/// Deserializes one of the scalars above from its YAML scalar. A number without
/// a suffix arrives as an integer rather than a string, so it takes the same
/// path as the text: bytes read it as itself, an interval rejects it for naming
/// no unit.
struct ScalarVisitor<T> {
    expecting: &'static str,
    parsed: PhantomData<T>,
}

impl<'de, T: FromStr<Err = String>> Visitor<'de> for ScalarVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.expecting)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        value.parse().map_err(E::custom)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        value.to_string().parse().map_err(E::custom)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        value.to_string().parse().map_err(E::custom)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        value.to_string().parse().map_err(E::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_written_with_a_base_1024_suffix_or_as_bytes() {
        for (text, expected) in [
            ("32g", 32 * 1024 * 1024 * 1024),
            ("512M", 512 * 1024 * 1024),
            ("128mb", 128 * 1024 * 1024),
            ("4096", 4096),
        ] {
            let size: ByteSize = serde_yaml_ng::from_str(text).unwrap();

            assert_eq!(size.as_bytes(), expected, "parsing `{text}`");
        }
    }

    #[test]
    fn an_unknown_size_suffix_is_rejected() {
        let error = serde_yaml_ng::from_str::<ByteSize>("32x").unwrap_err();

        assert!(
            error.to_string().contains("not a known size suffix"),
            "{error}"
        );
    }

    #[test]
    fn intervals_are_written_with_a_unit() {
        for (text, expected) in [
            ("500ms", Duration::from_millis(500)),
            ("30s", Duration::from_secs(30)),
            ("5m", Duration::from_secs(300)),
            ("1h", Duration::from_secs(3600)),
        ] {
            let interval: Interval = serde_yaml_ng::from_str(text).unwrap();

            assert_eq!(interval.as_duration(), expected, "parsing `{text}`");
        }
    }

    #[test]
    fn an_interval_without_a_unit_is_rejected() {
        let error = serde_yaml_ng::from_str::<Interval>("30").unwrap_err();

        assert!(error.to_string().contains("needs a unit"), "{error}");
    }

    #[test]
    fn a_fractional_value_names_the_whole_number_that_means_the_same() {
        for text in ["0.5g", "0.5"] {
            let error = serde_yaml_ng::from_str::<ByteSize>(text).unwrap_err();

            assert!(
                error.to_string().contains("is fractional"),
                "parsing `{text}`: {error}"
            );
        }
        let error = serde_yaml_ng::from_str::<Interval>("1.5s").unwrap_err();

        assert!(error.to_string().contains("is fractional"), "{error}");
    }

    #[test]
    fn an_interval_too_large_to_represent_is_rejected_rather_than_wrapped() {
        let error = serde_yaml_ng::from_str::<Interval>("1000000000000000000m").unwrap_err();

        assert!(error.to_string().contains("overflows"), "{error}");
    }

    #[test]
    fn a_zero_interval_is_rejected_rather_than_becoming_a_busy_loop() {
        for text in ["0s", "0ms"] {
            let error = serde_yaml_ng::from_str::<Interval>(text).unwrap_err();

            assert!(
                error.to_string().contains("longer than zero"),
                "parsing `{text}`: {error}"
            );
        }
    }
}
