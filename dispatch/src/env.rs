//! Environment helpers and system information.

use std::str::FromStr;

/// Strings at or below this length are stored inline in the 128-bit view
/// (no block reference needed).
pub const MAX_INLINE_STRING_VIEW: usize = 12;

/// Read an environment variable, parsing it to `T`. Returns `default` if the
/// variable is unset or cannot be parsed.
pub fn get_env_var_with_default<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
