//! Environment helpers and system information.

use std::str::FromStr;

/// Strings at or below this length are stored inline in the 128-bit view
/// (no block reference needed).
pub const MAX_INLINE_STRING_VIEW: usize = 12;

/// A tunable read from the environment variable `name` the first time it is
/// asked for and kept in `cell` from then on, `default` when the variable is
/// unset. Panics on a value that does not parse or that `is_valid` rejects:
/// a mistyped tunable must not quietly run the default.
pub fn get_tunable(
    cell: &std::sync::OnceLock<usize>,
    name: &str,
    default: usize,
    is_valid: impl Fn(usize) -> bool,
) -> usize {
    *cell.get_or_init(|| {
        let value = match std::env::var(name) {
            Ok(text) => text
                .parse()
                .unwrap_or_else(|_| panic!("{name}={text} is not a number")),
            Err(_) => default,
        };
        assert!(is_valid(value), "{name}={value} is not a valid setting");
        tracing::info!(name, value, default, "tunable");
        value
    })
}

/// Read an environment variable, parsing it to `T`. Returns `default` if the
/// variable is unset or cannot be parsed.
pub fn get_env_var_with_default<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
