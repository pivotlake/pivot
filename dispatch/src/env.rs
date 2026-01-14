use std::str::FromStr;

pub fn get_env_var_with_default<T: FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn get_total_memory() -> usize {
    32 * 1024 * 1024 * 1024
}
