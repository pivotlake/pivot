//! Subprocess wrappers for `objdump`, `nm`, `readelf`.
//!
//! `perf.data` parsing is now native (see [`crate::reader`]); these helpers
//! exist only for the source+assembly view, which still relies on `objdump`
//! to disassemble functions and `nm`/`readelf` to enumerate text-symbol
//! sizes / LOAD segment offsets when computing PIE load bases.

use std::process::Command;

use crate::{Error, Result};

/// Disassemble the entire binary in one shot.
pub fn run_objdump(binary: &str) -> Result<String> {
    let output = Command::new("objdump")
        .args(["-d", "-S", "-l", "--no-show-raw-insn", "--demangle"])
        .arg(binary)
        .output()?;
    if !output.status.success() {
        return Err(Error::Objdump(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Disassemble a specific address range with source interleaving.
pub fn run_objdump_function(binary: &str, start: u64, end: u64) -> Result<String> {
    let output = Command::new("objdump")
        .args(["-d", "-S", "-l", "--no-show-raw-insn", "--demangle"])
        .arg(format!("--start-address=0x{start:x}"))
        .arg(format!("--stop-address=0x{end:x}"))
        .arg(binary)
        .output()?;
    if !output.status.success() {
        return Err(Error::Objdump(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `nm -S <binary>` — used for the function-bounds table.
pub fn run_nm(binary: &str) -> Result<String> {
    let output = Command::new("nm").arg("-S").arg(binary).output()?;
    if !output.status.success() {
        return Err(Error::Nm(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `readelf -lW <binary>` — used by `compute_load_base_from_mapping`.
pub fn run_readelf_l(binary: &str) -> Result<String> {
    let output = Command::new("readelf").arg("-lW").arg(binary).output()?;
    if !output.status.success() {
        return Err(Error::Readelf(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
