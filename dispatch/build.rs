//! Copy-and-patch stencil compiler for the GROUP BY value fold.
//!
//! Compiles `…/group/values/cap/stencils.rs` standalone (large code model, so the
//! holes become absolute `movz/movk` immediates) and extracts each stencil's
//! machine code + relocations into `OUT_DIR/cap_stencils.rs`, which the runtime
//! (`cap/mod.rs`) copies and patches. See those files for the technique.
//!
//! The stencils are aarch64; on any other target arch we emit a stub table
//! (`SUPPORTED = false`) and the runtime falls back to the interpreted fold.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

const STENCIL_SRC: &str = "src/operations/unary/group/values/cap/stencils.rs";

// ELF aarch64 relocation type numbers. The `MOVW_UABS_G*` family spans 263..=269:
// `G0=263, G0_NC=264, G1=265, G1_NC=266, G2=267, G2_NC=268, G3=269`. The checked
// and `_NC` variants patch the same imm16 field, so we key only on the 16-bit
// group, which is `(r_type - 263) / 2`.
const R_AARCH64_MOVW_UABS_FIRST: u32 = 263;
const R_AARCH64_MOVW_UABS_LAST: u32 = 269;
const R_AARCH64_JUMP26: u32 = 282;
const R_AARCH64_CALL26: u32 = 283;

const AARCH64_RET: [u8; 4] = [0xc0, 0x03, 0x5f, 0xd6]; // `ret`, little-endian

fn main() {
    println!("cargo:rerun-if-changed={STENCIL_SRC}");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let gen_path = out_dir.join("cap_stencils.rs");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // The stencils are aarch64 machine code. For any other target, emit a stub so
    // the crate still builds; the runtime keys off `SUPPORTED` and never JITs.
    if target_arch != "aarch64" {
        std::fs::write(&gen_path, "pub const SUPPORTED: bool = false;\n").unwrap();
        emit_stub_statics(&gen_path);
        return;
    }

    let obj = out_dir.join("cap_stencils.o");
    compile_stencils(&obj);
    let generated = extract(&obj);
    std::fs::write(&gen_path, generated).unwrap();
}

/// Compile the stencil source to an aarch64 ELF object. We always target
/// `aarch64-unknown-linux-gnu` (ELF, uniform relocations) even on a macOS build
/// host — `--emit obj` needs no linker/sysroot, and the raw aarch64 bytes run on
/// any aarch64 hardware (macOS and Linux alike).
fn compile_stencils(obj: &PathBuf) {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let status = Command::new(rustc)
        .args([
            "--target",
            "aarch64-unknown-linux-gnu",
            "--emit",
            "obj",
            "--crate-type",
            "lib",
            "-C",
            "opt-level=3",
            "-C",
            "code-model=large",
            "-C",
            "panic=abort",
            "-C",
            "relocation-model=static",
            "-C",
            "overflow-checks=off",
            "-o",
        ])
        .arg(obj)
        .arg(STENCIL_SRC)
        .status()
        .expect("failed to spawn rustc for stencils");
    assert!(status.success(), "stencil compile failed");
}

/// A hole: where in the stencil to write a 16-bit chunk of a runtime value.
struct Movw {
    at: u32,
    hole: u8,  // 0 = OFF (cell byte offset), 1 = COL (column index)
    shift: u8, // 0 / 16 / 32 / 48
}

/// Parse the ELF object and render the generated patch table.
fn extract(obj: &PathBuf) -> String {
    use object::{Object, ObjectSection, ObjectSymbol, RelocationFlags, RelocationTarget};

    let data = std::fs::read(obj).unwrap();
    let file = object::File::parse(&*data).expect("parse stencil object");

    let mut ops: Vec<(String, Vec<u8>, Vec<Movw>)> = Vec::new();
    let mut frame: Option<(Vec<u8>, u32)> = None;

    for section in file.sections() {
        let Ok(name) = section.name() else { continue };
        let Some(stencil) = name.strip_prefix(".text.") else {
            continue;
        };
        let code = section.data().unwrap().to_vec();

        let mut movws: Vec<Movw> = Vec::new();
        let mut splice_at: Option<u32> = None;
        for (off, reloc) in section.relocations() {
            let sym = match reloc.target() {
                RelocationTarget::Symbol(i) => file.symbol_by_index(i).unwrap().name().unwrap().to_string(),
                _ => continue,
            };
            let RelocationFlags::Elf { r_type } = reloc.flags() else {
                continue;
            };
            match r_type {
                R_AARCH64_MOVW_UABS_FIRST..=R_AARCH64_MOVW_UABS_LAST => {
                    let shift = ((r_type - R_AARCH64_MOVW_UABS_FIRST) / 2) as u8 * 16;
                    let hole = match sym.as_str() {
                        "OFF" => 0,
                        "COL" => 1,
                        other => panic!("unexpected hole symbol {other}"),
                    };
                    movws.push(Movw { at: off as u32, hole, shift });
                }
                R_AARCH64_CALL26 | R_AARCH64_JUMP26 => {
                    assert_eq!(sym, "body", "unexpected branch reloc to {sym}");
                    splice_at = Some(off as u32);
                }
                other => panic!("unexpected reloc type {other} in {stencil}"),
            }
        }

        if let Some(at) = splice_at {
            assert!(frame.is_none(), "more than one loop frame stencil");
            frame = Some((code, at));
        } else {
            // A leaf op stencil: drop the trailing `ret` so copies can be spliced.
            assert!(
                code.ends_with(&AARCH64_RET),
                "op stencil {stencil} does not end in `ret`"
            );
            let body = code[..code.len() - 4].to_vec();
            ops.push((stencil.to_string(), body, movws));
        }
    }

    let (frame_code, splice_at) = frame.expect("no fold_loop frame stencil found");
    render(&ops, &frame_code, splice_at)
}

fn render(ops: &[(String, Vec<u8>, Vec<Movw>)], frame: &[u8], splice_at: u32) -> String {
    let mut s = String::new();
    s.push_str("// @generated by build.rs from cap/stencils.rs — do not edit.\n");
    s.push_str("pub const SUPPORTED: bool = true;\n\n");
    writeln!(s, "pub static FOLD_LOOP: Frame = Frame {{").unwrap();
    write_bytes(&mut s, "code", frame);
    writeln!(s, "    splice_at: {splice_at},").unwrap();
    s.push_str("};\n\n");

    for (name, code, holes) in ops {
        writeln!(s, "pub static {}: OpStencil = OpStencil {{", name.to_uppercase()).unwrap();
        write_bytes(&mut s, "code", code);
        s.push_str("    holes: &[\n");
        for h in holes {
            writeln!(
                s,
                "        Movw {{ at: {}, hole: {}, shift: {} }},",
                h.at, h.hole, h.shift
            )
            .unwrap();
        }
        s.push_str("    ],\n};\n\n");
    }
    s
}

fn write_bytes(s: &mut String, field: &str, bytes: &[u8]) {
    write!(s, "    {field}: &[").unwrap();
    for (i, b) in bytes.iter().enumerate() {
        if i % 12 == 0 {
            s.push_str("\n        ");
        }
        write!(s, "0x{b:02x}, ").unwrap();
    }
    s.push_str("\n    ],\n");
}

/// Stub statics for non-aarch64 targets so `cap/mod.rs` still type-checks.
fn emit_stub_statics(gen_path: &PathBuf) {
    let mut s = std::fs::read_to_string(gen_path).unwrap();
    s.push_str(
        "pub static FOLD_LOOP: Frame = Frame { code: &[], splice_at: 0 };\n\
         pub static OP_COUNT: OpStencil = OpStencil { code: &[], holes: &[] };\n\
         pub static OP_SUM_I16: OpStencil = OpStencil { code: &[], holes: &[] };\n\
         pub static OP_SUM_I32: OpStencil = OpStencil { code: &[], holes: &[] };\n\
         pub static OP_SUM_I64: OpStencil = OpStencil { code: &[], holes: &[] };\n",
    );
    std::fs::write(gen_path, s).unwrap();
}
