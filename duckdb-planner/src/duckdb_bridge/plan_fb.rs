//! FlatBuffers reader for the logical plan IR.
//!
//! `flatc` (run from `build.rs`) generates `plan_generated.rs` from
//! `schema/plan.fbs` into `OUT_DIR`; this module includes it and re-exports the
//! `pivot.plan` namespace flat so the rest of the crate refers to it as
//! `plan_fb::PlanNode`, `plan_fb::OperatorKind`, etc.

#![allow(
    clippy::all,
    clippy::pedantic,
    unused_imports,
    dead_code,
    non_snake_case,
    mismatched_lifetime_syntaxes
)]

include!(concat!(env!("OUT_DIR"), "/plan_generated.rs"));

pub use self::pivot::plan::*;
