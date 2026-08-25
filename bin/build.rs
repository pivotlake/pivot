//! Make sure the frontend's `dist/` directory exists before compilation, so
//! the `include_dir!` that embeds the dashboard always has something to read -
//! even on a checkout where `npm run build` hasn't run yet. When the real
//! bundle is built, its files replace the placeholder and a rebuild re-embeds.

use std::path::Path;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let dist = Path::new(&manifest).join("../web/frontend/dist");

    if !dist.join("index.html").exists() {
        let _ = std::fs::create_dir_all(&dist);
        let _ = std::fs::write(
            dist.join("index.html"),
            "<!doctype html><meta charset=\"utf-8\"><title>pivotdb</title>\
             <body style=\"font-family:sans-serif;background:#0a0c16;color:#e8eaf6;padding:48px\">\
             <h1>pivotdb dashboard</h1>\
             <p>The frontend hasn't been built. Run \
             <code>npm --prefix web/frontend install &amp;&amp; npm --prefix web/frontend run build</code>, \
             then rebuild the server.</p></body>",
        );
    }

    // Re-embed when the built bundle changes.
    println!("cargo:rerun-if-changed=../web/frontend/dist");
}
