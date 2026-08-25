//! Serves the built React frontend, embedded in the binary at compile time.
//! Unknown paths fall back to `index.html` so the client-side router owns deep
//! links (single-page-app routing).

use axum::http::{StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use include_dir::{Dir, include_dir};

/// The built frontend, embedded at compile time. `build.rs` guarantees the
/// directory exists (with at least a placeholder `index.html`) so this compiles
/// even before `npm run build`.
static UI_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../web/frontend/dist");

pub(super) async fn ui(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    // Serve the requested asset; fall back to index.html for SPA routes.
    let (bytes, name) = match read_asset(path) {
        Some(bytes) => (bytes, path),
        None => match read_asset("index.html") {
            Some(bytes) => (bytes, "index.html"),
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    "dashboard not built - run `npm --prefix web/frontend run build`",
                )
                    .into_response();
            }
        },
    };
    ([(header::CONTENT_TYPE, content_type(name))], bytes).into_response()
}

fn read_asset(path: &str) -> Option<Vec<u8>> {
    // Reject path traversal before touching the embedded tree.
    if path.split('/').any(|seg| seg == "..") {
        return None;
    }
    UI_DIR.get_file(path).map(|f| f.contents().to_vec())
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    }
}
