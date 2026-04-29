from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, send_from_directory

from .data import LoadedPerfData, serialize_function, serialize_summaries

# CRA build output (run `npm run build` in ticks/frontend/).
_BUILD_DIR = (Path(__file__).resolve().parent.parent / "frontend" / "build").resolve()


def create_app(data: LoadedPerfData) -> Flask:
    # Vite emits the bundle under build/assets/. The catch-all `/<path:asset>`
    # route below serves any file inside build/, including assets/.
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        index_html = _BUILD_DIR / "index.html"
        if not index_html.exists():
            return (
                "<h1>ticks: frontend not built</h1>"
                "<p>Run <code>cd ticks/frontend && npm install && npm run build</code> "
                "and restart the server.</p>",
                500,
            )
        return send_from_directory(str(_BUILD_DIR), "index.html")

    @app.route("/<path:asset>")
    def asset(asset):
        # Serve top-level CRA assets (favicon, manifest, etc.) from build/.
        candidate = (_BUILD_DIR / asset).resolve()
        try:
            candidate.relative_to(_BUILD_DIR)
        except ValueError:
            abort(404)
        if candidate.is_file():
            return send_from_directory(str(_BUILD_DIR), asset)
        # Fall back to SPA index for client-side routes.
        if (_BUILD_DIR / "index.html").exists():
            return send_from_directory(str(_BUILD_DIR), "index.html")
        abort(404)

    @app.route("/api/summary")
    def summary():
        return jsonify({
            "perf_data": data.perf_data,
            "binary": data.binary,
            "total_unweighted": data.total_uw,
            "total_weighted": data.total_w,
            "total_cycles": data.total_cycles,
            "skipped_lines": data.skipped_lines,
            "cache_summary": data.cache_summary,
            "pf_summary": data.pf_summary,
            "functions": serialize_summaries(data),
        })

    @app.route("/api/function/<path:name>")
    def function(name):
        return jsonify(serialize_function(data, name))

    return app
