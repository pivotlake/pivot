from __future__ import annotations

import threading
import webbrowser

import click

from .data import load_perf_data
from .server import create_app


@click.command()
@click.argument("perf_data", default="perf.data", type=click.Path(exists=True))
@click.option("--binary", default=None, type=click.Path(exists=True),
              help="Path to binary to disassemble (auto-detected if omitted).")
@click.option("--port", default=3000, type=int, help="HTTP port to serve on.")
@click.option("--host", default="127.0.0.1", help="HTTP bind host.")
@click.option("--no-browser", is_flag=True, help="Do not auto-open browser.")
def main(perf_data: str, binary: str | None, port: int, host: str, no_browser: bool) -> None:
    """Serve an interactive web UI for AMD IBS perf data on localhost:3000."""
    click.echo(f"Loading {perf_data}...", err=True)
    data = load_perf_data(perf_data, binary=binary)
    click.echo(
        f"Loaded {len(data.summaries)} functions "
        f"({data.total_cycles} cycle samples, {data.total_uw} IBS samples)",
        err=True,
    )

    app = create_app(data)
    url = f"http://{host}:{port}"

    from .server import _BUILD_DIR
    if not (_BUILD_DIR / "index.html").exists():
        click.echo(
            f"WARNING: frontend not built at {_BUILD_DIR}.\n"
            f"  cd {_BUILD_DIR.parent} && npm install && npm run build",
            err=True,
        )

    if not no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    click.echo(f"ticks: serving at {url}", err=True)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
