"""Launch orchestration: uvicorn thread + streamlit subprocess."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen


def _wait_for_api(api_port: int, timeout_seconds: float = 10.0) -> bool:
    """Poll the FastAPI health path until it responds or timeout expires."""
    deadline = time.time() + timeout_seconds
    url = f"http://localhost:{api_port}/api/repos"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(0.2)
    return False


def launch(db_path: str | None = None, api_port: int = 8000) -> None:
    """Start FastAPI (uvicorn) in a daemon thread and Streamlit as a subprocess."""
    import uvicorn

    from git_xrays.web.api import app

    app.state.db_path = db_path
    streamlit_port = api_port + 1

    def _run_api() -> None:
        uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="warning")

    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()

    if not _wait_for_api(api_port):
        print(f"Failed to start API server on http://localhost:{api_port}", file=sys.stderr)
        sys.exit(1)

    dashboard_path = str(
        __import__("pathlib").Path(__file__).parent / "dashboard.py"
    )

    print(f"API server:  http://localhost:{api_port}")
    print(f"Dashboard:   http://localhost:{streamlit_port}")
    print()

    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                dashboard_path,
                "--server.port", str(streamlit_port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--", f"--api-url=http://localhost:{api_port}",
            ],
            check=False,
        )
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\nShutting down.")
        sys.exit(0)
