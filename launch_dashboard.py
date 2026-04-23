#!/usr/bin/env python3
"""Start the Streamlit dashboard and open it in a browser.

Rationale
---------
``streamlit run dashboard.py`` is fine on its own, but we've set
``server.headless = true`` in ``.streamlit/config.toml`` so Streamlit does
not auto-open any browser (useful on servers, required on headless VMs).
This launcher restores "just double-click and go" behaviour, with one
extra convenience: on macOS it opens **Safari specifically** rather than
relying on whatever the system default happens to be.

Usage
-----
    python launch_dashboard.py [--port 8501] [--browser default|safari|none]
"""
from __future__ import annotations

import argparse
import platform
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
DASHBOARD = ROOT / "dashboard.py"


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _open_browser(url: str, choice: str) -> None:
    if choice == "none":
        return
    system = platform.system()
    if choice == "safari" or (choice == "default" and system == "Darwin"):
        if system != "Darwin":
            print(f"Safari was requested but this is {system}; falling back to default browser.")
            webbrowser.open(url)
            return
        # Safari ships with every macOS install, so this is safe. If the user
        # has removed it somehow, fall back to the OS default.
        try:
            subprocess.run(["open", "-a", "Safari", url], check=True)
            print(f"Opened {url} in Safari.")
            return
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"Could not open Safari ({exc}); falling back to default browser.")
    webbrowser.open(url)
    print(f"Opened {url} in your default browser.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--browser",
        choices=("default", "safari", "none"),
        default="default",
        help=(
            "'default' = Safari on macOS, system default elsewhere. "
            "'safari'  = force Safari (macOS only). "
            "'none'    = don't open a browser at all."
        ),
    )
    args = parser.parse_args()

    if not DASHBOARD.exists():
        print(f"dashboard.py not found at {DASHBOARD}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
        "--server.port", str(args.port),
        "--server.headless", "true",
    ]
    print(f"Starting dashboard on http://{args.host}:{args.port} …")
    # Inherit stdout/stderr so the user sees Streamlit's startup output.
    proc = subprocess.Popen(cmd, cwd=ROOT)

    # Hand Ctrl+C to the child and wait for it to shut down cleanly.
    def _forward_sigint(_signum, _frame):
        proc.send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT, _forward_sigint)

    try:
        if _wait_for_server(args.host, args.port):
            _open_browser(f"http://{args.host}:{args.port}", args.browser)
        else:
            print(
                f"Streamlit did not start listening on port {args.port} in time. "
                "Check the output above for errors.",
                file=sys.stderr,
            )
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
