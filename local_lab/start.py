"""Start the JEE lab as a detached local process and verify its health."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from storage import atomic_write


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_lab/runs"
SERVER = ROOT / "local_lab/server.py"
PYTHON = Path(sys.executable)
URL = "http://127.0.0.1:8766/"


def healthy() -> bool:
    try:
        with urlopen(URL + "api/health", timeout=1) as response:
            return response.headers.get("Server", "").startswith("JEEInteractive/") and (
                json.load(response).get("status") == "ready"
            )
    except (URLError, TimeoutError, ValueError, OSError):
        return False


def main() -> None:
    if healthy():
        print(f"JEE lab already available at {URL}")
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / "server-stdout.log").open("a") as out, (RUNS / "server-stderr.log").open("a") as err:
        process = subprocess.Popen(
            [str(PYTHON), str(SERVER), "--port", "8766"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True,
        )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if healthy():
            atomic_write(RUNS / "server-last-start.json", json.dumps({
                "pid": process.pid, "started_unix": time.time(), "url": URL,
                "note": "Verify liveness after any restart; this is not a login service.",
            }, indent=2) + "\n")
            print(f"JEE lab available at {URL} (PID {process.pid})")
            return
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError(f"JEE lab did not start; inspect {RUNS / 'server-stderr.log'}")


if __name__ == "__main__":
    main()
