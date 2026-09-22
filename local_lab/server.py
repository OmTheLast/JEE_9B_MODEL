"""Loopback-only browser interface for exploratory JEE model testing."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from storage import atomic_write


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local_lab/runs"
UI = ROOT / "local_lab/app.html"
WORKER = ROOT / "local_lab/worker.py"
MAX_BODY = 12 * 1024 * 1024
MAX_IMAGE = 8 * 1024 * 1024
CHOICES = {"base", "reliable01-step60", "coverage16-step60", "data60-step60"}
SUBJECTS = {"auto", "mathematics", "physics", "chemistry"}
EXAMS = {"auto", "main", "advanced"}
ANSWER_TYPES = {"auto", "options", "numeric"}
active = threading.Lock()


def save(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_image(path: Path, image: bytes) -> None:
    temporary = path.with_name(".question-" + uuid.uuid4().hex + ".png")
    try:
        with temporary.open("wb") as stream:
            stream.write(image)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def decode_image(data: str) -> bytes:
    if not data:
        return b""
    if not data.startswith("data:image/") or "," not in data:
        raise ValueError("Upload a PNG or JPEG image")
    header, encoded = data.split(",", 1)
    if header not in ("data:image/png;base64", "data:image/jpeg;base64"):
        raise ValueError("Only PNG and JPEG images are supported")
    if len(encoded) > MAX_IMAGE * 4 // 3 + 16:
        raise ValueError("Image exceeds 8 MB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise ValueError("Invalid image encoding") from error
    if len(raw) > MAX_IMAGE:
        raise ValueError("Image exceeds 8 MB")
    try:
        with Image.open(io.BytesIO(raw)) as picture:
            if picture.format not in ("PNG", "JPEG") or picture.width * picture.height > 20_000_000:
                raise ValueError("Image format or dimensions are unsupported")
            picture.load()
            picture = picture.convert("RGB")
            picture.thumbnail((2400, 2400))
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Image could not be decoded") from error


def validate_request(value: object) -> tuple[dict, bytes]:
    if not isinstance(value, dict):
        raise ValueError("Request must be an object")
    question = value.get("question", "")
    if not isinstance(question, str) or len(question) > 20_000:
        raise ValueError("Question text must be at most 20,000 characters")
    image_data = value.get("image", "")
    if not isinstance(image_data, str):
        raise ValueError("Invalid image field")
    image = decode_image(image_data)
    if not question.strip() and not image:
        raise ValueError("Enter a question or attach an image")
    choice = value.get("model", "base")
    subject = value.get("subject", "auto")
    exam = value.get("exam", "auto")
    answer_type = value.get("answer_type", "auto")
    tokens = value.get("max_tokens", 2048)
    if choice not in CHOICES or subject not in SUBJECTS or exam not in EXAMS or answer_type not in ANSWER_TYPES:
        raise ValueError("Invalid model or question setting")
    if type(tokens) is not int or tokens not in (512, 1024, 2048, 4096):
        raise ValueError("Token budget must be 512, 1024, 2048 or 4096")
    return {
        "question": question.strip(),
        "model": choice,
        "subject": subject,
        "exam": exam,
        "answer_type": answer_type,
        "max_tokens": tokens,
        "created_unix": time.time(),
        "purpose": "user_supplied_exploratory_interaction_not_training_or_evaluation",
    }, image


class Handler(BaseHTTPRequestHandler):
    server_version = "JEEInteractive/0.1"

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed_hosts:
            return False
        if origin and origin not in {"http://" + host}:
            return False
        return True

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict | list) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._local_request():
            self._json(403, {"error": "Local browser only"})
            return
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(200, {"status": "ready", "busy": active.locked()})
        elif path == "/api/runs":
            rows = []
            for run in DATA.glob("*/"):
                try:
                    request = json.loads((run / "request.json").read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                result = run / "result.json"
                halted = run / "HALTED.json"
                state = "incomplete"
                if halted.exists():
                    state = "halted"
                elif result.exists():
                    try:
                        outcome = json.loads(result.read_text())
                        state = ("token cap" if outcome.get("finish_reason") == "length"
                                 else outcome.get("status", "completed"))
                    except (OSError, json.JSONDecodeError):
                        state = "result unreadable"
                rows.append({
                    "id": run.name,
                    "question": request["question"][:160],
                    "model": request["model"],
                    "created_unix": request["created_unix"],
                    "status": state,
                })
            rows.sort(key=lambda row: row["created_unix"], reverse=True)
            self._json(200, rows[:30])
        elif path.startswith("/api/runs/"):
            run_id = path.removeprefix("/api/runs/")
            try:
                uuid.UUID(run_id)
            except ValueError:
                self._json(404, {"error": "Unknown run"})
                return
            run = DATA / run_id
            if not (run / "request.json").exists():
                self._json(404, {"error": "Unknown run"})
                return
            response = {"id": run_id, "request": json.loads((run / "request.json").read_text())}
            for key, name in (("result", "result.json"), ("inprogress", "inprogress.json"), ("halted", "HALTED.json")):
                file = run / name
                if file.exists():
                    response[key] = json.loads(file.read_text())
            self._json(200, response)
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._local_request():
            self._json(403, {"error": "Local browser only"})
            return
        if urlparse(self.path).path != "/api/solve":
            self._json(404, {"error": "Not found"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self._json(415, {"error": "Expected JSON"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= MAX_BODY:
                raise ValueError("Request exceeds 12 MB or is empty")
            value = json.loads(self.rfile.read(size))
            request, image = validate_request(value)
        except (ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
            return
        if not active.acquire(blocking=False):
            self._json(409, {"error": "Another local model run is active"})
            return
        try:
            run_id = str(uuid.uuid4())
            run = DATA / run_id
            run.mkdir(parents=True, exist_ok=False)
            save(run / "request.json", request)
            if image:
                save_image(run / "question.png", image)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write((json.dumps({"type": "accepted", "run_id": run_id}) + "\n").encode())
            self.wfile.flush()
            with (run / "worker-stderr.log").open("w") as stderr:
                process = subprocess.Popen(
                    [sys.executable, str(WORKER), "--run", str(run)],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                )
                disconnected = False
                assert process.stdout is not None
                for line in process.stdout:
                    if not disconnected:
                        try:
                            self.wfile.write(line.encode())
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            disconnected = True
                returncode = process.wait()
            save(run / "transport.json", {
                "worker_pid": process.pid, "worker_returncode": returncode,
                "client_disconnected": disconnected, "finished_unix": time.time(),
            })
            if returncode and not (run / "HALTED.json").exists():
                save(run / "HALTED.json", {"status": "halted", "error": "Worker exited; inspect worker-stderr.log"})
            if returncode and not disconnected:
                self.wfile.write((json.dumps({"type": "error", "message": "Worker failed; saved run for inspection"}) + "\n").encode())
                self.wfile.flush()
        finally:
            active.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Port must be between 1024 and 65535")
    DATA.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"JEE interactive test: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("JEE interactive server stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
