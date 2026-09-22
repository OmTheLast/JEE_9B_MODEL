"""Public-package checks that do not download or load model weights."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

import server
from worker import MODEL_CHOICES, parse_final


class LocalLabTests(unittest.TestCase):
    def test_public_model_names_exist(self):
        release = json.loads((server.ROOT / "release.json").read_text())
        self.assertIsNone(MODEL_CHOICES["base"])
        self.assertTrue(set(MODEL_CHOICES.values()) - {None} <= set(release["adapters"]))

    def test_image_and_request_boundary(self):
        image = Image.new("RGB", (15, 12), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        request, normalized = server.validate_request({"question": " ", "image": encoded})
        self.assertEqual(request["model"], "base")
        self.assertEqual(Image.open(io.BytesIO(normalized)).size, (15, 12))
        with self.assertRaisesRegex(ValueError, "question or attach"):
            server.validate_request({"question": " "})
        with self.assertRaisesRegex(ValueError, "Token budget"):
            server.validate_request({"question": "1+1", "max_tokens": True})

    def test_final_answer_check_is_format_only(self):
        self.assertEqual(parse_final('FINAL_ANSWER: {"type":"numeric","value":"9"}')["status"], "parseable")
        self.assertEqual(parse_final("No final line")["status"], "missing_or_multiple_final_lines")

    def test_loopback_and_history_endpoint(self):
        previous = server.DATA
        with tempfile.TemporaryDirectory() as directory:
            server.DATA = Path(directory)
            http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{http.server_port}"
            try:
                with urlopen(base + "/api/health") as response:
                    self.assertEqual(json.load(response), {"status": "ready", "busy": False})
                with urlopen(base + "/api/runs") as response:
                    self.assertEqual(json.load(response), [])
                foreign = Request(base + "/api/health", headers={"Origin": "http://evil.example"})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(foreign)
                self.assertEqual(raised.exception.code, 403)
            finally:
                http.shutdown()
                http.server_close()
                thread.join(timeout=2)
                server.DATA = previous


if __name__ == "__main__":
    unittest.main()
