"""Serve only the two security-test pages on loopback; no API or secrets exposed."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
PAGES = {
    "/tests/ui_security.html": BACKEND / "tests/ui_security.html",
    "/ui/index.html": BACKEND / "ui/index.html",
}


class TestPages(BaseHTTPRequestHandler):
    def do_GET(self):
        path = PAGES.get(self.path)
        if path is None:
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    with HTTPServer(("127.0.0.1", 8765), TestPages) as server:
        print("Abra http://127.0.0.1:8765/tests/ui_security.html; Ctrl+C encerra.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
