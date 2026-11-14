"""Login-node relay. Only render geometry and control messages enter this service."""
import argparse
import gzip
from collections import OrderedDict
import json
import math
import mimetypes
import os
import re
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from loads import WINDOWS, window_packet

BOUNDS = {"speed": (1, 30), "angle": (-20, 25), "flap": (0, 40),
          "aileron": (-25, 25), "elevator": (-25, 25), "rudder": (-25, 25)}
DEFAULT = dict(speed=30, angle=4, flap=0, aileron=0, elevator=0, rudder=0)


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def simplify_frame(data):
    """Simplify display polylines to 2 mm and 0.02 RGB; never touch solver data."""
    size = struct.unpack_from("<I", data)[0]
    meta = json.loads(data[4:4 + size])
    output = bytearray()
    chain = []

    def flush():
        if len(chain) < 2:
            return
        keep = {0, len(chain) - 1}
        pending = [(0, len(chain) - 1)]
        while pending:
            lo, hi = pending.pop()
            a, b = chain[lo], chain[hi]
            delta = [b[k] - a[k] for k in range(3)]
            length2 = sum(v * v for v in delta)
            worst, split = 1.0, None
            for i in range(lo + 1, hi):
                point = chain[i]
                t = max(0, min(1, sum((point[k] - a[k]) * delta[k] for k in range(3)) / max(length2, 1e-20)))
                error = sum((point[k] - a[k] - t * delta[k]) ** 2 for k in range(3)) / 0.002 ** 2
                color = max(abs(point[k] - a[k] - t * (b[k] - a[k])) for k in range(3, 6)) / 0.02
                error = max(error, color * color)
                if error > worst:
                    worst, split = error, i
            if split is not None:
                keep.add(split)
                pending.extend(((lo, split), (split, hi)))
        indices = sorted(keep)
        for lo, hi in zip(indices, indices[1:]):
            output.extend(struct.pack("<12f", *chain[lo], *chain[hi]))

    for segment in struct.iter_unpack("<12f", data[4 + size:]):
        a, b = segment[:6], segment[6:]
        if chain and (chain[-1][:3] != a[:3] or max(abs(chain[-1][k] - a[k]) for k in range(3, 6)) > 0.02):
            flush()
            chain = []
        if not chain:
            chain.append(a)
        chain.append(b)
    flush()
    meta["original_line_vertices"] = meta["line_vertices"]
    meta["line_vertices"] = len(output) // 24
    meta["display_tolerance_m"] = 0.002
    header = json.dumps(meta, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    return struct.pack("<I", len(header)) + header + output


class Relay:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.compressed = OrderedDict()
        path = directory / "desired.json"
        self.desired = json.loads(path.read_text()) if path.exists() else dict(
            sequence=1, reset=0, paused=False, controls=DEFAULT.copy())
        self.desired.setdefault("streamline_density", 2)
        self.save()

    def save(self):
        d = self.desired
        atomic(self.directory / "desired.json", json.dumps(d))
        values = [d["sequence"], d["reset"], int(d["paused"])] + [d["controls"][k] for k in BOUNDS] + [d["streamline_density"]]
        atomic(self.directory / "controls.txt", " ".join(map(str, values)) + "\n")

    def update(self, data):
        if not isinstance(data, dict) or set(data) - {"controls", "paused", "reset", "streamline_density"}:
            raise ValueError("Unknown control fields")
        if "streamline_density" in data:
            density = data["streamline_density"]
            if type(density) is not int or not 1 <= density <= 4:
                raise ValueError("Streamline density must be an integer from 1 to 4")
        controls = data.get("controls", {})
        if not isinstance(controls, dict) or set(controls) - BOUNDS.keys():
            raise ValueError("Unknown aircraft control")
        for key, value in controls.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Control values must be finite numbers")
            if not BOUNDS[key][0] <= value <= BOUNDS[key][1]:
                raise ValueError(f"{key} outside {BOUNDS[key]}")
        for key in ("paused", "reset"):
            if key in data and not isinstance(data[key], bool):
                raise ValueError(f"{key} must be boolean")
        with self.lock:
            self.desired["streamline_density"] = data.get("streamline_density", self.desired["streamline_density"])
            self.desired["controls"].update(controls)
            self.desired["paused"] = data.get("paused", self.desired["paused"])
            self.desired["reset"] += int(data.get("reset", False))
            self.desired["sequence"] += 1
            self.save()
            return json.loads(json.dumps(self.desired))


def make_handler(relay, static):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if len(args) >= 2 and str(args[1]) not in ("200", "304"):
                super().log_message(format, *args)

        def respond(self, status, data, kind="application/json", headers=None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            route = urlsplit(self.path).path
            compact = route == "/api/frame" and urlsplit(self.path).query == "compact=1"
            if route == "/api/status":
                with relay.lock:
                    state = json.loads(json.dumps(relay.desired))
                frame = relay.directory / "frame.bin"
                state["age"] = max(0, time.time() - frame.stat().st_mtime) if frame.exists() else None
                error = relay.directory / "error.txt"
                state["error"] = error.read_text() if error.exists() else None
                self.respond(200, json.dumps(state).encode())
                return
            if route == "/api/loads":
                try:
                    seconds = float(parse_qs(urlsplit(self.path).query).get("seconds", ["0.25"])[0])
                    if seconds not in WINDOWS:
                        raise ValueError("Unsupported history window")
                except ValueError as error:
                    self.respond(400, json.dumps({"error": str(error)}).encode())
                    return
                try:
                    path = relay.directory / "loads.bin"
                    with path.open("rb") as file:
                        stat = os.fstat(file.fileno())
                        etag = f'"loads-{stat.st_mtime_ns}-{stat.st_size}-{seconds}"'
                        if self.headers.get("If-None-Match") == etag:
                            self.respond(304, b"", headers={"ETag": etag})
                            return
                        data = file.read()
                    compressed = "gzip" in self.headers.get("Accept-Encoding", "")
                    key = (str(path), etag, compressed)
                    with relay.cache_lock:
                        if key not in relay.compressed:
                            packet = window_packet(data, seconds)
                            relay.compressed[key] = gzip.compress(packet, compresslevel=1) if compressed else packet
                            while len(relay.compressed) > 6:
                                relay.compressed.popitem(last=False)
                        packet = relay.compressed[key]
                    headers = {"ETag": etag, "Vary": "Accept-Encoding"}
                    if compressed:
                        headers["Content-Encoding"] = "gzip"
                    self.respond(200, packet, "application/octet-stream", headers)
                except FileNotFoundError:
                    self.respond(404, b'{"error":"Waiting for per-step load history"}')
                except (ValueError, KeyError, struct.error) as error:
                    self.respond(503, json.dumps({"error": str(error)}).encode())
                return
            if route == "/api/frame":
                path = relay.directory / "frame.bin"
            elif re.fullmatch(r"/api/mesh/[0-9]+", route):
                path = relay.directory / ("mesh-" + route.rsplit("/", 1)[-1] + ".bin")
            else:
                allowed = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js",
                           "/style.css": "style.css", "/charts.js": "charts.js", "/vendor/three.module.js": "vendor/three.module.js",
                           "/vendor/OrbitControls.js": "vendor/OrbitControls.js"}
                if route not in allowed:
                    self.respond(404, b'{"error":"Not found"}')
                    return
                path = static / allowed[route]
            try:
                # fstat and read the same inode: the worker publishes via atomic rename.
                with path.open("rb") as file:
                    stat = os.fstat(file.fileno())
                    etag = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if compact:
                        etag = etag[:-1] + '-compact"'
                    if self.headers.get("If-None-Match") == etag:
                        self.respond(304, b"", headers={"ETag": etag})
                        return
                    data = file.read()
                kind = "application/octet-stream" if route.startswith("/api/") else mimetypes.guess_type(path)[0]
                headers = {"ETag": etag, "Vary": "Accept-Encoding"}
                if "gzip" in self.headers.get("Accept-Encoding", "") and len(data) > 1024:
                    key = (str(path), etag)
                    with relay.cache_lock:
                        if key not in relay.compressed:
                            relay.compressed[key] = gzip.compress(simplify_frame(data) if compact else data, compresslevel=1)
                            while len(relay.compressed) > 6:
                                relay.compressed.popitem(last=False)
                        data = relay.compressed[key]
                    headers["Content-Encoding"] = "gzip"
                elif compact:
                    data = simplify_frame(data)
                self.respond(200, data, kind or "application/octet-stream", headers)
            except FileNotFoundError:
                self.respond(404, b'{"error":"Waiting for GPU worker"}')

        def do_POST(self):
            if urlsplit(self.path).path != "/api/controls":
                self.respond(404, b'{"error":"Not found"}')
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.respond(403, b'{"error":"Cross-origin control denied"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON control message")
                result = relay.update(json.loads(self.rfile.read(length)))
                self.respond(200, json.dumps(result).encode())
            except (ValueError, TypeError) as error:
                self.respond(400, json.dumps({"error": str(error)}).encode())
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10510)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    relay = Relay(args.directory.resolve())
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(relay, Path(__file__).resolve().parent))
    server.daemon_threads = True
    print(f"Wind3D live console on http://{args.bind}:{args.port}", flush=True)
    server.serve_forever()
