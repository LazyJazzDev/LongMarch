import gzip
import json
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from server import Relay, make_handler, simplify_frame


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.relay = Relay(self.directory)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.relay, Path(__file__).parent))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def post(self, value, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(self.url + "/api/controls", data=json.dumps(value).encode(), headers=headers)
        return urllib.request.urlopen(request)

    def test_controls_and_persistence(self):
        with self.post({"controls": {"speed": 18, "flap": 20}, "paused": True, "reset": True}) as response:
            state = json.load(response)
        self.assertEqual(state["sequence"], 2)
        self.assertEqual(state["reset"], 1)
        self.assertEqual((self.directory / "controls.txt").read_text().split(), ["2", "1", "1", "18", "4", "20", "0", "0", "0", "2"])
        self.assertEqual(Relay(self.directory).desired, state)

    def test_invalid_commands_do_not_mutate(self):
        for data in ({"controls": {"speed": 0}}, {"controls": {"angle": float("nan")}},
                     {"controls": {"flap": 41}}, {"controls": {"bad": 2}}, {"paused": 1}, {"reset": "yes"}, {"streamline_density": 0},
                     {"streamline_density": 5}, {"streamline_density": 1.5}, {"streamline_density": True}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post(data)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.relay.desired["sequence"], 1)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post({"reset": True}, "http://untrusted.invalid")
        self.assertEqual(error.exception.code, 403)

    def test_density_preserves_simulation_controls(self):
        before = self.relay.desired.copy()
        with self.post({"streamline_density": 4}) as response:
            state = json.load(response)
        self.assertEqual(state["controls"], before["controls"])
        self.assertEqual(state["reset"], before["reset"])
        self.assertEqual(state["paused"], before["paused"])
        self.assertEqual(state["streamline_density"], 4)
        self.assertEqual(Relay(self.directory).desired, state)
        self.assertEqual((self.directory / "controls.txt").read_text().split()[-1], "4")

    def test_compact_representation_has_distinct_etag(self):
        header = json.dumps(dict(version=1, line_vertices=6)).encode()
        header += b" " * (-len(header) % 4)
        points = [(i, 0, 0, 0.2, 0.4, 0.6) for i in range(4)]
        body = b"".join(struct.pack("<12f", *a, *b) for a, b in zip(points, points[1:]))
        (self.directory / "frame.bin").write_bytes(struct.pack("<I", len(header)) + header + body)
        with urllib.request.urlopen(self.url + "/api/frame") as response:
            original_etag = response.headers["ETag"]
        with urllib.request.urlopen(self.url + "/api/frame?compact=1") as response:
            compact_etag = response.headers["ETag"]
            data = response.read()
        self.assertNotEqual(original_etag, compact_etag)
        size = struct.unpack_from("<I", data)[0]
        self.assertEqual(json.loads(data[4:4 + size])["line_vertices"], 2)
        request = urllib.request.Request(self.url + "/api/frame?compact=1", headers={"If-None-Match": compact_etag})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 304)

    def test_load_window_transport(self):
        from loads import ROW
        header = json.dumps(dict(version=1, stride=44, count=1000)).encode()
        body = b"".join(ROW.pack(i, i * .001, .001, i, 2, 3, 4, 5, 6) for i in range(1, 1001))
        (self.directory / "loads.bin").write_bytes(struct.pack("<I", len(header)) + header + body)
        request = urllib.request.Request(self.url + "/api/loads?seconds=0.05", headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.headers["Content-Encoding"], "gzip")
            etag = response.headers["ETag"]
            data = gzip.decompress(response.read())
        length = struct.unpack_from("<I", data)[0]
        meta = json.loads(data[4:4 + length])
        rows = list(ROW.iter_unpack(data[4 + length:]))
        self.assertGreaterEqual(len(rows), 50)
        self.assertTrue(meta["contiguous"])
        self.assertEqual(rows[-1][0], 1000)
        request = urllib.request.Request(self.url + "/api/loads?seconds=0.05", headers={"If-None-Match": etag})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 304)
        with urllib.request.urlopen(self.url + "/api/loads?seconds=1") as response:
            self.assertNotEqual(response.headers["ETag"], etag)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + "/api/loads?seconds=nan")
        self.assertEqual(error.exception.code, 400)

    def test_only_display_data_is_exposed(self):
        (self.directory / "frame.bin").write_bytes(b"test-frame")
        with urllib.request.urlopen(self.url + "/api/frame") as response:
            self.assertEqual(response.read(), b"test-frame")
            etag = response.headers["ETag"]
        request = urllib.request.Request(self.url + "/api/frame", headers={"If-None-Match": etag})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 304)
        for path in ("/controls.txt", "/../desired.json", "/api/mesh/../../desired.json", "/frame.bin"):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(self.url + path)
            self.assertEqual(error.exception.code, 404)


class CompactFrameTest(unittest.TestCase):
    def test_curve_and_color_features_survive(self):
        points = [(i / 100, 0.2 if i == 50 else 0, 0, 1.0 if i == 75 else 0.2, 0.4, 0.6) for i in range(101)]
        body = b"".join(struct.pack("<12f", *a, *b) for a, b in zip(points, points[1:]))
        header = json.dumps(dict(version=1, line_vertices=200)).encode()
        header += b" " * (-len(header) % 4)
        compact = simplify_frame(struct.pack("<I", len(header)) + header + body)
        size = struct.unpack_from("<I", compact)[0]
        meta = json.loads(compact[4:4 + size])
        self.assertLess(meta["line_vertices"], 30)
        self.assertEqual(meta["original_line_vertices"], 200)
        vertices = list(struct.iter_unpack("<6f", compact[4 + size:]))
        self.assertTrue(any(abs(v[1] - 0.2) < 1e-6 for v in vertices))
        self.assertTrue(any(v[3] == 1 for v in vertices))
        for point in points:
            errors = []
            for a, b in zip(vertices[::2], vertices[1::2]):
                delta = [b[k] - a[k] for k in range(3)]
                t = max(0, min(1, sum((point[k] - a[k]) * delta[k] for k in range(3)) / max(sum(v*v for v in delta), 1e-20)))
                errors.append(sum((point[k] - a[k] - t*delta[k])**2 for k in range(3)))
            self.assertLessEqual(min(errors), 0.002**2)


if __name__ == "__main__":
    unittest.main()
