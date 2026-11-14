import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from server import Relay, make_handler


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
        self.assertEqual((self.directory / "controls.txt").read_text().split(), ["2", "1", "1", "18", "5", "20", "0", "0", "0"])
        self.assertEqual(Relay(self.directory).desired, state)

    def test_invalid_commands_do_not_mutate(self):
        for data in ({"controls": {"speed": 0}}, {"controls": {"angle": float("nan")}},
                     {"controls": {"flap": 41}}, {"controls": {"bad": 2}}, {"paused": 1}, {"reset": "yes"}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post(data)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.relay.desired["sequence"], 1)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post({"reset": True}, "http://untrusted.invalid")
        self.assertEqual(error.exception.code, 403)

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


if __name__ == "__main__":
    unittest.main()
