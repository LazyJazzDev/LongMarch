"""PTY checks for the standalone recorder: test_messages.py <demo_tui>."""
import pathlib
import select
import subprocess
import sys
import unittest

DEMO = str(pathlib.Path(sys.argv[1]).resolve())
# Reuse the isolated PTY harness; it consumes the executable argument.
from test_pty import Terminal


class MessageDemoTest(unittest.TestCase):
    def start(self, **kwargs):
        return Terminal([DEMO, "--dump"], **kwargs)

    def ready(self, terminal):
        terminal.until(lambda: b"Enter send" in terminal.output)
        self.assertIn(b"\x1b[?2004h", terminal.output)
        self.assertIn(b"\x1b[48;2;43;50;65m", terminal.output)

    def settle(self, terminal):
        while select.select([terminal.master], [], [], 0.03)[0]:
            terminal.read(0)

    def finish(self, terminal):
        terminal.send(b"\x03")
        terminal.finish(0)
        self.assertIn(b"\x1b[?2004l", terminal.output)
        self.assertIn(b"\x1b[<u", terminal.output)
        return terminal.output.split(b"\x1b[?1049l")[-1].replace(b"\r\n", b"\n").decode()

    def test_edit_click_shift_enter_send_and_clear(self):
        with self.start(columns=60, rows=14) as terminal:
            self.ready(terminal)
            terminal.send(b"abc\x1b[D\x7f\x1b[3~Z")  # aZ
            terminal.send(b"\x1b[<0;3;13MX\x1b[13;2uY\r")
            terminal.until(lambda: b"1 recorded" in terminal.output)
            terminal.send(b"next\r")
            terminal.until(lambda: b"2 recorded" in terminal.output)
            result = self.finish(terminal)
            self.assertEqual(result, "--- Message 1 ---\nX\nYaZ\n--- Message 2 ---\nnext\n")

    def test_unicode_bracketed_paste_and_vertical_navigation(self):
        with self.start(columns=60, rows=14) as terminal:
            self.ready(terminal)
            terminal.send(b"\x1b[200~" + "中a\nxy".encode() + b"\x1b[201~")
            terminal.until(lambda: "中a".encode() in terminal.output)
            self.assertNotIn(b"1 recorded", terminal.output)
            terminal.send(b"\x1b[AZ\x1b[B\x7f\r")
            terminal.until(lambda: b"1 recorded" in terminal.output)
            self.assertEqual(self.finish(terminal), "--- Message 1 ---\n中Za\nx\n")

    def test_xterm_shift_enter_and_legacy_alternative_newlines(self):
        with self.start(columns=60, rows=14) as terminal:
            self.ready(terminal)
            terminal.send(b"one\x1b[27;2;13~two\x1b\rthree\nfour\r")
            terminal.until(lambda: b"1 recorded" in terminal.output)
            self.assertEqual(self.finish(terminal), "--- Message 1 ---\none\ntwo\nthree\nfour\n")

    def test_history_scroll_resize_and_draft_preservation(self):
        with self.start(columns=60, rows=14) as terminal:
            self.ready(terminal)
            terminal.send(b"".join(f"message-{i:02}\r".encode() for i in range(15)))
            terminal.until(lambda: b"15 recorded" in terminal.output and b"message-14" in terminal.output)
            self.settle(terminal)
            terminal.output = b""
            terminal.send(b"\x1b[5~")
            terminal.until(lambda: b"History" in terminal.output)
            self.settle(terminal)
            self.assertNotIn(b"message-14", terminal.output)
            terminal.output = b""
            terminal.send(b"\x1b[<65;3;3M" * 10)
            terminal.until(lambda: b"message-14" in terminal.output)
            terminal.send(b"\x1b[200~long draft that wraps across several lines\nsecond line\x1b[201~")
            terminal.until(lambda: b"long draft" in terminal.output)
            self.settle(terminal)
            terminal.output = b""
            terminal.resize(20, 8)
            terminal.until(lambda: b"second line" in terminal.output)
            self.settle(terminal)
            terminal.output = b""
            terminal.resize(8, 3)
            terminal.until(lambda: b"Resize" in terminal.output)
            terminal.resize(60, 14)
            terminal.until(lambda: b"second line" in terminal.output)
            terminal.send(b"\r")
            terminal.until(lambda: b"16 recorded" in terminal.output)
            result = self.finish(terminal)
            for i in range(15):
                self.assertIn(f"message-{i:02}\n", result)
            self.assertTrue(result.endswith("long draft that wraps across several lines\nsecond line\n"))

    def test_input_cursor_blinks(self):
        with self.start(columns=60, rows=14) as terminal:
            self.ready(terminal)
            cursor_color = b"\x1b[48;2;226;232;240m"
            terminal.until(lambda: cursor_color in terminal.output)
            terminal.output = b""
            terminal.until(lambda: b"\x1b[48;2;43;50;65m" in terminal.output, timeout=2)
            self.assertNotIn(cursor_color, terminal.output)
            terminal.until(lambda: cursor_color in terminal.output, timeout=2)
            self.finish(terminal)

    def test_nonterminal_explains_requirement(self):
        result = subprocess.run([DEMO], input="", text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn("interactive terminal", result.stderr)
        self.assertNotIn("\x1b", result.stdout)


if __name__ == "__main__":
    unittest.main()
