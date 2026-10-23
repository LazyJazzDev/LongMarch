"""POSIX PTY integration checks: test_pty.py <probe> [demo ...].

Runs isolated pseudo-terminals; does not change the user's terminal. Optional
hello demos must support --backend metal --frames 1 (macOS smoke tests).
"""
import fcntl
import os
import pathlib
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import tempfile
import time
import unittest

PROBE = str(pathlib.Path(sys.argv[1]).resolve())
DEMOS = [str(pathlib.Path(p).resolve()) for p in sys.argv[2:]]
del sys.argv[1:]


class Terminal:
    def __init__(self, command=None, columns=120, rows=24, term="xterm-256color"):
        self.directory = tempfile.TemporaryDirectory(prefix="longmarch-tui-")
        self.master, self.slave = pty.openpty()
        self.resize(columns, rows)
        self.original = termios.tcgetattr(self.slave)
        self.output = b""
        self.process = subprocess.Popen(
            command or [PROBE], stdin=self.slave, stdout=self.slave,
            stderr=self.slave, env=dict(os.environ, TERM=term),
            start_new_session=True, cwd=self.directory.name,
        )

    def resize(self, columns, rows):
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

    def read(self, seconds=0.05):
        if select.select([self.master], [], [], seconds)[0]:
            self.output += os.read(self.master, 65536)

    def until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"PTY timeout; output={self.output[-3000:]!r}")
            self.read()

    def ready(self):
        self.until(lambda: b"[1/20]" in self.output)
        for control in (b"\x1b[?1049h", b"\x1b[?25l", b"\x1b[?1000h", b"\x1b[?1006h"):
            assert control in self.output
        assert not termios.tcgetattr(self.slave)[3] & (termios.ICANON | termios.ECHO | termios.ISIG)

    def send(self, data):
        os.write(self.master, data)

    def finish(self, expected, interactive=True):
        self.until(lambda: self.process.poll() is not None, timeout=30)
        while select.select([self.master], [], [], 0)[0]:
            self.read(0)
        assert self.process.returncode == expected, (self.process.returncode, self.output[-3000:])
        restored = termios.tcgetattr(self.slave)
        original = self.original.copy()
        # Darwin/BSD sets PENDIN when returning to canonical input. This is a
        # kernel reprocessing flag, not a changed echo/signal/input configuration.
        pending = getattr(termios, "PENDIN", 0)
        restored[3] &= ~pending
        original[3] &= ~pending
        assert restored == original, "termios configuration was not restored"
        if interactive:
            for control in (b"\x1b[?1006l", b"\x1b[?1000l", b"\x1b[?25h", b"\x1b[?1049l"):
                assert control in self.output
        else:
            assert b"\x1b[?1049h" not in self.output

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        os.close(self.master)
        os.close(self.slave)
        self.directory.cleanup()


class PtyMenuTest(unittest.TestCase):
    def test_keyboard_highlight_and_confirm(self):
        with Terminal() as terminal:
            terminal.ready()
            terminal.send(b"\x1b[B\x1b[B\x1b[A")
            terminal.until(lambda: b"[2/20]" in terminal.output)
            self.assertIn(b"\x1b[7m> item1", terminal.output)
            terminal.send(b"\r")
            terminal.finish(11)

    def test_fragmented_arrow_and_mouse(self):
        with Terminal() as terminal:
            terminal.ready()
            for chunk in (b"\x1b", b"[", b"B"):
                terminal.send(chunk)
                terminal.read(0.02)
            terminal.until(lambda: b"[2/20]" in terminal.output)
            for chunk in (b"\x1b[<", b"0;4;", b"5M"):
                terminal.send(chunk)
                terminal.read(0.02)
            terminal.until(lambda: b"[3/20]" in terminal.output)
            terminal.send(b"\r")
            terminal.finish(12)

    def test_mouse_single_double_click_and_release(self):
        with Terminal() as terminal:
            terminal.ready()
            terminal.send(b"\x1b[<0;8;7M\x1b[<0;8;7m")
            terminal.until(lambda: b"[5/20]" in terminal.output)
            self.assertIsNone(terminal.process.poll())
            terminal.send(b"\x1b[<0;8;7M")
            terminal.finish(14)

    def test_wheel(self):
        with Terminal() as terminal:
            terminal.ready()
            terminal.send(b"\x1b[<65;1;3M\x1b[<65;1;3M\x1b[<64;1;3M\r")
            terminal.finish(11)

    def test_escape_ctrl_c_ctrl_d_restore_modes(self):
        for key in (b"\x1b", b"\x03", b"\x04"):
            with self.subTest(key=key), Terminal() as terminal:
                terminal.ready()
                terminal.send(key)
                terminal.finish(0)

    def test_resize_two_rows_and_vertical_scroll(self):
        with Terminal() as terminal:
            terminal.ready()
            terminal.output = b""
            terminal.resize(20, 2)
            terminal.until(lambda: b"\x1b[2;1H" in terminal.output)
            self.assertNotIn(b"TUI integration probe", terminal.output)
            terminal.send(b"\x1b[B" * 19)
            terminal.until(lambda: b"> item19" in terminal.output)
            terminal.send(b"\r")
            terminal.finish(29)

    def test_clipping_and_timed_horizontal_scroll(self):
        with Terminal(columns=24, rows=6) as terminal:
            terminal.until(lambda: b"> item0" in terminal.output)
            start = terminal.output
            terminal.until(lambda: b"END_OF_DESCRIPTION" in terminal.output, timeout=8)
            self.assertNotEqual(start, terminal.output)
            for part in re.split(rb"\x1b\[[0-?]*[ -/]*[@-~]", terminal.output):
                self.assertLessEqual(len(part), 23)
            terminal.send(b"\x1b")
            terminal.finish(0)

    def test_dumb_terminal_falls_back(self):
        with Terminal(term="dumb") as terminal:
            terminal.until(lambda: b"Select option" in terminal.output)
            terminal.send(b"bad\n item3 \n")
            terminal.finish(13, interactive=False)
            self.assertIn(b"Invalid selection", terminal.output)

    def test_run_console_sized_pty_falls_back_without_changing_modes(self):
        for columns, rows in ((120, 1), (0, 0), (80, 0), (0, 24), (1, 24)):
            for data, expected in ((b"4\n", 13), (b"bad\n item3 \n", 13)):
                with self.subTest(size=(columns, rows), data=data), Terminal(columns=columns, rows=rows) as terminal:
                    terminal.until(lambda: b"Select option" in terminal.output)
                    self.assertEqual(termios.tcgetattr(terminal.slave), terminal.original)
                    self.assertIn(b"1. item0", terminal.output)
                    self.assertIn(b"20. item19", terminal.output)
                    terminal.send(data)
                    terminal.finish(expected, interactive=False)
                    self.assertNotIn(b"\x1b", terminal.output)

    def test_empty_term_falls_back(self):
        with Terminal(term="") as terminal:
            terminal.until(lambda: b"Select option" in terminal.output)
            terminal.send(b"4\n")
            terminal.finish(13, interactive=False)

    def test_redirected_streams(self):
        for data, expected in (("q\n", 0), ("", 0), ("4\n", 13), ("bad\nitem2\n", 12)):
            result = subprocess.run([PROBE], input=data, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected)
            self.assertIn("1. item0", result.stdout)
            self.assertIn("20. item19", result.stdout)
            self.assertNotIn("\x1b", result.stdout)

    @unittest.skipUnless(DEMOS, "Pass demo executables for macOS Metal launch checks")
    def test_demo_redirected_number_and_name_selection(self):
        for demo in DEMOS:
            for choice in ("1\n", "triangle\n"):
                with self.subTest(demo=demo, choice=choice), tempfile.TemporaryDirectory(prefix="longmarch-tui-") as directory:
                    result = subprocess.run(
                        [demo, "--backend", "metal", "--frames", "1"], input=choice,
                        text=True, capture_output=True, timeout=30, cwd=directory,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("Select option", result.stdout)
                    self.assertNotIn("\x1b[?1049h", result.stdout)

    @unittest.skipUnless(DEMOS, "Pass demo executables for macOS Metal launch checks")
    def test_demo_default_menu_launch(self):
        for demo in DEMOS:
            with self.subTest(demo=demo), Terminal([demo, "--backend", "metal", "--frames", "1"]) as terminal:
                terminal.until(lambda: b"\x1b[7m" in terminal.output)
                terminal.send(b"\r")
                terminal.finish(0)


if __name__ == "__main__":
    unittest.main()
