"""Native Windows message editor checks: test_messages_windows.py <demo_tui.exe>.

Uses isolated hidden consoles; never modifies the user's terminal.
"""
import subprocess
import sys
import ctypes as c
import unittest

from test_console import Console, Key, Mouse, Coord, Record, PROBE, k, check


class MessageConsole(Console):
    def __init__(self):
        super().__init__(PROBE, ("--dump",))

    def wait_text(self, text, timeout=8):
        if text == "TUI integration probe":
            text = "TUI Messages"
        super().wait_text(text, timeout)

    def type(self, text):
        for char in text:
            record = Record()
            record.kind = 1
            # Nonzero virtual-key codes select native Unicode handling.
            record.event.key = Key(True, 1, 0x41, 0, char, 0)
            self.send(record)

    def shift_enter(self):
        record = Record()
        record.kind = 1
        record.event.key = Key(True, 1, 0x0D, 0, "\0", 0x10)
        self.send(record)
        record.event.key.down = False
        self.send(record)



class NativeMessageTests(unittest.TestCase):
    def test_edit_unicode_newline_and_dump(self):
        with MessageConsole() as terminal:
            self.assertEqual(k.GetConsoleCP(), 65001)
            self.assertEqual(k.GetConsoleOutputCP(), 65001)
            terminal.type("abc")
            terminal.key(0x25)
            terminal.key(0x08)
            terminal.key(0x2E)
            terminal.type("Z")
            terminal.shift_enter()
            terminal.type("\u4e2d\u6587")
            terminal.key(0x0D)
            terminal.wait_text("1 recorded")
            terminal.wait_text("aZ")
            terminal.wait_text("\u4e2d\u6587")
            terminal.key(0x1B)
            terminal.finish(0)
            terminal.wait_text("--- Message 1 ---")
            terminal.wait_text("\u4e2d\u6587")

    def test_mouse_places_cursor(self):
        with MessageConsole() as terminal:
            terminal.type("abc")
            terminal.wait_text("abc")
            rows = terminal.rows()
            row = next(i for i, line in enumerate(rows) if line.startswith("> abc"))
            info = terminal.refresh()
            for down in (1, 0):
                record = Record()
                record.kind = 2
                record.event.mouse = Mouse(Coord(info.window.left + 2, info.window.top + row), down, 0, 0)
                terminal.send(record)
            terminal.type("X")
            terminal.key(0x0D)
            terminal.wait_text("Xabc")
            terminal.key(0x1B)
            terminal.finish(0)

    def test_nonterminal_explains_requirement(self):
        result = subprocess.run([PROBE], input="", text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn("interactive terminal", result.stderr)
        self.assertNotIn("\x1b", result.stdout)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
