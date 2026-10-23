"""Native Windows console integration tests. Run with probe, then optional demo paths.

Example: python test_console.py build/test/tui/tui_menu_probe.exe demo_rhi_hello.exe ...
Creates hidden, isolated console windows; never sends input to the user's terminal.
"""

import ctypes as c
from ctypes import wintypes as w
import pathlib
import subprocess
import sys
import time
import unittest

if sys.platform != "win32":
    raise SystemExit("This driver tests native Windows consoles; no WSL is used.")

PROBE = str(pathlib.Path(sys.argv.pop(1)).resolve())
DEMOS = [str(pathlib.Path(p).resolve()) for p in sys.argv[1:]]
sys.argv[1:] = []
k = c.WinDLL("kernel32", use_last_error=True)


class Coord(c.Structure):
    _fields_ = [("x", w.SHORT), ("y", w.SHORT)]


class Rect(c.Structure):
    _fields_ = [("left", w.SHORT), ("top", w.SHORT), ("right", w.SHORT), ("bottom", w.SHORT)]


class BufferInfo(c.Structure):
    _fields_ = [("size", Coord), ("cursor", Coord), ("attributes", w.WORD), ("window", Rect), ("maximum", Coord)]


class Key(c.Structure):
    _fields_ = [("down", w.BOOL), ("repeat", w.WORD), ("key", w.WORD), ("scan", w.WORD),
                ("character", w.WCHAR), ("control", w.DWORD)]


class Mouse(c.Structure):
    _fields_ = [("position", Coord), ("buttons", w.DWORD), ("control", w.DWORD), ("flags", w.DWORD)]


class Payload(c.Union):
    _fields_ = [("key", Key), ("mouse", Mouse), ("padding", c.c_byte * 16)]


class Record(c.Structure):
    _fields_ = [("kind", w.WORD), ("event", Payload)]


k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
k.CreateFileW.restype = w.HANDLE
k.AttachConsole.argtypes = [w.DWORD]
k.GetConsoleScreenBufferInfo.argtypes = [w.HANDLE, c.POINTER(BufferInfo)]
k.SetConsoleScreenBufferSize.argtypes = [w.HANDLE, Coord]
k.SetConsoleWindowInfo.argtypes = [w.HANDLE, w.BOOL, c.POINTER(Rect)]
k.ReadConsoleOutputCharacterW.argtypes = [w.HANDLE, w.LPWSTR, w.DWORD, Coord, c.POINTER(w.DWORD)]
k.ReadConsoleOutputAttribute.argtypes = [w.HANDLE, c.POINTER(w.WORD), w.DWORD, Coord, c.POINTER(w.DWORD)]
k.WriteConsoleInputW.argtypes = [w.HANDLE, c.POINTER(Record), w.DWORD, c.POINTER(w.DWORD)]
k.CloseHandle.argtypes = [w.HANDLE]


def check(result):
    if not result:
        raise c.WinError(c.get_last_error())


class Console:
    def __init__(self, executable=PROBE, args=()):
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        run_dir = pathlib.Path("out/tui/run").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([executable, *args], creationflags=subprocess.CREATE_NEW_CONSOLE,
                                        startupinfo=startup, cwd=run_dir)
        k.FreeConsole()
        deadline = time.monotonic() + 5
        while not k.AttachConsole(self.process.pid):
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError("Could not attach to test console")
            time.sleep(0.02)
        self.input = k.CreateFileW("CONIN$", 0xC0000000, 3, None, 3, 0, None)
        self.output = None
        self.wait_text("Module Selection" if executable != PROBE else "TUI integration probe")

    def refresh(self):
        # Opening CONOUT$ resolves the active alternate buffer, unlike a saved old handle.
        if self.output:
            k.CloseHandle(self.output)
        self.output = k.CreateFileW("CONOUT$", 0xC0000000, 3, None, 3, 0, None)
        info = BufferInfo()
        check(k.GetConsoleScreenBufferInfo(self.output, c.byref(info)))
        return info

    def rows(self):
        info = self.refresh()
        lines = []
        for y in range(info.window.top, info.window.bottom + 1):
            size = info.window.right - info.window.left + 1
            text = c.create_unicode_buffer(size + 1)
            count = w.DWORD()
            check(k.ReadConsoleOutputCharacterW(self.output, text, size, Coord(info.window.left, y), c.byref(count)))
            lines.append(text.value)
        return lines

    def wait_text(self, text, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if text in "\n".join(self.rows()):
                return
            if self.process.poll() is not None:
                raise AssertionError(f"Exited before displaying {text!r}: {self.process.returncode}")
            time.sleep(0.03)
        raise AssertionError(f"Missing {text!r}: {self.rows()!r}")

    def send(self, record):
        count = w.DWORD()
        check(k.WriteConsoleInputW(self.input, c.byref(record), 1, c.byref(count)))

    def key(self, virtual_key, repeat=1):
        record = Record()
        record.kind = 1
        record.event.key = Key(True, repeat, virtual_key, 0, "\0", 0)
        self.send(record)
        record.event.key.down = False
        self.send(record)

    def click(self, row):
        info = self.refresh()
        for pressed in (1, 0):
            record = Record()
            record.kind = 2
            record.event.mouse = Mouse(Coord(info.window.left + 2, info.window.top + row), pressed, 0, 0)
            self.send(record)

    def resize(self, columns, rows):
        self.refresh()
        check(k.SetConsoleWindowInfo(self.output, True, c.byref(Rect(0, 0, columns - 1, rows - 1))))
        time.sleep(0.15)

    def finish(self, expected):
        result = self.process.wait(timeout=30)
        if result != expected:
            raise AssertionError(f"Exit {result}, expected {expected} (100 means terminal state was not restored)")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        for handle in (getattr(self, "input", None), getattr(self, "output", None)):
            if handle:
                k.CloseHandle(handle)
        k.FreeConsole()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NativeMenuTests(unittest.TestCase):
    def test_keyboard_selection_and_restore(self):
        with Console() as terminal:
            terminal.key(0x28)
            terminal.key(0x28)
            terminal.key(0x26)
            terminal.key(0x0D)
            terminal.finish(11)

    def test_repeated_keys(self):
        with Console() as terminal:
            for _ in range(15):
                terminal.key(0x28)
            time.sleep(0.2)
            self.assertIn("> item15", "\n".join(terminal.rows()))
            terminal.key(0x0D)
            terminal.finish(25)

    def test_single_click_highlights_before_enter(self):
        with Console() as terminal:
            terminal.click(5)
            terminal.wait_text("> item3")
            info = terminal.refresh()
            attributes = (w.WORD * 1)()
            count = w.DWORD()
            check(k.ReadConsoleOutputAttribute(terminal.output, attributes, 1,
                                              Coord(info.window.left, info.window.top + 5), c.byref(count)))
            self.assertNotEqual(attributes[0], info.attributes)
            terminal.key(0x0D)
            terminal.finish(13)

    def test_fragmented_vt_mouse_reports(self):
        with Console() as terminal:
            for fragment in ("\x1b[<0;3;", "5M", "\x1b[<0;3;5m"):
                for char in fragment:
                    record = Record()
                    record.kind = 1
                    record.event.key = Key(True, 1, 0, 0, char, 0)
                    terminal.send(record)
                time.sleep(0.02)
            terminal.wait_text("> item2")
            terminal.key(0x0D)
            terminal.finish(12)

    def test_escape_cancels_and_restores(self):
        with Console() as terminal:
            terminal.key(0x1B)
            terminal.finish(0)

    def test_mouse_selection_and_confirmation(self):
        with Console() as terminal:
            terminal.click(4)
            time.sleep(0.08)
            terminal.click(4)
            terminal.finish(12)

    def test_resize_crops_and_scrolls_one_line(self):
        with Console() as terminal:
            terminal.resize(30, 8)
            before = terminal.rows()
            self.assertTrue(before[2].startswith("> item0"), before)
            self.assertTrue(before[3].startswith("  item1"), before)
            self.assertTrue(all(line[-1] == " " for line in before), before)
            terminal.wait_text("END_OF_DESCRIPTION", timeout=8)
            after = terminal.rows()
            self.assertIn("END_OF_DESCRIPTION", after[2])
            self.assertEqual(before[3], after[3])
            for _ in range(12):
                terminal.key(0x28)
            time.sleep(0.2)
            self.assertIn("> item12", "\n".join(terminal.rows()))
            terminal.key(0x0D)
            terminal.finish(22)

    def test_demo_default_menu_and_one_frame(self):
        for executable in DEMOS:
            with self.subTest(executable=executable):
                with Console(executable) as terminal:
                    terminal.key(0x1B)
                    terminal.finish(0)
                # Select the first module through the real menu, then render one frame.
                with Console(executable, ("--backend", "d3d12", "--frames", "1")) as terminal:
                    terminal.key(0x0D)
                    terminal.finish(0)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
