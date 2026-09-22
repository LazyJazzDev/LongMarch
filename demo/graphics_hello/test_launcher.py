"""Headless CLI/TUI regression checks; pass the built launcher as the first argument."""

import pathlib
import subprocess
import sys
import unittest

EXECUTABLE = str(pathlib.Path(sys.argv.pop(1)).resolve())
MODULES = (
    "triangle", "blend", "cube", "texture", "resize", "hdr", "sdr_sample",
    "raytracing", "ray_query",
)


class LauncherTest(unittest.TestCase):
    def run_launcher(self, *args, text=""):
        return subprocess.run(
            [EXECUTABLE, *args], input=text, text=True,
            capture_output=True, timeout=10,
        )

    def test_list_all_modules(self):
        result = self.run_launcher("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        names = [line.split()[1] for line in result.stdout.splitlines()]
        self.assertEqual(names, list(MODULES))
        self.assertNotIn("Device Name:", result.stdout)

    def test_help(self):
        result = self.run_launcher("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--module", "--tui", "--backend", "--frames", "--list"):
            self.assertIn(option, result.stdout)

    def test_invalid_arguments(self):
        for args in (
            ("--module", "missing"), ("--module",), ("--backend", "missing"),
            ("--frames", "0"), ("--frames", "-1"), ("--frames", "12x"),
            ("--frames", "999999999999999999999"), ("--unknown",),
            ("--module", "triangle", "--tui"),
        ):
            with self.subTest(args=args):
                result = self.run_launcher(*args)
                self.assertEqual(result.returncode, 1)
                self.assertTrue(result.stderr)
                self.assertNotIn("Device Name:", result.stdout)

    def test_tui_cancel_and_eof(self):
        for args, text in (((), "q\n"), (("--tui",), "quit\n"), ((), "")):
            with self.subTest(args=args, text=text):
                result = self.run_launcher(*args, text=text)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Module Selection", result.stdout)
                self.assertNotIn("Device Name:", result.stdout)

    def test_tui_invalid_selection_retries(self):
        result = self.run_launcher("--tui", text="0\n10\nmissing\n\n q \n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("Invalid selection."), 3)
        self.assertNotIn("Device Name:", result.stdout)


if __name__ == "__main__":
    unittest.main()
