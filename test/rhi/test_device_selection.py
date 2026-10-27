"""Run rhi_hello device-menu and explicit-selection integration checks.

Usage: python test_device_selection.py <demo_rhi_hello> --backend d3d12
Also runs on a single-device backend, where no device prompt should appear.
"""

import argparse
from pathlib import Path
import re
import subprocess
import unittest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("--backend", default="auto")
args = parser.parse_args()
executable = str(args.executable.resolve())


def run(extra=(), choice=""):
    return subprocess.run(
        [executable, "--backend", args.backend, "--module", "window", "--headless", "--frames", "1", *extra],
        input=choice, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


class DeviceSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = run(choice="q\n")
        if probe.returncode != 0:
            raise RuntimeError(probe.stdout + probe.stderr)
        cls.has_menu = "Device Selection" in probe.stdout
        cls.devices = re.findall(r"^\[(\d+)\] (.*?) \(", run(["--device", "1"]).stdout, re.MULTILINE)
        if not cls.devices:
            raise RuntimeError("Device enumeration did not produce any devices")

    def test_explicit_device_creates_the_requested_gpu(self):
        for number, name in self.devices:
            with self.subTest(number=number, name=name):
                result = run(["--device", number])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("Device Selection", result.stdout)
                self.assertIn(f"Selected device {number}: {name}", result.stdout)
                self.assertIn(f"Selected: {name}\n", result.stdout)

    def test_automatic_single_device_or_menu_selection(self):
        if len(self.devices) == 1:
            result = run()
            self.assertFalse(self.has_menu)
            self.assertIn(f"Selected: {self.devices[0][1]}\n", result.stdout)
        else:
            self.assertTrue(self.has_menu)
            number, name = self.devices[-1]
            result = run(choice=number + "\n")
            self.assertIn(f"Selected: {name}\n", result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_auto_skips_menu_and_creates_selected_device(self):
        result = run(["--device", "auto"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Device Selection", result.stdout)
        selected = re.search(r"^Selected: (.+)$", result.stdout, re.MULTILINE)
        self.assertIsNotNone(selected, result.stdout)
        name = selected.group(1)
        self.assertIn(name, [name for _, name in self.devices])
        self.assertIn(f"Selected device (auto): {name}", result.stdout)

    def test_last_device_option_wins(self):
        result = run(["--device", "auto", "--device", self.devices[-1][0]])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Selected: {self.devices[-1][1]}\n", result.stdout)
        result = run(["--device", self.devices[-1][0], "--device", "auto"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Selected device (auto):", result.stdout)

    def test_cancel_does_not_run_module(self):
        if not self.has_menu:
            self.skipTest("Single device is selected automatically")
        result = run(choice="q\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Selected:", result.stdout)

    def test_invalid_numbers_fail(self):
        for value in ["0", "-1", "1.5", "abc", "999999999999999999999", str(len(self.devices) + 1)]:
            with self.subTest(value=value):
                result = run(["--device", value])
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
                self.assertNotIn("Selected:", result.stdout)
        self.assertNotEqual(run(["--device"]).returncode, 0)


unittest.main(argv=[executable], verbosity=2)
