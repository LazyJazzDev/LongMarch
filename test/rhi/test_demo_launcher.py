"""Check the interactive RHI demo launcher's CLI without opening windows.

Usage: python test_demo_launcher.py <demo_rhi_hello> [--backend metal|vulkan|d3d12]
"""
import argparse
from pathlib import Path
import subprocess
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("--backend", default="metal")
args = parser.parse_args()


class LauncherTests(unittest.TestCase):
    def run_demo(self, *options, input="", code=0):
        result = subprocess.run(
            [str(args.executable.resolve()), *options], input=input,
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_list_and_help(self):
        self.assertEqual(self.run_demo("--list").splitlines(), [
            "window", "triangle", "cube", "texture", "blend", "nbody",
            "raytracing", "ray_query", "hdr", "imgui", "imgui_multi_window",
        ])
        help_text = self.run_demo("--help")
        self.assertIn("--device auto|N", help_text)
        self.assertNotIn("--frames", help_text)

    def test_removed_options_are_rejected(self):
        for option in ("--headless", "--all", "--frames", "--save-last-frame", "--font", "--font-size", "--tui"):
            with self.subTest(option=option):
                self.assertIn("Unknown option", self.run_demo(option, code=1))

    def test_invalid_selection_and_missing_values(self):
        self.run_demo("--backend", "invalid", code=1)
        self.run_demo("--backend", args.backend, "--module", "invalid", code=1)
        for option in ("--backend", "--module", "--device"):
            with self.subTest(option=option):
                self.run_demo(option, code=1)
                self.run_demo(option, "", code=1)
        for device in ("0", "-1", "1x", "999999999999999999999999999999999"):
            with self.subTest(device=device):
                output = self.run_demo("--backend", args.backend, "--module", "window", "--device", device, code=1)
                self.assertIn("Device must be auto or a number", output)

    def test_module_menu_cancel_and_invalid_choice(self):
        output = self.run_demo("--backend", args.backend, "--device", "auto", input="not_a_demo\n")
        self.assertIn("Invalid selection", output)
        self.assertIn("Module Selection", output)
        self.assertNotIn("\x1b", output)
        self.run_demo("--backend", args.backend, "--device", "auto")


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
