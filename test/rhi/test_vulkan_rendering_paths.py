"""Exercise native, KHR and fallback rendering on a dynamic-rendering-capable GPU.

Requires the Vulkan SDK API-dump, validation and Profiles layers.
A desktop is only required without --headless.
Usage: python test_vulkan_rendering_paths.py <rhi_hello> [<rhix_hello>] [--headless]
Profiles only constrain capabilities; the real driver still executes the commands.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def check_path(executable, mode, directory, headless=False):
    settings = [
        "khronos_validation.validate_sync = true",
        "khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG",
        "khronos_validation.report_flags = error,warn",
        "khronos_validation.log_filename = stdout",
        "lunarg_api_dump.detailed = false",
        "lunarg_api_dump.file = false",
    ]
    layers = ["VK_LAYER_LUNARG_api_dump", "VK_LAYER_KHRONOS_validation"]
    if mode != "core":
        profile = {
            "$schema": "https://schema.khronos.org/vulkan/profiles-0.8.2-312.json#",
            "capabilities": {
                "rendering": {
                    "features": {
                        "VkPhysicalDeviceDynamicRenderingFeatures": {
                            "dynamicRendering": mode == "khr"
                        }
                    }
                }
            },
            "profiles": {
                "VP_LONGMARCH_rendering": {
                    "version": 1,
                    "api-version": "1.2.0",
                    "label": "RHI rendering regression",
                    "description": "Constrain dynamic rendering on real hardware",
                    "capabilities": ["rendering"],
                }
            },
        }
        (directory / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        settings += [
            f"khronos_profiles.profile_dirs = {directory.as_posix()}",
            "khronos_profiles.profile_name = VP_LONGMARCH_rendering",
            "khronos_profiles.simulate_capabilities = SIMULATE_API_VERSION_BIT,SIMULATE_FEATURES_BIT",
            "khronos_profiles.default_feature_values = DEFAULT_FEATURE_VALUES_DEVICE",
        ]
        layers.append("VK_LAYER_KHRONOS_profiles")
    (directory / "vk_layer_settings.txt").write_text("\n".join(settings), encoding="utf-8")
    env = os.environ.copy()
    env["VK_INSTANCE_LAYERS"] = os.pathsep.join(layers)
    env["VK_LAYER_SETTINGS_PATH"] = str(directory)
    scenes = subprocess.check_output([str(executable), "--list"], text=True).splitlines()
    triangle = next(scene for scene in scenes if scene.endswith("triangle"))
    device_args = (["--device", "1"] if "--device" in subprocess.check_output(
        [str(executable), "--help"], text=True, encoding="utf-8") else [])
    result = subprocess.run(
        [str(executable), "--backend", "vulkan", "--frames", "3", *device_args]
        + (["--headless", "--all"] if headless else ["--module", triangle]),
        env=env, capture_output=True, timeout=60,
    )
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode or re.search(r"Validation Error|VUID-|SYNC-HAZARD|PROFILES ERROR", output):
        raise RuntimeError(f"{executable.name}/{mode}: execution or validation failed\n{output}")
    commands = {
        "core": ("vkCmdBeginRendering(", "vkCmdEndRendering("),
        "khr": ("vkCmdBeginRenderingKHR(", "vkCmdEndRenderingKHR("),
        "fallback": ("vkCmdBeginRenderPass(", "vkCmdEndRenderPass("),
    }
    scenes = subprocess.check_output([str(executable), "--list"], text=True).splitlines()
    rendering_scenes = len([scene for scene in scenes if scene not in ("window", "hello_window")])
    for path, pair in commands.items():
        for command in pair:
            count = output.count(command)
            expected = (63 * rendering_scenes if headless else 3) if path == mode else 0
            if count != expected:
                raise RuntimeError(f"{executable.name}/{mode}: {command} count {count}, expected {expected}\n{output}")
    for command in ("vkCreateRenderPass(", "vkCreateFramebuffer("):
        if bool(output.count(command)) != (mode == "fallback"):
            raise RuntimeError(f"{executable.name}/{mode}: unexpected {command} usage\n{output}")
    if headless:
        for command in ("vkCreateSwapchainKHR(", "vkAcquireNextImageKHR(", "vkQueuePresentKHR(",
                        "vkCreateWin32SurfaceKHR(", "vkCreateXlibSurfaceKHR(", "vkCreateXcbSurfaceKHR(",
                        "vkCreateWaylandSurfaceKHR("):
            if command in output:
                raise RuntimeError(f"Headless rendering used {command}")
    print(f"{executable.name}/{mode}/headless={headless}: passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executables", nargs="+", type=Path)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    for executable in args.executables:
        for mode in ("core", "khr", "fallback"):
            with tempfile.TemporaryDirectory(prefix="longmarch-rendering-") as temporary:
                check_path(executable.resolve(), mode, Path(temporary), args.headless)
