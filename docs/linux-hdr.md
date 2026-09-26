# Linux HDR presentation

Sparkium supports Vulkan HDR presentation using either linear scRGB or HDR10
(BT.2020 primaries with ST 2084/PQ encoding), selected from the actual window
surface formats. HDR requires a compatible GPU driver, compositor, output, and
desktop HDR setting. An advertised format is not a physical brightness measurement.

## Build and window-system compatibility

The default vcpkg manifest keeps the existing GLFW configuration. Native Wayland
is an optional manifest feature, and enabling it retains GLFW's X11 backend.
On Ubuntu, install `libwayland-dev`, `libxkbcommon-dev`, and `extra-cmake-modules`
alongside the existing X11 build dependencies. Then use a separate build directory:

```sh
cmake -S . -B build-wayland -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVCPKG_PATH=/path/to/vcpkg -DVCPKG_MANIFEST_FEATURES=wayland \
  -DLONGMARCH_DISABLE_CUDA=ON -DLONGMARCH_DISABLE_PYTHON=ON
cmake --build build-wayland --target demo_sparkium_gui sparkium_fallback_test
```

CUDA and Python are disabled only to keep this example's dependencies small.
The feature does not require disabling either one. The `wayland` feature is
Linux-only; Windows and macOS retain their native window backends.

On Linux, `LONGMARCH_WINDOW_SYSTEM` accepts:

| Value | Behavior |
| --- | --- |
| unset or `auto` | Try compiled-in Wayland when `WAYLAND_DISPLAY` exists; fall back to available X11 if initialization fails. |
| `x11` | Explicitly use X11, including XWayland in a Wayland session. |
| `wayland` | Require native Wayland; report an error if it is unavailable or not compiled in. |

Set the variable before starting the process. It controls both Vulkan's GLFW
initialization and the subsequent windows. The automatic fallback concerns window
system initialization, not HDR support: a valid SDR-only Wayland session stays
on Wayland. Older GLFW versions retain their native platform selection.

X11 remains supported for SDR. HDR is determined by surface capabilities, rather
than by a blanket platform ban. The Vulkan swapchain-color-space extension is
enabled only when available, so lack of that optional extension does not prevent
SDR initialization. No color-management protocol is required merely to run SDR.

Wayland owns window placement. Global positioning requests are ignored there;
the primary output's work area is used only as a window-size hint.

## Running

Enable HDR in the desktop's display settings, then run:

```sh
LONGMARCH_WINDOW_SYSTEM=wayland \
  ./build-wayland/demo/sparkium_gui/demo_sparkium_gui --backend vulkan --hdr
```

The GUI reports `HDR10 (PQ)` or `HDR (linear)` as appropriate. If the surface
offers neither supported encoding, `--hdr` prints the reason and continues in
SDR; the same reason is visible in the GUI. The interactive HDR toggle also
preserves SDR when the request is unsupported.

For compatibility checks, run the same binary with `LONGMARCH_WINDOW_SYSTEM=x11`.
An X11-only build does not gain native Wayland by changing environment variables.

## Color and brightness

The negotiation order is float16 linear scRGB, then 10-bit HDR10 in either
`A2B10G10R10` or `A2R10G10B10` layout. Unmatched formats, including float16
`BT709_LINEAR`, are not assumed to have scRGB semantics.

HDR10 uses this pipeline:

1. Develop the scene with its exposure and artistic grade into linear BT.709.
2. Resize and compose the scene and linearized ImGui into a floating-point image.
3. Convert BT.709 to BT.2020 and multiply by the chosen reference white in nits.
4. Encode ST 2084/PQ and transfer the result into the 10-bit swapchain.

PQ conversion is mandatory even when scRGB brightness alignment is disabled.
It never modifies the source film, accumulation, source image, or HDR export.
The floating-point intermediate preserves alpha; the swapchain uses opaque
desktop composition. UI blending happens before PQ encoding.

`HDR white (nits)` is a **manual** reference white, defaulting to 203 nits. It is
not an automatic query of GNOME's SDR white setting or the monitor's peak.
`Window::SetHDR10WhiteNits` accepts finite values from 1 to 10000 nits; the GUI
offers 80–400 nits. Linux display headroom remains unknown. The existing Windows
scRGB reference-white alignment and Metal presentation paths remain separate.

PQ values are bounded at 10000 nits. This is not display-adaptive tone mapping;
the compositor/display can perform further mapping. HDR metadata submission and
Wayland preferred-image-description feedback are not implemented here.

For Cornell Box, raise `Max exposure` above the bundled value of 1 (for example,
30–100) to retain scene highlights before presentation. An SDR screenshot cannot
verify the physical HDR luminance of the display.

## Validation

Interactive tests are opt-in (`LONGMARCH_TEST_HDR_WINDOWS` remains an alias):

```sh
LONGMARCH_WINDOW_SYSTEM=wayland LONGMARCH_TEST_HDR=1 \
  SPARKIUM_TEST_BACKEND=vulkan SPARKIUM_TEST_DEBUG=1 VK_LAYER_VALIDATE_SYNC=1 \
  ./build-wayland/test/sparkium/sparkium_fallback_test --gtest_filter='*HDR*'
```

Repeat with `LONGMARCH_WINDOW_SYSTEM=x11` and with an X11-only build. Tests cover
format negotiation, GPU PQ values and reference-white changes, source/alpha
preservation, scene/UI composition, resizing, repeated HDR/SDR transitions, and
continued SDR presentation after rejected HDR requests. HDR presentation tests
skip on SDR-only surfaces; the SDR rejection test runs there instead. Numeric PQ
readback is tested even on SDR desktops without presenting PQ images to them.

### Local validation, 2026-09-26

GNOME 50.1 / Wayland, RTX 3090 Ti, NVIDIA 595.91.07, M27P20P at 3840×2160,
desktop BT.2100 mode enabled; Ninja Release builds:

| Build / runtime | HDR-filtered tests | GUI |
| --- | --- | --- |
| Wayland feature / native Wayland | 8 pass, 1 SDR-only test skips | HDR10/PQ, 32 frames |
| Wayland feature / X11 backend | 7 pass, 2 HDR-surface tests skip | Unsupported HDR falls back to SDR, 8 frames |
| Default X11-only build | 7 pass, 2 HDR-surface tests skip | Automatic selection runs SDR, 8 frames |

The dual-backend GUI also exits successfully after 8 frames when `WAYLAND_DISPLAY`
is absent or names a nonexistent socket. Explicitly requesting Wayland from the
X11-only build exits with a clear unavailable-backend message. X11 tests here use
XWayland, not a separate Xorg login session.

GPU tests ran with Vulkan validation and synchronization validation enabled, with
no reported validation errors. The PQ test also switches its conversion pipeline
back to a controlled 3× scRGB reference-white scale to check that PQ settings do
not alter the existing linear path. The native Wayland GUI run used validation,
Cornell Box at 768×768, 1 sample/frame, 8 bounces, max exposure 100, and manual
white 203 nits. This is a bounded rendering/presentation check, not a subjective
image-quality judgment or a physical luminance measurement. The Wayland stack
emitted a GTK `gtk_disable_setlocale` startup warning without failing the run.

Windows/macOS, pure Xorg sessions, other GPUs/compositors, live desktop HDR
changes, and moving windows between HDR/SDR monitors were not tested in this run.

## References

- [Vulkan surface color spaces](https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html)
- [Wayland color-management protocol](https://wayland.app/protocols/color-management-v1)
- [GLFW build options](https://www.glfw.org/docs/latest/compile_guide.html)
- [Vulkan HDR metadata](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_hdr_metadata.html)
