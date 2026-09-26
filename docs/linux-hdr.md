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

Set the variable before starting the process. Every graphics `Core` attempts
GLFW initialization in its base constructor, before backend-specific setup
(including Vulkan WSI extension queries). Failure does not prevent headless
rendering. `Window` reuses the same initialization helper and requires success
when actually creating a window. The low-level Vulkan library keeps its native
GLFW initialization path without depending on the graphics/window layer.
Automatic fallback concerns window-system initialization, not HDR support:
a valid SDR-only Wayland session stays on Wayland. Older GLFW versions retain
their native platform selection.

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

The GUI reports `HDR` without exposing the backend encoding. If the surface
offers neither supported encoding, `--hdr` prints the reason and continues in
SDR; the same reason is visible in the GUI. The interactive HDR toggle also
preserves SDR when the request is unsupported.

`Window::SetHDR(bool)` requests the application's presentation mode; it does not
change the desktop HDR setting or scene exposure. It returns `0` on success and
nonzero on failure, with diagnostic details in the log. Unsupported requests
are rejected before changing the active Vulkan swapchain. Callers check the
status instead of catching capability exceptions; HDR format negotiation returns
an empty optional when no compatible format exists.

For compatibility checks, run the same binary with `LONGMARCH_WINDOW_SYSTEM=x11`.
An X11-only build does not gain native Wayland by changing environment variables.

## Color and brightness

The negotiation order is float16 linear scRGB, then 10-bit HDR10 in either
`A2B10G10R10` or `A2R10G10B10` layout. Unmatched formats, including float16
`BT709_LINEAR`, are not assumed to have scRGB semantics.

HDR10 uses this pipeline:

1. Develop the scene with its exposure and artistic grade into linear BT.709.
2. Resize and compose the scene and linearized ImGui into a floating-point image.
3. Convert BT.709 to BT.2020 and multiply by the fixed 203-nit content reference white.
4. Encode ST 2084/PQ and transfer the result into the 10-bit swapchain.

PQ conversion is mandatory even when scRGB brightness alignment is disabled.
It never modifies the source film, accumulation, source image, or HDR export.
The floating-point intermediate preserves alpha; the swapchain uses opaque
desktop composition. UI blending happens before PQ encoding.

The presentation layer uses **203 nits as PQ content reference white**, matching
the default reference white for the Wayland protocol's ST 2084 transfer function.
Encoding and content reference white are internal; the GUI exposes neither an
encoding choice nor a reference-white control. This content white is not a
measurement of desktop white or the monitor's peak. On the locally verified
NVIDIA/Mutter path, the driver declares PQ content with this default reference
white, and the compositor maps content reference white to its output reference
white. Substituting the output's reference white into the pixels without changing
their content description would apply the adjustment twice.

Applications submit linear HDR images; the backend selects the encoding and
performs any necessary conversion without a public output-encoding query or
PQ-reference-white getter/setter.
Linux display headroom remains unknown. The existing Windows scRGB reference-white
alignment and Metal presentation paths remain separate.

PQ values are bounded at 10000 nits. This is not display-adaptive tone mapping;
the compositor/display can perform further mapping. Application HDR metadata
submission is not implemented. Preferred-image-description feedback is observed
by the optional diagnostic below; the GUI does not override the compositor's
mapping using those output-side values.

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

The scene-size regression is also opt-in:

```sh
LONGMARCH_WINDOW_SYSTEM=wayland LONGMARCH_TEST_HDR=1 \
  ./build-wayland/test/sparkium/sparkium_fallback_test --gtest_filter='WindowResizeEventTest.*:VulkanWindowResizeTest.*'
```

It loads Cornell Box → Texture → Cornell Box, renders one sample per scene,
and resizes one window through their 1024×1024 / 2048×1024 logical dimensions
and half-size windows, without toggling HDR between resizes. Source images keep
the scene resolution; Film identity, resolution, accumulation and camera remain
unchanged when the window resizes. It checks
framebuffer/swapchain dimensions in SDR and HDR, and reads back both lower corners
of the HDR composition to detect stale targets or black borders. At 150% desktop
scaling, the original Wayland code reproduced a 3072×1536 framebuffer with a stale
1536×1536 swapchain. Vulkan keeps its `FramebufferResizeEvent` subscription so
presentation resources follow pixel sizes. Separately, GLFW's Wayland backend
can omit the logical-size callback during programmatic resizing. Only on Wayland,
`Window::Resize` checks the actual logical size after `glfwSetWindowSize` and
notifies subscribers if that size has not already been reported by a native
callback. Framebuffer callbacks never synthesize `ResizeEvent`.

The event regression checks programmatic resizing and duplicate suppression on
both platforms, and verifies that framebuffer callbacks alone do not synthesize
logical resize notifications.
The regression passes on native Wayland (SDR and PQ) and the default X11-only
build (SDR, via XWayland). The edge readback uses a uniform test image with ImGui after scene rendering;
it is not an image-quality comparison or physical display measurement.

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
Cornell Box at 768×768, 1 sample/frame, 8 bounces, max exposure 100, and content
white 203 nits. This is a bounded rendering/presentation check, not a subjective
image-quality judgment or a physical luminance measurement. The Wayland stack
emitted a GTK `gtk_disable_setlocale` startup warning without failing the run.

Windows/macOS, pure Xorg sessions, other GPUs/compositors, live desktop HDR
changes, and moving windows between HDR/SDR monitors were not tested in this run.

### X11 fallback capture

Actual client-window capture from code `9ea197b7`: GNOME 50.1 / XWayland,
Vulkan, Cornell Box at 768×768, Auto → Ray Query, 3652 accumulated spp,
1 sample/frame, 8 bounces, 0 EV, max exposure 100. Launching with `--hdr`
reports the unsupported surface and continues rendering in SDR. This shows
X11 fallback, not native Wayland HDR output or physical display luminance.

![X11 SDR fallback](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/8ef3107d99e9f29b840fa89346378c9238823a08/reports/linux-hdr-compat/x11-sdr-fallback.png)

Capture and metadata: [assets PR #31](https://github.com/LazyJazzDev/LongMarchAssetsLFS/pull/31).

### Reference-white protocol validation

An optional diagnostic opens an SDR white window and an HDR reference-white
window, reads each mapped surface's preferred image description, and observes
`preferred_changed` notifications. It does not set surface color descriptions
(Vulkan WSI owns those), change display settings, or measure physical luminance.
Only this opt-in target requires Wayland development tools and
`wayland-protocols >= 1.41`; normal builds acquire no new dependency.

```sh
cmake -S . -B build-wayland -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVCPKG_PATH=/path/to/vcpkg -DVCPKG_MANIFEST_FEATURES=wayland \
  -DLONGMARCH_ENABLE_WAYLAND_COLOR_PROBE=ON
cmake --build build-wayland --target wayland_color_probe
LONGMARCH_WINDOW_SYSTEM=wayland WAYLAND_DEBUG=client \
  ./build-wayland/test/sparkium/wayland_color_probe --seconds 30
```

`WAYLAND_DEBUG=client` exposes the WSI driver's submitted source description as
well as the probe's output-side feedback. Missing color-management support or
an X11 session returns skip status 77. An unavailable/ICC-only preferred
description is reported as unknown, not fabricated as a system reference white.
The probe binds protocol version 1 for compatibility, even when version 2 is
advertised. It does not require a new color-management version for the GUI.

A subsequent local check on GNOME/Mutter 50.1 and NVIDIA 595.91.07 temporarily
enabled desktop HDR and then restored the original SDR configuration. Both
windows received the same feedback and both live transitions were observed:

| Desktop mode | Preferred transfer / primaries | Preferred reference white | Primary-volume maximum |
| --- | --- | --- | --- |
| SDR, before | gamma 2.2 / BT.709 | 80 nits | 80 nits |
| HDR | PQ / BT.2020 | 203 nits | 10000 nits |
| SDR, restored | gamma 2.2 / BT.709 | 80 nits | 80 nits |

The 10000-nit value is the PQ container's maximum, **not the monitor's peak**.
PQ swapchains remained available with desktop HDR disabled, so format support
alone cannot prove physical HDR output.

The WSI trace showed `set_primaries_named(6)`, `set_tf_named(11)`, and perceptual
render intent, with no explicit `set_luminances`. The protocol therefore gives
this PQ source a 203-nit reference white. Mutter 50.1's
[`get_lum_mapping`](https://github.com/GNOME/mutter/blob/50.1/clutter/clutter/clutter-color-state-params.c)
uses `(target.ref / source.ref) * (source.max / target.max)` in normalized linear
light. In the HDR state above, SDR full white and PQ content white consequently
map to the same reference level. This is protocol/source-level validation, not
compositor pixel readback, a photometric test, or proof for every compositor.
No multi-monitor migration or SDR-white-slider sweep was performed.

The GPU regression also varies a controlled output-reference query through
80/203/400 nits and checks that PQ content white stays encoded at 203 nits, while
the separate scRGB path still applies its reference-white scale. This prevents
an accidental second output-white adjustment in the presentation shader.

## References

- [Vulkan surface color spaces](https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html)
- [Wayland color-management protocol](https://wayland.app/protocols/color-management-v1)
- [GLFW build options](https://www.glfw.org/docs/latest/compile_guide.html)
- [Vulkan HDR metadata](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_hdr_metadata.html)
