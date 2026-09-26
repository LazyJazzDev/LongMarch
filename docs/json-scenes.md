# JSON scenes

Sparkium loads version 1 `sparkium-scene` documents through `JsonScene::Load`.
The CLI renders a document, and the GUI discovers, selects, and reloads scenes.

The GUI uses each scene's film dimensions to choose its initial window size, then
renders at the actual framebuffer resolution (including HiDPI scaling). Scene
switches, automatic window fitting, and manual resizing update the film, display
image, and camera aspect together. Resizing restarts accumulation while preserving
exposure and other view settings. The `Resolution` row shows the current rendering
size. The CLI continues to use the dimensions in the scene document.
Mesh and texture paths are relative to the JSON document, independently of the
process working directory.

Supported scene data:

- Film dimensions and exposure settings, look-at camera, and renderer settings.
- Lambertian, specular, emissive, and Principled materials with texture inputs.
- OBJ meshes, inline triangle meshes, and generated spheres.
- Mesh instances with matrix/TRS or look-at transforms, and point lights.
- Automatic, rasterization, and hardware ray tracing pipeline selection.

The loader reports malformed documents, wrong field types, unsupported formats,
unknown references, and invalid camera/mesh settings through its error result.
Compute ray tracing fallback and Blender-specific extensions follow in separate
changes.

## Check the basic scenes

```sh
cmake --build cmake-build-rt-fallback --target demo_sparkium_cli demo_sparkium_gui -j8
python3 scripts/check_json_scenes.py \
  --cli cmake-build-rt-fallback/demo/sparkium_cli/demo_sparkium_cli \
  --output out/json-validation
```

Use the equivalent target paths in your configured build directory. Initialize
the matching LFS assets submodule before running the checker.

On Apple M5, the Vulkan build passed scene discovery, raster rendering of all
six basic demos at 96脳96 from a different working directory, and 25 invalid-input
cases. Both CLI and GUI targets compiled. This check does not cover interactive
GUI behavior, hardware ray tracing, or image equivalence with the C++ demos.
Per-case logs, images, and `results.json` are written to the output directory.

## HDR preview in the scene browser

All nine bundled scenes default to `renderer.pipeline: "auto"`, including the
Blender scenes. The GUI shows the resolved pipeline; an explicit user selection
can still override it.

The GUI offers an experimental **HDR preview** checkbox on Metal, D3D12 and Vulkan and an
**Exposure (EV)** slider. HDR preview develops the film into a floating-point
linear-sRGB image and enables the window's HDR/EDR presentation. Values above
1.0 are preserved rather than normalized or clipped to SDR white. For example,
a linear value of 4.0 becomes 8.0 at +1 EV. Output is bounded only to the finite
RGBA16Float presentation range (65504).

```sh
cmake-build-metal-only/demo/sparkium_gui/demo_sparkium_gui \
  assets/scenes/cornell_box/scene.json --backend metal --hdr
```

Without `--hdr`, startup remains SDR. The checkbox can switch viewing modes
without resetting accumulated samples. Exposure also affects only display
processing. HDR bypasses the scene's SDR view transform, gamma, and contrast;
turning it off restores those settings. The selected display mode persists when
changing or reloading scenes, while exposure is loaded from the new scene.
This preview does not change scene files or the CLI/export path.

On Windows, use `--backend d3d12 --hdr` or `--backend vulkan --hdr` and enable
HDR in Windows display settings. Both backends use a floating-point RGBA16
swapchain with linear scRGB presentation. Vulkan requires the surface to expose
`VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`; an unsupported HDR request reports
an error instead of silently presenting linear HDR through an SDR swapchain.
Visible HDR highlights on Metal require EDR headroom above 1.0 on the window's current screen; Metal logs current
and potential headroom when HDR is enabled. A floating-point output path alone
cannot make an SDR-only display brighter. See [Metal HDR presentation](metal-backend.md#hdr--edr-presentation).

Windows validation (RTX 3090 Ti, driver 596.49, Ninja Release): HDR film development,
surface-format preference, reference-white refresh/fallback, and D3D12/Vulkan
presentation/readback all pass (7 tests). The window tests exercise resize,
repeated HDR/SDR switching, both ImGui and plain
presentation with backend debugging and Vulkan synchronization validation enabled:

```powershell
$env:LONGMARCH_TEST_HDR_WINDOWS = '1'
$env:VK_LAYER_VALIDATE_SYNC = '1'
./cmake-build-ninja/test/sparkium/sparkium_fallback_test.exe --gtest_filter=*HDR*
```

Actual [D3D12 and Vulkan window captures](https://github.com/LazyJazzDev/LongMarchAssetsLFS/tree/3c93adf40f8771e74bbd06b3d3935d8cdc54594a/reports/sparkium-hdr-windows)
show the HDR control and resolved Auto pipeline at 768x768, 1 sample per frame,
8 bounces and maximum exposure 100. The PNG captures do not measure physical HDR
brightness or provide an equal-sample comparison. Metal was not retested here.

### Automatic HDR reference white

`Window` aligns HDR presentation to the system's current SDR reference white by
default. On Windows scRGB, 1.0 means 80 nits; the queried SDR white may be different.
For example, a configured 480-nit SDR white requires a 6x HDR presentation scale.
This is a display conversion, not a change to scene exposure, Film data or HDR export.
The GUI shows the queried white and the applied multiplier.

D3D12 and Vulkan compose the scene and ImGui into a floating-point intermediate,
then apply the reference-white multiplier once to RGB, preserving alpha. ImGui
vertex colors are decoded from sRGB for linear composition and restored after
upload. Custom ImGui textures on the HDR path must supply linear RGB. Incoming
HDR scene images must already be linear; SDR presentation remains unchanged.

The window API exposes:

- `GetDisplayBrightness()`: reference white in nits, native HDR white multiplier,
  HDR headroom, and validity/state flags. Zero nits or headroom means unknown.
- `RefreshDisplayBrightness()` and `DisplayBrightnessEvent()`: refresh explicitly
  and observe changes. Active HDR presentation also queries automatically, caching
  results for 500 ms, so monitor moves and system setting changes are picked up.
- `HDRReferenceWhiteScale()`: the applied scale (1 for SDR or disabled alignment).
- `SetHDRBrightnessAlignment(false)`: opt out when a caller already provides
  native display-scaled HDR values, avoiding double scaling.

Windows queries the window's monitor through DisplayConfig. If unavailable it
uses a 1x fallback with the reference marked unknown. Windows also matches the window's monitor across DXGI adapters and queries
`IDXGIOutput6::GetDesc1` for reported peak luminance internally.
When peak and SDR white are valid, estimated headroom is `max(1, peak / SDR white)`.
This driver-reported capability is not a real-time measurement. Invalid or
unavailable peak data leaves headroom unknown. Only the resulting `hdr_headroom`
is exposed through the C++ and Python APIs and displayed in the GUI.
Metal uses a 1x white
scale because its EDR surface already follows the system reference, and exposes
the screen's current EDR headroom; that path was not run on this Windows machine.

This corrects reference-white mismatch, but does not add a highlight tone mapper
or make HDR match the SDR Filmic look. Filmic/gamma/contrast still apply only to
SDR. GPU tests check scene RGB scaling, linear UI gray, alpha preservation,
unchanged input pixels, and refresh/fallback/opt-out behavior.

The **Render settings** panel exposes path-tracing samples per frame, maximum
bounces, alpha shadows, background, persistence, per-sample clamping, and
**Max exposure**. Changing these settings resets accumulation. Rasterization
instead exposes ambient light. **SDR view settings** selects Normalized,
Standard, or Filmic; gamma and contrast apply only to Filmic. Display settings
do not reset accumulation and SDR controls are disabled during HDR preview.

HDR development preserves the values it receives, but the renderer can already
have clipped them: **Max exposure** is a linear accumulated brightness limit,
not an EV adjustment. Cornell Box sets it to 1 even though its light emits 30.
Raise it to at least 30 (for example, 100) to preserve the light's HDR brightness;
keep **Sample clamp** high enough as well. These controls change the current
session only; Reload restores scene-file settings.

### Saving HDR images

The CLI can save an SDR PNG and a linear-sRGB Radiance RGBE `.hdr` from the
**same accumulated film**, without a second render:

```sh
demo_sparkium_cli scene.json --backend metal --pipeline ray_query --frames 64 \
  -o scene-sdr.png --hdr-output scene-linear.hdr
```

With 8 samples per dispatch this renders 512 spp. HDR export applies scene
exposure, bypasses the SDR view transform, and preserves values above 1 (up to
65504, as in GUI HDR development). RGBE is lossy, stores no alpha or embedded
color profile, and this export uses linear sRGB primaries by convention.
Accumulation limits (`max_exposure`, `clamping`) still apply upstream.

For HDR-capable browsers, convert that file to PQ AVIF using NumPy and FFmpeg
with the `libsvtav1` encoder:

```sh
python3 scripts/hdr_to_avif.py scene-linear.hdr scene-hdr.avif
```

This converts linear sRGB to BT.2020, maps linear 1 to 203 cd/m虏, applies ST 2084
(PQ), and writes a tagged 10-bit AVIF (YUV 4:2:0). It does not apply SDR tone
mapping; PQ luminance above 10000 cd/m虏 is clipped. The PNG uses the scene's SDR
view transform. Browser/OS/display HDR support and available EDR headroom are
required to see extended brightness. GitHub image proxies may transform images;
provide a direct original-file link as well as the inline image. SDR displays
may tone-map the HDR file, so screenshots cannot verify physical HDR brightness.
