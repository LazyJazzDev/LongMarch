# JSON scenes

Sparkium loads version 1 `sparkium-scene` documents through `JsonScene::Load`.
The CLI renders a document, and the GUI discovers, selects, and reloads scenes.
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
six basic demos at 96×96 from a different working directory, and 25 invalid-input
cases. Both CLI and GUI targets compiled. This check does not cover interactive
GUI behavior, hardware ray tracing, or image equivalence with the C++ demos.
Per-case logs, images, and `results.json` are written to the output directory.

## HDR preview in the scene browser

The GUI offers an experimental **HDR preview** checkbox on Metal and an
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

The first version exposes HDR preview only on Metal. Visible HDR highlights
require EDR headroom above 1.0 on the window's current screen; Metal logs current
and potential headroom when HDR is enabled. A floating-point output path alone
cannot make an SDR-only display brighter. See [Metal HDR presentation](metal-backend.md#hdr--edr-presentation).
