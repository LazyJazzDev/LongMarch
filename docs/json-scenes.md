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
