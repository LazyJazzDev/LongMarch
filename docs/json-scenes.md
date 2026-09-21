# JSON scenes

Sparkium loads version 1 `sparkium-scene` documents through `LoadScene(path)`.
The CLI renders a document, and the GUI discovers, selects, and reloads scenes.
Mesh and texture paths are relative to the JSON document, independently of the
process working directory. Loading eagerly resolves meshes and textures into an
independent `SceneDefinition`; it requires no renderer or device. Backends receive
that in-memory entity through `Renderer::SetScene` and never reopen its files.
See [the scene API](sparkium-scene-api.md) for ownership and library examples.

Supported scene data:

- Film dimensions and exposure settings, look-at camera, and renderer settings.
- Lambertian, specular, emissive, and Principled materials with texture inputs.
- OBJ/binary meshes, inline triangle meshes, generated spheres and binary hair.
- Mesh instances with matrix/TRS or look-at transforms, and point lights.
- Material node graphs with decoded texture references.
- Automatic, rasterization, hardware ray tracing, Ray Query and software tracing.

The loader reports malformed documents, wrong field types, unsupported formats,
unknown references, and invalid camera/mesh settings through exceptions.

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

## Renderer preferences and backend switching

`LoadSceneDocument(path)` returns the independent scene plus a separate
`preferred_pipeline` parsed from `renderer.pipeline`. `LoadScene(path)` returns
only the scene. Backend/API/pipeline selection is not stored in SceneDefinition.
GUI and CLI honor a supported document preference and otherwise use `auto`;
explicit user pipeline selections are checked strictly against the backend.
`Renderer::SetBackend` rebuilds from the retained scene without reopening files.
