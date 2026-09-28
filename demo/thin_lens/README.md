# Thin Lens

An interactive, procedural scene dedicated to `sparkium::CameraThinLens`.
Orange, teal and blue spheres occupy three depths, with a checkerboard floor
to help judge focus. Small background emitters show the
aperture's bokeh shape. No external models or textures are required.

```sh
cmake --build cmake-build-release --target demo_thin_lens -j8
cmake-build-release/demo/thin_lens/demo_thin_lens --backend metal
```

Choose the `demo_thin_lens` target in CLion. Omit `--backend metal` for automatic
backend selection on other platforms. Rendering uses native Ray Query when
available, otherwise native RT or software tracing; it never selects raster.

- **Near / orange**, **Mid / teal**, **Far / blue** focus on the subjects' front
  surfaces at distances 3.4, 5.4 and 8.4. Focus distance is measured along the
  camera's forward axis.
- **Aperture radius** controls depth of field. Larger values increase blur;
  zero matches the pinhole camera.
- **Shape**, **Rotation**, and **Horizontal ratio** change background bokeh.
- **Pinhole comparison** switches to the actual `CameraPinhole` implementation,
  preserving the thin-lens settings for switching back.
- Any control change resets accumulation. Leave the camera still to reduce noise.

The demo renders at 1100 × 700, eight samples per frame and four bounces. The
window displays accumulated sample count. `Reset lens` restores the default
middle focus, radius 0.22, six blades, zero rotation and ratio 1.

For repeatable renders without a window:

```sh
cmake-build-release/demo/thin_lens/demo_thin_lens --backend metal \
  --headless --frames 64 --output out/thin-lens/middle.png
cmake-build-release/demo/thin_lens/demo_thin_lens --backend metal \
  --headless --frames 64 --pinhole --output out/thin-lens/pinhole.png
cmake-build-release/demo/thin_lens/demo_thin_lens --backend metal \
  --headless --frames 64 --focus 3.4 --blades 0 --output out/thin-lens/near.png
```

`--frames N` also bounds interactive smoke tests. CLI optical options are
`--focus`, `--aperture`, and `--blades` (0 or 3–8). `--output` saves the developed
render without the UI; headless mode requires a positive frame count.

The spheres retain the coarse `Sphere(32, 16)` mesh. **Geometry offset** controls
Cycles' shadow-terminator geometry correction (default `0.1`, range `[0, 1]`);
zero disables it. This is a grazing-angle cutoff, not a world-space distance.
The slider updates the mesh and resets accumulation. Use `--geometry-offset 0`
or `--geometry-offset 0.1` for repeatable headless comparisons.

The correction uses Cycles' parabolic height estimate and local linear envelope,
with a smooth angular weight and separate reflection/transmission directions.
Only shadow visibility rays move; BSDF evaluation, light PDFs, silhouette and
indirect path origins stay unchanged. Meshes without vertex normals are unaffected.
Large values may alter nearby contact shadows.

For other scenes, configure each mesh through the public API:

```cpp
mesh.SetShadowTerminatorGeometryOffset(0.1f); // default, Cycles geometry cutoff
mesh.SetShadowTerminatorGeometryOffset(0.0f); // disable
float cutoff = mesh.GetShadowTerminatorGeometryOffset();
film.Reset(); // reset accumulated rendering after changing the mesh setting
```

Values outside `[0, 1]`, NaN and infinity throw `std::invalid_argument`.
The setting is shared by all instances of that mesh and does not rebuild its BLAS.
This does not enable Cycles' separate Shading Offset or Bump Map Correction.
Source and Apache-2.0 attribution: [Cycles adaptation](../../external/cycles/README.md).
