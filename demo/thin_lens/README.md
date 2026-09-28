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

The spheres retain the coarse `Sphere(32, 16)` mesh. Ray-traced direct-light
visibility uses a shadow-terminator position correction derived from vertex
normals: the shadow origin is interpolated toward the vertex tangent planes,
while BSDF evaluation and light sampling keep the actual hit position. This
reduces faceted self-shadow boundaries without adding geometry. It does not
smooth silhouettes or change indirect reflection rays. Flat faces, back-face
hits and transmission visibility retain their original shadow origins. Like
other position-offset corrections, it can soften very close contact shadows.
