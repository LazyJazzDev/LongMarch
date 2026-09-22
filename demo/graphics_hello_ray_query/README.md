# Graphics hello demos

All `graphics_hello_*` executables accept `--backend auto|metal|vulkan|d3d12`
and an optional positive `--frames N` limit. With no arguments they use the
platform default backend and run until the window closes. `--help` prints usage.
Only backends enabled in the build can be selected.

Build with Ninja (Release), explicitly selecting demo targets because they are
excluded from the default build:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target demo_graphics_hello_ray_query
build/demo/graphics_hello_ray_query/demo_graphics_hello_ray_query --backend metal
```

The ray query demo builds a triangle BLAS and an instance TLAS, updates the
instance transform every frame, and traces camera rays in an 8-by-8 compute
shader. Hits display triangle barycentrics; misses display a uniform background.
The shader uses `cs_6_5`, forces triangles opaque, and checks image bounds for
partial workgroups. It requires `DeviceRayQuerySupport()`; the current backend
implementation exposes this capability on supported Metal devices.

`graphics_hello_raytracing` demonstrates the separate ray tracing pipeline/SBT
API, which Metal does not implement. It exits with a diagnostic on Metal;
use `graphics_hello_ray_query` for native Metal acceleration-structure traversal.
The other seven demos use the raster graphics API and support Metal.
Metal presentation currently uses an SDR surface: `graphics_hello_hdr` still
renders to a floating-point image, but warns that display values are clipped.

For a short run with Metal API and shader validation:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  build/demo/graphics_hello_ray_query/demo_graphics_hello_ray_query \
  --backend metal --frames 10
```
