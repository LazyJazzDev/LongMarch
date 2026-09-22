# Graphics Hello

One executable, eleven built-in modules. Each module implements the `Module`
lifecycle interface and is created only when selected. Modules are statically
linked; no separate executables or dynamic plugins are needed.
Embedded modules share one shader bundle;
`external_shader` intentionally loads shader files from `assets/shaders/raytracing`.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target demo_graphics_hello

# Open the terminal selection menu (also available with --tui).
build/demo/graphics_hello/demo_graphics_hello

# Launch a module directly.
build/demo/graphics_hello/demo_graphics_hello --module triangle
build/demo/graphics_hello/demo_graphics_hello --module ray_query --backend metal

# Discover modules without initializing a graphics device.
build/demo/graphics_hello/demo_graphics_hello --list
```

The terminal UI lists module names and descriptions. Enter a number (1–11) or a
module name, followed by Enter. Invalid choices prompt again; `q`, `quit`, or EOF
exits without creating a window. This line-based menu works in IDE consoles and
with redirected input as well as native terminals. Closing a demo window exits
the executable normally.

| Module | Demonstration |
| --- | --- |
| `triangle` | Colored triangle |
| `blend` | Alpha blending |
| `cube` | Rotating cube |
| `texture` | Textured triangle |
| `resize` | Resizable window |
| `hdr` | HDR gradient |
| `sdr_sample` | SDR sampling |
| `raytracing` | Ray tracing pipeline and shader binding table |
| `rt_multi_shader_group` | Triangle and procedural sphere with multiple hit groups and a callable shader |
| `external_shader` | The same RT scene with shaders loaded from the assets directory |
| `ray_query` | Matching triangle/ellipsoid scene using inline triangle and procedural ray queries |

Shared options:

- `--module NAME`: select a module directly, bypassing the menu.
- `--tui`: explicitly open the menu; mutually exclusive with `--module`.
- `--backend auto|metal|vulkan|d3d12`: select a compiled backend; `auto` is the
  platform default. Startup logs show the module, actual API, device, and ray
  tracing/query capabilities.
- `--frames N`: exit after a positive number of rendered frames.
- `--list`, `--help`: list modules or print usage without opening a window.

Window titles retain the backend and demo name and show average FPS, refreshed
every half second. The first interval displays `FPS: --`.
All rotating geometry completes one revolution every two seconds, measured with
a monotonic clock from the first animation update, independently of frame rate.

Metal supports the seven raster modules and `ray_query` on compatible devices.
`raytracing`, `rt_multi_shader_group`, and `external_shader` require ray tracing
pipelines, which Metal does not implement;
its diagnostic recommends `--module ray_query`. Query support is checked
separately using `DeviceRayQuerySupport()`. Metal presentation currently uses
an SDR surface, so `hdr` warns that floating-point display values are clipped.

The ray query module matches `rt_multi_shader_group`: a rotating triangle at
x = -2 and a rotating, nonuniformly scaled sphere at x = +2, viewed from (0, 0, 5).
It builds triangle and AABB BLAS objects in a shared TLAS. The `cs_6_5` compute
shader intersects procedural candidates with an analytic unit sphere in object
space and commits the closest valid root. Triangle front/back colors and the
sphere's lighting/tint match the RT hit/callable shaders. Object-space directions
retain their length, and normals use the inverse transpose for correct ellipsoid
shading. Bounds checks handle partial 8-by-8 workgroups.

```sh
build/demo/graphics_hello/demo_graphics_hello --module rt_multi_shader_group --backend vulkan
build/demo/graphics_hello/demo_graphics_hello --module external_shader --backend d3d12
```

The external-shader module retains its runtime `VirtualFileSystem::LoadDirectory`
workflow and requires the assets submodule's `shaders/raytracing` files.

Validation examples:

```sh
python3 demo/graphics_hello/test_launcher.py build/demo/graphics_hello/demo_graphics_hello
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  build/demo/graphics_hello/demo_graphics_hello --module ray_query --backend metal --frames 10
```

The previous `demo_graphics_hello_*` targets are replaced by
`demo_graphics_hello --module NAME`. Module sources and shaders now live in
`modules/NAME/`; built-in shaders are embedded once through `shaders.cpp`.
The old `demo_graphics_rt_multi_shader_group` and `demo_graphics_external_shader`
executables are replaced by the corresponding launcher modules.
