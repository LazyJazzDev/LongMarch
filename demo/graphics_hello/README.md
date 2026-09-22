# Graphics Hello

One executable, nine built-in modules. Each module implements the `Module`
lifecycle interface and is created only when selected. Modules are statically
linked; no separate executables, dynamic plugins, or external shader files are
needed at runtime.

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

The terminal UI lists module names and descriptions. Enter a number (1–9) or a
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
| `ray_query` | Native acceleration-structure traversal in a compute shader |

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

Metal supports the seven raster modules and `ray_query` on compatible devices.
`raytracing` requires a ray tracing pipeline, which Metal does not implement;
its diagnostic recommends `--module ray_query`. Query support is checked
separately using `DeviceRayQuerySupport()`. Metal presentation currently uses
an SDR surface, so `hdr` warns that floating-point display values are clipped.

The ray query module builds a triangle BLAS and instance TLAS, updates the
instance transform every frame, and traces camera rays in an 8-by-8 compute
shader (`cs_6_5`). Hits display barycentrics; misses display a uniform background.

Validation examples:

```sh
python3 demo/graphics_hello/test_launcher.py build/demo/graphics_hello/demo_graphics_hello
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  build/demo/graphics_hello/demo_graphics_hello --module ray_query --backend metal --frames 10
```

The previous `demo_graphics_hello_*` targets are replaced by
`demo_graphics_hello --module NAME`. Module sources and shaders now live in
`modules/NAME/`; the executable embeds all shaders once through `shaders.cpp`.
