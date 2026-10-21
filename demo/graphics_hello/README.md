# Graphics Hello

One executable, twelve built-in modules. Each module implements the `Module`
lifecycle interface and is created only when selected. Modules are statically
linked; no separate executables or dynamic plugins are needed.
Embedded modules share one shader bundle;
`external_shader` intentionally loads shader files from `assets/shaders/raytracing`.

```sh
cmake --build cmake-build-release --target demo_graphics_hello

# Open the terminal selection menu (also available with --tui).
cmake-build-release/demo/graphics_hello/demo_graphics_hello

# Launch a module directly.
cmake-build-release/demo/graphics_hello/demo_graphics_hello --module triangle
cmake-build-release/demo/graphics_hello/demo_graphics_hello --module ray_query --backend metal

# Discover modules without initializing a graphics device.
cmake-build-release/demo/graphics_hello/demo_graphics_hello --list
```

The terminal UI lists module names and descriptions. Enter a number (1–12) or a
module name, followed by Enter. Invalid choices prompt again; `q`, `quit`, or EOF
exits without creating a window. This line-based menu works in IDE consoles and
with redirected input as well as native terminals. Closing a demo window exits
the executable normally.

| Module | Demonstration |
| --- | --- |
| `window` | Clear sky-blue image |
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
Graphics windows request presentation without vertical synchronization by default:
D3D12 uses a zero sync interval, Vulkan prefers immediate presentation (falling
back to FIFO VSync if unavailable), and Metal disables display synchronization.
The window system or driver may still constrain the visible refresh rate.
All rotating geometry completes one revolution every two seconds, measured with
a monotonic clock from the first animation update, independently of frame rate.

Metal supports the eight raster modules and `ray_query` on compatible devices.
`raytracing`, `rt_multi_shader_group`, and `external_shader` require ray tracing
pipelines, which Metal does not implement;
their diagnostics recommend `--module ray_query`. Query support is checked
separately using `DeviceRayQuerySupport()`. Metal supports HDR presentation through macOS EDR: `SetHDR(true)` switches to
`RGBA16Float` with extended linear sRGB, retaining values above SDR white (1.0).
The `hdr` module shows an HDR gradient above an SDR-white reference bar. Press
**H** to switch HDR/SDR presentation while keeping the rendered image unchanged.
Startup logs report the display's current and potential EDR headroom. Visible
highlight brightness requires headroom above 1.0; macOS adjusts it with display
capabilities, brightness, and system conditions. The HDR gradient retains the
existing shader's `pow(color, 2.2)` response, reaching about 11.2 times reference
white before the display's own range handling.

The ray query module matches `rt_multi_shader_group`: a rotating triangle at
x = -2 and a rotating, nonuniformly scaled sphere at x = +2, viewed from (0, 0, 5).
It builds triangle and AABB BLAS objects in a shared TLAS. The `cs_6_5` compute
shader intersects procedural candidates with an analytic unit sphere in object
space and commits the closest valid root. Triangle front/back colors and the
sphere's lighting/tint match the RT hit/callable shaders. Object-space directions
retain their length, and normals use the inverse transpose for correct ellipsoid
shading. Bounds checks handle partial 8-by-8 workgroups.

```sh
cmake-build-release/demo/graphics_hello/demo_graphics_hello --module rt_multi_shader_group --backend vulkan
cmake-build-release/demo/graphics_hello/demo_graphics_hello --module external_shader --backend d3d12
```

The external-shader module retains its runtime `VirtualFileSystem::LoadDirectory`
workflow and requires the assets submodule's `shaders/raytracing` files.

Validation examples:

```sh
python3 demo/graphics_hello/test_launcher.py cmake-build-release/demo/graphics_hello/demo_graphics_hello
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cmake-build-release/demo/graphics_hello/demo_graphics_hello --module ray_query --backend metal --frames 10
```

## Offscreen benchmarks

```sh
cmake-build-release/demo/graphics_hello/demo_graphics_hello --headless --all --backend vulkan --frames 100000
cmake-build-release/demo/graphics_hello/demo_graphics_hello --headless --module cube --backend d3d12 --frames 100000
```

`--headless` supports `window`, `triangle`, `cube`, and `texture`, corresponding to
`rhi_hello`'s `hello_window`, `hello_triangle`, `hello_cube`, and `hello_texture`.
`--all` runs these four modules and requires `--headless`; it cannot be combined
with `--module` or `--tui`. Other modules reject headless mode. The default is
600 measured frames when `--frames` is omitted. Python demos remain windowed.

No GLFW window, swapchain, Present, screenshot, or event loop is used. Each module
creates its own device and resources. The benchmark uses 1280x720 RGBA8 images,
three cyclic color targets and three configured frames in flight. Sixty warmup
frames and a GPU drain precede timing; the final GPU drain is included in timing.
Initialization, shader compilation and teardown are excluded. The reported FPS
measures CPU recording/submission plus completed GPU work, not GPU-only duration.

Headless geometry, formats and draw counts match the RHI scenes: clear only,
one indexed RGB triangle, a 36-index cube with D32 depth and a per-frame matrix
upload, and a 256x256 XOR texture on one indexed triangle without depth.
Cube rotation advances by 0.01 radians each frame, including warmup. Headless
triangle uses static vertex/index buffers and the RHI triangle dimensions.
Windowed modules retain their original formats and animation timing.

The libraries retain their own command, barrier, upload and synchronization
implementations: graphics uses its Core frame ring (waiting for the next slot
at the end of Submit), while RHI waits on a submission token before slot reuse.
Graphics dynamic uniform uploads use its existing transfer path; RHI records
uploads into its command context. Results compare these end-to-end APIs, rather
than identical native command streams. Run Release builds on the same GPU with
validation disabled, one benchmark process at a time, and compare repeated
runs (for example, three runs of 100000 frames and their medians).

The comparison script runs both executables serially, alternates their order,
prints median FPS and ratios, and writes raw logs plus JSON under `out/`:

```sh
python demo/graphics_hello/benchmark_compare.py --frames 100000 --runs 3
```

It defaults to D3D12 and Vulkan. Repeat `--backend` to select other compiled
backends. Validation layers are disabled by the script for measurement; run
Debug/validation smoke tests separately. Use a Release build directory and
check device logs to confirm that both libraries selected the same GPU.

## Python

`python/graphics_hello.py` is the Python counterpart, built on the installed
`long_march` module. It retains the original eleven windowed modules, options, menu, window
titles, FPS display and animation timing, and compiles the shaders in
`modules/NAME/shaders/`. Each module is also a standalone script,
`python/graphics_hello_NAME.py`, which accepts `--backend` and `--frames`.

```sh
python3 python/graphics_hello.py --module ray_query --backend vulkan
python3 python/graphics_hello_cube.py --backend d3d12 --frames 60
```

The previous `demo_graphics_hello_*` targets are replaced by
`demo_graphics_hello --module NAME`. Module sources and shaders now live in
`modules/NAME/`; built-in shaders are embedded once through `shaders.cpp`.
The old `demo_graphics_rt_multi_shader_group` and `demo_graphics_external_shader`
executables are replaced by the corresponding launcher modules.
