# RHIX container API window and device creation demo

Build `demo_rhix_hello` using the CLion Release or Debug CMake profile:

```sh
cmake-build-release/demo/rhix_hello/demo_rhix_hello vulkan
cmake-build-release/demo/rhix_hello/demo_rhix_hello d3d12
```

This demo uses `grassland::rhix` containers and dot-syntax methods. The companion
[rhi_hello](../rhi_hello/README.md) uses raw `grassland::rhi` interface pointers.
Both enumerate devices and capabilities, automatically select a device, and
clear a GLFW window to RGB (0.6, 0.7, 0.8). Window titles show FPS measured by
`grassland::FPSCounter`. Press **P** to toggle VSync/continuous presentation;
**Escape** or the close button exits. Continuous presentation may fall back to
VSync. The optional `metal` backend currently demonstrates device queries only.
Both demos print the window's supported RHI presentation-mode/pixel-format
combinations at startup via `GetSupportedPresentationOptions()`; format selection
remains automatic. See the pointer demo documentation for the query's SDR scope.

Containers adopt factory results, retain on copy, and transfer on move.
`context.Finish()` consumes the context, and `graphics.Submit()` consumes the
recorded-command container. Temporary images and submission tokens release their
references automatically; the backend retains accepted work until completion.
Parents are declared before their children so normal and early returns release
children first. Normal shutdown also demonstrates explicit `Release()` calls.
See the pointer demo documentation for the shared lifetime and backend contracts.
