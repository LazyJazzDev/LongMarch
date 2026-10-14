# RHI window and device creation demo

Build the `demo_rhi_hello` target using the CLion Release or Debug CMake profile.
Run the executable with `metal`, `vulkan`, or `d3d12` to select a compiled backend;
with no argument, it uses the platform default (Metal on Apple platforms).

```sh
cmake-build-release/demo/rhi_hello/demo_rhi_hello vulkan
cmake-build-release/demo/rhi_hello/demo_rhi_hello d3d12
```

The demo enumerates devices, prints their memory and capabilities, and creates
the automatically selected device. It links the independent `grassland_rhi`
module and does not use the old `grassland::graphics` runtime.

D3D12 and Vulkan then open a blue GLFW window. The title displays FPS measured
by `grassland::FPSCounter`, refreshed four times per second. Press **P** to switch between
VSync and continuous presentation; **Escape** or the close button exits. Continuous
presentation falls back to VSync when unavailable. `Window::GetPresentMode()`
reports the actual mode. Resizing and minimization are handled by the Window.
Metal retains the device-query-only behavior in this iteration.

```cpp
auto window = device.CreateWindow({1280, 720, "RHI"});
auto graphics = device.GetQueue(QueueType::Graphics);
// Inside the main-thread event loop:
auto image = window.AcquireFrame();
if (image) {
  auto context = device.CreateCommandContext(graphics);
  if (context.ClearColor(image, {0.6f, 0.7f, 0.8f, 1.0f}) && context.Present(image)) {
    auto done = graphics.Submit(context.Finish());
    // done.WaitHost() is optional; GPU completion is collected automatically.
  }
}
```

The first command path supports one full-image clear/present pair per context,
or an empty submission. Graphics and Transfer are distinct logical queue handles
sharing one persistent native graphics queue. `SubmitInfo::waits` accepts tokens
from either role on that Device; native queue order supplies those dependencies
without blocking the host. Contexts and recorded commands are move-only; Finish
and Submit consume their input even on failure.

Swapchain Images are **single-use borrowed frames**, not independently owned GPU
allocations. Release the caller's frame before the next acquisition. Retaining a
frame across resize violates the API contract; this implementation invalidates
the old image rather than silently redirecting it to another native image.
Already submitted GPU use is drained before swapchain replacement. Recording
retains the exact image immediately, and the Device retains submitted command
contexts until completion even if the caller drops the image or completion token.
An empty acquired frame can mean a minimized/out-of-date window; creation and
submission failures also return empty handles and log the native error.

Window, Image, Queue and CommandContext each map directly to their `I*` interface
and D3D12/Vulkan implementation in the corresponding object files. Window owns its
swapchain; Image retains that Window. Native command storage belongs to the backend
CommandContext, and completion collection belongs to Device. GLFW initialization,
events and deferred main-thread window destruction live in `backend_glfw_helper`.
Window operations and presentation must run on the main thread. The Device's
nested native lifetime keeps resource destruction safe without retaining the
completion collector through an in-flight resource cycle.

Build `test_rhi` in the same CLion profile. Set `LONGMARCH_RHI_WINDOW_TESTS=1` on
a desktop to include the native multi-window tests. Vulkan validation can be
enabled with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

Metal snapshots and retains the available native devices for stable IDs within
each Core. Creating a Device retains that native `MTLDevice` and its Core;
supported capabilities are exposed automatically, without a feature enable mask.
Ray Query follows `MTLDevice::supportsRaytracing`. The DXR/Vulkan-style ray tracing
pipeline and geometry shader flags are false; Metal intersectors inside compute
or render shaders do not imply those pipeline stages are available.

Metal reports host physical memory as shared/unified memory. Dedicated VRAM
capacity is unknown (zero): `recommendedMaxWorkingSetSize` is a performance
budget, not physical VRAM capacity, and is not substituted for it. Shared memory
is not additional VRAM or a guarantee that the application can allocate it all.

The Metal path tests enumeration and device creation, not rendering or ray tracing.
The Metal native tests also execute and verify a buffer fill after releasing the
calling Core and a sibling Device, using Metal directly. This is an ownership
test, not an implementation of RHI command submission.
