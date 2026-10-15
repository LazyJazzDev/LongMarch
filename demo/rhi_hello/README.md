# RHI pointer API window and device creation demo

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

`grassland::rhi` is the pointer API. Its `rhi_*.h` headers declare pure virtual
`ICore`, `IDevice`, `IWindow`, `IImage`, `IQueue`, `ICommandContext`,
`IRecordedCommands` and `ISubmissionToken` interfaces without backend state or
native API headers. Native implementations and shared implementation helpers stay
under `backend/`.

`grassland::rhix` is the optional RAII and dot-syntax extension, included through
`grassland/rhi/rhix/rhix.h`. Both layers ship in `grassland_rhi`; using the pointer
API does not require including or using rhix. This demo uses only raw interface
pointers and explicitly releases every owned reference, including on failure.
The companion [rhix_hello](../rhix_hello/README.md) demonstrates the same behavior
with containers. The following illustrates the optional container API:

```cpp
namespace rhi = grassland::rhi;
namespace rhix = grassland::rhix;

// Typed pointer container: a raw factory result transfers one owned reference.
rhix::ObjPtr<rhi::ICore> core = rhi::CreateCore();
if (!core) return;
rhix::ObjPtr<rhi::IDevice> device = core->CreateDeviceByAutoSelection();
if (!device) return;

auto shared = device;             // Retain: an additional reference.
auto moved = std::move(shared);   // Transfer: shared becomes empty.
auto borrowed = device.Get();     // Borrow only; no implicit pointer conversion.
auto retained = rhix::ObjPtr<rhi::IDevice>::Retain(borrowed);
// Never adopt the same owned reference twice or adopt a borrowed Get() result.
```

The dot wrappers use the same public interfaces and can also adopt factory results:

```cpp
rhix::Core core = rhi::CreateCore();
if (!core) return;
rhix::Device device = core.CreateDeviceByAutoSelection();
```

Without any container, use `rhi::ICore *core = rhi::CreateCore()` and release each
owned pointer with `Release()`, children before parents. Factories return one
owned reference or `nullptr`. Ordinary object parameters are borrowed. The pointer
`ICommandContext::Finish()` returns a separate owned `IRecordedCommands` reference;
the caller still releases its context reference. `IQueue::Submit()` borrows the
packet and retains accepted work. A packet can be submitted only once; unfinished
contexts and repeat submissions are rejected. rhix wraps these operations with
the consuming semantics described below.

D3D12 and Vulkan then open a blue GLFW window. The title displays FPS measured
by `grassland::FPSCounter`, refreshed four times per second. Press **P** to switch between
VSync and continuous presentation; **Escape** or the close button exits. Continuous
presentation falls back to VSync when unavailable. `Window::GetPresentMode()`
reports the actual mode. Resizing and minimization are handled by the Window.
Metal retains the device-query-only behavior in this iteration.

After window creation, both demos print `GetSupportedPresentationOptions()`:
the presentation-mode/pixel-format combinations implemented by RHI and supported
by this window. These are SDR combinations: D3D12 currently uses RGBA8 UNORM;
Vulkan supports the surface's RGBA8/BGRA8 UNORM formats with sRGB nonlinear color
space. Continuous mode is listed only when available without VSync fallback.
Pixel format selection remains automatic; this query does not add format selection
or report unimplemented HDR/native formats. An empty result indicates failure.

```cpp
auto window = device.CreateGLFWWindow({1280, 720, "RHI"});
auto graphics = device.GetQueue(QUEUE_TYPE_GRAPHICS);
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

The rhix dot wrappers derive from `rhix::ObjectContainer`, which stores a
`ObjPtr<rhi::IObject>` and provides `Release()`. Release clears only that handle's reference;
it does not force GPU completion or invalidate other copies. Destruction releases
the reference automatically. Contexts and recorded commands remain move-only.

The caller must release **all child handle copies before their parent**, including
copies stored in containers or other threads. Release frames and recordings before
their Window, all Device children before Device, and Device before Core. Stop
concurrent callers before releasing parents. Reversing this order violates the API
contract; children do not retain their parent automatically.

```cpp
// After leaving the rendering loop (local frames/contexts have been released):
graphics.Release();
window.Release();
device.Release();  // Drains submitted work, then destroys managed resources.
core.Release();
```

Swapchain Images are **single-use borrowed frames**, not independently owned GPU
allocations. Release the caller's frame before the next acquisition. Retaining a
frame across resize violates the API contract; this implementation invalidates
the old image rather than silently redirecting it to another native image.
Already submitted GPU use is drained before swapchain replacement. Recording
retains the exact image immediately, and the Device retains submitted command
contexts until completion even if the caller drops the image or completion token.
An empty acquired frame can mean a minimized/out-of-date window; creation and
submission failures also return empty handles and log the native error.

Window, Image, Queue and CommandContext wrap their corresponding public `rhi::I*`
interfaces. Common implementation bases and D3D12/Vulkan implementations stay
in the backend object files; rhix has no backend includes or Access bridge. Window owns its
swapchain. `D3D12Image` and `VulkanImage` provide common backend resource access
without depending on Window. Their `D3D12SwapChainImage` and
`VulkanSwapChainImage` subclasses carry the counted frame lease, borrowed Window
pointer, frame validity and presentation state. Vulkan acquisition synchronization
also belongs to the swapchain subclass. The frontend still receives `IImage`:
callers follow the ownership and lifetime contract of the API that produced it,
without querying an ownership category. Independent images and pooled buffers
can use their own backend subclasses when those creation APIs are implemented.
Device owns managed Windows and reclaims them once caller references and frame
leases are gone, during completion collection, subsequent window creation, or
Device shutdown. Returning a frame lease does not itself destroy its parent.
Recorded and submitted uses retain the Image, so its lease remains active until
those uses end. Native command storage belongs to the backend
CommandContext, and completion collection belongs to Device. GLFW initialization,
events and deferred main-thread window destruction live in `backend_glfw_helper`.
Window operations and presentation must run on the main thread. Device directly
owns the native device, queue and synchronization state; there is no separate
`BackendLifetime`. Queue and CommandContext access Device without owning it.
Device shutdown drains in-flight work and stops the collector before clearing
managed Windows and destroying the native device. Resource pools added later must
follow the same contract: children retain leases, and the higher-level owner
decides when the pool can be reset or destroyed. This change adds no general pool.

Build `test_rhi` in the same CLion profile. Set `LONGMARCH_RHI_WINDOW_TESTS=1` on
a desktop to include the native multi-window tests. Vulkan validation can be
enabled with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

Metal snapshots and retains the available native devices for stable IDs within
each Core. Creating a Device retains that native `MTLDevice` and borrows its Core;
supported capabilities are exposed automatically, without a feature enable mask.
Ray Query follows `MTLDevice::supportsRaytracing`. The DXR/Vulkan-style ray tracing
pipeline and geometry shader flags are false; Metal intersectors inside compute
or render shaders do not imply those pipeline stages are available.

Metal reports host physical memory as shared/unified memory. Dedicated VRAM
capacity is unknown (zero): `recommendedMaxWorkingSetSize` is a performance
budget, not physical VRAM capacity, and is not substituted for it. Shared memory
is not additional VRAM or a guarantee that the application can allocate it all.

The Metal path tests enumeration and device creation, not rendering or ray tracing.
The Metal native tests also execute and verify a buffer fill after releasing a
sibling Device, while keeping Core alive, using Metal directly. This is an ownership
test, not an implementation of RHI command submission.
