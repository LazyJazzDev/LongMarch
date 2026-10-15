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
and `ISubmissionToken` interfaces without backend state or
native API headers. Native implementations and shared implementation helpers stay
under `backend/`.

`grassland::rhi::ObjPtr<T>` is the interface pointer container, available through
`grassland/rhi/rhi_obj_ptr.h` or `grassland/rhi/rhi.h`.
`grassland::rhix` is the optional typed dot-syntax extension built on it, included through
`grassland/rhi/rhix/rhix.h`. Both layers ship in `grassland_rhi`; using the pointer
API does not require including or using rhix. This demo uses `rhi::ObjPtr<IXXX>` to manage owned references and calls
the pointer interfaces with `->`. Declaration order releases children before
parents, including on early returns.
The companion [rhix_hello](../rhix_hello/README.md) demonstrates the same behavior
with typed wrappers. The following illustrates the RHI pointer container:

```cpp
namespace rhi = grassland::rhi;
namespace rhix = grassland::rhix;

// Typed pointer container: a raw factory result transfers one owned reference.
rhi::ObjPtr<rhi::ICore> core = rhi::CreateCore();
if (!core) return;
rhi::ObjPtr<rhi::IDevice> device = core->CreateDeviceByAutoSelection();
if (!device) return;

auto shared = device;             // Retain: an additional reference.
auto moved = std::move(shared);   // Transfer: shared becomes empty.
auto borrowed = device.Get();     // Borrow only; no implicit pointer conversion.
auto retained = rhi::ObjPtr<rhi::IDevice>::Retain(borrowed);
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
`IQueue::Submit()` borrows the context, ends recording, and retains accepted work.
A context can be submitted only once; Debug rejects repeat submissions. The caller retains its own reference and may release it after Submit.
The rhix overload consumes a moved CommandContext, including on failure.

Caller-contract checks use `#if !defined(NDEBUG)` throughout RHI and rhix. They
cover invalid device IDs, object type/device mismatches, recording/submission
order, stale-frame use, window-thread requirements and calls on empty wrappers.
Release assumes callers satisfy these contracts; invalid calls have no guaranteed
failure return. Checking an empty handle with `bool`, releasing empty handles,
explicit state queries, minimized-window handling, allocation/native API failures
and GPU completion/error handling remain available in every configuration.

Backend command implementations call `RetainResource(IObject*)` before encoding
native resource use. CommandContext stores strong references in an object-identity
map, retaining each object once even when several commands use it. The same path
supports images and future buffer/binding-group objects without per-type storage.
Presentation requests are stored separately from resource lifetime and state tracking;
there is no single-frame field in the common CommandContext.
D3D12 and Vulkan directly implement CmdClearColor and CmdPresent. Each backend
tracks resource states locally to the context and inserts barriers as commands
use resources. D3D12 tracks native resource states and their defaults; Vulkan
tracks image layouts, pipeline stages and access masks, including same-layout
write dependencies. Before Close/vkEndCommandBuffer, every tracked resource is
restored to its default, even when no Present was recorded. Recording and
abandoning a context do not mutate a shared current-resource-state variable.
Swapchain defaults are D3D12 COMMON/PRESENT and Vulkan PRESENT_SRC_KHR. Vulkan
establishes that layout once on the first acquisition of each physical image,
using Window-owned initialization command storage and waiting for completion
before exposing it. Each acquisition semaphore is consumed only by the first
successful submission using that acquired image; later contexts use queue order.
Buffer/BindingGroup APIs and subresource-specific transitions are not implemented
in this iteration; the current state paths operate on full color images.

Unsubmitted contexts release all references on destruction; Device keeps submitted
contexts alive through GPU completion. Retaining a context longer also retains its
resources. Future binding commands must retain the concrete resources or immutable
binding snapshot used at recording time, rather than resolve mutable bindings at Submit.


D3D12 and Vulkan then open a blue GLFW window. The title displays FPS measured
by `grassland::FPSCounter`, refreshed four times per second. Press **P** to switch between
VSync and continuous presentation; **Escape** or the close button exits. Continuous
presentation falls back to VSync when unavailable. `Window::GetPresentMode()`
reports the actual mode. Window installs its own GLFW framebuffer-size and restore callbacks
at creation, rebuilding the swapchain when size events are dispatched (normally
by `PollEvents()`). `SetPresentMode()` also rebuilds internally. There is no public
rebuild function and the demos do not install resize callbacks. GLFW user pointers
remain available to callers; replacing the framebuffer-size or iconify callback requires
chaining the previous callback so the internal resize handling is preserved.

`GetNextImage()` returns an owned reference to Window's prepared next image.
Repeated calls return the same object until a context recording `CmdPresent(window)`
is submitted. Submission presents that image and prepares the next one before
returning. Dropping caller references does not advance the frame. Size/mode changes
replace the prepared image as part of rebuilding. Only minimization or zero size
returns an empty image; acquisition/rebuild work never occurs inside the getter.
Creation/mode-change failures are reported at those calls, and resize callback
failure is reported by `PollEvents()`. Device-loss recovery is not implemented.
Metal implements the same presentation and submission contract using CAMetalLayer
and retained-reference Metal command buffers. Graphics and transfer queues share
one native command queue. SDR BGRA8 supports both VSync and continuous presentation.

`CmdClearColor` records work on an Image, while `CmdPresent` takes the target
Window (`IWindow *` in RHI, `const rhix::Window &` in rhix). The current backend
captures that window's next image during recording and does not require a preceding clear.
Submission presents the recorded frame; it does not look up a different frame
or copy an arbitrary image into the window.

Window input uses GLFW's existing key definitions, e.g. `IsKeyDown(GLFW_KEY_P)`.
`IWindow::GLFWWindow()` (also exposed by rhix) returns a borrowed `GLFWwindow *`
for other GLFW operations on the main thread. Window retains ownership: callers
must not destroy the handle or terminate GLFW while RHI windows are alive.

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
auto image = window.GetNextImage();
if (image) {
  auto context = device.CreateCommandContext();
  if (!context) return;
  if (!context.CmdClearColor(image, {0.6f, 0.7f, 0.8f, 1.0f})) return;
  if (!context.CmdPresent(window)) return;
  auto done = graphics.Submit(std::move(context));
  if (!done) return;
  // done.WaitHost() is optional; GPU completion is collected automatically.
}
```

The command path supports interleaved clears of multiple images, clear-only contexts,
present-only contexts, multiple window presentations, and empty submissions. Device creates contexts without a queue argument.
Graphics and Transfer are logical roles with the same full graphics capabilities,
sharing one persistent native graphics queue. Either role can submit a context.
`SubmitInfo::waits` accepts tokens
from either role on that Device; native queue order supplies those dependencies
without blocking the host. rhix contexts are move-only; Submit consumes the
context even on failure. There is no separate Finish or recorded-commands object.

The rhix dot wrappers derive from `rhix::ObjectContainer`, which stores a
`ObjPtr<rhi::IObject>` and provides `Release()`. Release clears only that handle's reference;
it does not force GPU completion or invalidate other copies. Destruction releases
the reference automatically. Command contexts remain move-only.

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
An empty next image means a minimized or zero-sized window; creation and
submission failures also return empty handles and log the native error.

Window, Image, Queue and CommandContext wrap their corresponding public `rhi::I*`
interfaces. Common implementation bases and D3D12/Vulkan implementations stay
in the backend object files; rhix has no backend includes or Access bridge. Window owns its
swapchain. `D3D12Image` and `VulkanImage` provide common backend resource access
without depending on Window. Their `D3D12SwapChainImage` and
`VulkanSwapChainImage` subclasses carry a borrowed Window pointer, frame validity
and presentation state. Vulkan acquisition synchronization
also belongs to the swapchain subclass. The frontend still receives `IImage`:
callers follow the ownership and lifetime contract of the API that produced it,
without querying an ownership category. Independent images and pooled buffers
can use their own backend subclasses when those creation APIs are implemented.
Window lifetime belongs to its callers; Device does not retain or defer reclamation
of Windows. Release caller-held frames and recordings before their Window, and
perform swapchain replacement outside command recording without retaining old
frames. Window destruction and swapchain replacement wait for native GPU use
before destroying the underlying swapchain resources. No frame-lease counter is
needed. Contexts still strongly retain resources used by submitted commands.
Native command storage belongs to the backend
CommandContext, and completion collection belongs to Device. GLFW initialization,
events and deferred main-thread window destruction live in `backend_glfw_helper`.
Window operations and presentation must run on the main thread. Device directly
owns the native device, queue and synchronization state; there is no separate
`BackendLifetime`. Queue and CommandContext access Device without owning it.
Device shutdown drains in-flight work and stops the collector before destroying
the native device; caller-owned Windows must already have been released. Resource pools added later must
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

Metal tests cover enumeration, device creation, native buffer execution, ordered
RHI clears verified by pixel readback, multi-window presentation, retained old
image wrappers, resizing, and restoration after mode changes while minimized.
Window tests require `LONGMARCH_RHI_WINDOW_TESTS=1` on a desktop; use
`MTL_DEBUG_LAYER=1` to enable Metal API validation. Ray tracing is not exercised.
