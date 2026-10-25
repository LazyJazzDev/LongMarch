# RHI native validation

`CreateCore(api, CoreDesc{enable_debug})` selects native backend diagnostics.
Omitting the descriptor enables them when `NDEBUG` is undefined and disables them otherwise;
callers can explicitly pass `{false}` for profiling or `{true}` for Release diagnostics.
The caller must ensure only one Core exists at a time and release its children before releasing it.

D3D12 Core owns the debug controller and disables the layer on destruction after
caller-owned Devices have been released. Vulkan validation is configured per instance.
Requested but unavailable D3D12/Vulkan validation fails Core creation.

Metal uses `MTLCommandBufferErrorOptionEncoderExecutionStatus` for every RHI command
buffer when enabled, including queue-idle markers. Disabled Cores use no enhanced
error options. Command-buffer failures log the native error description and, when
enabled, its diagnostic user information. This setting is per Core and does not modify process environment variables.
Metal API Validation is separate: enable it through Xcode or `MTL_DEBUG_LAYER=1`
before launching the process. `enable_debug=false` does not disable that external layer.
External validation tools/environment overrides remain external on all backends.

D3D12 Devices forward native warning/error messages (including message IDs) to the
same console logger used by Vulkan through ID3D12InfoQueue1 callbacks. Callbacks are
unregistered when the Device is destroyed. Older runtimes without this interface
retain native debugger output and emit a console warning about unavailable forwarding.

Metal command-buffer descriptors are immutable per Device and reused for command
creation. Command execution, idle-marker and shader compilation failures use the
console logger; native NSError reports include domain and code. Enhanced command
buffer diagnostics also include user information. Metal API Validation assertions
are emitted directly by Apple's layer to stderr and may terminate the process
before a command buffer completes; they are not forwarded through an RHI callback.
`MTLLogState` handles shader logging and is not an API Validation callback.

## Format-aware image clears

RHI and RHIX use CmdClearImage(image, ClearValue). ColorAttachmentInfo and
DepthStencilAttachmentInfo store ClearColorValue and ClearDepthStencilValue
respectively in clear_value. The image format selects the
union member; initialize color.float32 for UNORM/SFLOAT colors, color.int32 for
SINT colors, color.uint32 for UINT colors, or depth_stencil for depth formats.

```cpp
ClearValue color{0.1f, 0.2f, 0.3f, 1.0f};
commands->CmdClearImage(color_image, color);

ClearValue integer{0xffffffffu, 16777217u, 0x80000001u, 0u};
commands->CmdClearImage(uint_image, integer);

ClearValue depth{ClearDepthStencilValue{1.0f, 0}};
commands->CmdClearImage(depth_image, depth);

RenderingInfo rendering;
rendering.color_attachments = {
    {uint_image, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, integer.color}};
rendering.depth_stencil.image = depth_image;
rendering.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
rendering.depth_stencil.stencil_load_op = ATTACHMENT_LOAD_OP_CLEAR;
rendering.depth_stencil.clear_value = depth.depth_stencil;
```

Four-component constructors select float32, int32 or uint32 from their argument
types. Use floating-point literals for UNORM/SFLOAT clears, signed integers for
SINT, and uint32_t values (or unsigned literals) for UINT; keep all four arguments
the same type to avoid ambiguous overloads.

Standalone depth clears also clear stencil when the image format includes it.
Rendering scopes retain independent depth/stencil load and store operations;
their default depth clear value remains 1 and stencil remains 0.

Vulkan passes the corresponding native clear member. Metal converts integer color
components to its double-valued clear color without losing 32-bit integer precision.
D3D12 uses integer UAV clears for integer targets and preserves descriptor bindings
and resource ordering; integer targets without UAV support report an error and fail recording.
Floating color and depth/stencil clears use their respective native clear commands.
Integer values never pass through float. Tests read back extreme signed/unsigned
32-bit values, 8-bit negative values, depth/stencil, and interleaved clear/dispatch
results.

## Buffer and image readback

Record downloads outside a rendering scope, before submitting the command context:

```cpp
ObjPtr<ICommandContext> commands = device->CreateCommandContext();
ObjPtr<IDownloadToken> bytes = commands->CmdDownloadBuffer(buffer.Get(), offset, size);
ObjPtr<IDownloadToken> pixels = commands->CmdDownloadImage(image.Get(), {x, y, width, height});
if (!bytes || !pixels) return false;
ObjPtr<ISubmissionToken> submitted = queue->Submit(commands.Get());
if (!submitted || !pixels->WaitHost()) return false;
std::vector<uint8_t> cpu(pixels->GetSize());
if (!pixels->CopyTo(cpu.data(), cpu.size())) return false;
```

RHIX exposes the same operations through `CommandContext` and `DownloadToken`.
Query status only after the submission call has returned, and publish tokens to
other threads only after submission. `GetStatus()` delegates to the associated
SubmissionToken, or returns PENDING when no submission is associated. The caller
manages abandonment/cancellation of recordings; it is not fed back into downloads.
`WaitHost()` delegates to SubmissionToken and returns false if none is associated.
Callers may release their SubmissionToken handle while retaining the DownloadToken.

Call `CopyTo()` only after observing COMPLETED or a successful `WaitHost()`, and
perform copies on a single thread. DownloadToken does not lock, duplicate completion
state, or recheck readiness in `CopyTo()`. Destination/capacity validation remains.
Release all tokens before their Device, like other RHI resource handles.

Each token holds an intrusive lease from a Device-owned `TemporalBufferPool`, grouped
into power-of-two size classes starting at 64 KiB. Upload and download share the same
pool implementation and the existing UploadPool lease/reclamation semantics. D3D12
and Vulkan use separate instances for their upload/readback allocation policies;
Metal uses one Shared-storage pool for both directions. The command context retains
the token until GPU completion. Dropping the caller's token early cannot recycle storage still being
written by the GPU. Keeping a completed token preserves its snapshot through later
submissions; its final release returns the lease for reuse. Idle capacity shrinks
against the peak occupancy of the previous second, just as
in the original UploadPool. The pool template is implemented and explicitly
instantiated in a `.cpp`; native buffer/lease types remain statically known. Tokens
do not retain the source image or buffer after
command retirement, so keeping a screenshot token does not block swapchain rebuilds.

Buffer ranges are byte offsets and sizes; empty or out-of-bounds ranges fail. Image
ranges are 2D rectangles, with zero width/height meaning the remaining extent from
x/y. The returned image has `width * height * ImageFormatSize(format)` bytes, ordered
left-to-right and top-to-bottom without row padding. Color values retain their native
encoding, including BGRA channel order, packed 10-bit channels and half floats.
Depth readback tests use D32 only. Stencil-bearing images are not read back by default;
stencil rendering tests verify the color output instead. All demo depth attachments
use pure D32 float depth. Cube capture downloads both color and D32 depth each frame,
then saves the final matching pair at exit. `cube_depth.bin` retains float32 device
depth; `cube_depth.png` is an 8-bit grayscale preview of depth 0–1, without gamma
conversion or linearization.

D16 returns uint16 normalized depth; D32 returns float depth. D24S8 readback is forbidden.
D3D12 requires a full-subresource transfer for depth images; the token crops the
requested rectangle while copying to CPU memory. Vulkan invalidates mapped memory
before reading when required by the allocation's memory type.

Swapchain downloads must be recorded before submission; recording after `CmdPresent`
includes its ImGui overlay. Vulkan swapchains always enable transfer-source usage;
swapchain readback support is assumed. Independent images support readback on all
three backends.

Concrete backend DownloadToken implementations store the lease and native copy
layout as members. CPU copy, row compaction and stencil packing live in their `.cpp`
files, without `std::function` copy callbacks or `shared_ptr` lease control blocks.
Tokens query and wait on the existing SubmissionToken completion state; the collector
does not publish a second completion state to each DownloadToken.

## Demo readback checks

All modules of `demo_rhi_hello` and `demo_rhix_hello` accept `--save-last-frame DIR`:

```text
demo_rhi_hello --backend vulkan --all --frames 30 --save-last-frame out/readback/vulkan
demo_rhix_hello --backend d3d12 --all --headless --frames 30 --save-last-frame out/readback/d3d12
```

Each rendered module saves `<module>.png`, `<module>.bin` and `<module>.txt` at normal
exit. The binary file contains the exact compact pixels; the text file records extent,
format, color space, row pitch and reference-white scale. PNG is an SDR preview:
HDR is decoded relative to the window's reference white, converted to sRGB and clipped
to its range. Use the binary file to inspect unclipped HDR values. Multi-window ImGui
saves separate numbered files. Modules skipped for unavailable features or headless
window requirements produce no capture. Different invocations should use separate
directories to preserve previous results.

With --save-last-frame, every module records an image download after CmdPresent on
every rendered frame (after rendering in headless mode), and directly retains only
the newest DownloadToken and its image metadata. The render loop does not query,
wait on, or copy this token. Normal exit waits on the final token, copies its pixels
to CPU memory, and writes the files. Without the option, no image download, CPU
image copy, or file output is performed. Each demo owns its recording/exit flow;
save_image only encodes already downloaded CPU pixels.

Both nbody modules download GPU velocities asynchronously and plot a 100-bin speed
histogram using the same binning and peak normalization as `nbody_cs`. Only one
velocity download is pending at a time. Zero maximum speed is handled explicitly;
non-finite velocities fail the demo. With frame capture enabled, `nbody_speeds.csv`
contains the latest completed histogram, including bin edges, particle counts and
normalized values. The curve can lag rendering by a frame or more while the GPU is
busy. `*RhiDownload*` tests cover byte ranges, color/depth/stencil rectangles, compact
rows, early token release, retained snapshots, abandoned recordings, rejected
submissions remaining unassociated, direct and concurrent waits after releasing the
submission handle, and temporal-pool lease retention/reuse/shrinking. The Metal test
also checks that both pool accessors refer to the same pool and reuse the same storage.

## Image formats and vertex input types

RHI image format names spell out channel widths, for example
`IMAGE_FORMAT_R8G8B8A8_UNORM`, `IMAGE_FORMAT_R32G32_UINT`, and
`IMAGE_FORMAT_R32G32B32A32_SFLOAT`. Signed integers use the `_INT` suffix;
backends map it to native SINT formats. Float formats use `_SFLOAT`.

| Channel width | Encodings | Channel counts |
|---|---|---|
| 8 bits | UINT, INT, UNORM | 1, 2, 3, 4 |
| 32 bits | UINT, INT, SFLOAT | 1, 2, 3, 4 |

Existing BGRA8 UNORM, RGBA16 float, and packed RGB10A2/BGR10A2 UNORM formats
remain available under expanded names. Depth formats are `D16_UNORM`,
`D32_SFLOAT`, and `D24_UNORM_S8_UINT`. There is no portable native 32-bit UNORM
or 8-bit SFLOAT texture format, so those encodings are not exposed.

A format definition is not a guarantee of native image support. Metal has no
three-channel formats in this set; D3D12 has no RGB8 formats, and RGB32 texture
usage depends on its format capabilities. RHI images require sampled and attachment
usage; unsupported combinations return an empty image instead of changing its
channel count. Vulkan checks the required color-format features before allocation.

`VertexFormat` is now `InputType`, with `INPUT_TYPE_FLOAT[2/3/4]`,
`INPUT_TYPE_INT[2/3/4]`, and `INPUT_TYPE_UINT[2/3/4]` (the unsuffixed name is scalar).
Attributes use `InputAttributeDesc`, whose `type` member selects the `InputType`.
Each component occupies 32 bits. `InputTypeSize` returns the byte size; existing
vertex binding, attribute location, offset, and stride semantics are unchanged.
All RHI/RHIX call sites use the new names; old enum/type aliases are not retained.

Integer render targets require matching integer shader outputs and cannot enable
blending. Clear values remain numeric floats in the public API (not reinterpreted
integer bits), so values beyond float's exact integer range should be written via
upload or shader output. Vulkan converts clear values to the appropriate integer
union member. Typed storage images must use the matching shader format, e.g.
`[[vk::image_format("r8ui")]] RHI_RW_TEXTURE2D(0, 0, uint, output_image);`
for `IMAGE_FORMAT_R8_UINT`. Without an explicit qualifier, Slang can infer R32
from `uint`, which does not match an R8 image. Integer sampling should use `Load`
or nearest filtering, not linear filtering.

`RhiFormats` tests cover native-byte upload/subrectangle readback, transfer and
render-pass integer clears, D16 cropped readback, all 12 input types through real
vertex fetch and rasterization, and integer storage writes/render-target output
including full-width uint32 values. Stencil-bearing images are not downloaded.

D24S8 must not be passed to `CmdDownloadImage`. All three backends reject it in
Debug before encoding a copy, report the error, return no download token, and mark
the context invalid for submission. Release retains the caller-contract policy;
callers must not rely on a runtime check. The Debug rejection test never submits
a native D24S8 readback.

## Fixed topology and rasterizer state

`ProgramDesc::topology` selects triangle list (default), triangle strip, line list,
line strip, or point list. It controls both indexed and non-indexed draws. The
primitive type is fixed by the Program, even where a native API sets it on an
encoder/list. Primitive restart, adjacency and tessellation patches are outside
this API; indexed strips must not rely on restart/cut index values. Point shaders
should write `PSIZE` explicitly (for example 1).

`ProgramDesc::rasterizer` contains only these settings:

- `cull_mode`: NONE (default), FRONT or BACK.
- `front_face`: COUNTER_CLOCKWISE (default) or CLOCKWISE.
- `fill_mode`: SOLID (default) or WIREFRAME. Fill mode controls triangle interiors;
  it does not turn triangle input into line topology. Wireframe requires native
  support; Vulkan Program creation fails if `fillModeNonSolid` is unavailable.

Front-face and viewport semantics follow the existing graphics implementation.
After perspective division, counterclockwise winding in NDC (+X right, +Y up)
means front-facing when `FRONT_FACE_COUNTER_CLOCKWISE` is selected. Viewport x/y
identify the top-left framebuffer pixel and width/height must be positive; NDC
+Y maps towards the top of the viewport. The selected face controls culling,
`SV_IsFrontFace` and front/back stencil operations together.

| Backend | Native viewport | Public CCW maps to |
|---|---|---|
| Vulkan | x, y + height, width, -height | VK_FRONT_FACE_COUNTER_CLOCKWISE |
| D3D12 | TopLeftX=x, TopLeftY=y, positive width/height | FrontCounterClockwise=TRUE |
| Metal | originX=x, originY=y, positive width/height | MTL::WindingCounterClockwise |

The public default is explicit and follows graphics, rather than inheriting
D3D12/Metal's native clockwise default. Existing RHI programs keep no culling,
solid fill and triangle-list assembly. Programs using `SV_IsFrontFace` or distinct
front/back stencil state should use the documented winding convention.

Vulkan fixes these states at pipeline creation, without requiring dynamic-state
extensions. D3D12 stores rasterizer state and topology class in its PSO and applies
the exact topology when binding. Metal applies cull, winding and fill state on
Program binding and every new render encoder, and uses the selected primitive type
for draws. No depth bias, multisampling or other rasterizer settings are added.

`RhiRasterizer` pixel tests cover both windings and face settings, all cull modes,
offset viewport placement and Y orientation, every topology with indexed/non-indexed
draws, wireframe edges versus interiors, scope restarts and in-scope Program changes.
Invalid enums are rejected; Debug also rejects zero/negative viewport extents.
