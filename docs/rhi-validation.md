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
D32 returns float depth. D24S8 returns little-endian words with depth in bits 0-23 and
stencil in bits 24-31; Metal's native D32S8 depth is quantized to that public layout.
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

Capture is optional and disabled by default. When enabled, it records a download
every frame so closing the window can save the last rendered frame, then waits only
at exit to write the files. This adds readback overhead and should not be used for
rendering performance comparisons.

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
