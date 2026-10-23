# rhi_hello

Nine independent modules use `rhi::ObjPtr<IXXX>` pointer APIs. Each module creates its own Core, Device, Window and Queue; main only selects the module/backend and parses arguments.

```text
demo_rhi_hello --list
demo_rhi_hello --module window --backend vulkan
demo_rhi_hello --module triangle --backend d3d12 --frames 120
```

The positional backend argument remains supported. Without `--module`, a line-based terminal menu opens (also available with `--tui`). Enter a module number or name; invalid input retries, and `q`, `quit`, or EOF exits without opening a window. `--tui` and `--module` are mutually exclusive; `--backend auto` uses the default backend. Specify `--module` for unattended runs; omit `--frames` to run until the window closes. P switches requested VSync mode, Escape exits, and the title displays `grassland::FPSCounter` statistics.

- `window`: original RGB (0.6, 0.7, 0.8) clear/presentation and supported presentation-combination output. D3D12, Vulkan and the existing Metal backend are supported.
- `triangle`: D3D12/Vulkan/Metal colored indexed triangle on that background. The module creates independent GPU-only vertex/index Buffers, uploads once through CommandContext, creates a Slang Program, then records bindings, viewport/scissor and DrawIndexed. Resize/minimize/restore and P-mode changes are supported.

- `cube`: the rotating colored cube from `graphics_hello`, with a fixed uniform-buffer BindingGroup and a D32 depth attachment. The depth image follows the window extent. Uniform uploads take a CPU snapshot and are ordered with draws on the same queue.
- `texture`: the XOR texture triangle from `graphics_hello`, using a sampled image and linear sampler in a versioned BindingGroup.
- `blend`: the two overlapping triangles from `graphics_hello/blend`, first opaque and then alpha 0.5, over the same RGB (0.6, 0.7, 0.8) background. Renders directly to the RGBA8/BGRA8 target using two indexed draws; includes the same window controls and headless timing as other modules.

- `nbody`: shared-memory tiled compute simulation of 65,536 particles followed by additive particle drawing.

The first six modules and the depth/stencil, MRT and BindingGroup APIs are implemented on D3D12, Vulkan and Metal. Metal bindings require argument-buffer Tier 2 support. D24S8 uses native Depth32Float_Stencil8 on Metal, so depth precision differs from native 24-bit formats.


## Shader and Program contract

Shader stores original Slang source, entry point and stage; creation does not compile it. Program creation captures shaders and the C++ vertex layout, retains their references, compiles Slang and creates the native pipeline. Graphics, Compute and RayTracing programs are immutable and ready to bind when creation succeeds; compilation, layout or native pipeline failure returns an empty handle.

`RHI_INPUT(location, type, name)` is expanded by a generated prefix into the C++ layout's Vulkan/Metal attribute locations or D3D12 input semantics. Source BOM is preserved by Shader and omitted only when concatenating the compilation input. C++ `PipelineLayoutDesc` and `BindingLayoutDesc` describe groups, binding numbers, resource types, fixed array capacities and shader visibility. Layouts are copied into Programs/Groups and remain fixed; group resources can be rebound. Generated `RHI_UNIFORM`, `RHI_STRUCTURED_BUFFER`, `RHI_RW_STRUCTURED_BUFFER`, `RHI_TEXTURE2D`, `RHI_RW_TEXTURE2D`, `RHI_SAMPLER` and corresponding `_ARRAY` macros use this layout. Resource macros take `(group,binding,type,name)`; sampler macros omit `type`. Arrays get their length from C++, not from reflection. Reflection checks used resources against the expected type, position, array count and stage; unused declarations may be eliminated.

D3D12 assigns b/t/u/s registers in binding order within each group/space, and uses separate resource/sampler tables. Vulkan preserves group/binding as set/binding. Programs also describe up to eight color outputs (RGBA8/BGRA8, plus RGBA16 float and RGB10A2 on D3D12/Vulkan; BGR10A2 on Vulkan) and an optional D32 or D24S8 depth/stencil format, depth compare/write, front/back stencil operations, masks and reference. Triangle-list topology and float/float2/float3/float4 vertex attributes remain supported; culling and runtime-sized descriptor arrays are outside this change.

## Buffer and command lifetime

CreateBuffer returns a fixed GPU-only allocation managed by VMA, D3D12MA or private Metal storage. Backend Buffer bases expose resource access; AllocatedBuffer subclasses own allocations. Nested UploadPool::Lease subclasses forward access to pool-owned upload allocations; custom GPU-only resource pools can later implement the same interfaces without changing the public Buffer type. Each Device owns its allocator, upload pool and attachment pools; children borrow their Device.

CmdUploadBuffer copies CPU bytes immediately into an automatically leased upload Buffer, then records a native GPU copy. The caller can release/reuse the CPU data after the call. CmdCopyBuffer copies between distinct Buffers. UploadPool owns the physical AllocatedBuffers and returns backend Buffer-derived UploadPool::Lease objects. Leases forward the Buffer interfaces and mapped Write to the underlying storage, borrow their pool, and return occupancy on destruction of their last reference. CommandContext retains each lease through its ordinary resource table, with no separate uploads list or manual return path. Storage is recycled after GPU completion and final command reference release, or when an unsubmitted context is abandoned. Upload storage is grouped into independent power-of-two size classes starting at 64 KiB. Acquire takes an idle buffer directly from its class. Each class tracks allocated/occupied counts and a monotonic queue of occupancy changes over the last second; Release discards idle buffers above that recent peak. Equal occupancy samples keep the latest timestamp. Trimming occurs on returns to that class, without a background sweep. There is no per-frame reset or CPU wait per upload. Callers still order overwrites against prior GPU readers using queue ordering/dependencies.

CmdBindProgram precedes vertex-buffer and group binding. Bind all required vertex slots and groups again after changing Program. CmdBeginRendering selects an ordered color-attachment array and optional depth/stencil attachment, with independent LOAD/CLEAR/DISCARD and STORE/DISCARD operations (including separate depth/stencil operations); CmdEndRendering explicitly closes the scope. Draw calls require an active scope. Upload, copy, standalone CmdClearColor, Present and submission require no active scope. A scope may contain no draws (for example, a clear-only pass). RTV/framebuffer leases remain backend details. CmdDraw and CmdDrawIndexed support instancing and base offsets. Contexts retain the exact Program, Buffer and Image objects they use. Barriers are recorded immediately and used resources return to default states when submission ends recording.

D3D12 leases RTV slots from Device-owned heap pages. Vulkan prefers dynamic rendering when the negotiated Vulkan 1.3 core feature or Vulkan 1.2 VK_KHR_dynamic_rendering feature is supported. Program uses attachment formats instead of a RenderPass, and Begin/End record the corresponding core/KHR commands; the attachment cache creates only ImageViews. Otherwise, it uses traditional RenderPass/Framebuffer objects: render passes are cached by format and load/store operations, image views/framebuffers by ordered physical-image tuple. Load/store variants remain compatible with the same Program pipeline. Vulkan synchronizes vertex/index/uniform/shader reads and storage writes at scope boundaries so bindings can change inside a scope without splitting it. Swapchain teardown retires cached attachments after GPU idle, while completed contexts can finish returning their CPU leases. Dynamic rendering is not required.

Callers release child handles before parent Device/Core handles, and release old frame references/recordings before resize/rebuild boundaries. Device shutdown drains submitted work. Metal uses tracked private buffers/textures, shared upload leases and explicit render encoders. Bindings set outside a rendering scope are applied when the encoder begins; changes inside the scope are encoded immediately. Metal compiles Slang to DX-layout SPIR-V, validates the reflected layout, then translates through SPIRV-Cross to MSL during Program creation. This preserves compact structured-buffer layouts across backends.

## Compute programs and nbody

`nbody` is available on D3D12, Vulkan and Metal, through menu item 6, `--module nbody`, and `--all`. It follows `nbody_cs`: 65,536 particles, 128-thread shared-memory tiles, softened all-pairs gravity, velocity integration, position copy, and additive camera-facing particle quads. It uses ten initial galaxy clusters with seed 42, a fixed 0.03 simulation step, and the original gravity/radius/speed constants. There is no ImGui, HDR output or tone-mapping pass; particles render directly into SDR UNORM targets. Existing P/Escape/FPS/window handling and headless timing conventions apply. The nbody module creates all its own resources.

```text
demo_rhi_hello --module nbody --backend vulkan
demo_rhi_hello --module nbody --backend d3d12 --headless --frames 300
```

Create a `SHADER_STAGE_COMPUTE` Shader, then `CreateComputeProgram(ComputeProgramDesc)`. The descriptor captures the shader and PipelineLayout; creation compiles Slang with the same RHI resource macros and validates binding type/count/visibility, structured-buffer stride and thread-group size. `SHADER_VISIBILITY_COMPUTE` is bit 4; `SHADER_VISIBILITY_ALL` includes vertex, fragment and compute. Dispatch arguments count workgroups, not threads.

`CmdBindComputeProgram`, `CmdBindGroup`, then `CmdDispatch(x,y,z)` records native compute commands immediately. Graphics and compute share one active program selection and group-binding table: switching either Program requires rebinding groups. Compute binding/Dispatch must occur outside Begin/EndRendering. Group versions are retained at bind time; later BindResource does not change existing bindings. Barriers are inserted before each Dispatch, including repeated writes with no intervening rebind, and compute/copy/graphics transitions are automatic. Dispatch does not submit or wait on the CPU. D3D12 uses the existing transient descriptor heaps for both pipeline kinds; Vulkan uses compute descriptor visibility and bind points. Vulkan selects a queue family supporting both graphics and compute. The rhix package provides the corresponding ComputeProgram, creation and command wrappers; the nbody module itself uses the pointer API.

## Binding lifetime and descriptor pages

`CreateBindingGroup(layout)` allocates binding slots without initializing them. `BindResource(binding, resource)` writes one slot; typed vectors write arrays, with an optional `first_element` for partial updates. Buffers, images, and sampler objects are retained. Create immutable samplers with CreateSampler(SamplerDesc), then bind the resulting object. Use `BufferView{buffer, offset, size}` for a byte subrange; size zero selects the remaining buffer. `BINDING_STRUCTURED_BUFFER` and `BINDING_RW_STRUCTURED_BUFFER` require an explicit byte `BindingDesc::stride`; Program creation checks it against Slang reflection for each used shader binding, including binding arrays. Other binding types ignore stride. `BINDING_BYTE_ADDRESS_BUFFER` and its RW variant use raw byte views and the `RHI_BYTE_ADDRESS_BUFFER(G,B,N)` / `RHI_RW_BYTE_ADDRESS_BUFFER(G,B,N)` macros (also available with `_ARRAY`). Vulkan compiles with `-fvk-use-dx-layout` and requires scalarBlockLayout, so D3D12 and Vulkan share DX buffer packing instead of backend-specific element padding. C++ upload structures must match shader field offsets and matrix ordering; stride equality alone does not validate every field. Binding failures leave the previous contents unchanged. All layout slots must be initialized before `CmdBindGroup`.

`CmdBindGroup` captures the current binding version, including its resources and descriptor storage. Later `BindResource` calls affect only subsequent `CmdBindGroup` calls, even before Draw or Submit. Unreferenced versions are updated in place; a captured version is copied only when changed. Commands keep old versions alive until canceled or retired after GPU completion. Vulkan allocates a new descriptor set on a fork; D3D12 versions own CPU-only descriptor ranges. Commands retain these versions without copying CPU descriptors at each bind, then copy changed tables into shader-visible storage at Draw. Device-owned D3D12TransientDescriptorPool instances lease whole resource/sampler heaps in size classes starting at 32 descriptors. CommandContext requests at least twice its previous page capacity and enough room for all currently bound tables; the terminal class is capped at the native heap limit, where subsequent pages retain that size. On rollover all bound tables of that heap type migrate together, while old pages remain referenced until command retirement. Each class caches its recent one-second occupancy peak, using the same monotonic-queue policy as UploadPool; it trims surplus idle pages on return. CPU descriptor storage, RTV and DSV continue using the separate fragment allocator. D3D12 creates final structured/raw descriptors during BindResource, independently of Program; there is no reflected-stride cache. Barrier tracking reads the captured resources. Buffer, Image and Sampler writes stay typed through native descriptor creation; the common version stores only per-binding arrays of strong object references for lifetime and barrier tracking, without duplicating binding contents. Initialization state and native descriptor storage are backend-owned. Vulkan writes a contiguous binding-array update with one VkWriteDescriptorSet. D3D12 clones initialized CPU descriptors with CopyDescriptorsSimple; Vulkan clones initialized descriptor-set ranges with VkCopyDescriptorSet and shares native ImageView/Sampler ownership. These snapshots preserve bindings, not resource contents: updating pixels or buffer bytes still requires proper GPU ordering. CPU access to the same BindingGroup must be externally synchronized.

Uniform offsets must satisfy backend alignment (256 bytes on D3D12 and Metal); D3D12 allocations must include the rounded 256-byte CBV range. Structured buffer offset/range must be multiples of the Layout stride on all backends, checked during BindResource, even for unused bindings. Raw buffer ranges use four-byte alignment. Vulkan additionally enforces its native descriptor offset alignment. Layout compatibility includes structured stride, so different strides require different Layouts. Texture bindings accept independent color images; swapchain frames and depth sampling remain unsupported. Independent images automatically enable storage usage when their format supports it; RW texture bindings require that usage. Metal versions own shared argument buffers and a buffer-view size table for GetDimensions. They retain buffers, images and immutable ISampler objects through the common reference table. Each Metal sampler creates its argument-buffer-compatible native state once in CreateSampler; binding and cloning reuse it without recreating native samplers. Sparse cloning copies only descriptor storage and initialization state while retaining the exact referenced resources; updates fork captured versions. Dense argument IDs include unused layout entries, making storage compatible across stages and Programs. Vertex attributes use buffer slots 8–30; slots 0–7 are reserved for binding groups.

`CmdUploadImage` snapshots complete RGBA8/BGRA8 rows with the supplied CPU row pitch, leases upload storage and records a GPU copy. Metal copies rows directly into the shared staging lease, without an intermediate CPU pixel vector. It is allowed outside rendering scopes. Changing bindings within a scope is supported, but a resource must not simultaneously be an attachment and shader input, or be bound for conflicting read/write uses. Dependent storage accesses between draws require separate End/Begin scopes; this version does not add an in-scope barrier command. Bind texture groups outside a scope when transitioning from a different image use. Vulkan returns color targets to their default layout at EndRendering, permitting their use in a later scope.

D3D12 RTV, DSV, CBV/SRV/UAV and sampler pools allocate contiguous ranges from page-local rings. A range never straddles a page; wrap padding stays occupied until preceding leases return. Out-of-order release marks a range free, while the tail advances only through the released prefix. When a ring empties it resets; one normal idle page is cached, extra idle pages and oversized pages are destroyed. GPU resource pages use 4096 descriptors and sampler pages use 256; an oversized request gets its own page within native limits. Each CommandContext reserves shader-visible ranges and appends only changed tables. When one heap type runs out of space, all still-bound tables of that type move to a new range and both types of root tables are rebound. Earlier ranges remain alive through GPU completion. Unchanged tables are reused while capacity remains; CPU descriptor storage stays with the captured binding versions.

Vulkan groups own descriptor-set leases from pages of up to 64 sets, with fixed descriptor budgets. Allocation grows a new page on exhaustion/fragmentation. Final lease destruction frees the set; an empty page is reset, and only one idle page is cached. Group destruction also releases its views and samplers. Attachment views/framebuffers have their own leases. CommandContext holds groups and leases through its ordinary resource-use table; no separate descriptor return queue is needed. Submitted work keeps them alive until GPU completion and final command reference release; abandoning an unsubmitted context releases them too. Device/Core still must outlive their children.

## Vulkan rendering-path regression

With Vulkan SDK validation, API-dump and Profiles layers installed and a Vulkan 1.3+ GPU supporting dynamic rendering, run `python test/rhi/test_vulkan_rendering_paths.py <demo executable> [<second demo executable>]` from the repository root. With `--headless`, this runs all available modules; windowed mode runs the triangle. It checks native capabilities, a Vulkan 1.2 KHR profile, and a Vulkan 1.2 profile with dynamic rendering disabled. It verifies the actual native Begin/End commands and that dynamic rendering creates no RenderPass or Framebuffer objects. Profile simulation changes reported capabilities; it does not substitute for testing on older hardware.


Headless benchmarks (D3D12/Vulkan/Metal) render to independent 1280x720 RGBA8 images without creating a GLFW window, swapchain, or presenting. `window` measures its clear pass; `triangle` measures the same indexed triangle draw as the windowed scene. Each module independently creates its own Core, Device and resources.

```sh
demo_rhi_hello --headless --all --backend vulkan --frames 10000
demo_rhi_hello --headless --module triangle --backend d3d12 --frames 10000
```

`--all` runs all nine modules sequentially on the selected backend, continues after a failed module, and returns nonzero if any module fails. It cannot be combined with `--module` or `--tui`. Each scene prints its measured frame count, wall time and average FPS. Headless runs use 60 untimed warmup frames and at most 3 in-flight frames; target reuse and the final timing boundary wait for GPU completion. Setup, shader compilation, image initialization and initial uploads are outside the measurement. The cube's per-frame uniform update is included. This is end-to-end rendering throughput including CPU recording/submission and synchronization, not GPU timestamp duration. Use Release without validation layers for performance comparisons.

Without a positive `--frames`, headless runs measure 3000 frames per scene; windowed `--all` runs 300 frames per scene. Normal individual windowed demos remain interactive. `--all` selects scenes, not backends. Metal supports eight scenes in both windowed and headless mode; `raytracing` is skipped because Metal RT pipelines are not implemented.

## Regression tests

Build `test_rhi` and run `ctest --test-dir cmake-build-debug/test/rhi --output-on-failure`. Set `LONGMARCH_RHI_WINDOW_TESTS=1` to include window tests, `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` and `VK_LAYER_VALIDATE_SYNC=1` for Vulkan validation. CTest runs each GoogleTest case in its own process. This also isolates the backends: on the tested Windows NVIDIA driver, globally enabling D3D12's debug layer and then destroying an acquired Vulkan window in the same process triggers an internal D3D12 destruction error, even without recording commands.

Pixel regressions cover MRT plus depth rejection, depth-only stencil writes preserved into a color pass, fixed uniform ranges, texture/structured-buffer arrays, group changes across descriptor pages, release before submission, and storage image/buffer writes read in a subsequent scope. Buffer-view size queries also cover uniform/structured/raw arrays and mutation after capture. Metal tests can use `MTL_DEBUG_LAYER=1`; apply it only to Metal tests (`ctest -R '(/2$|RhiMetal)'`), separately from Vulkan tests. Pool tests cover contiguous ranges, wrap padding, out-of-order release, oversized pages, descriptor-set reuse and idle-page trimming.

## Blending

`ProgramDesc::blend_states` is empty by default (blending disabled), or contains exactly one `BlendState` per color attachment. `BlendState(true)` selects source-alpha RGB blending and source-over alpha accumulation, matching graphics_hello. The six-argument constructor configures independent RGB/alpha factors and operations. Supported factors are zero/one, source/destination color and alpha, and their inverses; operations are add, subtract, reverse subtract, min and max. Color factors in the alpha equation use their alpha component. States are compiled into the pipeline during Program creation; differing attachment states require Vulkan independentBlend support. Metal mappings are implemented but this change was verified on Windows D3D12/Vulkan only.

Run `--module blend --backend d3d12` (or `vulkan`), or add `--headless --frames 100000` for offscreen throughput. Both `--list`, the interactive menu and `--all` include blend. Each module owns its device/window/resource setup independently.

Nbody reports FPS and estimated TFLOP/s in the window title and headless output. Following `nbody_cs`, each particle interaction counts as 20 FLOP: `FLOP/frame = 20 * 65536^2 = 85,899,345,920`, and `TFLOP/s = FLOP/frame * FPS / 1e12`. This is an algorithmic estimate using complete frame time (simulation, copy and rendering), not hardware instruction counts or isolated compute timing. Headless output also reports FLOP/frame.

Backend layout ownership is shared by graphics and compute programs through D3D12ProgramBase (root signature and root-parameter mapping) and VulkanProgramBase (descriptor-set layouts and pipeline layout). These internal bases have no IObject identity or reference counting; concrete programs retain responsibility for shader compilation and pipeline creation.

Metal graphics and compute programs share `MetalProgramBase` argument-buffer layout validation and Slang/SPIR-V-to-MSL compilation. Compute threadgroup dimensions come from Slang reflection and are checked against device/pipeline limits. Each Dispatch uses a separate serial compute encoder; tracked resource residency synchronizes dependent dispatches, copies and rendering, including indirect argument-buffer resources. Switching programs requires rebinding groups.

## Acceleration structures and ray queries (D3D12 / Vulkan / Metal)

`raytracing` uses a RayTracingProgram with ray-generation, miss, triangle closest-hit,
procedural intersection, sphere closest-hit and callable shaders, matching
`graphics_hello/rt_multi_shader_group`. `ray_query` matches `graphics_hello/ray_query`
using inline queries from a ComputeProgram. Both use the reference 60-degree camera,
triangle at x = -2, scaled ellipsoid at x = 2, two-second Y-axis rotation, front/back
face coloring, blue-gray sphere lighting and tan background. They compact the
triangle BLAS once and refit the TLAS each frame. The RHI output uses RGBA8 UNORM
(the current supported color format) rather than the reference float image; all
reference shader colors are in [0, 1]. Each module
owns its own setup, output image and fullscreen display pass. Both APIs expose these
modules through the terminal menu, `--module`, `--all` and `--headless`. An unavailable
ray-tracing feature is reported as `SKIPPED`. Metal supports `ray_query`, including triangle and procedural AABB intersections; `raytracing` requires D3D12 or Vulkan.

```sh
demo_rhi_hello --module raytracing --backend vulkan
demo_rhix_hello --headless --module ray_query --backend d3d12 --frames 3000
```

`IBottomLevelAccelerationStructure` and `ITopLevelAccelerationStructure` derive from
`IAccelerationStructure`. Creation allocates fixed result storage; it does not build
anything. A BLAS contains either triangle geometries (float3 positions, optional
16/32-bit indices) or AABB geometries (float3 minimum followed by float3 maximum).
A TLAS has a fixed allocation capacity; BUILD accepts an actual instance count up to that capacity. Instance transforms are row-major 3x4 matrices.
The instance custom index and hit-group offset are 24-bit values, and masks are 8-bit.

`CmdBuildBottomLevelAccelerationStructure` and `CmdBuildTopLevelAccelerationStructure`
record native Build or Update immediately, outside rendering scopes. Update requires
`allow_update` at creation and an earlier Build ordered before the update. Geometry type/count,
primitive count, vertex format/stride, index format/topology, opacity and active versus
inactive primitives must remain unchanged during Update. Positions, bounds, instance
transforms/masks and BLAS references may change. Changing a TLAS instance count requires BUILD; exceeding its capacity or changing BLAS geometry counts requires a new AS.
No implicit resizing or CPU wait occurs. Calls accessing the same AS are externally
synchronized; submit its modifying contexts in recording order before recording later
uses in another context. CommandContext inserts build/read barriers automatically.

Scratch storage is shared on a single physical queue, with barriers between builds.
Each Device caches one buffer, grows it in power-of-two sizes (at least 64 KiB), and
shrinks it on acquisition when the last second's peak request is below half its capacity.
Replaced buffers survive through references held by recorded or submitted commands.
Build commands retain input buffers, scratch buffers and exact BLAS references until GPU completion and final
context release. TLAS owns its submitted BLAS dependencies; replacing these dependencies
happens on submission, so abandoning a recording leaves the previous TLAS intact.
Resources only point weakly to their Device. Destroy Device/Core after all children.

Metal records native AS encoders, converts instances to user-ID descriptors, and uses tracked scratch/AS resources. Argument-buffer bindings and their indirect BLAS resources are declared for compute and render encoders. Compute ray queries require `supportsRaytracing`; queries from graphics shaders additionally require native `supportsRaytracingFromRender`. Metal RT pipelines and SBTs are not implemented.

Compaction is explicit: create with `allow_compaction`, Build, record
`CmdQueryCompactedSize`, submit and wait for its token, then call `GetCompactedSize`.
Copy the original description, set `compacted_size` to that result, create a new AS and
record `CmdCompactAccelerationStructure(source, destination)`. The destination must
use the same shape/counts/build flags as the source and sufficient queried capacity.
Its physical identity stays fixed. A compact destination accepts compact copies and
Update when enabled, but cannot receive an ordinary full Build. Rebuild TLAS references
when replacing a BLAS with its compact copy. Changing the contents of a BLAS also
requires rebuilding or updating every TLAS that uses it before tracing. `GetCompactedSize` never waits and is only
valid after its query submission completes. Serialization is outside this implementation.

Bind TLAS through `BINDING_ACCELERATION_STRUCTURE`, `BindResource(binding, tlas)` and
`RHI_ACCELERATION_STRUCTURE(group, binding, name)` (or its `_ARRAY` variant). It uses the
same captured-binding lifetime contract as other resources. RayTracingProgram captures
original Shader sources and creates its native pipeline and default shader binding table before returning.
ProgramBase supplies the native global binding layout. Miss, hit-group and callable
record indices follow their descriptor vectors; a hit group with an intersection shader
is procedural. There are no local root parameters or user-defined SBT record data.
`max_payload_size` and `max_attribute_size` set D3D12 pipeline configuration; Vulkan
reflects shader payloads and checks the native hit-attribute limit. Bind with
`CmdBindRayTracingProgram`, bind groups, then `CmdDispatchRays(width, height, depth)`.
Slang uses SM 6.5 for ray pipelines and shaders with AS bindings, preserving SM 6.0 for
existing shaders and the shared DX-compatible Vulkan layout.

The standalone [TLAS pressure test](../tlas_pressure_test/README.md) is built as
`demo_tlas_pressure_test` and is not included in this demo's module list or `--all`.

## HDR display

```text
demo_rhi_hello --module hdr --backend d3d12
demo_rhi_hello --module hdr --backend vulkan
demo_rhi_hello --module hdr --backend metal
demo_rhi_hello --module hdr --backend vulkan --headless --frames 12
```

`hdr` uses nominal 20, 80, 203, 400 and 1000 nit patches. For the PQ comparison, linear 1.0 maps to 100 nits instead of 80, producing PQ targets of 25, 100, 253.75, 500 and 1250 nits, with neutral patches above Rec.709 red/green/blue patches. H cycles the surface's available color spaces; Escape closes the window. Bounded window runs of at least nine frames also cycle automatically. The demo starts in HDR when a supported HDR encoding is available; otherwise it explicitly reports SDR. Headless runs exercise SDR, floating-point scRGB and 10-bit PQ rendering without claiming HDR display validation. This diagnostic waits after each frame and is not a throughput benchmark. Metal also runs this module in windowed and headless modes.

`WindowDesc::color_space` selects the initial encoding. `GetSupportedPresentationOptions()` reports `(present_mode, format, color_space)` combinations; `SetColorSpace()` explicitly rebuilds the swapchain, and `GetColorSpace()` reports the last successfully configured drawable swapchain. Release caller-held frames and recordings before switching, just as for resize/rebuild. Unsupported requests fail without silently selecting SDR or replacing the current frame; native failures can invalidate the swapchain. When minimized, a successful request may remain pending until restoration. A shader/pipeline must match the acquired image format and active color space.

- `COLOR_SPACE_SRGB_NONLINEAR`: SDR sRGB encoded output in RGBA8/BGRA8 UNORM.
- `COLOR_SPACE_EXTENDED_SRGB_LINEAR`: linear Rec.709/scRGB in RGBA16 float, where 1.0 represents 80 cd/m^2. Values above 1.0 preserve highlights.
- `COLOR_SPACE_HDR10_ST2084`: Rec.2020 primaries and ST.2084/PQ absolute luminance in RGB10A2 UNORM (also BGR10A2 on Vulkan).

Present performs no implicit transfer-function conversion, gamut conversion or tone mapping. The demo implements the sRGB encoding, Rec.709-to-Rec.2020 conversion and PQ encoding in its fragment shader. `CmdUploadImage` accepts native packed pixels and byte row pitch; RGBA16 float uses eight bytes per pixel, 10-bit formats use four.

D3D12 HDR options require Windows HDR enabled on the monitor containing the window, and native color-space support is checked again when configuring the swapchain. Vulkan enables `VK_EXT_swapchain_colorspace` when available and uses advertised surface format/color-space pairs. Surface support does not prove the physical display is currently emitting HDR luminance; the compositor and OS HDR settings still matter. Neither HDR metadata nor OS HDR settings are changed by the demo. Resize and explicit settings operations rebuild outside command recording; submission failures do not trigger implicit reconstruction.

Metal maps SDR to BGRA8Unorm / sRGB, linear HDR to RGBA16Float / extended linear sRGB, and PQ to RGB10A2Unorm / ITU-R BT.2100 PQ. HDR enables CAMetalLayer EDR content; returning to SDR disables it. Core Animation performs display color matching. The layer accepts these encodings on SDR displays too, so enumeration does not assert HDR headroom. macOS maps extended-linear values relative to its current reference white; the demo's nominal 80-nit scale is not a physical luminance calibration or a guarantee that scRGB and PQ patches match on every display. No mastering metadata is supplied.

Independent Metal images also support BGR10A2Unorm; RGBA16Float supports storage writes. HDR upload/readback tests cover all three new formats, including padded rows and highlights above 1.0. Shader source files use UTF-8 without BOM to permit source concatenation.
