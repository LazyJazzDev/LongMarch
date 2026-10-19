# rhi_hello

Two independent modules use `rhi::ObjPtr<IXXX>` pointer APIs. Each module creates its own Core, Device, Window and Queue; main only selects the module/backend and parses arguments.

```text
demo_rhi_hello --list
demo_rhi_hello --module hello_window --backend vulkan
demo_rhi_hello --module hello_triangle --backend d3d12 --frames 120
```

The positional backend argument remains supported. Without `--module`, a line-based terminal menu opens (also available with `--tui`). Enter a module number or name; invalid input retries, and `q`, `quit`, or EOF exits without opening a window. `--tui` and `--module` are mutually exclusive; `--backend auto` uses the default backend. Specify `--module` for unattended runs; omit `--frames` to run until the window closes. P switches requested VSync mode, Escape exits, and the title displays `grassland::FPSCounter` statistics.

- `hello_window`: original RGB (0.6, 0.7, 0.8) clear/presentation and supported presentation-combination output. D3D12, Vulkan and the existing Metal backend are supported.
- `hello_triangle`: D3D12/Vulkan colored indexed triangle on that background. The module creates independent GPU-only vertex/index Buffers, uploads once through CommandContext, finalizes a Slang Program, then records bindings, viewport/scissor and DrawIndexed. Resize/minimize/restore and P-mode changes are supported.

## Shader and Program contract

Shader stores original Slang source, entry point and stage; creation does not compile it. Program captures shaders and the C++ vertex layout, owns their references, and compiles/creates the native pipeline at Finalize. After successful finalization it is immutable; repeated Finalize is a no-op.

`RHI_INPUT(location, type, name)` is expanded by a generated prefix into the C++ layout's Vulkan locations or D3D12 input semantics. Source BOM is preserved by Shader and omitted only when concatenating the compilation input. Resource-binding layouts/groups and resource macros are deferred: a shader requiring global resource parameters is rejected at Finalize. This iteration supports triangle-list topology, one UNORM color target, no depth, culling or blending, and float/float2/float3/float4 vertex attributes.

## Buffer and command lifetime

CreateBuffer returns a fixed GPU-only allocation managed by VMA or D3D12MA. Backend Buffer bases expose resource access; AllocatedBuffer subclasses own allocations. UploadBufferLease subclasses forward access to pool-owned upload allocations; custom GPU-only resource pools can later implement the same interfaces without changing the public Buffer type. Each Device owns its allocator, upload pool and attachment pools; children borrow their Device.

CmdUploadBuffer copies CPU bytes immediately into an automatically leased upload Buffer, then records a native GPU copy. The caller can release/reuse the CPU data after the call. CmdCopyBuffer copies between distinct Buffers. UploadPool owns the physical AllocatedBuffers and returns separate backend Buffer-derived UploadBufferLease objects. Leases forward the Buffer interfaces and mapped Write to the underlying storage, borrow their pool, and return occupancy on destruction of their last reference. CommandContext retains each lease through its ordinary resource table, with no separate uploads list or manual return path. Storage is recycled after GPU completion and final command reference release, or when an unsubmitted context is abandoned. Upload storage is grouped into independent power-of-two size classes starting at 64 KiB. Acquire takes an idle buffer directly from its class. Each class tracks allocated/occupied counts and a monotonic queue of occupancy changes over the last second; Release discards idle buffers above that recent peak. Equal occupancy samples keep the latest timestamp. Trimming occurs on returns to that class, without a background sweep. There is no per-frame reset or CPU wait per upload. Callers still order overwrites against prior GPU readers using queue ordering/dependencies.

CmdBindProgram precedes vertex-buffer binding. Bind all required vertex slots again after changing Program. CmdBeginRendering selects one color image and its LOAD/CLEAR/DISCARD and STORE/DISCARD operations; CmdEndRendering explicitly closes the scope. Draw calls require an active scope. Upload, copy, standalone CmdClearColor, Present and submission require no active scope. A scope may contain no draws (for example, a clear-only pass). RTV/framebuffer leases remain backend details. CmdDraw and CmdDrawIndexed support instancing and base offsets. Contexts retain the exact Program, Buffer and Image objects they use. Barriers are recorded immediately and used resources return to default states when submission ends recording.

D3D12 leases RTV slots from Device-owned heap pages. Vulkan prefers dynamic rendering when the negotiated Vulkan 1.3 core feature or Vulkan 1.2 VK_KHR_dynamic_rendering feature is supported. Program uses attachment formats instead of a RenderPass, and Begin/End record the corresponding core/KHR commands; the attachment cache creates only ImageViews. Otherwise, it uses traditional RenderPass/Framebuffer objects: render passes are cached by format and load/store operations, image views/framebuffers by physical image. Load/store variants remain compatible with the same Program pipeline. Vulkan synchronizes vertex/index buffer reads at scope boundaries so bindings can change inside a scope without splitting it. Swapchain teardown retires cached attachments after GPU idle, while completed contexts can finish returning their CPU leases. Dynamic rendering is not required.

Callers release child handles before parent Device/Core handles, and release old frame references/recordings before resize/rebuild boundaries. Device shutdown drains submitted work. Metal retains window functionality; these new drawing/Buffer operations currently report unsupported.

## Vulkan rendering-path regression

With Vulkan SDK validation, API-dump and Profiles layers installed and a Vulkan 1.3+ GPU supporting dynamic rendering, run `python test/rhi/test_vulkan_rendering_paths.py <demo executable> [<second demo executable>]` from the repository root. This runs the triangle under native capabilities, a Vulkan 1.2 KHR profile, and a Vulkan 1.2 profile with dynamic rendering disabled. It verifies the actual native Begin/End commands and that dynamic rendering creates no RenderPass or Framebuffer objects. Profile simulation changes reported capabilities; it does not substitute for testing on older hardware.
