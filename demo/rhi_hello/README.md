# rhi_hello

Two independent modules use `rhi::ObjPtr<IXXX>` pointer APIs. Each module creates its own Core, Device, Window and Queue; main only selects the module/backend and parses arguments.

```text
demo_rhi_hello --list
demo_rhi_hello --module hello_window --backend vulkan
demo_rhi_hello --module hello_triangle --backend d3d12 --frames 120
```

The positional backend argument remains supported. The default module is `hello_window`; omit `--frames` to run until the window closes. P switches requested VSync mode, Escape exits, and the title displays `grassland::FPSCounter` statistics.

- `hello_window`: original RGB (0.6, 0.7, 0.8) clear/presentation and supported presentation-combination output. D3D12, Vulkan and the existing Metal backend are supported.
- `hello_triangle`: D3D12/Vulkan colored indexed triangle on that background. The module creates independent GPU-only vertex/index Buffers, uploads once through CommandContext, finalizes a Slang Program, then records bindings, viewport/scissor and DrawIndexed. Resize/minimize/restore and P-mode changes are supported.

## Shader and Program contract

Shader stores original Slang source, entry point and stage; creation does not compile it. Program captures shaders and the C++ vertex layout, owns their references, and compiles/creates the native pipeline at Finalize. After successful finalization it is immutable; repeated Finalize is a no-op.

`RHI_INPUT(location, type, name)` is expanded by a generated prefix into the C++ layout's Vulkan locations or D3D12 input semantics. Source BOM is preserved by Shader and omitted only when concatenating the compilation input. Resource-binding layouts/groups and resource macros are deferred: a shader requiring global resource parameters is rejected at Finalize. This iteration supports triangle-list topology, one UNORM color target, no depth, culling or blending, and float/float2/float3/float4 vertex attributes.

## Buffer and command lifetime

CreateBuffer returns a fixed GPU-only allocation managed by VMA or D3D12MA. Backend Buffer bases expose resource access; AllocatedBuffer subclasses own allocations. Future pool-backed subclasses can implement the same interfaces without changing the public Buffer type. Each Device owns its allocator, upload pool and attachment pools; children borrow their Device.

CmdUploadBuffer copies CPU bytes immediately into an automatically leased upload Buffer, then records a native GPU copy. The caller can release/reuse the CPU data after the call. CmdCopyBuffer copies between distinct Buffers. Upload storage is recycled after GPU completion and final CommandContext release, or immediately when an unsubmitted context is abandoned. There is no per-frame reset or CPU wait per upload. Callers still order overwrites against prior GPU readers using queue ordering/dependencies.

CmdBindProgram precedes vertex-buffer binding. Bind all required vertex slots again after changing Program. CmdBindRenderTarget selects one image; render-pass scopes and RTV/framebuffer leases are backend details. CmdDraw and CmdDrawIndexed support instancing and base offsets. Contexts retain the exact Program, Buffer and Image objects they use. Barriers are recorded immediately and used resources return to default states when submission ends recording.

D3D12 leases RTV slots from Device-owned heap pages. Vulkan uses ordinary RenderPass/Framebuffer objects, preserving the Vulkan 1.2 baseline; render passes are cached by format, image views/framebuffers by physical image. Swapchain teardown retires cached attachments after GPU idle, while completed contexts can finish returning their CPU leases. Dynamic rendering is not required.

Callers release child handles before parent Device/Core handles, and release old frame references/recordings before resize/rebuild boundaries. Device shutdown drains submitted work. Metal retains window functionality; these new drawing/Buffer operations currently report unsupported.
