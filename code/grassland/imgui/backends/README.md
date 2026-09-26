# ImGui renderer backend adaptations

These renderer sources are based on Dear ImGui commit
`5f0acadf7db1b6d469f0771284e2e3f7818443ce`, matching `external/imgui`.
The upstream MIT license is included in LICENSE.txt. The ImGui core, public
backend headers and GLFW backend still come from that pinned submodule.

CMake compiles these files directly. It does not generate, patch or rewrite C++.

The local changes are limited to vertex upload:

- Include grassland/imgui/vertex_upload.h.
- Use GPUVertex for GPU buffer sizes, offsets and strides, leaving ImDrawVert unchanged.
- Call UploadVertices instead of copying packed ImGui vertices to GPU buffers.
- Use float4 color input formats in D3D12, Vulkan and Metal.
- Metal consumes normalized float color directly, without dividing it by 255.

The shared upload helper chooses normalized sRGB or linear RGB through the
scoped ImGuiLinearColors state. Alpha is unchanged and original draw lists are
never modified. Rendering shaders, textures, descriptors and synchronization
otherwise retain the upstream implementation.

When upgrading external/imgui, explicitly update these sources and compare
against the matching upstream backends. Preserve upstream formatting to keep
that comparison focused. Run the HDR vertex-upload and GPU presentation tests
on the supported backends; Metal requires a native macOS build and runtime check.
